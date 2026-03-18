# Chidori GPU-over-IP: Full System Architecture Trace

Complete code-path trace from server startup through client connection to GPU execution.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Wire Protocol](#2-wire-protocol)
3. [Network I/O Primitives](#3-network-io-primitives)
4. [Handle Mapping (Server-Side)](#4-handle-mapping-server-side)
5. [Server Startup and Dispatch Loop](#5-server-startup-and-dispatch-loop)
6. [Client Library: LD_PRELOAD Interception](#6-client-library-ld_preload-interception)
7. [Connection Establishment](#7-connection-establishment)
8. [RPC Mechanics: Async vs Sync](#8-rpc-mechanics-async-vs-sync)
9. [TCP_CORK Batching](#9-tcp_cork-batching)
10. [Server-Side CUDA Dispatch](#10-server-side-cuda-dispatch)
11. [Representative Handler Traces](#11-representative-handler-traces)
12. [Client-Side Caching](#12-client-side-caching)
13. [Environment Variables](#13-environment-variables)
14. [End-to-End Call Chain Examples](#14-end-to-end-call-chain-examples)

---

## 1. System Overview

Chidori is a CUDA-over-IP system. An application on a client machine (which has no GPU) makes standard CUDA API calls. Those calls are intercepted by a shared library (`libchidori.so`) loaded via `LD_PRELOAD`, serialized into a binary wire protocol, sent over TCP to a server (`chidori-worker`) running on a machine with a real GPU, executed on the physical hardware, and results returned.

**Key files:**

| File | Role | Lines |
|------|------|-------|
| `include/chidori_protocol.h` | Wire protocol definitions (headers, func IDs, flags) | 325 |
| `src/protocol/chidori_net.c` | Network I/O primitives (read_full, write_full, send/recv) | 83 |
| `src/server/chidori_worker.c` | Server main(), socket accept, dispatch loop | 385 |
| `src/server/cuda_dispatch.c` | Server-side CUDA execution (3144 lines, giant switch statement) | 3144 |
| `src/server/handle_map.h` | Wire-handle to real-handle mapping | 52 |
| `src/client/chidori_client.c` | Client interception library (5001 lines of CUDA API shims) | 5001 |

---

## 2. Wire Protocol

**File:** `include/chidori_protocol.h`

### Request Header (16 bytes, line 41-47)

```
Offset  Size  Field
0       4     msg_len        Total message length including header
4       2     func_id        CUDA function identifier (chidori_func_id_t enum)
6       2     flags          CHIDORI_FLAG_* bitmask
8       4     seq_num        Sequence number for request/response matching
12      4     payload_size   Size of payload following this header
```

The struct is `__attribute__((packed))` -- no padding.

### Response Header (12 bytes, line 52-57)

```
Offset  Size  Field
0       4     msg_len        Total message length including header
4       4     seq_num        Matching request sequence number
8       4     result         CUresult / cudaError_t return value
```

Followed by optional payload bytes.

### Message Flags (lines 21-24)

| Flag | Value | Meaning |
|------|-------|---------|
| `CHIDORI_FLAG_SYNC` | 0x0001 | Client blocks for response |
| `CHIDORI_FLAG_ASYNC` | 0x0002 | No immediate response needed |
| `CHIDORI_FLAG_BATCH_MORE` | 0x0004 | More messages follow (reserved) |
| `CHIDORI_FLAG_ERROR` | 0x0008 | Response indicates error (reserved) |

### Function ID Ranges (lines 66-296)

| Range | Category | Example IDs |
|-------|----------|-------------|
| 0x0001-0x000F | Initialization | CU_INIT, CU_DRIVER_GET_VERSION |
| 0x0010-0x002F | Device Management | CU_DEVICE_GET, CU_DEVICE_GET_ATTRIBUTE |
| 0x0030-0x004F | Context Management | CU_CTX_CREATE, CU_CTX_SYNCHRONIZE |
| 0x0050-0x006F | Module Management | CU_MODULE_LOAD_DATA, CU_MODULE_GET_FUNCTION |
| 0x0070-0x009F | Memory Management | CU_MEM_ALLOC, CU_MEMCPY_H_TO_D |
| 0x00A0-0x00BF | Execution Control | CU_LAUNCH_KERNEL |
| 0x00C0-0x00DF | Stream Management | CU_STREAM_CREATE, CU_STREAM_SYNCHRONIZE |
| 0x00E0-0x00FF | Event Management | CU_EVENT_CREATE, CU_EVENT_RECORD |
| 0x0100-0x01FF | CUDA Runtime API | CUDA_MALLOC, CUDA_MEMCPY, CUDA_DEVICE_SYNCHRONIZE |
| 0x0200-0x02FF | cuBLAS | CUBLAS_CREATE, CUBLAS_SGEMM, CUBLAS_GEMM_EX |
| 0x0300-0x03FF | cuDNN | CUDNN_CREATE, CUDNN_CONVOLUTION_FORWARD |

Both sides assume **native x86_64 byte order** (no endian conversion). This is stated in the header comment at line 5.

---

## 3. Network I/O Primitives

**File:** `src/protocol/chidori_net.c` (83 lines)

### `chidori_read_full(fd, buf, len)` -- line 12

Blocking read loop. Reads exactly `len` bytes, retrying on `EINTR`. Returns 0 on success, -1 on EOF/error.

### `chidori_write_full(fd, buf, len)` -- line 26

Blocking write loop. Writes exactly `len` bytes, retrying on `EINTR`. Returns 0 on success, -1 on error.

### `chidori_send_request(fd, hdr, payload, payload_len)` -- line 40

1. Sets `hdr->payload_size = payload_len`
2. Sets `hdr->msg_len = HEADER_SIZE + payload_len`
3. Writes the 16-byte header
4. Writes the payload (if non-empty)

Two `write_full` calls per request.

### `chidori_recv_request(fd, hdr)` -- line 55

Single `read_full` of 16 bytes into the header struct.

### `chidori_send_response(fd, seq_num, result, payload, payload_len)` -- line 59

**Critical detail (line 62):** If `fd == -1`, the function returns immediately. This is how async requests are handled on the server -- the dispatch loop passes `fd = -1` for async-flagged requests, so no response is ever sent.

1. Builds a 12-byte response header
2. Writes header
3. Writes payload (if non-empty)

### `chidori_recv_response(fd, hdr)` -- line 80

Single `read_full` of 12 bytes into the response header struct.

---

## 4. Handle Mapping (Server-Side)

**File:** `src/server/handle_map.h` (52 lines)

Simple fixed-size array-based map (max 8192 entries, with overflow logging). Maps wire protocol `uint64_t` handles to real CUDA pointers/handles.

```c
typedef struct {
    uint64_t keys[HANDLE_MAP_MAX];  // Wire handles
    uint64_t vals[HANDLE_MAP_MAX];  // Real CUDA pointers cast to uint64_t
    int count;
} handle_map_t;
```

- `hmap_put`: Append-only O(1) insert (line 24)
- `hmap_get`: Linear scan O(n) lookup (line 33)
- `hmap_remove`: Swap-with-last O(n) remove (line 41)

The server maintains separate handle maps for each CUDA object type (`cuda_dispatch.c` lines 35-56):
- `g_modules`, `g_functions`, `g_streams`, `g_events` (Driver API)
- `g_rt_streams`, `g_rt_events` (Runtime API)
- `g_cublas_handles`
- `g_cudnn_handles`, `g_cudnn_tensor_descs`, `g_cudnn_filter_descs`, `g_cudnn_conv_descs`, `g_cudnn_activation_descs`, `g_cudnn_pooling_descs`, `g_cudnn_dropout_descs`, `g_cudnn_rnn_descs`
- `g_ctx_to_dev` (context handle to device ordinal)

Wire handles are allocated sequentially starting from `0x10000` via `alloc_handle()` (line 60-62).

---

## 5. Server Startup and Dispatch Loop

**File:** `src/server/chidori_worker.c` (385 lines)

### Two Operating Modes

#### Mode 1: Agent-mediated (default) -- `main()` at line 308

Command line: `chidori-worker --id <session_id> --ipc_write <fd> --ipc_read <fd> --pcibus <pci>`

1. **`parse_args()`** (line 41): Parses CLI options. Hardcodes `ipc_write_fd = 3`, `ipc_read_fd = 4` (line 72-73) because Go's `exec.Cmd.ExtraFiles` always maps to fd 3, 4 in the child process.

2. **Socket reception loop** (line 332): Waits for client sockets forwarded by the Go-based Juice Agent.

3. **`receive_socket_from_agent()`** (line 79): Uses `recvmsg()` with `SCM_RIGHTS` to receive a file descriptor over a Unix domain socket IPC channel. The agent hijacks the client's TCP connection and passes the raw fd to the worker.

4. **Clear O_NONBLOCK** (line 347-353): The Go agent sets sockets non-blocking (required by Go's netpoller). The worker must clear this flag for its blocking I/O to work.

5. **IPC handshake** (line 356-367): Writes a `1` byte to `ipc_write_fd` signaling readiness, then reads a confirmation byte from the agent.

6. **Lazy CUDA init** (line 370-377): `cuda_dispatch_init()` is called only on the first client connection. This avoids blocking the agent during `cuInit()` which can take several seconds.

7. **Enter dispatch loop** (line 379): `dispatch_loop(client_fd)` for each connection.

#### Mode 2: Standalone TCP (--standalone) -- `standalone_mode()` at line 265

1. Calls `cuda_dispatch_init()` immediately (line 268)
2. Creates TCP listen socket on specified port (default 43211)
3. Sets `SO_REUSEADDR` (line 277)
4. Sets `TCP_NODELAY` on accepted connections (line 296)
5. Calls `dispatch_loop()` for each accepted client
6. Sequential: one client at a time

### The Batch-Read Dispatch Loop -- `dispatch_loop()` at line 122

This is the performance-critical server loop. Instead of two blocking reads per request (header + payload), it uses a single large `recv()` to pull ALL available TCP data into a 256KB buffer, then parses and dispatches every complete request in a tight inner loop.

**Buffer:** 256KB static array (line 132-133)
```c
#define BATCH_BUF_SIZE (256 * 1024)
char buf[BATCH_BUF_SIZE];
```

**Main loop structure (starting at line 142):**

```
Step 1: recv() all available data into buf (line 144)
Step 2: Inner parse loop (line 157):
    For each complete request in buffer:
        - Cast buf+offset to chidori_request_t*
        - If entire message fits in buffer (COMMON PATH, line 161):
            Zero-copy dispatch -- payload pointer is directly into buf
        - If message exceeds 256KB (OVERSIZED PATH, line 183):
            Allocate big_payload, copy partial data, read_full() the rest
        - Otherwise: partial message, break and wait for more data
Step 3: Compact buffer -- memmove remaining bytes to front (line 238)
```

**Async handling in dispatch loop (line 174):**
```c
int dispatch_fd = (req->flags & CHIDORI_FLAG_ASYNC) ? -1 : client_fd;
```
For async requests, the dispatch function receives `fd = -1`, so `chidori_send_response()` at the bottom of each handler will skip the response write entirely.

---

## 6. Client Library: LD_PRELOAD Interception

**File:** `src/client/chidori_client.c` (5001 lines)

The library is compiled as `libchidori.so` and loaded via:
```
LD_PRELOAD=libchidori.so ./my_cuda_app
```

### Interception Mechanism

The library provides public symbol definitions for all intercepted CUDA functions. Because `LD_PRELOAD` libraries are searched before the real CUDA libraries in the dynamic linker's search order, calls to `cuInit()`, `cudaMalloc()`, `cublasSgemm_v2()`, etc. resolve to Chidori's versions.

Each intercepted function:
1. Serializes arguments into a packed struct (the wire payload)
2. Calls either `chidori_rpc()` (sync) or `chidori_rpc_async()` (async)
3. Deserializes the response payload into output parameters

### Global State (lines 46-49)

```c
static int g_server_fd = -1;           // TCP socket to server
static pthread_mutex_t g_lock;         // Global mutex for all RPC
static uint32_t g_seq_num = 0;         // Monotonic sequence counter
static int g_initialized = 0;          // cuInit called?
```

### API Coverage

The client intercepts:
- **Driver API** (lines 547-965): `cuInit`, `cuDeviceGet*`, `cuCtx*`, `cuModule*`, `cuMemAlloc`, `cuMemcpy*`, `cuLaunchKernel`, `cuStream*`, `cuEvent*`
- **Runtime API** (lines 967-1356): `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaStream*`, `cudaEvent*`, `cudaGetDeviceProperties`
- **cuBLAS** (lines 1358-2100+): `cublasCreate`, `cublasSgemm`, `cublasGemmEx`, `cublasSgemv`, `cublasSaxpy`, `cublasSnrm2`, `cublasSdot`, `cublasSscal`, `cublasScopy`, batched GEMM variants
- **cuDNN** (lines 2100+): `cudnnCreate`, tensor/filter/convolution/activation/pooling descriptors, convolution forward/backward, batch norm, softmax
- **cublasLt stubs** (lines 4700-4866): Lightweight matmul desc stubs returning SUCCESS (no RPC)
- **NVRTC stubs** (lines 4910-4990): Compile stubs returning SUCCESS (no RPC)
- **Destructor** (line 4995): `chidori_cleanup()` closes the socket on library unload

---

## 7. Connection Establishment

**File:** `src/client/chidori_client.c`, `chidori_connect()` at line 153

### Step 1: Parse Config

Reads `CHIDORI_CFG_OVERRIDE` env var (line 157). Expected format:
```json
{"id":"session-uuid","servers":["host:port"]}
```

Minimal JSON parsing via `json_get_string()` (line 60) and `json_get_array_first()` (line 75).

### Step 2: Create TCP Socket -- `tcp_connect()` at line 90

1. `getaddrinfo()` for DNS resolution (line 98)
2. `socket()` creation (line 104)
3. **`TCP_NODELAY` set immediately** (line 112): Disables Nagle's algorithm
4. `connect()` to server (line 114)

### Step 3: Fd Duplication to High Numbers (lines 123-135)

```c
int high_fd = dup(fd);       // First dup
close(fd);
fd = high_fd;
int higher = fcntl(fd, F_DUPFD, 500);  // Try to get fd >= 500
close(fd);
fd = higher;
```

**Why:** PyTorch and CUDA runtime internals aggressively close low file descriptors (3, 4, 5). By moving the socket to fd 500+, the connection survives runtime initialization.

### Step 4: TCP Keepalive (lines 139-148)

```c
SO_KEEPALIVE = 1
TCP_KEEPIDLE = 10 seconds    // Start probing after 10s idle
TCP_KEEPINTVL = 5 seconds    // Probe every 5s
TCP_KEEPCNT = 3              // Give up after 3 failed probes
```

### Step 5: Connection Mode Selection (line 190-305)

**Direct connect** (`CHIDORI_DIRECT_CONNECT=1`): Simply TCP connect to the worker's standalone port. No protocol upgrade.

**Agent-mediated** (default): After TCP connect to the agent:
1. Send HTTP POST to `/v1/connect/session/{session_id}` (lines 220-228)
2. Read HTTP 101 Switching Protocols response byte-by-byte looking for `\r\n\r\n` (lines 249-291)
3. Set 30-second `SO_RCVTIMEO` during the HTTP handshake, then clear it (lines 251, 280-282)
4. After the 101 response, the raw TCP stream carries binary Chidori protocol

---

## 8. RPC Mechanics: Async vs Sync

### Async RPC -- `chidori_rpc_async()` at line 366

Used for fire-and-forget operations where the client does not need the result immediately.

Call chain:
1. Check `CHIDORI_TRACE` / `CHIDORI_VERBOSE` env vars
2. Lock `g_lock` mutex
3. Lazy-connect if needed (line 384)
4. Enable TCP_CORK if not already active (line 391-397)
5. Build request header with `CHIDORI_FLAG_ASYNC` (line 401)
6. `chidori_send_request()` -- sends header + payload
7. Unlock mutex
8. **Does NOT read any response**

**Operations that use async:** `cuMemcpyHtoD`, `cuMemcpyDtoD`, `cuMemFree`, `cuCtxSetCurrent`, `cuStreamDestroy`, `cuEventRecord`, `cuEventDestroy`, `cuLaunchKernel`, `cudaMemcpy(HtoD)`, `cudaMemcpy(DtoD)`, `cudaFree`, `cudaMemset`, `cudaStreamDestroy`, `cudaEventRecord`, `cudaEventDestroy`, all `cublasSgemm`/`cublasGemmEx`/`cublasSgemv`/etc (GEMM family), `cublasSetStream`, `cublasSetMathMode`, `cublasSscal`, `cublasSaxpy`, `cublasScopy`

### Sync RPC -- `chidori_rpc()` at line 421

Used when the client needs a return value or output data.

Call chain:
1. Check `CHIDORI_TRACE` / `CHIDORI_VERBOSE` env vars
2. Lock `g_lock` mutex
3. **Uncork TCP** (line 443-448): Disables TCP_CORK to flush any batched async writes
4. **Flush pending async responses** (line 452-453): Calls `chidori_flush_responses()` -- reads and drains all pending response headers/payloads from previously-sent async calls (even though the client didn't need them, the server still sent responses for sync-flagged requests; but actually, the server skips responses for async requests, so `g_pending_responses` should be 0 in normal operation -- this is a safety mechanism)
5. Lazy-connect if needed
6. Build request header with `CHIDORI_FLAG_SYNC` (line 465)
7. `chidori_send_request()` -- sends header + payload
8. `chidori_recv_response()` -- reads 12-byte response header
9. Read response payload if present (lines 489-526):
   - Read into caller's buffer (up to `*out_payload_len` bytes)
   - Drain any excess bytes the caller didn't request
10. Unlock mutex, return `resp.result`

**Operations that use sync:** `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem`, `cuDeviceGetAttribute` (cache miss), `cuCtxCreate`, `cuCtxGetCurrent` (cache miss), `cuCtxSynchronize`, `cuMemAlloc`, `cuMemcpyDtoH`, `cuMemGetInfo`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncGetAttribute`, `cuStreamCreate`, `cuStreamSynchronize`, `cuEventCreate`, `cuEventSynchronize`, `cuEventElapsedTime`, `cudaDeviceSynchronize`, `cudaGetDeviceProperties`, `cudaStreamSynchronize`, `cublasCreate`, `cublasSnrm2`, `cublasSdot` (return scalars)

---

## 9. TCP_CORK Batching

**Client-side** (`chidori_client.c` lines 364, 391-397, 443-448):

```c
static int g_cork_active = 0;
```

**Enable (during async sends):**
When `chidori_rpc_async()` fires and cork is not yet active, it sets `TCP_CORK = 1` (line 393-394). This tells the kernel to buffer outgoing data instead of sending it immediately. Subsequent async writes accumulate in the kernel's send buffer.

**Disable (on sync call):**
When `chidori_rpc()` fires, it sets `TCP_CORK = 0` (line 445-446) before sending the sync request. This flushes all previously corked data as a single TCP segment (or minimal number of segments), then the sync request goes out immediately.

**Effect:** A burst of N async operations (e.g., multiple `cuLaunchKernel` + `cuMemcpyHtoD` calls) coalesce into fewer TCP packets. The sync call (e.g., `cudaDeviceSynchronize`) uncorks and flushes everything at once.

**Server-side:** No TCP_CORK. The server sets `TCP_NODELAY` on accepted connections (line 296 in standalone mode).

---

## 10. Server-Side CUDA Dispatch

**File:** `src/server/cuda_dispatch.c` (3144 lines)

### Initialization -- `cuda_dispatch_init()` at line 94

1. Initialize all 16+ handle maps via `hmap_init()` (lines 97-112)
2. `cuInit(0)` -- initialize the CUDA driver (line 117)
3. `cuDeviceGetCount()` -- detect GPUs, cap at 16 (lines 124-131)
4. For each GPU: `cuDeviceGet()`, `cuDeviceGetName()`, `cuDeviceTotalMem()` -- enumerate and log (lines 134-147)
5. If `--pcibus` was specified: `cuDeviceGetByPCIBusId()` to find the target device (lines 153-167)
6. `cuDevicePrimaryCtxRetain()` + `cuCtxSetCurrent()` on the default device (lines 169-183)

### Shutdown -- `cuda_dispatch_shutdown()` at line 188

Releases primary contexts for all devices via `cuDevicePrimaryCtxRelease()`.

### Main Dispatch -- `cuda_dispatch()` at line 200

```c
int cuda_dispatch(int fd, chidori_request_t *req, void *payload)
```

Parameters:
- `fd`: Client socket for sending response. **-1 for async** (no response sent).
- `req`: Parsed request header
- `payload`: Pointer to payload bytes (may point directly into the batch buffer)

The function is one giant `switch(req->func_id)` statement spanning lines 213-3143. Each case:
1. Deserializes the payload via `memcpy()` into local variables
2. Calls the real CUDA function
3. Calls `chidori_send_response()` with the result and any output payload

### Context Management -- `ensure_device_context()` at line 74

Lazily retains the primary context for a device ordinal and sets it as current. Called when device switching is needed (e.g., `cudaSetDevice`).

---

## 11. Representative Handler Traces

### cuMemAlloc (sync, returns data)

**Client** (`chidori_client.c` line 666):
```
cuMemAlloc_v2(dptr, bytesize)
  -> payload: size(uint64_t)
  -> chidori_rpc(CHIDORI_CU_MEM_ALLOC, ...)
  -> receives: dptr(uint64_t) in response payload
```

**Server** (`cuda_dispatch.c` line 486):
```
case CHIDORI_CU_MEM_ALLOC:
  -> memcpy(&size, payload, 8)
  -> cuMemAlloc(&dptr, size)
  -> chidori_send_response(fd, seq, result, &dptr64, 8)
```

### cuMemcpyHtoD (async, sends bulk data)

**Client** (`chidori_client.c` line 677):
```
cuMemcpyHtoD_v2(dst, srcHost, byteCount)
  -> payload: dst(8) + size(8) + data(byteCount)   [malloc'd]
  -> chidori_rpc_async(CHIDORI_CU_MEMCPY_H_TO_D, ...)
  -> no response read
```

**Server** (`cuda_dispatch.c` line 504):
```
case CHIDORI_CU_MEMCPY_H_TO_D:
  -> memcpy(&dst, payload, 8)
  -> memcpy(&size, payload+8, 8)
  -> data = payload + 16       [ZERO COPY from batch buffer]
  -> cuMemcpyHtoD(dst, data, size)
  -> chidori_send_response(fd=-1, ...) -> SKIPPED (async)
```

### cublasSgemm (async, no data transfer -- all device pointers)

**Client** (`chidori_client.c` line 1463):
```
cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
  -> 72-byte packed struct:
     handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
     + alpha_value(4) + A_devptr(8) + lda(4) + B_devptr(8) + ldb(4)
     + beta_value(4) + C_devptr(8) + ldc(4)
  -> chidori_rpc_async(CHIDORI_CUBLAS_SGEMM, ...)
```

**Key detail:** alpha and beta are HOST pointers to scalar values. The client dereferences them (line 1490: `req.alpha = *alpha`) and sends the *values* over the wire. Matrix pointers A, B, C are device pointers sent as uint64_t.

**Server** (`cuda_dispatch.c` line 1534):
```
case CHIDORI_CUBLAS_SGEMM:
  -> Deserialize 72 bytes into local variables
  -> Resolve wire_handle -> real cublasHandle_t via hmap_get()
  -> cublasSgemm(handle, transa, transb, m, n, k,
                  &alpha,                    // local stack variable
                  (float*)(uintptr_t)A, lda, // cast wire ptr to real device ptr
                  (float*)(uintptr_t)B, ldb,
                  &beta,
                  (float*)(uintptr_t)C, ldc)
  -> chidori_send_response(fd=-1, ...) -> SKIPPED (async)
```

### cublasGemmEx (async, mixed-precision)

**Client** (`chidori_client.c` line 1642):
```
cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
             beta, C, Ctype, ldc, computeType, algo)
  -> 96-byte packed struct
  -> alpha/beta sent as 8-byte buffers (max scalar size)
  -> actual scalar size determined by chidori_cublas_scalar_size(computeType) (line 1380)
  -> chidori_rpc_async(CHIDORI_CUBLAS_GEMM_EX, ...)
```

**Server** (`cuda_dispatch.c` line 1666):
```
case CHIDORI_CUBLAS_GEMM_EX:
  -> Deserialize: alpha_buf[8], beta_buf[8] as raw byte buffers
  -> cublasGemmEx(handle, ..., alpha_buf, A, Atype, lda, B, Btype, ldb,
                   beta_buf, C, Ctype, ldc, computeType, algo)
```

### cudaDeviceSynchronize (sync, barrier)

**Client** (`chidori_client.c` line 1230):
```
cudaDeviceSynchronize()
  -> chidori_rpc(CHIDORI_CUDA_DEVICE_SYNCHRONIZE, NULL, 0, NULL, NULL)
  -> UNCORKS TCP (flushes all pending async writes)
  -> Sends sync request
  -> BLOCKS waiting for response
```

**Server** (`cuda_dispatch.c` line 1105):
```
case CHIDORI_CUDA_DEVICE_SYNCHRONIZE:
  -> cudaDeviceSynchronize()   // Waits for ALL GPU work to finish
  -> chidori_send_response(fd, seq, err, NULL, 0)
```

### cuLaunchKernel (async, complex serialization)

**Client** (`chidori_client.c` lines 787-918):

Wire format: 44-byte header + 1-byte has_params + optional [4-byte buffer_size + buffer_data]

```
func_handle(8) + gridX(4) + gridY(4) + gridZ(4)
+ blockX(4) + blockY(4) + blockZ(4)
+ sharedMem(4) + stream_handle(8)
+ has_params(1)
+ [buffer_size(4) + buffer_data(N)]
```

Two paths for kernel parameters:
1. **`extra` array** (line 811-824): Contains `CU_LAUNCH_PARAM_BUFFER_POINTER` / `CU_LAUNCH_PARAM_BUFFER_SIZE` tags. The buffer is sent directly.
2. **`kernelParams` array** (line 825-873): Issues an extra RPC to `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_PARAM_SIZE)` to learn the total parameter buffer size, then packs arguments contiguously with 8-byte alignment.

**Server** (`cuda_dispatch.c` lines 613-698 for Driver API, 1349-1448 for Runtime API):

1. Deserialize the 44-byte header
2. Resolve func_handle and stream_handle via handle maps
3. If has_params: query actual kernel param size via `cuFuncGetAttribute`, pass buffer through `CU_LAUNCH_PARAM_BUFFER_POINTER`/`CU_LAUNCH_PARAM_BUFFER_SIZE` extra array
4. Call `cuLaunchKernel(func, grid, block, shared, stream, NULL, extra)`

### CUDA_REGISTER_FUNCTION (sync, returns param layout)

**Server** (`cuda_dispatch.c` lines 1284-1347):

On the Runtime API path, when a function is registered, the server:
1. Calls `cuModuleGetFunction()` to get the real CUfunction
2. Calls `cuFuncGetParamInfo()` (CUDA 12.3+) in a loop (up to 64 params) to discover the **exact offset and size** of each kernel parameter
3. Returns: `func_handle(8) + total_param_size(4) + num_params(4) + [offset(4) + size(4)] x N`

This allows the client to serialize kernel parameters with correct layout.

---

## 12. Client-Side Caching

### Device Attribute Cache -- `g_attr_cache` (line 582)

```c
#define ATTR_CACHE_SIZE 512
static struct { int dev; int attrib; int value; int valid; } g_attr_cache[ATTR_CACHE_SIZE];
```

Both `cuDeviceGetAttribute()` (line 585) and `cudaDeviceGetAttribute()` (line 1036) check this cache before issuing an RPC. Cache hits avoid a network round-trip.

**Why it matters:** PyTorch queries 100+ device attributes during initialization. Without the cache, each would be a sync RPC round-trip.

### Context Cache -- `g_cached_ctx` (line 607)

```c
static CUcontext g_cached_ctx = NULL;
static int g_cached_ctx_valid = 0;
```

`cuCtxGetCurrent()` (line 633) returns the cached context immediately if valid, avoiding an RPC. Updated by `cuCtxCreate_v2`, `cuCtxSetCurrent`, `cuDevicePrimaryCtxRetain`.

`cuCtxSetCurrent()` (line 626) is sent as **async** (no round-trip) because the client can update its local cache immediately.

### Device Count Cache -- `g_device_count_cache` (line 991)

```c
static int g_device_count_cache = -1;
```

`cudaGetDeviceCount()` caches after first successful RPC.

### Current Device Cache -- `g_current_device` (line 1006)

```c
static int g_current_device = 0;
```

`cudaGetDevice()` (line 1008) returns the cache without any RPC (zero network cost). Updated by `cudaSetDevice()`.

### Last Error Cache -- `g_last_error` (line 989)

```c
static cudaError_t g_last_error = 0;
```

`cudaGetLastError()` (line 1325) returns and clears the local cache. `cudaPeekAtLastError()` (line 1331) returns without clearing. Updated by every Runtime API call.

---

## 13. Environment Variables

| Variable | Checked in | Effect |
|----------|-----------|--------|
| `CHIDORI_CFG_OVERRIDE` | Client (line 157) | JSON config with session ID and server addresses. Required for connection. |
| `CHIDORI_DIRECT_CONNECT` | Client (line 190) | If `"1"`, bypass HTTP/agent protocol and connect directly to worker TCP port. |
| `CHIDORI_TRACE` | Client (lines 323-327) | If `"1"`, logs every RPC call with timestamp, func_id, payload size, and cumulative async/sync counts. |
| `CHIDORI_VERBOSE` | Client (lines 367-369, 423-425) | If `"1"`, logs per-RPC details (func_id, payload size, pending count, result). |
| `CHIDORI_SERVER_VERBOSE` | Server dispatch_loop (line 124), cuda_dispatch (line 202) | If `"1"`, logs every dispatched request (func_id, payload_size, seq_num), batch sizes, kernel launch details, parameter hex dumps. |

---

## 14. End-to-End Call Chain Examples

### Example 1: PyTorch `torch.matmul(A, B)` on remote GPU

```
1. PyTorch calls cublasGemmEx(handle, ...)

2. Client libchidori.so::cublasGemmEx()  [line 1642]
   -> Serialize 96-byte packed struct (handle, dims, alpha/beta scalars, device ptrs, types)
   -> chidori_rpc_async(CHIDORI_CUBLAS_GEMM_EX, payload, 96)  [line 1686]
      -> Lock mutex
      -> TCP_CORK ON (if not already)  [line 391-397]
      -> chidori_send_request(g_server_fd, hdr={func=0x0214, flags=ASYNC, seq=N}, payload, 96)
         -> write_full(fd, header, 16)
         -> write_full(fd, payload, 96)
      -> Unlock mutex
      -> Return CUDA_SUCCESS immediately (no response read)

3. TCP layer: Data sits in kernel send buffer (corked)

4. Later, PyTorch calls cudaDeviceSynchronize()

5. Client libchidori.so::cudaDeviceSynchronize()  [line 1230]
   -> chidori_rpc(CHIDORI_CUDA_DEVICE_SYNCHRONIZE, NULL, 0, NULL, NULL)  [line 1231]
      -> Lock mutex
      -> TCP_CORK OFF  [line 443-448]  -- FLUSHES all buffered data
      -> chidori_send_request(g_server_fd, hdr={func=0x0106, flags=SYNC, seq=N+1}, NULL, 0)
      -> chidori_recv_response(g_server_fd, &resp)  -- BLOCKS
      -> Unlock mutex, return resp.result

6. Network: TCP segment arrives at server containing BOTH the GemmEx and Synchronize requests

7. Server dispatch_loop::recv()  [line 144]
   -> Reads both messages into 256KB buffer in ONE recv() call

8. Server inner parse loop  [line 157]:
   Request 1: CHIDORI_CUBLAS_GEMM_EX (async, fd=-1)
     -> cuda_dispatch(-1, req, payload)
        -> Deserialize, hmap_get() for handle
        -> cublasGemmEx(...) -- ENQUEUES on GPU
        -> chidori_send_response(-1, ...) -> NO-OP (fd=-1)

   Request 2: CHIDORI_CUDA_DEVICE_SYNCHRONIZE (sync, fd=client_fd)
     -> cuda_dispatch(client_fd, req, NULL)
        -> cudaDeviceSynchronize() -- WAITS for GPU
        -> chidori_send_response(client_fd, seq, 0, NULL, 0)
           -> write_full(fd, response_header, 12)

9. Client: recv() returns 12-byte response, cudaDeviceSynchronize() returns 0

10. PyTorch continues execution
```

### Example 2: `cudaMalloc` + `cudaMemcpy(HtoD)` + kernel launch + `cudaMemcpy(DtoH)`

```
cudaMalloc(&d_A, size)
  -> SYNC RPC: send CHIDORI_CUDA_MALLOC, receive device pointer
  -> 1 round-trip

cudaMemcpy(d_A, host_data, size, cudaMemcpyHostToDevice)
  -> ASYNC RPC: send CHIDORI_CUDA_MEMCPY with kind=1
  -> Payload: dst(8) + size(8) + kind(4) + data(size_bytes)
  -> NO round-trip (corked, buffered)

kernel<<<grid, block>>>(d_A, d_B)  [via cudaLaunchKernel]
  -> ASYNC RPC: send CHIDORI_CUDA_LAUNCH_KERNEL
  -> NO round-trip (still corked)

cudaMemcpy(host_result, d_B, size, cudaMemcpyDeviceToHost)
  -> SYNC RPC: send CHIDORI_CUDA_MEMCPY with kind=2
  -> UNCORKS TCP (all 3 previous messages flush)
  -> Server processes all 3 in batch, then processes DtoH:
     reads from GPU memory, sends data in response payload
  -> 1 round-trip (carries result data back)
```

Total network round-trips: 2 (one for malloc, one for the final DtoH memcpy).
The HtoD memcpy and kernel launch were free -- batched into the sync call's TCP flush.

---

## File Reference (Absolute Paths)

```
/Users/fabricio/Desktop/compute-related-repos/chidori/chidori-worker/include/chidori_protocol.h
/Users/fabricio/Desktop/compute-related-repos/chidori/chidori-worker/src/protocol/chidori_net.c
/Users/fabricio/Desktop/compute-related-repos/chidori/chidori-worker/src/server/chidori_worker.c
/Users/fabricio/Desktop/compute-related-repos/chidori/chidori-worker/src/server/cuda_dispatch.c
/Users/fabricio/Desktop/compute-related-repos/chidori/chidori-worker/src/server/handle_map.h
/Users/fabricio/Desktop/compute-related-repos/chidori/chidori-worker/src/client/chidori_client.c
```

---

## 15. Client-Side Optimizations

These optimizations reduce sync RPC count and improve WAN throughput.

### TCP Socket Buffer Tuning

**Client** (`chidori_client.c:114-118`) and **Server** (`chidori_worker.c:299-301`): Set 16MB `SO_RCVBUF`/`SO_SNDBUF` on both ends. At 1Gbps with 55ms RTT, the bandwidth-delay product is 6.875MB — default 208KB buffers waste 97% of the pipe.

### cudaSetDevice Short-Circuit

**`chidori_client.c:1024`**: Returns `cudaSuccess` immediately when the requested device matches `g_current_device`. PyTorch calls `cudaSetDevice(0)` 20-40 times per inference token — this eliminated ~30 sync RPCs per token.

### Version Call Caching

Five one-shot caches that avoid repeated sync RPCs for immutable values:
- `cuDriverGetVersion` (`g_driver_version_cached`)
- `cudaRuntimeGetVersion` (`g_runtime_version_cached`)
- `cudaDriverGetVersion` (`g_cuda_driver_version_cached`)
- `cublasGetVersion_v2` (`g_cublas_version_cached`)
- `cudnnGetVersion` (`g_cudnn_version_cached`)

### cuBLAS State Tracking

`cublasSetStream`/`cublasSetMathMode` are async RPCs that also update local cache variables. The corresponding `cublasGetStream`/`cublasGetMathMode` return cached values on hit, avoiding sync RPCs. PyTorch calls these before every GEMM operation.

### cudaFuncGetAttributes Cache

256-entry cache keyed by function pointer (`g_func_attr_cache`). Function attributes are immutable — first call per function is an RPC, subsequent calls are local. Eliminates ~5 sync RPCs per token after warmup.

### cublasLt to cublasGemmEx Translation

Client-side descriptor pool (`lt_matmul_desc_t`, `lt_matrix_layout_t`) tracks matrix dimensions, dtypes, transpose flags, and batch parameters set via `cublasLtMatmulDescSetAttribute`/`cublasLtMatrixLayoutSetAttribute`. When `cublasLtMatmul` is called, it extracts the descriptor state and dispatches as either `cublasGemmEx` (non-batched) or `cublasGemmStridedBatchedEx` (batched) async RPC. Uses `g_last_cublas_handle` (tracked from `cublasCreate_v2`) since `cublasLtHandle_t` is a different type.

**Limitation**: `cublasLtMatmulAlgoGetHeuristic` returns 0 algorithms. PyTorch 2.6's FP32 `F.linear(input, weight, bias)` requires at least 1 algorithm for the batched GEMM path and has no fallback. FP16 and bias=False paths work via `cublasGemmEx` directly.

### Handle Map Expansion

`HANDLE_MAP_MAX` increased from 1024 to 8192. Overflow now logs to stderr with the key that failed to insert.

---

## 16. Performance Results

Benchmarked on Tesla V100-SXM3-32GB (server) with various client configurations.

### Compute (Zero Overhead)

| Benchmark | Chidori Loopback | Native | Overhead |
|-----------|-----------------|--------|----------|
| FP32 matmul (2000x 4096x4096) | 14.6 TFLOPS | 15.25 TFLOPS | ~0% |

GPU compute is identical to native once data is on the device. The TCP layer is invisible for compute-bound operations.

### Inference (Latency-Bound)

| Network | tok/s | Sync RPCs/token | Bottleneck |
|---------|-------|-----------------|------------|
| Loopback | 10.2 | 3.1 | GPU compute + RPC overhead |
| WAN 39ms RTT | 0.8 | 3.1 | 3 RPCs x 39ms = 117ms/token |

Inference is autoregressive — token N must complete before token N+1 starts. The 3 remaining sync RPCs per token (`cudaStreamSynchronize` x2 + `cuMemcpyDtoH` x1) are the theoretical floor.

### Training (Near-Native)

| Network | tok/s | Native | Overhead |
|---------|-------|--------|----------|
| Loopback | 321K | 330K | 2.9% |
| WAN 39ms RTT | 14.2K | — | Compute-bound, latency hidden |

Training hides latency because forward/backward passes are compute-heavy GEMM operations sent as async RPCs. Sync points are rare (once per optimizer step).

---

## 17. Known Blockers

### cublasLt Heuristic (FP32 Training with Bias)

`cublasLtMatmulAlgoGetHeuristic` returns `*returnAlgoCount = 0`. PyTorch 2.6 requires at least 1 algorithm for FP32 batched GEMM (`F.linear` with bias and 3D input). No fallback exists in PyTorch. Returning 1 algorithm with zeroed struct causes a NULL function pointer segfault in PyTorch's internal code. **Fix**: Implement as server-side RPC returning real heuristic results.

### Fat Binary Size Detection

`cuModuleLoadData` uses hardcoded `image_len = 64 * 1024` for ELF CUBINs. Backward kernels exceeding 64KB are silently truncated, causing `lazy_register_function` to return handle 0 → `cudaErrorInvalidDeviceFunction` during `backward()`. **Fix**: Parse ELF header (`e_shoff + e_shnum * e_shentsize`) for actual binary size.

### SDPA / Flash Attention

`scaled_dot_product_attention` uses CUTLASS kernels that have their own dispatch mechanism bypassing `cuLaunchKernel`. Current workaround: `torch.backends.cuda.enable_flash_sdp(False)` + `attn_implementation="eager"` in HuggingFace models.

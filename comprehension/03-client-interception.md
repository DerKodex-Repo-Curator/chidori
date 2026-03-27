# Chidori Client Interception

## LD_PRELOAD Mechanism

`libchidori.so` is loaded before all other shared libraries via `LD_PRELOAD`. It overrides approximately 280 CUDA symbols (Driver API, Runtime API, cuBLAS, cuBLASLt, cuDNN, NVML, NVRTC). When the application calls `cudaMalloc()`, the dynamic linker resolves it to `libchidori.so`'s `cudaMalloc()` implementation, which serializes the call over TCP instead of executing it locally.

Additional proxy libraries are built from the same source to intercept cuBLAS and cuBLASLt symbols specifically:
- `libcublas.so.12` / `libcublas.so.13` -- intercepts cuBLAS symbols
- `libcublasLt.so.12` / `libcublasLt.so.13` -- intercepts cublasLt symbols
- `libnvidia-ml.so.1` -- NVML proxy (separate source file `chidori_nvml.c`)

The client library is typically symlinked as `libcuda.so.1` so that `LD_PRELOAD=build/libcuda.so.1` causes it to be loaded before the real NVIDIA driver.

## Initialization Flow

The client library initializes through a multi-step sequence spanning constructor functions, lazy initialization, and first-use triggers.

### Step 1: auto_prefer_cublaslt() -- __attribute__((constructor))

Runs before the application's `main()`. Sets the environment variable `TORCH_BLAS_PREFER_CUBLASLT=1` (with `setenv(..., 0)` so it does not overwrite an existing value). This forces PyTorch to use the cublasLt path for all GEMM operations, which works correctly with the interception layer because cublasLt routing handles descriptors properly.

### Step 2: init_real_dlsym() -- called lazily

Uses `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5")` to obtain the real glibc `dlsym` function pointer. This is necessary because `libchidori.so` overrides `dlsym()` itself, and calling `dlsym(RTLD_NEXT, "dlsym")` from within the override would cause infinite recursion. All internal dlsym calls in the library go through the `__real_dlsym()` wrapper.

### Step 3: init_real_cuda() -- called from cuInit, cuDeviceGetCount, or __cudaRegisterFatBinary

Triggered when any of these entry points are first called. This step initializes the partial offloading subsystem:

1. Checks `CHIDORI_LOCAL_DEVICES` environment variable. If not set, returns immediately (all-remote mode).
2. Sets `g_partial_offload = 1` and parses the local device mask (e.g., "0" means device 0 is local).
3. Loads the real NVIDIA CUDA driver via explicit `dlopen()`. Uses `realpath()` to resolve symlinks and avoid loading itself (since `libchidori.so` is symlinked as `libcuda.so.1`). Searches multiple system paths: `/usr/lib/libcuda.so.1`, `/usr/lib64/libcuda.so.1`, `/usr/lib/x86_64-linux-gnu/libcuda.so.1`.
4. Sets `g_real_cuda_handle_for_dlopen` so that subsequent `dlopen("libcuda.so.1")` calls (from cuBLAS etc.) return the real driver handle.
5. Loads approximately 45 real CUDA function pointers into the `g_real` dispatch table using `dlsym()` on the real driver handle.
6. Counts local GPUs from the device mask.
7. If `CHIDORI_DEFERRED_INIT=1`, replays any pending fat binary registrations that were queued during library loading.

### Step 4: chidori_connect() -- called on first RPC

Establishes the TCP connection to the remote GPU worker. Triggered lazily when `g_server_fd < 0` at the start of any RPC call.

1. Reads `CHIDORI_CFG_OVERRIDE` environment variable (JSON format: `{"id":"test","servers":["127.0.0.1:43211"]}`).
2. Parses session ID and server address from the JSON.

**Direct mode** (`CHIDORI_DIRECT_CONNECT=1`):
- Calls `tcp_connect(host, port)` to establish a direct TCP connection to `chidori-worker` standalone mode.

**Agent mode** (default):
- Connects to the Go agent's address.
- Sends an HTTP `POST /v1/connect/session/{id}` request with `Connection: Upgrade` header.
- Reads the HTTP `101 Switching Protocols` response byte-by-byte, looking for `\r\n\r\n` end-of-headers.
- After the 101 response, the TCP stream carries raw Chidori CUDA protocol.

**TCP socket configuration** (both modes):
- `TCP_NODELAY`: disable Nagle's algorithm for low latency.
- `SO_RCVBUF` / `SO_SNDBUF`: 16MB socket buffers for WAN throughput (BDP = bandwidth * RTT).
- `dup()` + `fcntl(F_DUPFD, 500)`: move fd to a high number (>= 500) to avoid collision with PyTorch/cuBLAS internal file descriptors.
- `SO_KEEPALIVE` with 10s idle, 5s interval, 3 probes.

### Step 5: chidori_cleanup() -- __attribute__((destructor))

Runs when the shared library is unloaded (process exit or `dlclose`). Closes `g_server_fd` and sets it to -1.

## dlopen/dlsym Interception

### dlopen Override

When `g_dlopen_redirect_active` or `g_in_real_cuda > 0` is true, calls to `dlopen("libcuda.so.1")` or `dlopen("libcuda.so")` return `g_real_cuda_handle_for_dlopen` (the real NVIDIA driver handle). This ensures that cuBLAS's internal `dlopen("libcuda.so.1")` during initialization gets the real driver, not our interception library.

### dlsym Override

When `g_in_real_cuda > 0` (inside a real CUDA call), dlsym is NOT intercepted -- cuBLAS internal operations need to resolve their own symbols to real implementations.

When NOT inside a real CUDA call, dlsym checks `g_intercepted_cublas_symbols[]` (48 symbols including cuBLAS and cublasLt functions). If the requested symbol matches, our version is returned via `dlsym(RTLD_DEFAULT, symbol)`. This prevents PyTorch from bypassing LD_PRELOAD when it resolves cuBLAS symbols via `dlsym(cublas_handle, "cublasGemmEx")`.

### cuGetProcAddress Override

Checks `g_intercepted_symbols[]` for driver API symbols that need virtual device routing. Returns our version for intercepted symbols.

### cudaGetDriverEntryPoint Override

Same pattern as cuGetProcAddress -- checks intercepted symbols and returns our version for those that need device routing.

## RPC Mechanism

### chidori_rpc() -- Synchronous RPC

```
1. Acquire g_lock (global mutex)
2. Disable TCP_CORK if active (flush batched async writes)
3. Flush pending async responses (drain g_pending_responses from socket)
4. If g_server_fd < 0, call chidori_connect()
5. Build chidori_request_t with CHIDORI_FLAG_SYNC
6. chidori_send_request (16-byte header + payload)
7. chidori_recv_response (12-byte header)
8. Read response payload into caller buffer
9. Drain extra bytes if caller buffer is smaller than response
10. Release g_lock
11. Return (CUresult)resp.result
```

If the connection fails at any point, the fd is closed, set to -1, and `CUDA_ERROR_NOT_INITIALIZED` is returned. This allows reconnection on the next call.

### chidori_rpc_async() -- Asynchronous RPC

```
1. Check replay intercept (if in replay mode, collect H2D data instead)
2. Acquire g_lock
3. If g_server_fd < 0, call chidori_connect()
4. Enable TCP_CORK (batch multiple writes into one TCP segment)
5. Build chidori_request_t with CHIDORI_FLAG_ASYNC
6. chidori_send_request (16-byte header + payload)
7. Release g_lock
8. Return CUDA_SUCCESS immediately (no response awaited)
```

The server sees `CHIDORI_FLAG_ASYNC` and passes `fd=-1` to `cuda_dispatch()`, causing `chidori_send_response()` to return immediately without writing.

### chidori_flush_responses()

Called when transitioning from async to sync mode. Drains `g_pending_responses` response headers and payloads from the socket. Reads into a 4096-byte drain buffer. If the connection dies during flush, closes fd and resets state.

## Thread Safety

| Variable | Type | Scope | Purpose |
|----------|------|-------|---------|
| `g_lock` | `pthread_mutex_t` | Global | Protects all TCP send/recv operations |
| `g_alloc_track_lock` | `pthread_mutex_t` | Global | Protects allocation tracking array |
| `g_in_real_cuda` | `int __thread` | Per-thread | Reentrance guard -- >0 means inside a real CUDA call, forward to real driver |
| `g_dlopen_redirect_active` | `int __thread` | Per-thread | Controls dlopen redirect scope during cuBLAS loading |
| `g_current_device` | `int __thread` | Per-thread | Current active device ordinal (matches CUDA semantics) |
| `g_last_error` | `cudaError_t __thread` | Per-thread | Per-thread CUDA error state (matches `cudaGetLastError` semantics) |

There is a single shared TCP connection (`g_server_fd`). All threads serialize through `g_lock` for TCP operations. Per-thread connections are not used.

## Global State

### Connection State

| Variable | Description |
|----------|-------------|
| `g_server_fd` | TCP socket to remote server (-1 when disconnected) |
| `g_seq_num` | Monotonically increasing sequence counter for request/response matching |
| `g_lock` | Global mutex for all TCP operations |
| `g_cork_active` | TCP_CORK state (1 = batching enabled) |
| `g_pending_responses` | Count of unflushed async responses in the socket receive buffer |
| `g_async_count` / `g_sync_count` | Counters for logging/debugging |
| `g_initialized` | 1 after first cuInit call |

### Real CUDA Driver State (Partial Offloading)

| Variable | Description |
|----------|-------------|
| `g_real_cuda_handle` | dlopen handle for real libcuda.so |
| `g_real_cudart_handle` | RTLD_NEXT for libcudart.so |
| `g_real_cublas_handle` | dlopen handle for real libcublas.so (loaded lazily) |
| `g_real_cuda_handle_for_dlopen` | Set after init_real_cuda for dlopen redirect |
| `g_real` | Dispatch table of ~45 real CUDA function pointers |
| `g_partial_offload` | 1 if CHIDORI_LOCAL_DEVICES is set |
| `g_local_device_mask` | Bitmask: bit N = device N is local |
| `g_local_device_count` | Number of local GPUs |
| `g_local_ctx` | Real CUDA primary context for local device |
| `g_real_cuda_inited` | 1 after real cuInit(0) called |
| `g_device_count_cache` | Cached device count (-1 = not cached) |

### Fat Binary and Kernel Registration

| Variable | Description |
|----------|-------------|
| `g_fat_binaries[1024]` | Stored fat binary data pointers |
| `g_registered_functions[65536]` | Function registrations with param info (hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize) |
| `g_pending_fatbins[1024]` | Pending fat binary registrations for deferred init mode |
| `g_pending_funcs[65536]` | Pending function registrations for deferred init mode |
| `g_fatbin_passthrough` | 1 if CHIDORI_FATBIN_PASSTHROUGH=1 (skip all fatbin tracking) |

### Tracking Arrays

| Variable | Description |
|----------|-------------|
| `g_event_track[4096]` | Event-to-device mapping for cross-device backward (handle, device) |
| `g_alloc_track[16384]` | GPU pointer range to device ownership (ptr, size, device) |
| `g_bulk_pending[4096]` | Combined alloc+transfer pairs for model loading (placeholder, alloc_size) |
| `g_attr_cache[512]` | Device attribute cache (dev, attrib, value, valid) |
| `g_driver_version` / `g_driver_version_cached` | Cached driver version |

### Replay State

| Variable | Description |
|----------|-------------|
| `g_replay_mode` | 0=off, 1=recording, 2=replaying |
| `g_replay_step_count` | Counts training steps (sync calls) |
| `g_replay_warmup` | Record on step N (default 3, env: CHIDORI_REPLAY_WARMUP) |
| `g_replay_seq_id` | Current replay sequence ID |
| `g_replay_fingerprint` | Fingerprint of recorded sequence |
| `g_replay_h2d_collected[64]` | H2D data collected during replay intercept |
| `g_replay_h2d_count` | Number of collected H2D entries |
| `g_replay_call_hash` | Running fingerprint for validation during replay |

### Memory Pool State

| Variable | Description |
|----------|-------------|
| `pool_base` | Base address of current pool chunk |
| `pool_offset` | Current offset within pool (bump pointer) |
| `pool_size` | Total size of current pool chunk |

### cublasLt Descriptor Pools

| Variable | Description |
|----------|-------------|
| `g_lt_descs[1024]` | cublasLt matmul descriptors (computeType, scaleType, transa, transb, epilogue, bias_pointer) |
| `g_lt_layouts[4096]` | cublasLt matrix layouts (dtype, rows, cols, ld, batch_count, strided_batch_offset) |
| `g_lt_desc_free[1024]` | Free list for matmul descriptors |
| `g_lt_layout_free[4096]` | Free list for matrix layouts |

### Event Pool

| Variable | Description |
|----------|-------------|
| `g_event_pool[256]` | Pre-allocated remote event handles |
| `g_event_pool_count` | Current number of available events in pool |
| `g_event_pool_inited` | 1 after first fill |

## CUDA Driver API Interceptions (~30 functions)

All Driver API functions follow the same pattern:

```
1. Reentrance guard: if g_in_real_cuda > 0, forward to real driver
2. Local device check: if partial offload and device is local, forward to real driver
3. Device index remapping: local devices use real driver, remote devices subtract local count
4. Serialize parameters into payload buffer
5. Call chidori_rpc() or chidori_rpc_async()
6. Deserialize response
7. Update tracking arrays (event_track, alloc_track) if applicable
8. Set g_last_error
9. Return result
```

### Key Special Cases

**cuDeviceGetCount**: Returns only the local device count (from `g_local_device_count`) when `g_partial_offload` is set. This is critical because NVIDIA libraries (cuBLAS, cublasLt, cuDNN) cache this value during their init. Returning an inflated virtual count causes `CUBLAS_STATUS_NOT_INITIALIZED` for FP16 operations. PyTorch discovers devices through `cudaGetDeviceCount` (Runtime API) instead.

**cuLaunchKernel**: Performs cross-device pointer translation in kernel parameter buffers. GPU pointers that belong to the remote server are used as-is. Local GPU pointers are translated for remote use when necessary. The kernel function handle is looked up from `g_registered_functions`.

**cuMemcpyDtoD**: When the source and destination are on different devices (one local, one remote), data is staged through a CPU buffer -- read from one side, written to the other. When both are remote, only the 24-byte descriptor (dst, src, size) is sent over the wire.

**cuMemAlloc**: When `g_partial_offload` is set and the current device is local, routes to the real driver even when `g_in_real_cuda == 0`. This avoids unnecessary network round-trips for local allocations.

**cuDeviceGetAttribute**: Uses a 512-entry attribute cache (`g_attr_cache[]`) to avoid repeated RPCs. PyTorch queries 100+ device attributes during initialization.

## CUDA Runtime API Interceptions (~26 functions)

### Key Special Cases

**cudaGetDeviceCount**: Returns the merged count (local + remote) when partial offloading is active. This is cached after the first call. Without partial offloading, queries the remote server.

**cudaSetDevice**: For local devices, uses the real driver directly (bypasses the remote RPC). This avoids cudart error 36 (cudaErrorDeviceUnavailable) which occurs when the remote server does not have the local device.

**cudaMalloc**: Uses a bump allocator with 512MB+ chunks. When the current pool chunk is exhausted, a new chunk is allocated from the server via RPC. Allocations within a chunk are satisfied locally with zero network round-trips.

```
cudaMalloc(size):
  aligned_size = align_up(size, 256)
  if pool_offset + aligned_size <= pool_size:
    return pool_base + pool_offset (bump pointer, no RPC)
  else:
    new_pool_size = max(aligned_size * 2, 512MB)
    RPC: cudaMalloc(new_pool_size) -> pool_base
    pool_offset = aligned_size
    return pool_base
```

**cudaMemcpy**: Handles all four transfer directions:
- **HtoH**: Local memcpy, no network.
- **HtoD**: Sent as async RPC + bulk transfer optimization during model loading.
- **DtoH**: Synchronous RPC, data returned in response payload.
- **DtoD**: If both pointers are remote, sends only 24 bytes. If cross-device (one local, one remote), stages through CPU buffer.

**cudaDeviceSynchronize**: In replay mode, builds a `REPLAY_EXECUTE` payload containing all collected H2D data and sends it as a single RPC instead of individual calls. This is the replay state machine boundary.

**cudaEventCreate/Destroy**: Uses an event pool of 256 pre-allocated events on the server. Creation returns a pre-allocated handle instantly (zero network cost). Destruction returns the handle to the pool. When the pool drops below 16 entries, it is refilled in batches of 64 using pipelined sync RPCs.

**cudaGetLastError / cudaPeekAtLastError**: Returns `g_last_error` (per-thread). The error is set by each intercepted function.

## cuBLAS Interceptions (27 functions)

All cuBLAS operations are sent via `chidori_rpc_async()` (fire-and-forget). No response is expected from the server.

Key functions:
- **cublasGemmEx**: Critical for PyTorch FP16 GEMM. Serializes the operation type, matrix dimensions, data types, and device pointers into the RPC payload.
- **cublasSgemmBatched**: Serializes pointer arrays inline. Each batch entry's A, B, C pointers are written sequentially into the payload.

Pattern for all cuBLAS functions:
```
1. Reentrance guard: if g_in_real_cuda > 0, forward to real cuBLAS
2. Local device routing: if partial offload and local device, use real cuBLAS handle
3. Serialize all parameters into a stack-allocated buffer
4. chidori_rpc_async(func_id, buffer, size)
5. Return CUBLAS_STATUS_SUCCESS
```

## cublasLt Interceptions (20+ functions)

Client-side descriptor pools are used to avoid network round-trips for descriptor creation/destruction.

**Descriptor types:**
- `lt_matmul_desc_t`: 28 bytes (computeType, scaleType, transa, transb, epilogue, bias_pointer)
- `lt_matrix_layout_t`: 40 bytes (dtype, rows, cols, ld, batch_count, strided_batch_offset)

**cublasLtMatmul**: Translates client-side descriptor structures into a 308-byte RPC payload:
- 28 bytes: matmul descriptor fields (transa, transb, computeType, scaleType, epilogue, bias_ptr)
- 160 bytes: 4 matrix layouts (A, B, C, D), each 40 bytes
- 8 bytes: alpha scalar
- 8 bytes: beta scalar
- 32 bytes: data pointers (A, B, C, D)
- 8 bytes: stream handle
- 64 bytes: algorithm (opaque cublasLtMatmulAlgo_t data)

**Descriptor lifecycle:**
- `cublasLtMatmulDescCreate` / `cublasLtMatrixLayoutCreate`: Allocates from static arrays (`g_lt_descs[1024]`, `g_lt_layouts[4096]`) with free lists. Returns a pointer to the struct as the opaque handle.
- `cublasLtMatmulDescDestroy` / `cublasLtMatrixLayoutDestroy`: Pushes the index back onto the free list. No RPC.
- `cublasLtMatmulDescSetAttribute` / `cublasLtMatrixLayoutSetAttribute`: Updates the local struct. No RPC.

## cuDNN Interceptions (47+ functions)

All cuDNN descriptors are opaque uint64 handles managed by the server. The client sends descriptor creation requests and receives handles, then passes those handles in subsequent operations.

**Descriptor categories:**
- Tensor descriptors (create, set 4D, set ND, get 4D, destroy)
- Filter descriptors (create, set 4D, set ND, destroy)
- Convolution descriptors (create, set 2D, set ND, destroy, math type, group count)
- Activation descriptors (create, set, destroy, forward, backward)
- Pooling descriptors (create, set 2D, destroy, forward, backward)
- Dropout descriptors (create, set, destroy, forward)
- RNN descriptors (create, destroy)

All cuDNN operations are sent as synchronous or asynchronous RPCs depending on the operation type.

## Symbol Versioning (chidori_client.map)

A GNU linker version script is applied at link time using `-Wl,--version-script=src/client/chidori_client.map`. This is critical for PyTorch compatibility.

**Problem:** PyTorch's ELF imports reference versioned symbols like `cublasGemmEx@libcublas.so.12`. Without versioned exports from our library, the dynamic linker skips our unversioned symbols and resolves to the real cuBLAS library.

**Solution:** The version script declares our symbols under the `libcublas.so.12` and `libcublasLt.so.12` version tags:

```
libcublas.so.12 {
    global:
        cublasCreate_v2;
        cublasDestroy_v2;
        cublasGemmEx;
        ... (35 total symbols)
};

libcublasLt.so.12 {
    global:
        cublasLtCreate;
        cublasLtDestroy;
        cublasLtMatmul;
        ... (25 total symbols)
};
```

Without this version script, PyTorch bypasses our interception entirely for cuBLAS/cublasLt symbols.

## Error Handling

### TCP Failures
- Any `chidori_write_full` or `chidori_read_full` failure closes the fd and sets `g_server_fd = -1`.
- A `[chidori] SOCKET CLOSED` message is logged to stderr with file:line information.
- The next RPC call triggers `chidori_connect()` to re-establish the connection.

### RPC Failure Return
- When the connection cannot be established or a send/recv fails, the function returns `CUDA_ERROR_NOT_INITIALIZED` (3).

### Protocol Desync Prevention
- Response payloads are always fully drained, even when the caller does not want them (e.g., extra bytes read into a 4096-byte drain buffer).
- Pending async responses are flushed before any synchronous RPC.

### Extra Response Bytes
- If the caller's output buffer is smaller than the response payload, extra bytes are drained into a 4096-byte buffer to prevent protocol desync.

## Logging

Controlled by environment variables (all output to stderr):
- `CHIDORI_DEBUG=1`: Connection events, device routing, errors. Implies VERBOSE=1.
- `CHIDORI_TRACE=1`: Every sync/async RPC with func_id, timing, and counters.
- `CHIDORI_VERBOSE=1`: Per-RPC payload size, pending count, result codes.

## Memory Allocation Patterns

| Allocation Type | Strategy | Lifetime |
|----------------|----------|----------|
| Fat binary data | Raw pointer into .nv_fatbin ELF section | Process lifetime |
| RPC payloads | `malloc()` per call, `free()` after send | Per-call |
| Kernel param buffer | `malloc()` per launch | Per-launch |
| Cross-device DtoD staging | `malloc()` per call | Per-call |
| cudaMalloc pool | Static bump allocator with 512MB+ server-allocated chunks | Until pool exhausted |
| cublasLt descriptors | Static arrays (`g_lt_descs[1024]`, `g_lt_layouts[4096]`) with free lists | Process lifetime (reusable) |
| Event pool | Static `uint64_t[256]` array | Process lifetime (reusable) |
| Bulk transfer tracking | Static `g_bulk_pending[4096]` array | Per-model-load |
| Allocation tracking | Static `g_alloc_track[16384]` array | Process lifetime (append-only) |
| Replay H2D collection | `malloc()` per collected entry, freed after REPLAY_EXECUTE | Per-training-step |

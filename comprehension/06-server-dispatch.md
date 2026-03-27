# Server Dispatch

## Server Main Loop (`chidori_worker.c`)

The server supports two operational modes, selected by command-line arguments.

### Standalone Mode

```
chidori-worker --standalone <port>
```

Direct TCP listener on the specified port. Accepts one client at a time (no multiplexing). Used for development and testing.

**Socket setup:**
- `SO_REUSEADDR`: allows quick server restart without TIME_WAIT
- `TCP_NODELAY`: disables Nagle's algorithm for low latency
- `SO_RCVBUF` / `SO_SNDBUF`: 16 MB send and receive buffers

### Agent-Mediated Mode

```
chidori-worker --id <session_id> --ipc_write <fd> --ipc_read <fd>
```

Production mode. The server does not open its own listening socket. Instead, it receives pre-accepted client TCP sockets from the Go agent via Unix domain socket `SCM_RIGHTS` message passing.

**Startup sequence:**
1. `receive_socket_from_agent(ipc_read_fd)`: calls `recvmsg()` to extract the TCP file descriptor from the auxiliary data of a `SCM_RIGHTS` message
2. Clear `O_NONBLOCK`: Go's netpoller sets non-blocking mode on all file descriptors; the C worker needs blocking I/O for its `recv()` loop
3. **Signal readiness**: write a single byte to `ipc_write_fd` BEFORE performing CUDA initialization. This prevents the agent from timing out while waiting for the worker to become ready, since `cuInit` can take several seconds
4. **Wait for confirmation**: read a byte from `ipc_read_fd`. The agent sends this after it has set up any necessary state
5. **Lazy CUDA init**: CUDA initialization (`cuInit`) happens only on the first connection, not at startup. This allows the worker process to start quickly and defer the multi-second CUDA driver initialization

## Batch-Read Dispatch Loop (`dispatch_loop`)

This is the performance-critical heart of the server. It uses a 256 KB receive buffer (`BATCH_BUF_SIZE`) to amortize syscall overhead by processing multiple requests from a single `recv()` call.

### Step 1: Bulk recv

```c
n = recv(client_fd, buf + buf_len, BATCH_BUF_SIZE - buf_len, 0);
```

- Data is appended to any leftover partial data from the previous iteration (`buf_len` tracks how much unprocessed data remains)
- `EINTR` is retried (signal interruption)
- Return of 0 means client disconnected (clean close)
- Return < 0 means error (after EINTR retry)

### Step 2: Parse and dispatch (inner while loop)

The inner loop processes all complete requests in the buffer:

1. Cast the buffer at the current offset to `chidori_request_t` (a 16-byte packed struct)
2. Calculate `total_msg = 16 + req->payload_size`
3. Three code paths based on how much data is available:

**Common path (total fits in buffer):**
- Zero-copy dispatch: `payload` points directly into the receive buffer, no allocation
- For sync calls: `dispatch_fd = client_fd` (used to send the response)
- For async calls (`CHIDORI_FLAG_ASYNC` set): `dispatch_fd = -1`
- Calls `cuda_dispatch(dispatch_fd, req, payload)`

**Oversized path (total_msg > 256 KB):**
- Allocates a separate `big_payload` buffer
- Copies the partial data already in the receive buffer
- Reads the remaining data directly from the socket with additional `recv()` calls
- Dispatches against the assembled big payload
- Frees the buffer after dispatch

**Partial path (not enough data in buffer):**
- Breaks out of the inner loop to return to Step 1 for more `recv()`

### Step 3: Compact buffer

After the inner loop exits (partial path or buffer exhausted):
```c
memmove(buf, buf + processed, buf_len);
```

Any remaining partial data is moved to the front of the buffer for the next iteration.

### Async vs Sync

- **Sync** (`dispatch_fd = client_fd`): After `cuda_dispatch()` executes the CUDA call, it calls `chidori_send_response(fd, ...)` which writes the result back to the client
- **Async** (`dispatch_fd = -1`): `chidori_send_response(fd=-1, ...)` is a no-op. The CUDA call executes but no response is sent. This enables fire-and-forget compute calls -- hundreds of kernel launches can be queued without the overhead of per-call responses

Async is used for compute-phase operations (kernel launches, memory copies) where the client does not need an immediate result. The client can later synchronize via stream or event operations to ensure completion.

## `cuda_dispatch()` Function (~4300 lines)

**Signature:** `int cuda_dispatch(int fd, chidori_request_t *req, void *payload)`

This is a giant switch statement on `req->func_id` that dispatches to real CUDA API calls. It is organized by function ID ranges corresponding to the different CUDA libraries.

### Pre-dispatch Logic

Before the switch:

1. **Verbose logging check**: Lazily evaluates whether verbose logging is enabled (avoids string formatting overhead when not logging)
2. **Replay recording**: If a replay is being recorded and `func_id < 0x0400`, the entire request (header + payload) is snapshotted for later replay
3. **Graph capture tracing**: If `g_capturing_stream` is set (the server is in CUDA graph capture mode), certain calls are logged for debugging

### Dispatch Cases by Range

#### INITIALIZATION (0x0001-0x0002)

- **`cuInit`**: Returns `CUDA_SUCCESS` immediately. The server already initialized CUDA during its own startup.
- **`cuDriverGetVersion`**: Calls the real `cuDriverGetVersion`, returns the 4-byte version integer.

#### DEVICE MANAGEMENT (0x0010-0x0017)

- **`cuDeviceGetCount`**: Returns 1 **always**. The server has a fixed GPU configuration. Returning 1 prevents multi-GPU initialization issues in frameworks that try to enumerate all devices.
- **`cuDeviceGet`**: Validates the ordinal (must be 0), returns a `CUdevice` handle.
- **`cuDeviceGetName`**: Calls the real API, returns the device name string.
- **`cuDeviceTotalMem`**: Calls the real API, returns `uint64_t` total memory.
- **`cuDeviceGetAttribute`**: Calls the real API, returns the attribute value as `int`.
- **`cuDeviceGetUuid`**: Calls the real API, returns the 16-byte `CUuuid` struct.
- **`cuDeviceComputeCapability`**: Calls the real API, returns `{major, minor}` pair.

#### CONTEXT MANAGEMENT (0x0030-0x003C)

- **`cuCtxCreate` / `cuDevicePrimaryCtxRetain`**: Calls `ensure_device_context(ordinal)` to create or retrieve the CUDA context for the given device, then returns the real context pointer.
- **`cuCtxDestroy` / `cuCtxRelease` / `cuCtxReset`**: **NO-OP**. The server manages context lifecycle independently. Client requests to destroy a context are silently ignored because the context must persist for the server's operation.
- **`cuCtxSetCurrent`**: Calls the real `cuCtxSetCurrent`, and updates `g_current_ctx` and `g_current_dev` globals.
- **`cuCtxSynchronize`**: Calls `cuCtxSynchronize`, but **skipped during graph capture** to avoid interfering with the capture process.
- **`cuCtxSetLimit` / `cuCtxGetLimit`**: Direct wrappers around the real CUDA driver calls.

#### MODULE MANAGEMENT (0x0050-0x0055)

- **`cuModuleLoadData`**: Receives raw PTX or CUBIN data in the payload, calls the real `cuModuleLoadData`, and stores the resulting `CUmodule` handle in `g_modules`.
- **`cuModuleLoad`**: Loads a module from the server's filesystem using the path in the payload.
- **`cuModuleUnload`**: Looks up the module handle in `g_modules`, calls the real `cuModuleUnload`, and removes the entry from the map.
- **`cuModuleGetFunction`**: Looks up the module in `g_modules`, calls the real `cuModuleGetFunction`, and stores the `CUfunction` handle in `g_functions`.
- **`cuModuleGetGlobal`**: Looks up the module, calls the real `cuModuleGetGlobal`, returns `{dptr, size}`.

#### MEMORY MANAGEMENT (0x0070-0x0081)

- **`cuMemAlloc`**: Calls the real `cuMemAlloc`, returns the device pointer as `uint64_t`.
- **`cuMemFree`**: Calls the real `cuMemFree`.
- **`cuMemcpyHtoD`**: Payload layout: `dst(8) + size(8) + data(size)`. The data is embedded directly in the payload. Calls the real `cuMemcpyHtoD`.
- **`cuMemcpyDtoH`**: Payload layout: `src(8) + size(8)`. The server allocates a host buffer, calls `cuMemcpyDtoH` to copy from GPU, sends the data as the response payload, then frees the host buffer.
- **`cuMemcpyDtoD`**: Payload layout: `dst(8) + src(8) + size(8)` = **24 bytes total**. This is the **DtoD zero-network optimization** -- no actual data crosses the wire because the data stays on the server's GPU.
- **`cuMemset` variants**: Payload layout: `dst(8) + val(1/2/4) + count(8)`. Sets device memory to a constant value.
- **`cuMemGetInfo`**: Calls the real API, returns `{free, total}` as two `uint64_t` values.
- **`cuMemHostAlloc` / `cuMemFreeHost` / `cuMemHostGetDevicePointer` / `cuPointerGetAttribute` / `cuMemAllocManaged`**: Return `NOT_SUPPORTED`. Pinned memory and managed memory are not supported over the network.

#### EXECUTION CONTROL (0x00A0-0x00A4)

- **`cuLaunchKernel`** (the most complex dispatch case):
  - **44-byte fixed header** in the payload containing: `func_handle(8) + gridDimX/Y/Z(4*3) + blockDimX/Y/Z(4*3) + sharedMemBytes(4) + stream_handle(8)`
  - **Optional parameter buffer** following the header (variable length)
  - **Zero-dimension optimization**: If any grid dimension is 0, returns success immediately (no-op). This avoids launching an empty grid, which some CUDA versions handle poorly.
  - **Stream resolution**: Tries `g_streams` first, then `g_rt_streams`, then falls back to `g_capturing_stream` (during graph capture).
  - **Parameter handling**: Uses `CU_LAUNCH_PARAM_BUFFER_POINTER` and `CU_LAUNCH_PARAM_BUFFER_SIZE` extra arguments to pass kernel parameters as a contiguous buffer. This is more efficient than the older array-of-pointers approach.
  - **Parameter size**: Uses `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_PARAM_SIZE, ...)` to determine the actual parameter buffer size.
  - **No-parameter path**: Falls through to `NULL` extra args for kernels that take no parameters.

#### STREAM MANAGEMENT (0x00C0-0x00C5)

Uses the **dual-map lookup pattern**: checks `g_streams` first, then `g_rt_streams`.

- **Create**: Allocates a wire handle via `alloc_handle()`, stores the mapping, returns the wire handle.
- **Destroy**: Removes the handle from **both** maps.
- **Synchronize**: If handle is 0, calls `cuCtxSynchronize` (synchronizes the default stream). Otherwise, looks up the real stream and calls `cuStreamSynchronize`.

#### EVENT MANAGEMENT (0x00E0-0x00E5)

Same dual-map pattern as streams (`g_events` then `g_rt_events`).

- **Create/Destroy**: Standard handle map operations.
- **Record/Query**: Maps wire handle to real event, calls real API.
- **ElapsedTime**: Takes two event handles, calls `cuEventElapsedTime`, returns float milliseconds.

#### RUNTIME API (0x0100-0x011E)

Mirrors the Driver API but uses `cudaError_t` error codes instead of `CUresult`.

- **`cudaGetDeviceCount`**: Returns 1 (same as Driver API).
- **`cudaMalloc` / `cudaFree`**: Direct wrappers around real API.
- **`cudaMemcpy`**: Unified layout with a `kind` field distinguishing `HtoH`, `HtoD`, `DtoH`, and `DtoD` directions. Each direction has different payload and response handling.
- **`cudaMemcpyAsync`**: Same as `cudaMemcpy` but with an additional stream field.
- **`cudaGetDeviceProperties`**: Calls the real API, serializes the entire `cudaDeviceProp` struct (~700+ bytes) into the response.
- **`RegisterFatBinary`**: 3-tier fallback strategy:
  1. **Cache check**: Look up in `g_fat_cache` by FNV-1a hash. If found, return the cached handle.
  2. **Direct load**: Try `cuModuleLoadData` with the raw payload. If it works, done.
  3. **Fat binary parsing**: Parse the CUDA fat binary format to extract PTX/CUBIN for the server's GPU architecture, decompress if gzip-compressed, then load.
- **`RegisterFunction`**: Associates a kernel function name with a module handle. Uses **bit 63 tagging** to determine whether the module was loaded via Runtime API (bit 63 set) or Driver API, and routes to the appropriate launch path.
- **`cudaLaunchKernel`**: Same logic as `cuLaunchKernel` but returns Runtime API error codes.

#### CUDA GRAPH API (0x0120-0x0129)

- **`StreamBeginCapture`**: Forces relaxed capture mode (`mode=1`). Sets `g_capturing_stream` to the captured stream.
- **`StreamEndCapture`**: Clears `g_capturing_stream`, stores the resulting graph handle in `g_graphs`.
- **`GraphCreate` / `GraphDestroy`**: Standard handle map operations.
- **`GraphInstantiate`**: Instantiates a graph into an executable, stores in `g_graph_execs`.
- **`GraphExecDestroy`**: Removes from `g_graph_execs`.
- **`GraphLaunch`**: Looks up graph executable and stream, calls `cuGraphLaunch`.
- **`GraphExecUpdate`**: Updates a graph executable from a graph node list.

#### cuBLAS (0x0200-0x02A2)

- **`cublasCreate`**: Calls `ensure_device_context(0)` to ensure a valid CUDA context, then calls `cudaGetLastError()` to clear any stale errors before creating the cuBLAS handle. This prevents cuBLAS from seeing leftover error codes from previous operations.
- **`cublasHgemm`**: Not dispatched directly to the real cuBLAS. Instead, converted to a `cublasGemmEx` call with `CUDA_R_32F` accumulation type. This works around a NaN issue in Qwen models with Grouped Query Attention (GQA) where FP16 accumulation produces NaN values.
- **`cublasSgemmBatched`**: The batched GEMM API passes arrays of pointers (one per matrix). The server must allocate device-side pointer arrays, copy the pointer values from the client payload, call the real `cublasSgemmBatched`, then free the device arrays.
- **`cublasLtMatmulAlgoHeuristic`**: Reconstructs cuBLASLt descriptors from the wire format, queries heuristics. Forces FP32 accumulation when the compute type is FP16.
- **`cublasLtMatmul`**: 308-byte payload. Uses a **persistent 32 MB workspace** (`cuMemAlloc` once, reused across calls) rather than allocating and freeing workspace memory per call.

#### cuDNN (0x0300-0x03A2)

All cuDNN descriptor types follow a uniform create/destroy/set/get pattern:

- **Descriptor operations**: Create allocates a wire handle and calls the real `cudnnCreate*`. Destroy removes from the map. Set/Get operations serialize the descriptor fields.
- **Convolution**: Forward, backward data, and backward filter paths.
- **Activation**: Forward and backward.
- **Pooling**: Forward and backward.
- **BatchNorm**: Forward training, forward inference, and backward.
- **Softmax**: Forward and backward.
- **Dropout**: Stub that returns `NOT_SUPPORTED`. Dropout state management is deferred to the client.
- **Workspace size queries**: Return the workspace size needed for a given operation descriptor, allowing the client to allocate appropriately.

#### BULK TRANSFER (0x0500)

- **`AllocAndTransfer`**: A compound RPC that combines `cuMemAlloc` + `cuMemcpyHtoD` + optional `cuMemsetD8` into a single round trip. Payload: `size(8) + memset_val(1) + do_memset(1) + data(size)`. This reduces the number of round trips for common allocation+initialization patterns.

#### GRAPH DECODE (0x0501)

Server-side autoregressive decode loop for transformer inference:

1. For each token in the generation loop:
   - Write the current position to a fixed GPU memory address
   - Launch the CUDA graph (containing the full transformer forward pass)
   - Synchronize the stream
   - Read the logits from GPU memory
   - CPU argmax to select the next token (uses an FP16 bit trick that avoids converting to float32)
   - Write the selected token back to the GPU position buffer
2. Response: `num_generated(4) + token_ids(4 * N)`

This keeps the entire decode loop on the server, avoiding per-token round trips for the forward pass.

#### GRAPH DECODE SPLIT (0x0502)

Pipeline-parallel inference where the client runs some layers and the server runs others:

1. Client sends activations from its layers via HtoD transfer
2. Server updates the position counter in GPU memory
3. Server launches its CUDA graph (server-side layers)
4. Server synchronizes and reads logits via DtoH transfer
5. Server performs CPU argmax
6. Server returns the selected token ID

#### VIRTUAL MEMORY (0x0600-0x0607)

Thin wrappers around the CUDA Driver API virtual memory management functions (`cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease`, `cuMemCreate`, `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`).

- **`cuMemSetAccess`**: Hardcodes device 0 with read/write access. The server only has one GPU, so the access descriptor is always the same.

#### REPLAY CONTROL (0x0400-0x0403)

- **`BeginRecord` / `EndRecord` / `Execute` / `Invalidate`**: Delegates to `replay.c` for recording and replaying sequences of CUDA operations. Used for debugging and performance profiling.

#### DEFAULT

Any unrecognized function ID returns `CUDA_ERROR_NOT_SUPPORTED`. This prevents the server from silently ignoring calls that the client expects to succeed.

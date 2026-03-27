# 11. Partial GPU Offloading

## Concept

The client machine has a **local GPU** AND connects to a **remote GPU server**. Some computation runs locally (via the real CUDA driver behind LD_PRELOAD), while other computation is serialized and sent to the remote server over TCP.

**Architecture constraint:** One client + one remote server. Never multi-server. The client uses its own GPU for some layers and sends overflow layers to the single remote server.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `CHIDORI_LOCAL_DEVICES` | Comma-separated local device indices (e.g., `"0"`) |
| `CHIDORI_REAL_CUDA_PATH` | Path to the real `libcuda.so` on the client machine |
| `CHIDORI_FATBIN_PASSTHROUGH=1` | Pass fat binaries through to both local and remote GPUs |
| `CHIDORI_DEFERRED_INIT=1` | Defer real CUDA registration until first kernel launch |

---

## Initialization (`init_real_cuda`)

The `init_real_cuda` function runs during library load to set up the dual-GPU environment:

1. **Parse `CHIDORI_LOCAL_DEVICES`** into `g_local_device_mask` (a `uint32` bitmask, one bit per device).
2. **Load the REAL `libcuda.so.1`** from system paths:
   - `/usr/lib/libcuda.so.1`
   - `/usr/lib64/libcuda.so.1`
   - `/usr/lib/x86_64-linux-gnu/libcuda.so.1`
   - Uses `realpath()` to resolve symlinks (prevents accidentally loading the interception library itself).
   - Loaded with `RTLD_LAZY | RTLD_LOCAL`.
3. **Set `g_real_cuda_handle_for_dlopen`** -- used later by cuBLAS redirect to find the real driver.
4. **Load ~45 real CUDA function pointers** via `dlsym` from the real library.
5. **Count local GPUs** from the bitmask (number of set bits).

---

## Device Count Merging

The system reports different device counts depending on which API is queried:

- **`cuDeviceGetCount` (Driver API):** Returns ONLY the local device count.
  - This is critical because cuBLAS and cublasLt cache this value internally. It must match the real driver's count to avoid errors.
- **`cudaGetDeviceCount` (Runtime API):** Returns the MERGED count (local + remote).
  - PyTorch discovers devices through this API combined with NVML.
- **`nvmlDeviceGetCount`:** Returns the merged count (via the `libnvidia-ml.so.1` shim library).

---

## Device Routing Logic

Every intercepted CUDA function follows this decision pattern:

1. **Reentrance guard:** If `g_in_real_cuda > 0`, forward directly to the real driver (prevents infinite recursion when real CUDA calls back into intercepted functions).
2. **Local device check:** If `is_local_device(g_current_device)`, forward to the real driver.
3. **Remote device:** Otherwise, serialize the call and send it via RPC to the remote server.
4. **Device index remapping:** Remote device index = `device - g_local_device_count` (the server always uses device 0).

---

## Cross-Device Operations

### `cuMemcpyDtoD` (Cross-Device)

When copying between a local GPU and the remote GPU, data must be staged through CPU memory:

```
Local -> Remote:
  1. cuMemcpyDtoH(ptr_local, cpu_buffer, size)   -- real driver, local GPU
  2. RPC: cuMemcpyHtoD(ptr_remote, cpu_buffer, size)  -- remote GPU

Remote -> Local:
  1. RPC: cuMemcpyDtoH(cpu_buffer, ptr_remote, size)  -- remote GPU
  2. cuMemcpyHtoD(ptr_local, cpu_buffer, size)   -- real driver, local GPU
```

`alloc_track_device(ptr)` determines which GPU owns a given pointer to decide the staging direction.

### `cuLaunchKernel` (Cross-Device Pointer Translation)

When launching a kernel on the remote GPU that references local GPU memory pointers:

1. Scans the kernel parameter buffer for pointers.
2. Range-checks each pointer against `g_alloc_track[]`.
3. For local GPU pointers: copies data to remote GPU memory, substitutes the pointer in the parameter buffer.
4. This is **critical for the backward pass** where autograd kernels receive tensors stored on the local GPU.

### `cuMemset` (Routes by Address, Not Device)

Unlike most operations that route based on `g_current_device`, `cuMemset` routes based on the memory address:

1. Uses `alloc_track_device()` to determine which GPU owns the target address.
2. Forwards to the appropriate GPU (local or remote).
3. Ensures backward pass correctness where gradient zeroing must happen on the correct device.

---

## Fat Binary Handling

- **`__cudaRegisterFatBinary`:** Stores the fat binary locally and registers it with BOTH the real CUDA driver AND the remote server.
- **Lazy registration:** Fat binaries are registered on-demand with client-side deduplication to avoid redundant registrations.
- **Deferred mode** (`CHIDORI_DEFERRED_INIT=1`): Queues registrations and replays them after real CUDA initialization completes. This avoids ordering issues during library load.

---

## `cudaSetDevice` Special Case

- **For local devices:** Uses the real driver's `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` directly, bypassing `cudart` to avoid error 36 (CUDA error: context already current and in use).
- **For remote devices:** Skips the RPC entirely. The server always uses device 0, so setting the device on the client side only updates the local routing variable.

---

## Event Tracking

- **`g_event_track[4096]`:** Maps event handles to the device on which they were created.
- **Critical for cross-device backward pass:** An event created on `cuda:0` (local) that is later synced on `cuda:1` (remote) must be routed to the correct GPU. Without this tracking, the sync would be sent to the wrong device, causing deadlocks or incorrect results.

---

## Allocation Tracking

- **`g_alloc_track[16384]`:** Maps GPU pointer ranges to device ownership.
- Protected by `g_alloc_track_lock` (`pthread_mutex_t`).
- **Used by:**
  - Cross-device `cuMemcpyDtoD` staging direction.
  - `cuLaunchKernel` pointer translation.
  - `cuMemset` address-based routing.
- Each entry stores: base pointer, size, device index.

---

## NVML Proxy (`libnvidia-ml.so.1`)

A separate shared library built from `chidori_nvml.c`:

- **`nvmlDeviceGetCount`:** Returns the merged device count by calling `cudaGetDeviceCount` via `dlsym(RTLD_DEFAULT, "cudaGetDeviceCount")`.
- **All other NVML functions:** Forwarded directly to the real `libnvidia-ml.so.1`.
- This ensures PyTorch's device discovery (which queries NVML) sees both local and remote GPUs.

---

## Test Coverage

### Basic Tests

- **`test_partial_offload_basic.py`:** Device enumeration -- verifies 2 devices are visible, both are valid CUDA devices.

### Compute Tests

- **`test_partial_offload_compute.py`:** Independent matmul operations on both devices.

### Transfer Tests

- **`test_partial_offload_transfer.py`:** Data integrity when transferring tensors between local and remote devices.

### Split Model Tests

- **`test_partial_offload_split_model.py`:** 2-layer model split across devices with full backward pass.
- **`test_partial_offload_device_map.py`:** TinyLlama-1.1B with `accelerate` `device_map="auto"` for automatic layer distribution.

### Comprehensive Tests

- **`test_partial_offload_full.py`:** 16+ tests covering all partial offload operations.
- **`test_partial_offload_advanced.py`:** 13 advanced tests including large matmul, FP16 precision, transformer models, gradient accumulation, and split model training.

### Split Inference Tests

- **`test_split_inference.py`:** TinyLlama split inference via `accelerate.dispatch_model`.
- **`test_split_graph.py`:** Split inference with `StaticCache` for KV cache management.

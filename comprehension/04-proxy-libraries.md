# Proxy Libraries

## Why Proxy Libraries Are Needed

Chidori's core interception mechanism is `LD_PRELOAD` -- the client library (`libchidori.so`) is loaded before all other shared libraries and intercepts CUDA API calls by providing its own implementations of CUDA symbols. However, `LD_PRELOAD` alone is insufficient for intercepting calls made by **other libraries** that link against CUDA at the ELF level.

The problem manifests with cuBLAS specifically. PyTorch links against `libcublas.so` using **versioned ELF symbols** -- for example, `cublasGemmEx@libcublas.so.12`. The GNU dynamic linker resolves versioned symbol references by matching the version tag against the `DT_NEEDED` entry and version definitions in the shared library providing the symbol. Even when `libchidori.so` is loaded first via `LD_PRELOAD`, the linker will still resolve `cublasGemmEx@libcublas.so.12` to the **real** `libcublas.so.12` on the system, because that library actually defines the version `libcublas.so.12`.

The solution is to build **proxy libraries** that have the same soname and versioned symbols as the real cuBLAS library. When `LD_LIBRARY_PATH` is set to the build directory, the dynamic linker finds our proxy `libcublas.so.12` (or `.13`) instead of the real one, and all versioned references resolve to our intercepted implementations.

## Build Process

The same source files (`chidori_client.c` + `chidori_net.c`) are compiled five times with different `-Wl,-soname` values and different linker version scripts. Each output is a shared library that masquerades as a different real library:

| Output File | Soname | Source | Purpose |
|---|---|---|---|
| `build/libcublas.so.12` | `libcublas.so.12` | `chidori_client.c` + `chidori_net.c` | cuBLAS proxy for CUDA 12.x |
| `build/libcublasLt.so.12` | `libcublasLt.so.12` | `chidori_client.c` + `chidori_net.c` | cuBLASLt proxy for CUDA 12.x |
| `build/libcublas.so.13` | `libcublas.so.13` | `chidori_client.c` + `chidori_net.c` | cuBLAS proxy for CUDA 13.x (Arch Linux) |
| `build/libcublasLt.so.13` | `libcublasLt.so.13` | `chidori_client.c` + `chidori_net.c` | cuBLASLt proxy for CUDA 13.x |
| `build/libnvidia-ml.so.1` | `libnvidia-ml.so.1` | `chidori_nvml.c` | NVML proxy |

The Makefile target `proxy-libs` (or `all`) builds all five. Each cuBLAS variant uses a different `-Wl,--version-script` argument to define the appropriate symbol version set.

## GNU Linker Version Script (`chidori_client.map`)

The linker version script is applied at link time via `-Wl,--version-script=chidori_client.map`. This is the mechanism that makes the dynamic linker match versioned references to our proxy.

The file defines two versioned symbol groups:

**`libcublas.so.12` version (35 symbols):**
- Handle management: `cublasCreate`, `cublasCreate_v2`, `cublasDestroy`, `cublasDestroy_v2`, `cublasSetStream`, `cublasSetStream_v2`, `cublasGetStream`, `cublasGetStream_v2`, `cublasGetVersion`, `cublasGetVersion_v2`, `cublasGetCudartVersion`
- GEMM family: `cublasGemmEx`, `cublasGemmStridedBatchedEx`, `cublasSgemm`, `cublasSgemm_v2`, `cublasDgemm`, `cublasDgemm_v2`, `cublasHgemm`, `cublasHgemm_v2`
- BLAS1: `cublasSaxpy_v2`, `cublasDaxpy_v2`, `cublasSscal_v2`, `cublasDscal_v2`
- Batched: `cublasSgemmBatched`, `cublasSgemmStridedBatched`, `cublasDgemmBatched`, `cublasDgemmStridedBatched`, `cublasHgemmBatched`, `cublasHgemmStridedBatched`
- Other: `cublasSgeam`, `cublasDgeam`

**`libcublasLt.so.12` version (25 symbols):**
- Handle management: `cublasLtCreate`, `cublasLtDestroy`
- Matmul: `cublasLtMatmul`, `cublasLtMatmulAlgoGetHeuristic`, `cublasLtMatmulAlgoGetHeuristic_v2`
- Descriptors: `cublasLtMatrixLayoutCreate`, `cublasLtMatrixLayoutDestroy`, `cublasLtMatmulDescCreate`, `cublasLtMatmulDescDestroy`, `cublasLtMatmulPreferenceCreate`, `cublasLtMatmulPreferenceDestroy`
- Layouts: `cublasLtMatrixLayoutSetAttribute`, `cublasLtMatrixLayoutGetAttribute`
- Preferences: `cublasLtMatmulPreferenceSetAttribute`, `cublasLtMatmulPreferenceGetAttribute`, `cublasLtMatmulDescSetAttribute`, `cublasLtMatmulDescGetAttribute`
- Heuristics: `cublasLtMatmulAlgoHeuristicResultInit`

When PyTorch loads and resolves `cublasGemmEx@libcublas.so.12`, the dynamic linker finds our `libcublas.so.12` proxy in `LD_LIBRARY_PATH`, sees it exports `cublasGemmEx` under the `libcublas.so.12` version, and binds to our implementation.

## NVML Proxy (`libnvidia-ml.so.1`)

The NVML proxy is built from a separate source file, `chidori_nvml.c`, and serves a different purpose than the cuBLAS proxies.

**Why it exists:** PyTorch calls `nvmlDeviceGetCount` via Python's `ctypes` module **before** it calls `cudaGetDeviceCount`. This happens during PyTorch initialization when it probes available GPUs. In partial offload mode (where the client has a local GPU and also connects to a remote server), the client needs to report a merged device count that includes both local and remote devices.

**Key behavior:**
- `nvmlDeviceGetCount`: In partial offload mode, returns the merged count (local + remote). Uses `dlsym(RTLD_DEFAULT, "cudaGetDeviceCount")` to resolve to our intercepted version of `cudaGetDeviceCount`, which itself knows about remote devices.
- All other NVML functions: loaded from the real `libnvidia-ml.so.1` via `dlopen` and forwarded. The proxy is transparent for any NVML call it does not specifically intercept.

## dlopen Redirect Mechanism

cuBLAS internally calls `dlopen("libcuda.so.1")` during its own initialization to get direct access to the CUDA driver. If this dlopen resolves to our `libchidori.so` (which is also linked as `libcuda.so.1`), cuBLAS would call our intercepted CUDA functions, creating infinite recursion and broken behavior.

The redirect mechanism prevents this:

**`g_dlopen_redirect_active`** (`__thread` thread-local flag):
- When set to 1, our intercepted `dlopen` function checks if the requested library name matches `"libcuda.so.1"`
- If it matches: returns `g_real_cuda_handle_for_dlopen`, which is the handle obtained from a real `dlopen` of the NVIDIA driver library (resolved by its full filesystem path, bypassing `LD_LIBRARY_PATH`)
- This ensures cuBLAS gets the real NVIDIA driver functions, not our intercepted versions

**`g_in_real_cuda`** (`__thread` reentrance guard):
- Incremented around every call to a real CUDA function on the client side
- When > 0: also enables dlopen redirect (same behavior as `g_dlopen_redirect_active`)
- Prevents any intercepted CUDA calls made by the real CUDA driver (or cuBLAS calling into the driver) from being re-intercepted

## Symbol Interception in dlsym

The client library intercepts `dlsym` itself, which is necessary because libraries like cuBLAS and cuDNN use `dlsym` to look up CUDA functions at runtime. Our intercepted `dlsym` must return our proxy implementations for cuBLAS/cuBLASLt symbols while passing everything else through.

**Getting the real dlsym:**
- Uses `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5")` to get the real `dlsym` function pointer
- `dlvsym` with a version tag avoids infinite recursion (calling `dlsym` to find `dlsym`)

**Lookup logic (in order):**
1. If `g_in_real_cuda > 0`: pass through directly to real `dlsym`. This handles the case where the real CUDA driver or cuBLAS is doing its own symbol lookups internally.
2. Check the symbol name against `g_intercepted_cublas_symbols[]` (48 entries covering cuBLAS and cuBLASLt functions).
3. If found: return our implementation via `dlsym(RTLD_DEFAULT, symbol_name)`, which finds the symbol in our already-loaded proxy library.
4. If not found: pass through to real `dlsym`.

This mechanism is what allows cuBLAS/cuBLASLt calls from PyTorch to be routed through our proxy library without PyTorch needing any modification.

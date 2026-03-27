# libcudart.so.13.1.80 String Table Analysis

**Binary**: `/opt/cuda/targets/x86_64-linux/lib/libcudart.so.13.1.80`
**Size**: 740KB (740,800 bytes)
**Text section**: 0xe8e0..0x907c4 (531,172 bytes / 519 KB)
**Exported symbols**: 449 (all in .dynsym)
**Strings extracted**: 3,889 (min 6 chars)

---

## 1. Binary Architecture

### Dynamic Dependencies (NEEDED)

```
libdl.so.2          ← dlopen/dlsym/dlvsym/dlmopen (ALL Driver API resolved this way)
libpthread.so.0     ← mutexes, rwlocks, TLS, spinlocks, condvars
librt.so.1          ← semaphores, shm, clock_gettime
libstdc++.so.6      ← C++ runtime (C++ ABI but no mangled exports)
libm.so.6           ← math
libgcc_s.so.1       ← unwind, stack protector
libc.so.6           ← everything else
ld-linux-x86-64.so.2
```

### Critical: NO Direct Link to libcuda.so

**libcudart.so has ZERO imports from libcuda.so.** All Driver API functions (`cuModuleLoad`, `cuLaunchKernel`, `cuCtxSetCurrent`, etc.) are resolved at runtime via `dlsym()` / `dlvsym()` from `libcuda.so.1`. The string `"libcuda.so.1"` is at file offset `0x9413c`.

This means:
- The library opens `libcuda.so.1` with `dlopen()` at first use
- Each `cu*` function is resolved by name via `dlsym(RTLD_DEFAULT, "cuModuleLoad")` etc.
- The library stores the dlopen handle (visible at runtime_state+0xd8)

### Key GLIBC Imports

| Import | Purpose |
|--------|---------|
| `dlmopen` | Open libcuda.so in isolated namespace (avoids symbol conflicts) |
| `dlvsym` | Versioned dlsym (resolves e.g. `cuCtxGetDevice@@CUDA_12.0`) |
| `getenv` | Reads CUDA environment variables |
| `pthread_key_create/del/get/set` | Thread-local storage for per-thread CUDA state |
| `pthread_spin_lock/unlock` | Fast spinlocks for hot paths |
| `mmap/mprotect/munmap` | Code patching, JIT buffer allocation |
| `shm_open/shm_unlink/shmat/shmdt/shmget/shmctl` | Shared memory for multi-process |
| `socket/socketpair/connect/bind/listen/sendmsg/recvmsg` | IPC (likely for multi-GPU, MPS) |
| `__tls_get_addr` | Thread-local storage access (module mgmt state?) |
| `uname` | System identification |
| `/proc/self/maps`, `/proc/self/status`, `/proc/self/exe` | Runtime introspection |

---

## 2. Registration & Resolution Symbols (Exported)

| Offset | Symbol | Role |
|--------|--------|------|
| `0x23c60` | `cudaGetExportTable` | Undocumented export table interface |
| `0x23cc0` | `__cudaRegisterFatBinary` | Register fat binary (legacy nvcc path) |
| `0x23cf0` | `__cudaRegisterFatBinaryEnd` | End batch registration |
| `0x23d20` | `__cudaUnregisterFatBinary` | Unregister at dlclose/shutdown |
| `0x23d60` | `__cudaRegisterVar` | Register `__device__` variables |
| `0x23de0` | `__cudaRegisterManagedVar` | Register `__managed__` variables |
| `0x23e60` | `__cudaRegisterHostVar` | Register `__device__` host-accessible vars |
| `0x23ea0` | `__cudaRegisterFunction` | Register kernel metadata (name, stub, params) |
| `0x23f10` | `__cudaInitModule` | Register entire module (CUDA 12+ nvcc path) |
| `0x23ff0` | `__cudaPushCallConfiguration` | Push launch config for `<<<>>>` syntax |
| `0x24090` | `__cudaPopCallConfiguration` | Pop launch config |
| `0x24140` | `__cudaGetKernel` | Internal: resolve kernel by name? |
| `0x24240` | `__cudaLaunchKernel` | Internal: launch from `<<<>>>` |
| `0x24260` | `__cudaRegisterUnifiedTable` | CUDA unified memory registration |
| `0x242b0` | `__cudaGetProcAddress` | Get function pointer for any CUDA symbol |
| **`0x79600`** | **`cudaGetFuncBySymbol`** | **Resolve host stub → device function** |
| `0x79810` | `cudaGetKernel` | Resolve kernel by name (public API) |

### Key Offset Ranges

| Range | Contents |
|-------|----------|
| `0x23c60..0x242b0` | Registration/init infrastructure (legacy + new) |
| `0x4ccf0..0x548c0` | Device, stream, event management |
| `0x57070..0x79600` | Kernel launch, memcpy, memory ops |
| `0x79600..0x79810` | Function resolution (cudaGetFuncBySymbol, cudaGetKernel) |
| `0x7fc60` | `cudaLaunchKernel` (public) |

---

## 3. Error Strings (Complete, with File Offsets)

### Lifecycle & Initialization Errors

| File Offset | String | Error Code |
|-------------|--------|------------|
| `0x98391` | `cudaErrorInitializationError` | 3 |
| `0x98412` | `cudaErrorCudartUnloading` | **4** |
| `0x9842b` | `driver shutting down` | (detail for 4) |
| `0x9856a` | `cudaErrorDeviceUninitialized` | 35 |
| `0x98ad6` | `system not yet initialized` | (detail) |
| `0x9909c` | `cudaErrorStartupFailure` | 127 |
| `0x99aaa8` | `startup failure in cuda runtime` | (detail for 127) |
| `0x99138` | `profiler not initialized: call cudaProfilerInitialize()` | (detail) |
| `0x99118` | `cudaErrorProfilerNotInitialized` | 126 |

### Kernel & Module Errors

| File Offset | String | Error Code |
|-------------|--------|------------|
| `0x9854e` | `cudaErrorInvalidKernelImage` | 34 |
| `0x986f1` | `cudaErrorInvalidPtx` | 37 |
| `0x98705` | `a PTX JIT compilation failed` | (detail for 37) |
| `0x9873f` | `cudaErrorJitCompilerNotFound` | 42 |
| `0x99568` | `PTX JIT compiler library not found` | (detail for 42) |
| `0x99590` | `cudaErrorUnsupportedPtxVersion` | 211 |
| `0x995f0` | `the provided PTX was compiled with an unsupported toolchain.` | (detail for 211) |
| `0x99610` | `cudaErrorJitCompilationDisabled` | 213 |
| `0x99638` | `PTX JIT compilation was disabled` | (detail for 213) |
| **`0x99370`** | **`cudaErrorNoKernelImageForDevice`** | **209** |
| **`0x99390`** | **`no kernel image is available for execution on the device`** | (detail for 209) |
| `0x98c5f` | `cudaErrorFunctionNotLoaded` | 214 |

### Context Errors

| File Offset | String | Error Code |
|-------------|--------|------------|
| `0x988df` | `cudaErrorContextIsDestroyed` | 68 |
| `0x988e2` | `context is destroyed` | (detail) |
| `0x9a880` | `cudaErrorIncompatibleDriverContext` | 501 |
| `0x9902a` | `incompatible driver context` | (detail) |

### Driver Compatibility Errors

| File Offset | String | Error Code |
|-------------|--------|------------|
| `0x984a1` | `CUDA driver is a stub library` | (detail) |
| `0x94149` | `Failed to load CUDA driver!` | (init error) |
| `0x94165` | `Insufficient driver: %d < %d` | (init error) |
| `0x98fa0` | `cudaErrorInsufficientDriver` | 35 |
| `0x98af1` | `cudaErrorSystemDriverMismatch` | 81 |
| `0x991b0` | `cudaErrorCallRequiresNewerDriver` | 222 |
| `0x991d8` | `API call is not supported in the installed CUDA driver` | (detail) |
| `0x9a788` | `CUDA driver version is insufficient for CUDA runtime version` | (detail) |
| `0x99be8` | `system has unsupported display driver / cuda driver combination` | (detail) |
| `0x9848c` | `cudaErrorStubLibrary` | 34 |

### Symbol & Function Errors

| File Offset | String | Error Code |
|-------------|--------|------------|
| `0x98806` | `cudaErrorSymbolNotFound` | 61 |
| `0x98835` | `named symbol not found` | (detail) |
| `0x98d70` | `cudaErrorInvalidSymbol` | 79 |
| `0x99810` | `cudaErrorSharedObjectSymbolNotFound` | 400 |
| `0x99870` | `shared object symbol not found` | (detail) |

### Device Errors

| File Offset | String | Error Code |
|-------------|--------|------------|
| `0x984bf` | `cudaErrorDevicesUnavailable` | 9 |
| `0x984db` | `cudaErrorNoDevice` | 10 |
| `0x984ed` | `cudaErrorInvalidDevice` | 11 |
| `0x9851b` | `cudaErrorDeviceNotLicensed` | 13 |
| `0x986d5` | `cudaErrorDeviceAlreadyInUse` | 36 |
| `0x98bbe` | `cudaErrorCdpNotSupported` | 206 |
| `0x98bb2` | `cudaErrorCdpVersionMismatch` | 207 |

---

## 4. Internal State & Diagnostic Strings

### Module Management

| File Offset | String | Significance |
|-------------|--------|--------------|
| `0x91cd0` | `CUDA_ENABLE_MODULE_LAZY_LOADING` | **Environment variable** controlling module init behavior |
| `0x920ff` | `cuModuleGetLoadingMode` | Driver API: query loading mode (lazy/eager) |
| `0x91280` | `Duplicate entry kernels named "%s" detected. The Runtime will use the one that was registered first` | Kernel deduplication warning |
| `0x90f45` | `unrecognized error code` | Error string fallback |
| `0x9110b` | `kernelPtr` | Field name in device properties or internal struct |

### Context Management

| File Offset | String | Significance |
|-------------|--------|--------------|
| `0x91128` | `Invalid ctx specified. Only contexts explicitly created via CUDA Runtime APIs can be destroyed.` | **Runtime-created contexts vs Driver API contexts** |
| `0x911d8` | `Cannot use this function with a non-primary current context` | Primary context requirement |
| `0x91e75` | `cuDevicePrimaryCtxRetain` | dlsym name for primary context |

### Driver Loading

| File Offset | String | Significance |
|-------------|--------|--------------|
| `0x9413c` | `libcuda.so.1` | Driver library filename (dlopen target) |
| `0x94149` | `Failed to load CUDA driver!` | Fatal init error |
| `0x94165` | `Insufficient driver: %d < %d` | Version check (runtime > driver) |
| `0x91d47` | `cuInit` | First dlsym call |
| `0x7b96` | `libcudart.so.13` | SONAME |

### Device Properties / Init

| File Offset | String | Significance |
|-------------|--------|--------------|
| `0x91cf0` | `Invalid device number %d provided, must be in the range of 0 to %d` | Bounds check |
| `0x90fc6` | `mipmapDesc` | Internal struct field name |
| `0x9101e` | `mipmappedArrayPtr` | Internal struct field |
| `0x9104e` | `driverVersion` | Device property |
| `0x9105c` | `runtimeVersion` | Device property |
| `0x910f3e` | `device` | Property category |

### Shared Memory & IPC

| File Offset | String | Significance |
|-------------|--------|--------------|
| `0x9e8fb` | `/cudart.shm.%x.%x.%llx` | Shared memory path template |
| `0x9e922` | `/cudart.shm.%08llx%08llx` | Alt shared memory path |
| `0x9e631` | `/proc/self/status` | Process introspection |
| `0x9e760` | `/proc/self/exe` | Executable path |
| `0x9e8af` | `/proc/self/maps` | Memory map introspection |
| `0x9e950` | `/proc/self/task/%d/comm` | Thread name introspection |

---

## 5. Driver API Function Names (dlsym targets)

The Runtime resolves these from `libcuda.so.1` at startup. Listed by category:

### Core Context

| dlsym Name | Purpose |
|------------|---------|
| `cuInit` | Initialize driver |
| `cuDeviceGet` | Get device handle |
| `cuDeviceGetCount` | Count devices |
| `cuDeviceGetName` | Device name |
| `cuDeviceTotalMem_v2` | Total memory |
| `cuDeviceGetAttribute` | Device capability query |
| `cuDeviceGetByPCIBusId` | PCI bus lookup |
| `cuDeviceGetPCIBusId` | PCI bus query |
| `cuDeviceGetUuid` | GPU UUID |
| `cuDeviceGetP2PAttribute` | Peer access capability |
| `cuDeviceCanAccessPeer` | Peer access check |
| `cuDevicePrimaryCtxRetain` | **Primary context retain** |
| `cuDevicePrimaryCtxRelease` | Primary context release |
| `cuDevicePrimaryCtxSetFlags_v2` | Primary context flags |
| `cuDevicePrimaryCtxGetState` | Primary context state |
| `cuDevicePrimaryCtxReset` | Primary context reset |
| `cuCtxGetApiVersion` | Context API version |
| `cuCtxGetDevice` | Context's device |
| `cuCtxGetDevice_v2` | v2 |
| `cuCtxGetCacheConfig` | Cache config |
| `cuCtxSetCacheConfig` | Cache config |
| `cuCtxResetPersistingL2Cache` | L2 cache |
| `cuCtxSetCurrent` | **Set current context** |
| `cuCtxGetCurrent` | Get current context |
| `cuDriverGetVersion` | Driver version |
| `cuGetErrorString` | Error string |
| `cuGetErrorName` | Error name |
| `cuGetProcAddress` | Symbol lookup |
| `cuGetProcAddress_v2` | v2 |

### Module Loading (Key for Chidori)

| dlsym Name | Purpose |
|------------|---------|
| `cuModuleLoad` | Load module from file |
| `cuModuleLoadData` | **Load module from memory** |
| `cuModuleLoadFatBinary` | Load fat binary |
| `cuModuleUnload` | Unload module |
| `cuModuleGetFunction` | **Get kernel handle** |
| `cuModuleGetGlobal_v2` | Get device variable |
| `cuModuleGetTexRef` | Get texture ref |
| `cuModuleGetSurfRef` | Get surface ref |
| `cuModuleGetLoadingMode` | Query loading mode |

### Kernel Launch

| dlsym Name | Purpose |
|------------|---------|
| `cuLaunchKernel` | Launch kernel |
| `cuLaunchKernel_ptsz` | Per-thread-stream version |
| `cuLaunchKernelEx` | Extended launch |
| `cuLaunchKernelEx_ptsz` | Extended per-thread-stream |
| `cuLaunchCooperativeKernel_ptsz` | Cooperative launch |
| `cuLaunchHostFunc` | Host function callback |
| `cuLaunchHostFunc_ptsz` | Per-thread-stream |
| `cuFuncSetCacheConfig` | Kernel cache config |
| `cuKernelSetCacheConfig` | Kernel cache config (new) |
| `cuKernelSetAttribute` | Kernel attribute |
| `cuKernelGetAttribute` | Kernel attribute |
| `cuKernelGetFunction` | Kernel from library |
| `cuKernelGetName` | Kernel name |
| `cuKernelGetParamInfo` | Kernel parameter info |

### Linking

| dlsym Name | Purpose |
|------------|---------|
| `cuLinkCreate` | Create link state |
| `cuLinkAddData` | Add data to link |
| `cuLinkAddFile` | Add file to link |
| `cuLinkComplete` | Complete linking |
| `cuLinkDestroy` | Destroy link state |

### Memory

| dlsym Name | Purpose |
|------------|---------|
| `cuMemAlloc` | Allocate device memory |
| `cuMemAlloc_v2` | v2 |
| `cuMemFree` | Free device memory |
| `cuMemFree_v2` | v2 |
| `cuMemcpy` | Memory copy |
| `cuMemcpyAsync` | Async memory copy |
| `cuMemHostRegister_v2` | Register host memory |
| `cuMemHostUnregister` | Unregister host memory |
| `cuMemHostGetDevicePointer_v2` | Get device pointer |
| `cuMemPoolExportToShareableHandle` | Shareable pool handle |
| `cuMemPoolImportFromShareableHandle` | Import pool handle |
| `cuDeviceGetDefaultMemPool` | Default mem pool |
| `cuDeviceSetMemPool` | Set mem pool |

---

## 6. Environment Variables

| File Offset | Variable | Purpose |
|-------------|----------|---------|
| `0x91cd0` | `CUDA_ENABLE_MODULE_LAZY_LOADING` | Controls module loading deferral |

Note: Other well-known CUDA env vars (`CUDA_VISIBLE_DEVICES`, `CUDA_LAUNCH_BLOCKING`, `CUDA_CACHE_MAXSIZE`, `CUDA_CACHE_DISABLE`, `CUDA_FORCE_PTX_JIT`, etc.) are NOT present as strings in libcudart.so. They may be read via `getenv()` using inline string constants or handled by the driver.

---

## 7. What's NOT in the String Table (Equally Informative)

### No Compression Strings
- No "zstd", "decompress", "compress", "inflate", "deflate", "snappy", "lz4", "a2"
- The a2 decompression is likely inlined machine code with no string references
- zstd may be linked statically (no symbol) or also inlined

### No Architecture Strings
- No "sm_89", "sm_90", "sm_120", "compute_89", etc.
- Architecture matching uses numeric comparisons, not string comparisons
- The only arch reference: `0x3852` "aarch64" (platform detection)

### No Format/Magic Strings
- No "0xba55ed50" (fat binary magic) as a string
- No "ELF", "PTX", "cubin" as searchable strings
- Binary format detection is done byte-by-byte with immediate constants

### No C++ Mangled Names
- All exports are C-linkage (`T` type in nm, no `_Z` prefixes)
- NVIDIA strips C++ symbols from the public ABI
- Internal C++ classes/methods exist but are invisible in the dynamic symbol table

### No Debug Strings
- No "assert", "sanity", "invariant", "should_not_reach"
- No file/line references (stripped in release build)
- The only debug-like string: `"unrecognized error code"` (fallback in error string lookup)

---

## 8. String Offset to Virtual Address Mapping

The library is PIE with the first LOAD segment at vaddr=0, so file offsets == virtual addresses for the first ~740KB. For cross-referencing with `objdump -d`:

```
virtual_address = file_offset  (for offsets < 0xb3000)
```

To find which function references a string:
```bash
objdump -d /opt/cuda/targets/x86_64-linux/lib/libcudart.so.13.1.80 | grep -B5 "0xOFFSET"
```

Example: `0x91280` = "Duplicate entry kernels" → referenced by the registration dedup code.

---

## 9. Files Generated

| File | Contents | Lines |
|------|----------|-------|
| `libcudart_strings_all.txt` | All strings (min 6 chars), no offsets | 3,889 |
| `libcudart_strings_offset.txt` | All strings with hex file offsets | 3,889 |
| `libcudart_symbols_exported.txt` | All exported symbols with offsets | 449 |
| `libcudart_error_strings_offsets.txt` | Error strings with file offsets | 70 |
| `libcudart_internal_strings_offsets.txt` | Internal strings (env vars, paths) with offsets | 11 |

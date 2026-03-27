# 09 -- Fat Binary Handling

## Overview

CUDA applications ship GPU code in "fat binaries" -- containers holding PTX (portable intermediate representation) and/or SASS (native GPU machine code) for multiple GPU architectures. When a client intercepts `cuRegisterFatBinary` via `LD_PRELOAD`, it receives the raw fat binary blob from the application (compiled for the client's GPU architecture) and must load it on the server (which may have a different GPU architecture).

This is the most complex subsystem in Chidori because it must handle architecture mismatches, multiple compression formats, and fallback between Driver API and Runtime API paths.

## Fat Binary Format

### Magic Numbers

| Magic | Value | Description |
|---|---|---|
| ELV | `0xba55ed50` | CUDA fat binary v1 (ELF container) |
| FATBIN | `0x466243b1` | CUDA fat binary v2 (used by Runtime API) |

### Structure

A fat binary contains entries for multiple GPU architectures. Each entry holds either:

- **PTX**: Portable intermediate code that can be JIT-compiled for any architecture (slow but universal)
- **SASS**: Native machine code for a specific SM version (fast, but architecture-specific)

Entries may be compressed (Zstd, LZ4, or NVIDIA's a2-prefix format).

## Three-Tier Fallback Strategy

The `RegisterFatBinary` handler in `cuda_dispatch.c` (~320 lines) implements a four-tier fallback strategy to load the fat binary on the server.

### Tier 1: Cache Lookup

```c
static struct {
    uint64_t hash;         // FNV-1a hash of the fat binary
    uint32_t payload_size; // Collision check (hash + size must both match)
    uint64_t wire_handle;  // Previously allocated handle
} g_fat_cache[FAT_CACHE_MAX]; // 1024 entries
```

**Hash computation**: FNV-1a over the first 4KB + last 4KB + total size of the payload. This provides a fast probabilistic match without hashing the entire (potentially large) fat binary.

**Purpose**: PyTorch registers the same fat binary multiple times (once per CUDA kernel in the same compilation unit). Without caching, each registration would trigger `cuModuleLoadData`, which is expensive (~10ms). With caching, subsequent registrations of the same binary return the cached handle immediately.

**Collision safety**: Both hash AND payload_size must match, reducing collision probability to near-zero for practical workloads.

### Tier 2: Direct cuModuleLoadData

If the cache misses, the server attempts to load the raw fat binary directly:

```c
result = cuModuleLoadData(&mod, payload);
```

This works when the fat binary contains SASS code matching the server's GPU architecture. For example, if the client has an Ada Lovelace GPU (sm_89) and the server also has sm_89, the SASS code in the fat binary will execute directly on the server.

**On success**: The module handle is stored in `g_modules`, and the fat binary is cached in `g_fat_cache` for future registrations.

### Tier 3: Fat Binary Parsing + Decompression (~220 lines)

When Tier 2 fails (returns `CUDA_ERROR_INVALID_IMAGE`), it typically means the fat binary does not contain SASS for the server's architecture. Tier 3 parses the fat binary to extract compatible code.

**Step 1: Validate magic**
Checks that the payload starts with `0xba55ed50`.

**Step 2: Get server SM version**
```c
cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
```

**Step 3: Scan entries and select by priority**

The parser scans all entries in the fat binary and selects the best match:

| Priority | Match Type | Description |
|---|---|---|
| a (best) | ELF with exact SM match | `sm_major.sm_minor` exactly matches server |
| b | ELF with same major arch, compatible minor | Same major, server minor >= entry minor |
| c | Any older ELF | Any SASS entry, regardless of version |
| d (worst) | PTX fallback | Portable code, JIT-compiled on server (slowest) |

**Step 4: Decompression**

If the selected entry is compressed, it must be decompressed before loading:

| Format | Magic Bytes | Method |
|---|---|---|
| Zstd | `28 b5 2f fd` | `ZSTD_decompress()` from libzstd (linked at build time: `-lzstd`) |
| LZ4 | `04 22 4d 18` | Custom inline LZ4 block decompressor, specifically for CUDA 11.x fat binaries |
| a2-prefix | `a2 7f` | NVIDIA format marker; strips the 2-byte marker and zero-pads to 16-byte alignment |

**Step 5: Load decompressed data**
```c
result = cuModuleLoadData(&mod, decompressed_data);
```

### Tier 4: Runtime API Fallback

If all Driver API attempts fail (Tier 2 and Tier 3 both fail), the system falls back to the CUDA Runtime API registration path:

**Step 1: Resolve the Runtime API registration function**
```c
typedef void* (*register_fn_t)(void**);
register_fn_t rt_register = dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
```

**Step 2: Build a `__fatBinC_Wrapper` struct**
This struct has magic `0x466243b1` (FATBIN format, not ELV) and wraps the fat binary data for the Runtime API.

**Step 3: Call Runtime API registration**
```c
void* rt_handle = rt_register(&wrapper);
```

**Step 4: Bit 63 tagging**
The handle returned by the Runtime API is stored with bit 63 set:
```c
wire_handle = rt_handle | (1ULL << 63);
```

This high-bit tag is critical: it tells `RegisterFunction` (see below) to use the Runtime API path instead of the Driver API path, because the Runtime API and Driver API have different handle namespaces.

## Function Registration (`RegisterFunction`, ~130 lines)

After a fat binary is loaded, the application registers individual kernel functions from it via `cuRegisterFunction`. This subsystem must resolve function names to executable `CUfunction` handles.

### Same-Architecture Path (bit 63 NOT set)

When the fat binary was loaded via Driver API (Tier 2 or Tier 3):

```c
cuModuleGetFunction(&func, module, func_name);
```

Straightforward lookup by name in the loaded module.

### Cross-Architecture Path (bit 63 IS set)

When the fat binary was loaded via Runtime API fallback (Tier 4), the module handle is a Runtime API handle, not a Driver API handle. The system must bridge between the two APIs:

**Step 1: Allocate a function stub**
```c
void* stub = alloc_function_stub();  // Returns 16-byte-aligned executable address
```

**Step 2: Register with CUDA Runtime**
```c
g_rt_RegisterFunction(rt_handle, stub, func_name, ...);
```

The CUDA Runtime API requires `hostFun` to be a real executable address, even though it will never actually be called. The stub satisfies this requirement.

**Step 3: Force Runtime re-initialization**
After registering, the CUDA Runtime must be re-initialized to pick up the new function:

```c
ensure_device_context(0);   // Ensure CUcontext is current
cuCtxSynchronize();          // Flush any pending operations
cudaGetLastError();          // Clear error state
cudaGetDevice();             // Force device query
cudaSetDevice(0);            // Re-set device
cudaMalloc(&dummy, 1);       // Force lazy initialization
```

This sequence is necessary because the CUDA Runtime lazily initializes many internal structures. The `cudaMalloc` call is the key trigger -- it forces the runtime to walk its registered function table and resolve symbols.

**Step 4: Get the real function handle**
```c
cudaGetFuncBySymbol(&func, stub);
```

Now that the runtime has resolved the stub to a real kernel, `cudaGetFuncBySymbol` returns the actual `CUfunction` handle.

### Parameter Info Query (CUDA 12.3+)

For kernel launch, the client needs to know the exact parameter layout (offsets and sizes). This is obtained via:

```c
cuFuncGetParamInfo(func, &param_info);
```

This API is only available in CUDA 12.3+. For backward compatibility, it is resolved via `dlsym` at runtime:

```c
typedef CUresult (*fn_t)(CUfunction, void*, size_t*);
fn_t getParamInfo = dlsym(RTLD_DEFAULT, "cuFuncGetParamInfo");
```

If unavailable, the system falls back to a default parameter layout (assumes packed, no alignment padding).

### Response Format

```
func_handle(8) + total_param_size(4) + num_params(4) + [offset(4) + size(4)] x num_params
```

## Function Stub Mechanism

The stub system provides dummy executable addresses for the Runtime API registration path.

```c
// Global stub page
static void* g_stub_page = NULL;   // mmap'd 64KB page
static int g_stub_offset = 0;      // Next available offset in the page
```

**Allocation**:
- Page is `mmap`'d with `PROT_READ | PROT_WRITE | PROT_EXEC` (RWX -- required for JIT-like use)
- Filled with `0xC3` bytes (x86_64 `RET` instruction -- if ever called, returns immediately)
- Returns 16-byte-aligned addresses (CUDA alignment requirement)
- Maximum 4096 stubs (64KB / 16 bytes per stub)

**Why stubs are needed**: The CUDA Runtime API (`__cudaRegisterFunction`) requires the `hostFun` parameter to be a valid host-code address. It uses this address as a key in its internal function table. The stub never executes -- it is only used as an identifier.

## Client-Side Fat Binary Handling

### `__cudaRegisterFatBinary` Interception

When the application calls `__cudaRegisterFatBinary`, the client library:

1. **Parses the magic** to determine the fat binary size (different formats have different header layouts)
2. **Stores locally** in a client-side cache
3. **Sends to server** via the normal RPC path (function ID for RegisterFatBinary)

### Lazy Registration

Fat binaries are not eagerly sent to the server. Registration happens on-demand -- typically when the first kernel from that binary is launched. This avoids sending fat binaries for libraries that are loaded but never used.

### Client-Side Deduplication

Before sending a fat binary to the server, the client computes the same FNV-1a hash and checks if it has already been sent. If so, it reuses the existing wire handle.

### Flush (Pipelined Batch Registration)

When registration does happen, the client uses pipelined I/O:
1. Sends all pending registration requests without waiting for responses
2. Then reads all responses in order

This amortizes network latency across multiple registrations.

### Partial Offload Mode

In partial offload (client has a local GPU), fat binaries are registered with both:
- The real CUDA driver (via `dlsym` to the original `__cudaRegisterFatBinary`)
- The remote server (via RPC)

### Deferred Mode

Registrations can be queued for later replay. This is used when the client needs to establish a replay sequence that includes one-time setup calls.

## Cross-Architecture Realities

### Same Architecture (e.g., both Ada Lovelace sm_89)

- **Tier 2** (`cuModuleLoadData`) succeeds
- Simplest path, no parsing needed
- SASS code executes natively on the server
- No performance penalty

### Different Architecture, Same Generation (e.g., Ampere sm_80 to Ada sm_89)

- **Tier 2** fails (no sm_89 SASS in the client's sm_80 fat binary)
- **Tier 3** parses the fat binary, finds PTX or older SASS
- If PTX is available, it is JIT-compiled for sm_89 (adds ~100ms one-time cost)
- If only SASS for sm_80 is available and server is sm_89, PTX fallback is required

### Very Different Architecture (e.g., Maxwell sm_50 to Hopper sm_90)

- May require PTX fallback (Tier 3d) which is significantly slower
- PTX for very old architectures may not be compatible with very new GPUs
- In the worst case, Tier 4 (Runtime API fallback) is used as a last resort
- The Runtime API has its own internal fat binary handling that may succeed where the Driver API path fails

### Performance Impact

| Scenario | First Registration | Subsequent |
|---|---|---|
| Same arch, cached | ~0ms (cache hit) | ~0ms (cache hit) |
| Same arch, not cached | ~10ms (cuModuleLoadData) | ~0ms (cache hit) |
| Cross-arch, PTX available | ~100-500ms (JIT compile) | ~0ms (cache hit) |
| Cross-arch, PTX only | ~100-500ms (JIT compile) | ~0ms (cache hit) |
| Tier 4 fallback | ~50-200ms (Runtime API) | ~0ms (cache hit) |

The one-time cost of fat binary loading is amortized over the lifetime of the training/inference session. The cache ensures that repeated registrations (common in PyTorch) are essentially free.

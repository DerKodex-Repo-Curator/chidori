# CUDA Runtime API Module Management: Reverse-Engineering Findings

**Date**: 2026-03-27
**Binary**: libcudart.so.13.1.80 (CUDA 13.1.80)
**GPU**: NVIDIA GeForce RTX 4090 Laptop (sm_89, driver 590.48.01)
**Context**: Chidori GPU-over-IP server — understanding why cudaGetFuncBySymbol fails in a server context

---

## Executive Summary

**Initial hypothesis (DISPROVEN)**: The field at `runtime_state->0x108->0x540` is an "init gate" that blocks `cudaGetFuncBySymbol` when zero. The probe tool proved this is WRONG — the field is 0 in ALL states and `cudaGetFuncBySymbol` still succeeds.

**Actual findings**:
1. `cudaGetFuncBySymbol` works **regardless of initialization ordering** — creating a Driver API context before registration does NOT break it.
2. The field at `state->0x108->0x540` is NOT a gate — the "deinit path" is actually an **alternative resolver** that bootstraps module management on-the-fly.
3. The Runtime API handles **zstd-compressed PTX** and **cross-architecture JIT** correctly.
4. Error 209 (`cudaErrorNoKernelImageForDevice`) occurs only when the fat binary contains **no compatible code** for the device (wrong architecture cubins, no compatible PTX).
5. The Chidori server's actual problem is the **a2 proprietary compression format** used by pre-built CUDA 12.x pip wheels — the Runtime API's internal decompression may not work when called outside normal program startup.

---

## Test Results (All Reproducible)

### Test 1: Register BEFORE context creation (sm_89 fatbin on sm_89 GPU)

```
[A] __cudaRegisterFatBinary: handle=0x560f14922260 (NO context yet)
[B] __cudaRegisterFunction: stub=0x7ff37c4c6000 name='simple_add'
[C] cuDevicePrimaryCtxRetain: 0 ctx=0x560f14912e60
[D] cudaGetFuncBySymbol: err=0 (cudaSuccess) func=0x560f15375330

*** PROVEN: cudaGetFuncBySymbol succeeds ***
```

### Test 2: Context BEFORE registration (sm_89 fatbin on sm_89 GPU)

```
[A] Driver API context created: ctx=0x55f596eb8e60
[B] cudaSetDevice(0) + cudaFree(0) completed
[C] __cudaRegisterFatBinary: handle=0x55f5979061f0
[D] __cudaRegisterFunction: stub=0x7f181a619000 name='simple_add'
[E] cudaGetFuncBySymbol: err=0 (cudaSuccess) func=0x55f59791b3a0

*** UNEXPECTED: cudaGetFuncBySymbol succeeded ***
*** The initialization ordering does NOT matter on CUDA 13.1 ***
```

### Test 3: Cross-arch (sm_120 SASS + compute_89 PTX) on sm_89 GPU

```
[E] cudaGetFuncBySymbol: err=0 (cudaSuccess) func=0x555c806297e0

*** PROVEN: Runtime API JIT-compiles compatible PTX for local GPU ***
```

### Test 4: Cross-arch (sm_120 SASS only, no PTX) on sm_89 GPU

```
[E] cudaGetFuncBySymbol: err=209 (cudaErrorNoKernelImageForDevice) func=(nil)

*** PROVEN: Error 209 occurs only when NO compatible code exists ***
```

### Test 5: Cross-arch + zstd-compressed PTX on sm_89 GPU

```
Fat binary entries:
  type=1 sm=89 uncomp=432 comp=431 flags=0x40 (zstd compressed PTX)
  type=2 sm=120 uncomp=8608 comp=0 flags=0x40 (raw sm_120 SASS)

[E] cudaGetFuncBySymbol: err=0 (cudaSuccess) func=0x556250c72940

*** PROVEN: Runtime API decompresses zstd PTX and JIT-compiles it ***
```

### Test 6: Runtime state probe — field values at each stage

```
--- STATE 1: Before any Runtime API call ---
  [0x0d0] lifecycle state:    0
  [0x0f8] vtable (module):    0x0
  [0x100] vtable (func):      0x0
  [0x108] module mgmt ptr:   (nil)
  [0x540] INIT FLAG:         N/A (no module mgmt object)

--- STATE 2: After cudaMalloc ---
  [0x0d0] lifecycle state:    1
  [0x0f8] vtable (module):    0x7fb3f0ef68a0  ← POPULATED
  [0x100] vtable (func):      0x7fb3f0ef4e60  ← POPULATED
  [0x108] module mgmt ptr:   0x7fb3f1030240  ← POPULATED
  [0x540] INIT FLAG:         0               ← STILL ZERO!

--- STATE 3: After __cudaRegisterFatBinary ---
  [0x540] INIT FLAG:         0               ← STILL ZERO!

--- STATE 4: After cuDevicePrimaryCtxRetain ---
  [0x540] INIT FLAG:         0               ← STILL ZERO!

--- STATE 5: After cudaGetFuncBySymbol (SUCCESS) ---
  [0x540] INIT FLAG:         0               ← STILL ZERO!

*** DISPROVED: The 0x540 field is NOT a gate ***
```

---

## Disassembly Analysis: Corrected Interpretation

### cudaGetFuncBySymbol (0x79600) — Corrected Control Flow

```
Guard 1: getRuntimeState() == NULL? → return error 4
Guard 2: checkShuttingDown()? → return error 4
Guard 3: state->0x108->0x540 == 0?
    |
    ├── YES (0x540 == 0): Call ALTERNATIVE RESOLVER at 0x23160
    |     └── 0x23160 calls 0x466d0 (bootstrap), then 0x28e30 (resolve)
    |     └── This path CAN succeed — it just uses a different code path
    |
    └── NO  (0x540 != 0): Call NORMAL RESOLVER via vtable dispatch
          └── Uses pre-initialized module management infrastructure
```

**Key correction**: The branch at `0x79664 je 0x79800` is NOT an error path. It's a routing decision between two resolution strategies:
- Normal path (0x540 != 0): Uses cached vtable-based resolution
- Alternative path (0x540 == 0): Bootstraps module management on-the-fly, then resolves

Both paths can return success or failure. The error code 4 in `%eax` is just the default value, overwritten by the actual resolution result.

### Alternative Resolver at 0x23160 — Detailed Flow

```
0x23160: Entry
0x23184: call 0x466d0  ← Module management bootstrap (sets up if needed)
0x23189: test %eax, %eax
0x2318b: jne  0x231c0  ← Bootstrap failed → error
0x2318d: call getRuntimeState()
0x23192: lea  local_var, %rsi
0x23196: mov  %r14, %rcx      ← symbol (hostFun)
0x23199: mov  %r13, %rdx      ← func_ptr output
0x2319c: mov  $0x1, %r8d     ← flag = 1
0x231a5: call 0x28e30          ← Core resolution function
0x231aa: return result
```

### The Singleton Factory at 0x8e360

Two independent globals, two independent singletons:

| Global Address | Purpose | Set During |
|---|---|---|
| 0x2b93c4 | Runtime state slot | First Runtime API call |
| 0x2b93d0 | Shutdown state slot | First Runtime API call |

`getRuntimeState()` (0x2a800):
- Type ID at 0x26920
- Global slot at 0x2b93c4
- Returns pointer from slot+4 (0x2b93c8)

`checkShuttingDown()` (0x42b00):
- Type ID at 0x45670
- Global slot at 0x2b93d0
- Checks lifecycle state at result->0xd0 (2 = SHUTTING_DOWN)

### Runtime State Object Layout (Measured)

```
runtime_state @ 0x55a78694b020:
  +0x000: [vtable ptr]
  +0x0d0: lifecycle state: 0 = uninit, 1 = active, 2 = shutting_down
  +0x0d4: shutdown error code
  +0x0d8: dlsym handle for libcuda.so
  +0x0f8: vtable ptr (module ops) ← NULL until cudaMalloc
  +0x100: vtable ptr (func resolution) ← NULL until cudaMalloc
  +0x108: ptr → module management subsystem ← NULL until cudaMalloc
           └ +0x540: uint32_t = 0 (purpose unknown, NOT a gate)
  +0x110: ptr → __cudaInitModule subsystem ← NULL until cudaMalloc
           └ +0x004: uint32_t = 0 (purpose unknown)
```

### __cudaRegisterFatBinary (0x23cc0)

```
23ccc: call getRuntimeState()      ← ensures singleton exists
23cd5: mov  %rbx, %rdx             ← fat binary wrapper ptr
23cdb: call 0x27020                ← internal handler
23ce0: mov -0x18(%rbp), %rax       ← return handle to caller
```

Internal handler (0x27020):
```
27029: mov  $0x68, %edi             ← allocate 104-byte structure
27039: call allocation function
27047: call 0x26730                ← parse fat binary, store metadata
27051: mov  %rbx, 0x0(%r13)       ← store handle to output
```

### __cudaRegisterFunction (0x23ea0)

```
23ec4: call getRuntimeState()
23ef1: call 0x27fd0                ← maps hostFun → kernel metadata
```

### __cudaInitModule (0x23f10) — CUDA 12+ Registration

```
23f1b: call getRuntimeState()
23f49: mov  0x110(%rbx), %rax     ← different sub-object (not 0x108!)
23f50: mov  [state->0x110]->0x4, %edx  ← different check field
23fa8: call 0x46710                ← module initialization (always called)
```

---

## The Real Problem: a2 Proprietary Compression

### What Works

| Scenario | Result | Error |
|---|---|---|
| Same-arch matching cubin (uncompressed) | SUCCESS | - |
| Same-arch matching cubin (zstd PTX) | SUCCESS | - |
| Cross-arch with compatible PTX (uncompressed) | SUCCESS | - |
| Cross-arch with compatible PTX (zstd) | SUCCESS | - |
| Context before registration (any compatible) | SUCCESS | - |
| Registration before context (any compatible) | SUCCESS | - |

### What Fails

| Scenario | Result | Error |
|---|---|---|
| Cross-arch with NO compatible PTX (sm_120 only on sm_89) | FAIL | 209 |
| a2-compressed cubins (pre-built pip wheels) | FAIL | 209/200 |

### The a2 Format

Found in pre-built CUDA 12.x pip wheels. Characteristics:
- First byte: `0xa2` (proprietary magic)
- Followed by ELF-like header: `7f 45 4c 46` (ELF magic)
- Non-standard ELF fields: `e_type=0x8000`, `e_machine=0x100`, OSABI=`0x33`
- The `flags` field in the fat binary entry header does NOT indicate a2 compression (flags=0x0)
- The a2 prefix byte is the only indicator

The a2 format is used by nvcc 12.x for cubin entries in fat binaries. nvcc 13.1 switched to standard zstd compression for PTX entries but may still use a2 for cubin entries (unconfirmed).

### Why the Runtime API Path Fails with a2

The Runtime API (`libcudart.so`) has internal a2 decompression code that works when `__cudaRegisterFatBinary` is called from a normal program's static constructor (`.init_array`). However, when called manually from a server context via `dlsym`, the decompression may fail because:

1. **Missing internal state**: The a2 decompression path may require state that's only set up during normal program startup (before `main()`), not during lazy initialization.

2. **Thread-local state**: The a2 decompressor may check thread-local variables that are only initialized during the CRT startup sequence.

3. **Different libcudart.so version**: The Chidori server links against libcudart.so.13, but the fat binaries were compiled with nvcc 12.x. The a2 format may have version-specific variations.

### The Working Solution

The Chidori server's current working path bypasses the Runtime API entirely:

1. Parse the `0xba55ed50` fat binary container
2. Find entries matching the server GPU architecture (or compatible PTX)
3. If `flags & 0x40` and first 4 bytes = `28 b5 2f fd`: decompress with standard `ZSTD_decompress()`
4. Load decompressed cubin/PTX via `cuModuleLoadData()` (Driver API)
5. Get CUfunction via `cuModuleGetFunction()` (Driver API)

This works for:
- nvcc 13.1 fat binaries (zstd compression)
- Source-built PyTorch with nvcc 13.1

This does NOT work for:
- Pre-built pip wheels with CUDA 12.x (a2 compression)
- Any fat binary with a2-compressed entries

---

## Files in This Directory

| File | Purpose |
|---|---|
| `simple.cu` | Test CUDA kernel (simple_add, simple_mul) |
| `simple_sm89.fatbin` | Test fat binary (sm_89, uncompressed) |
| `test_isolated_context_first.c` | Proof: context BEFORE registration |
| `test_isolated_register_first.c` | Proof: registration BEFORE context |
| `probe_runtime_state.c` | Reads internal runtime state fields |
| `crossarch_compressed.fatbin` | Cross-arch fatbin with zstd PTX |
| `Makefile` | Build system |

### How to Reproduce

```bash
cd /home/fabricio/chidori/research/cudart-internals
make test
```

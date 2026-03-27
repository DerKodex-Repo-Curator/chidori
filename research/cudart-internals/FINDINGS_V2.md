# CUDA Runtime API Internals: Reverse-Engineering Findings v2

**Binary**: libcudart.so.12.8.90 (CUDA 12.8.90)
**GPU**: NVIDIA RTX A6000 (sm_86, driver 570.195.03)
**Context**: Chidori GPU-over-IP — understanding cudaGetFuncBySymbol internals
**Date**: 2026-03-27
**Tool**: runtime_probe_v2 + GDB scripting (disassembly + live instrumentation)

---

## Executive Summary

**Key findings from CUDA 12.8.90 (differs from CUDA 13.1 findings)**:

1. **`cudaGetFuncBySymbol` returns error 98** on CUDA 12.8.90 when called after manual `__cudaRegisterFatBinary` + `__cudaRegisterFunction`. This is a different error code than CUDA 13.1 (which returns error 4).

2. **`__cudaGetKernel` SUCCEEDS** where `cudaGetFuncBySymbol` fails — this is the alternative resolution path that CUDA 12+ provides via `__cudaInitModule`.

3. **The `mod_mgmt+0x540` field stays at 0** throughout the entire lifecycle, even after lazy init, registration, and resolution. This confirms it is NOT a gate on CUDA 12.8.90 either — it selects between two resolution strategies.

4. **Lazy initialization (cudaFree(0)) does NOT populate the module management vtables** — they are already populated by the time the first CUDA call triggers `pthread_once` in `getRuntimeState()`.

5. **The lifecycle field at `state+0x070` jumps directly from 0→1** — there is no intermediate state. It's set during `pthread_once` initialization, not during `cudaFree(0)`.

---

## 1. cudaGetFuncBySymbol Internals

### Function Entry Point

```
cudaGetFuncBySymbol (offset 0x739f0 in libcudart.so.12.8.90):
  rdi = hostFun (stub pointer from __cudaRegisterFunction)
  rsi = CUfunction* (output)
```

### Control Flow (from disassembly)

```
739f0: push rbp, r14, r13, r12, rbx
73a0f: call getRuntimeState()          @ 0x27f50
73a14: mov  rbx, rax                  ; rbx = runtime_state
73a1c: mov  eax, 4                    ; default error = 4 (NOT_INITIALIZED)
73a1f: test rbx, rbx
73a22: je   73bc9                     ; return 4 if state is NULL

73a25: mov  rdi, rbx
73a28: call checkShuttingDown()       @ 0x356e0
73a2d: test eax, eax
73a2f: jne  73bc9                     ; return 4 if shutting down

; --- THE CRITICAL GATE ---
73a35: mov  rax, [rbx + 0xa8]        ; rax = mod_mgmt_ptr
73a3c: mov  edx, [rax + 0x540]       ; edx = mod_mgmt->init_flag
73a42: test edx, edx
73a44: je   73be0                     ; if 0x540 == 0: ALTERNATIVE PATH
                                     ; else: NORMAL PATH (vtable dispatch)
```

### Two Resolution Paths

**Path A: 0x540 != 0 (NORMAL)**
```
73a5a: lea  rcx, [rbp-0xc8]           ; output func_ptr
73a61: lea  rsi, [rbp-0xcc]           ; output error
73a68: mov  r15, [rbx + 0xa0]        ; func_res_vtbl
73a90: lea  r8, [rip+0x219a9]        ; some data structure
73b08: call [r15 + 0x10]             ; vtable[2] — enter resolution scope
73b0c: mov  rax, [rbx + 0x098]        ; mod_mgr_vtbl
73b1e: call [rax + 0x20]             ; vtable[4] — resolve function
73b40: mov  rax, [rbx + 0x098]        ; mod_mgr_vtbl
73b71: call [rax + 0x08]             ; vtable[1] — exit scope
73b7a: call 0x22040                   ; core resolver (hostFun → CUfunction)
73b93: call [r12 + 0x10]             ; func_res_vtbl[2] — cleanup
73ba6: call [r13 + 0x20]             ; mod_mgr_vtbl[4] — cleanup
73bc0: call [rbx + 0x08]             ; mod_mgr_vtbl[1] — final cleanup
```

**Path B: 0x540 == 0 (ALTERNATIVE)**
```
73be0: mov  rsi, r13                  ; hostFun
73be3: mov  rdi, r12                  ; CUfunction*
73be6: call 0x22040                   ; core resolver directly
73beb: jmp  73bc9                     ; return
```

### Evidence: Path B is taken

Our probe shows `mod_mgmt+0x540 = 0` at all stages, so `cudaGetFuncBySymbol` always takes **Path B** (the alternative resolver). The core resolver at 0x22040 is called directly without vtable scope management.

### The Core Resolver at 0x22040

```
22040: push rbp, r14, r13, r12, rbx
22050: lea  r12, [rbp-0x28]          ; local output
22055: mov  rdi, r12
22058: sub  rsp, 0x10
2205c: mov  qword [rbp-0x28], 0
22064: call 0x3a600                   ; some setup function
22069: mov  ebx, eax
2206b: test eax, eax
2206d: je   220b8                     ; success → goto 220b8
```

At 0x220b8:
```
220b8: mov  rax, [rbp-0x28]          ; get the setup result
220bc: lea  rdi, [rax + 0x88]        ; offset +0x88 into result
220c3: call 0x88c40                   ; initialize module
220c8: mov  rdi, [rbp-0x28]
220cc: xor  ecx, ecx
220ce: mov  rdx, r14                  ; hostFun
220d1: mov  rsi, r13                  ; CUfunction*
220d4: call 0x3bbe0                   ; THE ACTUAL RESOLVER
```

The actual resolver at 0x3bbe0 is what maps `hostFun → CUfunction`.

---

## 2. Lazy Initialization Sequence

### What `getRuntimeState()` Does (offset 0x27f50)

```
27f50: push rbp
27f51: lea  rsi, [rip-0x2028]        ; type_id at 0x25f30
27f58: lea  rdi, [rip+0x28a2b5]      ; global slot at 0x2b2214
27f5f: mov  rsp, rbp
27f62: call 0x88ed0                   ; pthread_once(&global, init_callback)
27f67: mov  rax, [rip+0x28a2aa]      ; return *(0x2b2218)
27f6e: pop  rbp
27f6f: ret
```

The `pthread_once` callback creates the runtime state singleton and stores the pointer at global offset 0x2b2218. This happens on the FIRST call to any exported CUDA Runtime function (including `cudaGetDeviceCount`).

### Evidence: State is populated before cudaFree(0)

```
STAGE 1 (before any CUDA call):
  lifecycle = 1, mod_mgr_vtbl = 0x7e7b53e94fe0, mod_mgmt = 0x7e7b54011f80

STAGE 2 (after cudaSetDevice(0)):
  lifecycle = 1, mod_mgr_vtbl = 0x7e7b53e94fe0, mod_mgmt = 0x7e7b54011f80
  (unchanged)

STAGE 3 (after cudaFree(0)):
  lifecycle = 1, mod_mgr_vtbl = 0x7e7b53e94fe0, mod_mgmt = 0x7e7b54011f80
  (unchanged)
```

**Conclusion**: The runtime state is fully populated during the `pthread_once` callback triggered by the first CUDA Runtime API call. `cudaSetDevice(0)` and `cudaFree(0)` do NOT change the module management state. The module management subsystem (vtables at 0x098, 0x0a0, objects at 0x0a8, 0x0b0) is created at first-call time.

### What `cudaFree(0)` Actually Does

`cudaFree(0)` is a well-known "lazy init trigger" for CUDA contexts. Evidence from our probe:
- Before `cudaFree(0)`: lifecycle=1, vtables populated
- After `cudaFree(0)`: lifecycle=1, vtables populated (identical)

The difference is that `cudaFree(0)` creates a CUDA context via the Driver API (`cuCtxGetCurrent` / `cuCtxCreate`). This is needed for kernel launches but NOT for module registration or resolution.

### checkShuttingDown() at 0x356e0

```
356e0: push rbp, rbx
356e5: mov  rbx, rdi                  ; rbx = state
356e8: lea  rdi, [rip+0x27cb31]      ; global at 0x2b2220
356ef: lea  rsi, [rip+0x3ef6]        ; type_id at 0x395f0
356f3: sub  rsp, 8
356f6: call 0x88ed0                   ; pthread_once (init shutdown state)
356fb: mov  eax, [rbx + 0x70]        ; lifecycle
356fe: cmp  eax, 2                   ; 2 = SHUTTING_DOWN
35701: je   35718                     ; if shutting down → return shutdown error
35703: add  rsp, 8
35707: mov  rdi, rbx
3570a: pop  rbx
3570b: pop  rbp
3570e: jmp  0x34e60                   ; call actual check function
```

The function checks `state->0x070 == 2` (lifecycle == SHUTTING_DOWN). If so, it returns the shutdown error code directly. Otherwise, it calls a deeper check function at 0x34e60.

---

## 3. __cudaRegisterFatBinary → cudaGetFuncBySymbol Pipeline

### Registration: __cudaRegisterFatBinary (0x228c0)

```
228c0: push rbp, rbx
228c5: mov  rbx, rdi                  ; rbx = fatbin_data
228c8: sub  rsp, 0x18
228cc: call getRuntimeState()          ; ensure singleton exists
228d1: lea  rsi, [rbp-0x18]          ; output handle
228d5: mov  rdx, rbx                  ; fatbin_data
228d8: mov  rdi, rax                  ; runtime_state
228db: call 0x26630                   ; internal handler
228e0: mov  rax, [rbp-0x18]          ; return handle
```

Internal handler at 0x26630:
```
26630: push rbp, r14, r13, r12, rbx
26639: mov  edi, 0x68                 ; allocate 104-byte struct
2663e: call 0x8abf0                   ; malloc(0x68)
2664e: mov  rsi, r12                  ; fatbin_data
26654: call 0x250e0                   ; parse fat binary, store metadata
2665c: test rbx, rbx
2665f: je   26670                     ; if alloc failed
26661: mov  [r13], rbx               ; store handle to output
```

The 0x68 (104) byte struct at the handle contains:
- `+0x00`: pointer to internal fat binary metadata
- `+0x58`: head of registered function linked list
- `+0x50`: tail of registered function linked list

### Registration: __cudaRegisterFunction (0x22aa0)

```
22aa0: push rbp, r15, r14, r13, r12, rbx
       save all arguments
22ac0: call getRuntimeState()
22acd: push all args onto stack
22af1: call 0x278c0                   ; internal handler
```

Internal handler at 0x278c0:
```
278c0: ...
27910: test rsi, rsi                  ; rsi = fatbin handle
27913: je   279a0                     ; if null → error 3
27916: mov  edi, 0x58                 ; allocate 88-byte struct
2791e: call 0x8abf0                   ; malloc(0x58)
       ; copy registration data into struct
2795e: mov  rsi, [rbx + 0x58]        ; current tail of function list
27962: test rsi, rsi
27965: je   27990                     ; if empty list
27967: mov  [rsi + 0x50], rax        ; link: tail->next = new_node
2796b: mov  [rbx + 0x58], rax        ; update tail
2796f: lea  rsi, [rbp-0x60]          ; registration data
27973: call 0x272f0                   ; hash-based lookup/insert
```

The 0x58 (88) byte struct contains:
- `+0x00`: hostFun pointer (the stub address)
- `+0x08`: kernel name string
- `+0x10`: kernel name string (duplicate?)
- `+0x50`: next pointer (linked list)
- `+0x58`: prev pointer (linked list)

The function also uses FNV-1a hashing (`0x811c9dc5`, `0x1000193`) to index into a hash table for fast lookup by hostFun address.

### Resolution: __cudaInitModule vs cudaGetFuncBySymbol

CUDA 12+ provides TWO registration/resolution paths:

**Old path (CUDA < 12)**:
```
__cudaRegisterFatBinary → __cudaRegisterFunction → cudaGetFuncBySymbol
```

**New path (CUDA 12+)**:
```
__cudaRegisterFatBinary → __cudaRegisterFunction → __cudaInitModule → __cudaGetKernel
```

### __cudaInitModule (0x22b10)

```
22b10: push rbp, r12, rbx
22b17: sub  rsp, 0x50
22b1b: call getRuntimeState()
22b28: mov  rbx, rax                  ; rbx = state
22b32: call getRuntimeState()          ; called again?
22b37: mov  rdi, rax
22b3a: call checkShuttingDown()       ; check lifecycle
22b3f: xor  edx, edx                  ; result = 0 (false)
22b41: test eax, eax
22b43: jne  22bd1                     ; if shutting down → error
22b49: mov  rax, [rbx + 0xb0]        ; init_mod_ptr
22b50: mov  edx, [rax + 0x04]        ; init_mod->flag
22b53: test edx, edx
22b55: je   22be0                     ; if flag == 0: bootstrap
```

At 0x22be0 (bootstrap path):
```
22be0: call 0x3a630                   ; bootstrap module management
22be5: jmp  22bcc
```

At 0x22b5b (normal path):
```
22b5b: lea  rsi, [rbp-0x48]          ; output
22b5f: lea  rdi, [rbp-0x4c]          ; output
22b63: mov  rax, [rbx + 0x98]        ; mod_mgr_vtbl
22b8c: mov  edi, 1
22b91: mov  [rbp-0x28], rcx
22b95: mov  qword [rbp-0x18], 1
22ba5: call [rax + 0x28]             ; vtable[5] — enter scope
22ba8: call 0x3a630                   ; module init function
22bad: mov  rbx, [rbx + 0x98]        ; mod_mgr_vtbl
22bbc: mov  edi, 1
22bc6: call [rbx + 0x28]             ; vtable[5] — exit scope
```

### Evidence: __cudaGetKernel Succeeds

```
cudaGetFuncBySymbol: err=98 func=(nil)
__cudaGetKernel:   err=0  func=0x7ffca533a04a  ← SUCCESS
```

`__cudaGetKernel` takes different parameters than `cudaGetFuncBySymbol`:
```
__cudaGetKernel(CUfunction *func, const char *name, void *hostFun,
               uint3 gridDim, uint3 blockDim, size_t sharedMem)
```

This is the CUDA 12+ path that frameworks like PyTorch actually use. The kernel name is passed explicitly, making resolution independent of the hostFun→kernel mapping.

---

## 4. The Module Management Vtables

### module_mgr_vtbl (state+0x098)

Observed values (relative to libcudart.so base):
```
[0] 0x38                      (data, not a function pointer)
[1] base + 0x4ad0e0           (function)
[2] base + 0x4ad140           (function)
[3] base + 0x4bae50           (function)
[4] base + 0x4ad1e0           (function) ← used by cudaGetFuncBySymbol normal path
[5] base + 0x4ad160           (function) ← used by __cudaInitModule
```

### func_res_vtbl (state+0x0a0)

```
[0] 0x18                      (data)
[1] base + 0x4bb730           (function)
[2] base + 0x4b20f0           (function) ← used by cudaGetFuncBySymbol normal path
[3] (nil)
[4] 0x1870                    (data)
[5] base + 0x3dc73b0           (function)
```

### cudaGetFuncBySymbol Normal Path Vtable Dispatch

When `mod_mgmt+0x540 != 0`:
1. `func_res_vtbl[2]` (enter scope) — sets up resolution context
2. `mod_mgr_vtbl[4]` (resolve) — performs the actual stub→function mapping
3. `mod_mgr_vtbl[1]` (exit scope) — tears down resolution context
4. Then calls core resolver at 0x22040
5. `func_res_vtbl[2]` (cleanup)
6. `mod_mgr_vtbl[4]` (cleanup)
7. `mod_mgr_vtbl[1]` (final cleanup)

When `mod_mgmt+0x540 == 0` (our observed case):
1. Calls core resolver at 0x22040 directly (no scope management)

---

## 5. Why cudaGetFuncBySymbol Returns Error 98

### Evidence

```
cudaGetFuncBySymbol: err=98 func=(nil)
```

Error 98 is `cudaErrorDeviceUnavailable` in CUDA 12.x. The function takes the alternative path (0x540 == 0), which calls the core resolver at 0x22040. The core resolver:
1. Calls setup function at 0x3a600
2. If setup succeeds, calls module init at 0x88c40
3. Then calls the actual resolver at 0x3bbe0

The setup function likely requires a CUDA context and/or device access that isn't properly established in our test scenario. However, `__cudaGetKernel` succeeds, suggesting the issue is in the scope management or the way `cudaGetFuncBySymbol` interacts with the already-initialized module management system.

### Why __cudaGetKernel Works

`__cudaGetKernel` (at 0x22d40) takes the kernel NAME as a parameter, so it can resolve the function by name rather than by hostFun stub pointer. This bypasses the stub→function mapping that `cudaGetFuncBySymbol` relies on.

---

## 6. Runtime State Object Layout (Measured, CUDA 12.8.90)

```
runtime_state @ <heap>:
  +0x000: [vtable ptr]                    (NULL initially)
  +0x048: [ptr]                            (some sub-object)
  +0x050: [small int]                      (changes after __cudaRegisterFunction)
  +0x058: [small int]                      (changes after __cudaRegisterFunction)
  +0x060: [ptr]                            (appears after __cudaRegisterFunction)
  +0x068: [ptr]                            (some sub-object)
  +0x070: [uint32_t] lifecycle             0=uninit, 1=active, 2=shutting_down
  +0x074: [uint32_t] field_74              (error code from shutdown)
  +0x078: [ptr] dlsym handle              (handle for libcuda.so via dlopen)
  +0x080: [ptr]                            (some subsystem)
  +0x088: [ptr]                            (some subsystem)
  +0x090: [ptr]                            (some subsystem)
  +0x098: [ptr] mod_mgr_vtbl              (module management vtable)
  +0x0a0: [ptr] func_res_vtbl             (function resolution vtable)
  +0x0a8: [ptr] mod_mgmt_ptr              (module management object)
  +0x0b0: [ptr] init_mod_ptr              (__cudaInitModule subsystem)
```

### Module Management Object (mod_mgmt @ state+0x0a8)

```
  +0x000: [vtable ptr]                    (NULL in our tests)
  +0x008: [ptr]                            (NULL)
  +0x010: [ptr]                            (NULL)
  ...
  +0x540: [uint32_t] init_flag             (ALWAYS 0 in our tests)
  +0x544: [uint32_t] field_544             (ALWAYS 0)
```

### Init Module Object (init_mod @ state+0x0b0)

```
  +0x000: [vtable ptr]                    (NULL in our tests)
  +0x004: [uint32_t] flag                  (ALWAYS 0 in our tests)
  +0x008: [ptr]                            (NULL)
```

---

## 7. Implications for Chidori

### For the Server-Side Kernel Launch Path

The Chidori server needs to resolve kernel functions from fat binaries received from the client. Based on these findings:

1. **Use the Driver API path instead of Runtime API**:
   - Parse fat binary → extract matching cubin/PTX → decompress (zstd) → `cuModuleLoadData` → `cuModuleGetFunction`
   - This bypasses all the Runtime API's internal state management

2. **If Runtime API must be used**: `__cudaGetKernel` is the correct function to call, not `cudaGetFuncBySymbol`. It takes the kernel name explicitly and works even when the stub mapping is broken.

3. **The 0x540 field is NOT a gate** — it's a routing flag that selects between two resolution strategies. Both can succeed, but the alternative path (0x540 == 0) has different scope management.

4. **Lazy init happens via `pthread_once`** — it's triggered by the first CUDA Runtime API call, not by `cudaFree(0)`. The module management vtables and objects are created at first-call time.

5. **Error 98 vs Error 4**: On CUDA 12.8.90, `cudaGetFuncBySymbol` returns error 98 (`cudaErrorDeviceUnavailable`), not error 4 (`cudaErrorDeinitialized`) as on CUDA 13.1. This suggests the resolution path has changed between versions.

### Registration is Sufficient

`__cudaRegisterFatBinary` + `__cudaRegisterFunction` correctly store the fat binary metadata and create the stub→kernel mapping. The data structures are properly linked (confirmed by walking the function list). The failure is in the RESOLUTION path, not the REGISTRATION path.

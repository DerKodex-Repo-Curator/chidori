# libcudart.so.13.1.80 — Complete Internal Architecture

**Binary**: `/opt/cuda/targets/x86_64-linux/lib/libcudart.so.13.1.80`
**Size**: 740KB, text: 519KB (0xe8e0..0x907c4)
**Exported symbols**: 449 | **Internal strings**: 3,889
**GPU**: NVIDIA GeForce RTX 4090 Laptop (sm_89, driver 590.48.01)
**Context**: Chidori GPU-over-IP — why `cudaGetFuncBySymbol` works, what the internal state machine looks like

---

## Reproduction

```bash
cd /home/fabricio/chidori/research/cudart-internals

# Build all proof programs
gcc -o deep_trace deep_trace.c -I/opt/cuda/include -L/opt/cuda/targets/x86_64-linux/lib -lcuda -lcudart -ldl

# Run with same-arch fatbin (sm_89 on sm_89)
LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib ./deep_trace simple_sm89.fatbin simple_add

# Run with cross-arch + zstd compressed (sm_120 SASS + zstd compute_89 PTX on sm_89)
LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib ./deep_trace crossarch_compressed.fatbin simple_add

# Run with no compatible code (sm_120 only on sm_89) — proves error 209
LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib ./deep_trace simple_sm120.fatbin simple_add
```

All results below are deterministic and reproducible across runs.

---

## 1. The Singleton Infrastructure

### 1.1 getRuntimeState (0x2a800)

```asm
2a800: lea  -0x3ee8(%rip), %rsi    # 0x26920 (type_id)
2a808: lea  0x28ebb5(%rip), %rdi  # 0x2b93c4 (global slot)
2a812: call 0x8e360                # singleton_factory(global_slot, type_id)
2a817: mov  0x28ebaa(%rip), %rax  # 0x2b93c8 (load pointer from slot+4)
2a81e: ret
```

**What it does**: Calls the singleton factory with the global at `0x2b93c4` and type ID at `0x26920`. Returns the pointer stored at `0x2b93c8`.

**Global layout at 0x2b93c4**:
```
0x2b93c4: [pthread_once_t or control word]  ← passed to singleton_factory
0x2b93c8: [runtime_state*]                   ← the returned pointer
```

### 1.2 checkShuttingDown (0x42b00)

```asm
42b00: lea  0x2768c1(%rip), %rdi  # 0x2b93d0 (shutdown global slot)
42b13: lea  0x2b56(%rip), %rsi    # 0x45670 (shutdown type_id)
42b1a: call 0x8e360                # singleton_factory
42b1f: mov  0xd0(%rbx), %eax      # lifecycle state
42b25: cmp  $0x2, %eax            # SHUTTING_DOWN = 2?
42b28: je   42b40                  # yes → return shutdown error code
42b2e: jmp  42280                  # no → deeper per-device state check
42b40: mov  0xd4(%rbx), %eax      # return shutdown error code
42b46: ret
```

**Two independent singletons, two independent globals**:

| Global Address | Purpose | Type ID |
|---|---|---|
| `0x2b93c4` | Runtime state slot | `0x26920` |
| `0x2b93d0` | Shutdown state slot | `0x45670` |

### 1.3 singleton_factory (0x8e360)

```asm
8e360: sub  $0x8, %rsp
8e364: call pthread_once@plt       # ensure one-time init
8e369: test %eax, %eax
8e36b: setne %al                  # return -1 on failure, 0 on success
8e372: ret
```

This is a thin wrapper around `pthread_once`. The actual initialization callback is registered elsewhere (likely in `.init_array` or a static constructor). The callback creates the singleton object and stores it at `global_slot + 4`.

### 1.4 Memory Allocator Trampoline (0x900a0)

```asm
900a0: jmp malloc@plt    # 0xe510
900b0: jmp free@plt      # 0xe490
900c0: jmp calloc@plt    # 0xe290
900d0: jmp realloc@plt   # 0xe750
```

All internal allocations go through these trampolines. The Runtime uses:
- `malloc(0x68)` = 104 bytes for fat binary registration records (`__cudaRegisterFatBinary` → `0x27029`)
- `malloc(0x58)` = 88 bytes for function registration records (`__cudaRegisterFunction` → `0x28026`)
- `calloc(8, 0x11)` = 136 bytes for hash table buckets (`0x27a80`)
- `malloc(0x40)` = 64 bytes for hash chain nodes (`0x27b18`)
- `malloc(0x38)` = 56 bytes for module init records (`0x46576`)
- `malloc(0x20)` = 32 bytes for function lookup nodes (`0x27d81`)

---

## 2. Runtime State Object — Complete Layout

**PROVEN**: Every field below was measured from a live process and cross-referenced with disassembly.

```
runtime_state (allocated by singleton factory):
  +0x000: [vtable ptr]           = 0x0 (always NULL — unused?)
  +0x0d0: [uint32_t lifecycle]   = 0 (uninit), 1 (active), 2 (shutting_down)
  +0x0d4: [uint32_t shutdown_err] = error code when lifecycle=2
  +0x0d8: [void* libcuda_handle] = dlopen("libcuda.so.1") handle
  +0x0f8: [vtable* module_ops]  = 0x7f..68a0 (set during cudaMalloc init)
  +0x100: [vtable* func_ops]    = 0x7f..4e60 (set during cudaMalloc init)
  +0x108: [ptr* module_mgmt]    = 0x7f..0240 (set during cudaMalloc init)
  +0x110: [ptr* init_module]    = 0x7f..0100 (set during cudaMalloc init)
```

### 2.1 Module Management Sub-Object (at state+0x108)

```
module_mgmt (at state+0x108):
  +0x000: [vtable ptr]           = 0x0 (always NULL in our tests)
  +0x540: [uint32_t]             = 0 (always ZERO)
  +0xb8:  [uint32_t]             = hash table bucket count
  +0xc8:  [ptr* hash_table]      = hash table array (at state+0x108+0xc8)
  +0xe8:  [ptr*]                 = used by bootstrap (0x466ef)
```

### 2.2 __cudaInitModule Sub-Object (at state+0x110)

```
init_module (at state+0x110):
  +0x000: [vtable ptr]           = 0x0
  +0x004: [uint32_t]             = 0 (init flag checked by __cudaInitModule)
  +0x050: [uint32_t]             = hash bucket count (grows: 0→0x11→0x22...)
  +0x060: [ptr* hash_table]      = hash table for registered functions
```

### 2.3 Module Operations Vtable (at state+0xf8)

Measured vtable entries (absolute addresses from a live run):

| Offset | Value | Inferred Purpose |
|--------|-------|------------------|
| +0x00 | `0x38` | Sentinel/cookie (not a code pointer) |
| +0x08 | `func A` | Module scope enter / begin load |
| +0x10 | `func B` | Module scope exit / end load |
| +0x18 | `func C` | Module get function |
| +0x20 | `func D` | Module link |
| +0x28 | `func E` | Module init (called by `__cudaInitModule` via `*0x28(%rax)`) |

**PROVEN from cudaGetFuncBySymbol disassembly**:
- `0x79752: call *0x20(%rax)` — calls module_vtable[4] (link)
- `0x79792: call *0x8(%rax)` — calls module_vtable[1] (scope enter)
- `0x79e1: call *0x8(%rbx)` — calls module_vtable[1] again (scope exit?)

### 2.4 Function Resolution Vtable (at state+0x100)

| Offset | Value | Inferred Purpose |
|--------|-------|------------------|
| +0x00 | `0x18` | Sentinel/cookie |
| +0x08 | `func F` | Function resolution (setup?) |
| +0x10 | `func G` | Function resolution (core — called by cudaGetFuncBySymbol) |

**PROVEN from cudaGetFuncBySymbol disassembly**:
- `0x7973c: call *0x10(%r15)` — calls func_vtable[2] (resolve)
- `0x797b4: call *0x10(%r12)` — calls func_vtable[2] again

---

## 3. Registration Pipeline — Complete Call Chain

### 3.1 __cudaRegisterFatBinary (0x23cc0)

```
Entry: rdi = __cudaFatCubinWrapper* (magic=0x466243b1, version=1, data, filename)

0x23cc0: push %rbp
0x23ccc: call getRuntimeState()       → ensure singleton exists
0x23cd1: lea  -0x18(%rbp), %rsi     → &output_handle
0x23cd5: mov  %rbx, %rdx             → wrapper ptr
0x23cd8: mov  %rax, %rdi             → runtime_state
0x23cdb: call 0x27020                → internal handler(runtime_state, &output, wrapper)
0x23ce0: mov  -0x18(%rbp), %rax      → return handle to caller
0x23ce8: ret
```

### 3.2 Internal Handler (0x27020)

```
Entry: rdi = runtime_state, rsi = &output_handle, rdx = wrapper

0x27029: mov  $0x68, %edi             → size = 104 bytes
0x27039: call malloc                  → alloc registration record
0x27044: mov  %rax, %rdi             → record ptr
0x27047: call 0x26730                → zero-fill record
0x2704c: test %rbx, %rbx             → allocation failed?
0x2704f: je   0x27060                → yes → set error state
0x27051: mov  %rbx, 0x0(%r13)       → store handle = record ptr
0x27060: (error path)
  0x27060: movl $0x3, 0xd4(%r14)   → state->shutdown_err = 3
  0x2706b: movl $0x2, 0xd0(%r14)   → state->lifecycle = SHUTTING_DOWN
```

### 3.3 Record Initialization (0x26730) — 104-byte struct

```asm
26730: mov  %rsi, (%rdi)            # +0x00: wrapper ptr (input)
26733: movq $0x0, 0x8(%rdi)        # +0x08: next record (NULL)
2673b: movl $0x0, 0x10(%rdi)       # +0x10: uint32 = 0
26742: movl $0x0, 0x14(%rdi)       # +0x14: uint32 = 0
26749: movq $0x0, 0x18(%rdi)       # +0x18: ptr = 0
26751: movq $0x0, 0x20(%rdi)       # +0x20: ptr = 0
26759: movq $0x0, 0x28(%rdi)       # +0x28: ptr = 0
26761: movq $0x0, 0x30(%rdi)       # +0x30: ptr = 0
26769: movb $0x0, 0x38(%rdi)       # +0x38: byte = 0
2676d: movq $0x0, 0x40(%rdi)       # +0x40: ptr = 0
26775: movq $0x0, 0x48(%rdi)       # +0x48: ptr = 0
2677d: movq $0x0, 0x50(%rdi)       # +0x50: ptr = 0 (function list head)
26785: movq $0x0, 0x58(%rdi)       # +0x58: ptr = 0 (function list tail)
2678d: movl $0x0, 0x60(%rdi)       # +0x60: uint32 = 0 (hash bucket count)
26794: movb $0x0, 0x64(%rdi)       # +0x64: byte = 0
```

**Fat binary registration record (0x68 = 104 bytes)**:

```
struct fatbin_record {
    +0x00: void* wrapper;         // __cudaFatCubinWrapper*
    +0x08: fatbin_record* next;  // linked list (NULL = end)
    +0x10: uint32_t flags;       // 0
    +0x14: uint32_t reserved;    // 0
    +0x18: void* ptr1;           // 0 (set later?)
    +0x20: void* ptr2;           // 0
    +0x28: void* ptr3;           // 0
    +0x30: void* ptr4;           // 0
    +0x38: uint8_t flag;         // 0 (set to 1 on function registration)
    +0x40: void* ptr5;           // 0
    +0x48: void* ptr6;           // 0
    +0x50: func_record* func_head; // linked list of registered functions
    +0x58: func_record* func_tail; // linked list tail
    +0x60: uint32_t bucket_count; // hash table size (0 initially)
    +0x64: uint8_t flag2;        // 0
};
```

### 3.4 __cudaRegisterFatBinaryEnd (0x23cf0)

```asm
23cf0: call getRuntimeState()       → ensure singleton
23d08: jmp  0x27080                → tail call to end handler
```

### 3.5 End Handler (0x27080)

```asm
27080: call 0x26710                → init a mutex/lock (0x2b93e0 global)
270a7: call 0x8e270                → init another sync primitive
270b0: mov  (%r15), %r9d           → check record->flags
270bc: mov  $0x11, %esi            → bucket count = 17
270c1: mov  $0x8, %edi             → entry size = 8
270ca: call calloc                → calloc(17, 8) = hash table
```

**After FatBinaryEnd**: The record has a 17-bucket hash table allocated, and the state's +0x000 field changes from 0x0 to 0x11 (17 in decimal). This is the hash table bucket count being stored as a "vtable" cookie.

### 3.6 __cudaRegisterFunction (0x23ea0)

```
Entry: rdi=handle, rsi=stub(hostFun), rdx=name, r8=-1, r9=tex,
       stack: func_name, device_name, thread_limit, ...

0x23ec4: call getRuntimeState()
0x23ef1: call 0x27fd0                → internal handler (10 parameters)
```

### 3.7 Internal Function Handler (0x27fd0)

```
Entry: rdi=runtime_state, rsi=handle(record), rdx=stub, rcx=name,
       r8=-1, r9=tex, stack: [func_name, device_name, thread_limit, ...]

0x28020: test %rsi, %rsi           → handle null?
0x28026: mov  $0x58, %edi           → size = 88 bytes
0x2802e: call malloc                → alloc function record
0x28038: mov  0x58(%rbx), %rdx     → record->func_tail (linked list)
0x2803c: movups %xmm3, (%rax)       → copy parameters to record
0x2806e: mov  0x58(%rbx), %rsi     → record->func_tail
0x28075: je   0x280a0               → tail is NULL?
0x28077: mov  %rax, 0x50(%rsi)     → tail->next = new_record
0x2807b: mov  %rax, 0x58(%rbx)     → record->func_tail = new_record
0x28089: call 0x27a00               → store in hash table (FNV-1a)
```

**Function registration record (0x58 = 88 bytes)**:

```
struct func_record {
    +0x00: void* wrapper;       // fatbin_record*
    +0x08: void* stub;          // hostFun (the key for lookup)
    +0x10: char* name;          // kernel name
    +0x18: char* func_name;     // mangled function name
    +0x20: void* tex;           // texture reference
    +0x28: void* device_name;   // device name
    +0x30: void* thread_limit;  // thread limit
    +0x38: int32_t bdim;        // block dimensions
    +0x3c: int32_t gdim;        // grid dimensions
    +0x40: int32_t wSize;        // warp size
    +0x48: void* ptr1;
    +0x50: func_record* next;   // linked list
};
```

### 3.8 Hash Table Insert (0x27a00)

The function uses **FNV-1a hash** (constant `0x811c9dc5`, prime `0x01000193`) to insert into the hash table at `record+0x60`:

```asm
27a42: xor  $0x811c9dc5, %eax    # FNV offset basis
27a50: imul $0x1000193, %eax     # FNV prime
...
27aca: div  %edi                   # hash % bucket_count → bucket index
27acc: mov  (%rsi,%rdx,8), %rdi  # hash_table[bucket]
```

The hash chain is a singly-linked list with `record->+0x08` (the stub pointer) used as the search key:
```asm
27ae8: cmp  0x8(%rdi), %rcx       # compare stub pointers
27aec: jne  27ae0                 # not match → follow next pointer
27af2: movb $0x1, 0x38(%r13)     # found → set flag byte
```

---

## 4. cudaGetFuncBySymbol — Complete Disassembly Trace

### 4.1 Three-Guard Check

```asm
; === GUARD 1: Runtime state null check ===
7962f: call 0x2a800                 # getRuntimeState()
79634: mov  %rax, %rbx             # rbx = runtime_state
79637: mov  $0x4, %eax              # eax = cudaErrorCudartUnloading (default)
7963c: test %rbx, %rbx
7963f: je   0x797ea                 # NULL → return 4

; === GUARD 2: Shutdown check ===
79645: mov  %rbx, %rdi
79648: call 0x42b00                 # checkShuttingDown(state)
7964d: test %eax, %eax
7964f: jne  0x797ea                 # shutting down → return 4

; === GUARD 3: Module management init flag ===
79655: mov  0x108(%rbx), %rax      # module_mgmt = state->0x108
7965c: mov  0x540(%rax), %edx      # init_flag = module_mgmt->0x540
79662: test %edx, %edx
79664: je   0x79800                 # ZERO → ALTERNATIVE RESOLVER
                                    # NON-ZERO → NORMAL RESOLVER (falls through)
```

### 4.2 Normal Resolver Path (0x540 != 0)

This path is never taken in our tests (0x540 is always 0), but the disassembly shows:

```asm
; Build parameter block on stack
7966a: lea  -0xcc(%rbp), %rsi      # &error_code
79671: lea  -0xc8(%rbp), %rcx      # &func_ptr
79685: movq $0x0, -0xac(%rbp)     # clear local
7968c: lea  0x21c7d(%rip), %r8     # address 0x9b310 (constant)
796bd: lea  0x40(%r14), %r9       # stack buffer for module info
796c1: movhps -0xd8(%rbp), %xmm1   # pack params
796d7: mov  %rdi, -0xd8(%rbp)     # save func_ptr to stack
796e5: movaps %xmm1, -0x100(%rbp) # store packed params
796f3: mov  %r12, -0xd8(%rbp)     # save symbol to stack
796fa: movhps -0xd8(%rbp), %xmm2   # pack more params
7971e: rep stos %rax, (%rdi)       # zero-fill buffer (0x150 bytes)

; First module vtable dispatch
79735: mov  0x100(%rbx), %r15      # r15 = func_vtable
7973c: call *0x10(%r15)            # func_vtable[2](resolve_setup)

; Module scope operations
79740: mov  0xf8(%rbx), %rax       # rax = module_vtable
79747: lea  0x8(%r14), %r15       # module info buffer
79752: call *0x20(%rax)            # module_vtable[4](link)

; Second module vtable dispatch (scope enter)
7976f: mov  0xf8(%rbx), %rax       # module_vtable
79792: call *0x8(%rax)             # module_vtable[1](scope_enter)

; Alternative resolver (always called in normal path too!)
7979b: call 0x23160                 # alternative resolver(symbol, func_ptr)

; Second function vtable dispatch
797a0: mov  0x100(%rbx), %r12      # func_vtable
797b4: call *0x10(%r12)            # func_vtable[2](resolve)

; Second module scope operation
797b9: mov  0xf8(%rbx), %r13       # module_vtable
797c7: call *0x20(%r13)            # module_vtable[4](link again)

; Finalize
797cb: mov  0xf8(%rbx), %rbx       # module_vtable
797e1: call *0x8(%rbx)             # module_vtable[1](scope_exit?)
```

**Key insight**: The normal path calls `0x23160` (the alternative resolver) at `0x7979b` regardless. The alternative resolver is not a fallback — it's a core part of the resolution pipeline in both paths.

### 4.3 Alternative Resolver Path (0x540 == 0)

```asm
79800: mov  %r13, %rsi              # rsi = symbol (hostFun)
79803: mov  %r12, %rdi              # rdi = func_ptr output
79806: call 0x23160                # alternative resolver(symbol, func_ptr)
7980b: jmp  0x797ea                # return
```

### 4.4 Alternative Resolver (0x23160) — The Real Work

```asm
23160: push %rbp
23166: mov  %rsi, %r14              # r14 = symbol (hostFun)
2316b: mov  %rdi, %r13              # r13 = func_ptr output
23170: lea  -0x28(%rbp), %r12       # &local_var (output of bootstrap)
23175: mov  %r12, %rdi              # &local_var
23184: call 0x466d0                # bootstrap(runtime_state, &local_var)
23189: test %eax, %eax              # bootstrap failed?
2318b: jne  0x231c0                # YES → error path

; Bootstrap succeeded, do core resolution
2318d: call 0x2a800                # getRuntimeState()
23196: mov  %r14, %rcx             # rcx = symbol
23199: mov  %r13, %rdx             # rdx = func_ptr output
2319c: mov  $0x1, %r8d             # r8 = 1 (flag: force resolve)
231a5: call 0x28e30                # CORE RESOLUTION
231aa: mov  %eax, %ebx              # save result
231ac: test %eax, %eax              # success?
231ae: jne  0x231c2                # NO → error path
231b4: mov  %ebx, %eax              # return 0
231be: ret

; Error path: bootstrap or resolution failed
231c0: mov  %eax, %ebx              # save error code
231c5: call 0x4cac0                # get thread-local error state
231cd: call 0x46510                # set error code: state->0x80 = error
231d2: test %rdi, %rdi              # thread-local state null?
231d9: je   0x231b0                # yes → return error
231dd: call 0x46510                # set error on thread-local state too
```

### 4.5 Bootstrap (0x466d0)

```asm
466d0: push %rbp
466d5: mov  %rdi, %rbx              # rbx = runtime_state
466d8: sub  $0x8, %rsp
466dc: call 0x2a800                # getRuntimeState()
466e4: mov 0xd0(%rax), %edx        # read lifecycle (unused - value overwritten)
466ea: mov  $0x1, %edx              # edx = 1 (flag = "already initialized")
466ef: mov 0xe8(%rax), %rdi        # rdi = state->0xe8 (another sub-object)
466fc: jmp  0x47350                # REAL bootstrap
```

**Key detail**: The bootstrap reads `state->0xe8` (a sub-object we haven't explored) and calls the real bootstrap at `0x47350` with a flag of 1, meaning "force initialization."

### 4.6 Real Bootstrap (0x47350)

```asm
47350: push %rbp
47364: mov  %rsi, %r12              # r12 = &output_local_var
47368: mov  %rdi, %rbx              # rbx = state->0xe8
47376: mov  (%rdi), %rax            # vtable from state->0xe8
47379: xor  %esi, %esi              # esi = 0
47383: mov  %r13, %rdi              # &output_local_var
47386: call *0x10(%rax)            # vtable[2] — module mgmt setup
47389: test %r14b, %r14b           # flag (always 1 from 0x466ea)
4738c: je   0x473e8                # flag=0 → simpler path

; flag=1 path: check result, potentially create module mgmt
4738e: test %eax, %eax
47390: jne  0x473b0               # setup failed → create new module mgmt
47392: mov  -0x28(%rbp), %rdx      # return existing module mgmt
47396: xor  %eax, %eax              # return 0 (success)
473a8: ret

; Setup failed → create module mgmt from scratch
473b0: mov  %rbx, %rdi
473b3: call 0x46bb0                # create module mgmt object
473b8: test %eax, %eax
473ba: jne  0x4739c               # creation failed → use existing
473c0: lea  -0x29(%rbp), %r14
473c3: call 0x26710                # init a sync primitive
473cb: mov  %rbx, %rdi
473ce: call 0x46d70                # create another sub-object
473d7: call 0x26720                # another sync primitive init
473df: jmp  0x47392                # return the created object
```

### 4.7 Core Resolution (0x28e30)

```asm
28e30: push %rbp
28e38: mov  %rdx, %r13              # r13 = func_ptr output
28e44: sub  $0x8, %rsp
28e49: call 0x28cc0                # LOOKUP: find stub in hash table
28e4e: mov  %eax, %ebx              # result
28e50: test %eax, %eax
28e52: jne  0x28e6d                # lookup failed → return error

; Lookup succeeded, now resolve
28e54: test %r13, %r13              # func_ptr output null?
28e57: je   0x28e6d                # yes → return error
28e59: lea  0x290380(%rip), %rax   # 0x2b91e0 (global function pointer)
28e60: mov  (%r12), %rsi           # r12 = &local_var from lookup
28e67: call *(%rax)                # indirect call → ACTUAL RESOLUTION
28e69: test %eax, %eax
28e6b: jne  0x28e80                # failed → call error setter
28e6d: return ebx                   # return 0 or error code
```

The function pointer at `0x2b91e0` is the actual kernel resolution function. It:
1. Finds the matching cubin/PTX in the fat binary
2. If needed, decompresses (zstd) and JIT-compiles PTX
3. Loads the module via `cuModuleLoadData` internally
4. Gets the function via `cuModuleGetFunction`
5. Stores the result

### 4.8 Hash Table Lookup (0x28cc0)

Uses the same FNV-1a hash as insertion:

```asm
28f44: xor  $0x811c9dc5, %eax    # FNV offset basis
28f50: imul $0x1000193, %eax     # FNV prime
...
28fc3: div  %r8d                   # hash % bucket_count
28fc6: mov  (%rcx,%rdx,8), %r14  # hash_table[bucket]
```

Then walks the chain:
```asm
28fec: mov  (%r14), %r14            # follow next pointer
28ff0: cmp  %rbx, 0x8(%r14)       # compare stub pointer
28ff4: jne  28fe0                  # not match → continue
28ff2: mov  0x10(%r14), %r14      # match → return found record
```

---

## 5. __cudaInitModule (0x23f10) — The CUDA 12+ Path

```asm
23f1b: call getRuntimeState()        # ensure singleton
23f3a: call checkShuttingDown()     # check lifecycle
23f43: jne  0x23fd1                # shutting down → return false

; Check initModule sub-object
23f49: mov  0x110(%rbx), %rax      # state->0x110
23f50: mov  0x4(%rax), %edx        # initModule->0x4
23f53: test %edx, %edx
23f55: je   0x23fe0                # ZERO → skip vtable, just bootstrap

; initModule->0x4 is non-zero: do vtable operations first
23f63: mov  0xf8(%rbx), %rax       # module_vtable
23fa5: call *0x28(%rax)            # module_vtable[5](module init)
23fa8: call 0x46710                # bootstrap

; Return success/failure
23fc9: test %eax, %eax
23fce: sete %dl                    # dl = (eax == 0)
23fd5: mov  %edx, %eax             # return true/false

; initModule->0x4 is zero: bootstrap only
23fe0: call 0x46710                # bootstrap
23fe5: jmp  0x23fcc
```

### 5.1 Module Init (0x46710)

```asm
46710: push %rbp
46718: lea  -0x8(%rbp), %rdi       # &local_var
4671c: call 0x466d0                # bootstrap(runtime_state, &local_var)
46722: ret
```

A thin wrapper that calls the same bootstrap as the alternative resolver.

---

## 6. __cudaGetKernel (0x24140) — By-Name Resolution

```asm
24140: push %rbp
2414d: test %rdi, %rdi              # output null?
24150: je   0x24210                # yes → error ("no kernel image")
24159: test %rsi, %rsi              # name null?
2415c: je   0x241e0                # yes → error ("duplicate kernels")
24165: mov  %rsi, (%r12)            # store name
24169: call getRuntimeState()
24171: call checkShuttingDown()
2417a: je   0x241b0                # not shutting down → resolve
241b0: call getRuntimeState()
241b5: mov  $0x1, %r8d              # flag = 1
241bb: mov  %r13, %rcx             # name
241be: xor  %edx, %edx             # func_ptr = NULL (not needed)
241c0: mov  %r12, %rsi             # &local_var
241c3: mov  %rax, %rdi             # runtime_state
241c6: call 0x28e30                # SAME core resolution as cudaGetFuncBySymbol!
```

**Key finding**: `__cudaGetKernel` uses the exact same core resolution function (`0x28e30`) as `cudaGetFuncBySymbol`. The difference is:
- `cudaGetFuncBySymbol` looks up by hostFun stub pointer
- `__cudaGetKernel` looks up by kernel name string

### 6.2 Error Handling (0x26680)

```asm
26680: push %rbp
266c4: mov  0x293bbd(%rip), %rax  # 0x2ba288 (global error handler)
266ce: test %rax, %rax
266d0: lea  0x10(%rbp), %rcx       # &error_code
266fd: call *0x8(%rax)             # global_error_handler(format, args...)
26708: leave; ret
```

The error strings at `0x9110b` ("kernelPtr") and `0x90f00` (format string) are referenced by `__cudaGetKernel` when it calls this error handler.

---

## 7. Key Proofs

### Proof 1: The 0x540 Field Is NOT a Gate

**Reproduction**: Run `./deep_trace simple_sm89.fatbin simple_add`

**Result**: `state->0x108->0x540 = 0` in ALL states, including after `cudaGetFuncBySymbol` returns `cudaSuccess`.

**Conclusion**: The `je 0x79800` at `0x79664` does NOT return error 4. It routes to the alternative resolver, which bootstraps module management on-the-fly and succeeds. Both paths (0x540==0 and 0x540!=0) can return success.

### Proof 2: Initialization Ordering Does Not Matter

**Reproduction**: Run `./test_isolated_context_first simple_sm89.fatbin simple_add`

**Result**: Creates Driver API context BEFORE registration. `cudaGetFuncBySymbol` returns `cudaSuccess`.

**Conclusion**: On CUDA 13.1.80, the Runtime API handles externally-created contexts gracefully. The original hypothesis about initialization ordering causing failures was wrong.

### Proof 3: The Alternative Resolver IS the Normal Path

**Reproduction**: Compare disassembly:
- Normal path (0x540 != 0): calls `0x23160` at `0x7979b`
- Alternative path (0x540 == 0): calls `0x23160` at `0x79806`

**Result**: Both code paths call the same `0x23160` function. The only difference is whether additional vtable dispatches happen before and after.

### Proof 4: Cross-Arch + zstd PTX Works

**Reproduction**: Run `./deep_trace crossarch_compressed.fatbin simple_add`

**Result**: sm_120 SASS + zstd-compressed compute_89 PTX on sm_89 GPU. `cudaGetFuncBySymbol` returns `cudaSuccess` — the Runtime decompresses zstd and JIT-compiles PTX internally.

### Proof 5: No Compatible Code → Error 209

**Reproduction**: Run `./deep_trace simple_sm120.fatbin simple_add`

**Result**: sm_120 SASS only (no PTX) on sm_89 GPU. `cudaGetFuncBySymbol` returns error 209 (`cudaErrorNoKernelImageForDevice`).

### Proof 6: Fat Binary Record Is 0x68 Bytes, Function Record Is 0x58 Bytes

**Reproduction**: Disassembly of `0x27029` (`mov $0x68, %edi; call malloc`) and `0x28026` (`mov $0x58, %edi; call malloc`).

**Result**: The registration uses fixed-size records. The hash table grows from 17 buckets.

### Proof 7: Hash Function Is FNV-1a

**Reproduction**: Disassembly of `0x27a42` and `0x28f44` both show `xor $0x811c9dc5` and `imul $0x1000193`.

**Result**: Both insertion (`0x27a00`) and lookup (`0x28e30`) use FNV-1a hash with the hostFun stub pointer as the key.

### Proof 8: libcudart.so Has Zero Direct Imports from libcuda.so

**Reproduction**: `nm -D /opt/cuda/targets/x86_64-linux/lib/libcudart.so.13.1.80 | grep ' U ' | grep cuda`

**Result**: No `cu*` symbols in the undefined imports. All Driver API functions are resolved via `dlopen("libcuda.so.1")` + `dlsym("cuInit")` + `dlvsym("cuCtxGetDevice@@CUDA_12.0")`.

---

## 8. Complete Internal Function Map

### Registration & Resolution

| Offset | Size | Purpose | Called By |
|--------|------|---------|-----------|
| `0x23c60` | 0x60 | `cudaGetExportTable` | External |
| `0x23cc0` | 0x30 | `__cudaRegisterFatBinary` | nvcc `.init_array` |
| `0x23cf0` | 0x1d | `__cudaRegisterFatBinaryEnd` | nvcc `.init_array` |
| `0x23d20` | 0x40 | `__cudaUnregisterFatBinary` | dlclose cleanup |
| `0x23d60` | 0x80 | `__cudaRegisterVar` | nvcc |
| `0x23de0` | 0x80 | `__cudaRegisterManagedVar` | nvcc |
| `0x23e60` | 0x40 | `__cudaRegisterHostVar` | nvcc |
| `0x23ea0` | 0x69 | `__cudaRegisterFunction` | nvcc `.init_array` |
| `0x23f10` | 0xe1 | `__cudaInitModule` | nvcc 12+ `.init_array` |
| `0x24140` | 0xff | `__cudaGetKernel` | Runtime internal |
| `0x24240` | 0x0b | `__cudaLaunchKernel` | nvcc (thunk) |
| **`0x79600`** | **0x210** | **`cudaGetFuncBySymbol`** | **Chidori server** |
| `0x79810` | ? | `cudaGetKernel` | External |

### Internal Functions

| Offset | Size | Purpose |
|--------|------|---------|
| `0x26680` | 0x89 | Error reporter (sets state->0xd0/0xd4) |
| `0x26710` | 0xc | Sync primitive init (global 0x2b93e0) |
| `0x26720` | 0xc | Sync primitive init variant |
| `0x26730` | 0x69 | Zero-fill 0x68-byte fat binary record |
| `0x267a0` | 0x45 | Module mgmt lookup (called from 0x28ed0) |
| `0x27020` | 0x3e | Register fat binary internal handler |
| `0x27080` | ? | Register fat binary end handler |
| `0x27a00` | ~0x600 | Hash table insert (FNV-1a, rebalancing) |
| `0x27fd0` | 0x102 | Register function internal handler |
| `0x28cc0` | ~0x50 | Hash table lookup |
| `0x28e30` | ~0x70 | Core resolution (lookup + resolve) |
| `0x2a800` | 0x20 | `getRuntimeState()` — singleton accessor |
| `0x42b00` | 0x50 | `checkShuttingDown()` — lifecycle check |
| `0x46510` | 0x7 | Error code setter (state->0x80 = code) |
| `0x46550` | ~0x100 | Module record management |
| `0x466d0` | 0x2d | Bootstrap wrapper |
| `0x46710` | 0x13 | Module init wrapper |
| `0x47350` | ~0x100 | Real bootstrap (creates module mgmt) |
| `0x4cac0` | 0x50 | Thread-local error state accessor |
| `0x8e360` | 0x18 | Singleton factory (pthread_once) |
| `0x900a0` | 0x5 | `malloc` trampoline |
| `0x900b0` | 0x5 | `free` trampoline |
| `0x900c0` | 0x5 | `calloc` trampoline |

---

## 9. The a2 Problem (Unsolved)

### What We Know

- a2 compressed cubins appear in pre-built CUDA 12.x pip wheels
- First byte: `0xa2` (proprietary magic)
- The Runtime API's internal decompression works in normal program startup
- It may fail when called from a server context via `dlsym`
- No compression-related strings exist in libcudart.so (all inlined)

### What We Don't Know

- The a2 decompression algorithm
- Whether a2 is version-specific (12.0 vs 12.1 vs 12.2)
- Whether the decompression depends on thread-local or process-global state
- Why exactly the server context path fails with a2

### The Working Workaround

The Chidori server bypasses the Runtime API entirely:
1. Parse `0xba55ed50` fat binary container
2. Find entries matching server GPU architecture (or compatible PTX)
3. If `flags & 0x40` and bytes = `28 b5 2f fd`: zstd decompress
4. Load via `cuModuleLoadData()` (Driver API)
5. Get function via `cuModuleGetFunction()` (Driver API)

This works for nvcc 13.1 fat binaries (zstd) but NOT for pre-built CUDA 12.x pip wheels (a2).

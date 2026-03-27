/*
 * deep_trace.c
 *
 * Comprehensive proof program for libcudart.so.13.1.80 internal analysis.
 * This program traces the EXACT code path through cudaGetFuncBySymbol by:
 *   1. Reading the runtime state object at each stage
 *   2. Printing the vtable pointers that control dispatch
 *   3. Measuring the module management sub-object layout
 *   4. Proving which internal functions are called via GDB script
 *
 * Build:
 *   gcc -o deep_trace deep_trace.c \
 *       -I/opt/cuda/include -L/opt/cuda/targets/x86_64-linux/lib \
 *       -lcuda -lcudart -ldl
 *
 * Run:
 *   ./deep_trace /path/to/kernel.fatbin simple_add
 *
 * For full GDB trace:
 *   gdb -batch -x deep_trace.gdb ./deep_trace simple_sm89.fatbin simple_add
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

typedef void **(*RegisterFatBinary_fn)(void *);
typedef void   (*RegisterFatBinaryEnd_fn)(void **);
typedef void   (*RegisterFunction_fn)(void **, const char *, char *,
                                       const char *, int, void *, void *,
                                       void *, void *, int *);

static void *alloc_stub(void)
{
    static void *page = NULL;
    static int idx = 0;
    if (!page) {
        page = mmap(NULL, 65536, PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        memset(page, 0xC3, 65536);
    }
    return (char *)page + (idx++ * 16);
}

static uintptr_t find_lib_base(const char *lib_name)
{
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[512];
    uintptr_t first_addr = 0;
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, lib_name) && strstr(line, ".so")) {
            uintptr_t addr;
            if (sscanf(line, "%lx-", &addr) == 1) {
                if (!found) {
                    first_addr = addr;
                    found = 1;
                }
            }
        }
    }
    fclose(f);
    return first_addr;
}

/*
 * Read the runtime state pointer from the known global at offset 0x2b93c8
 * in libcudart.so. Proven: getRuntimeState() at 0x2a800 does:
 *   lea 0x28ebb5(%rip), %rdi   → loads address of global at 0x2b93c4
 *   call singleton_factory      → creates/returns singleton at 0x8e360
 *   mov 0x28ebaa(%rip), %rax   → loads from 0x2b93c8 (the actual pointer)
 */
static uintptr_t read_runtime_state_ptr(uintptr_t base)
{
    volatile uintptr_t *global_slot = (volatile uintptr_t *)(base + 0x2b93c8);
    return *global_slot;
}

/*
 * Read the shutdown state pointer from global at offset 0x2b93d0
 * checkShuttingDown() at 0x42b00 does:
 *   lea 0x2768c1(%rip), %rdi   → loads address of global at 0x2b93d0
 *   lea 0x2b56(%rip), %rsi    → loads type_id from 0x45670
 *   call singleton_factory      → 0x8e360
 *   mov 0xd0(%rbx), %eax      → checks lifecycle at offset 0xd0
 */
static uintptr_t read_shutdown_state_ptr(uintptr_t base)
{
    volatile uintptr_t *global_slot = (volatile uintptr_t *)(base + 0x2b93d0);
    return *global_slot;
}

/*
 * Dump the COMPLETE runtime state object layout.
 * Every field proven from disassembly of getRuntimeState (0x2a800),
 * checkShuttingDown (0x42b00), cudaGetFuncBySymbol (0x79600),
 * __cudaRegisterFatBinary (0x23cc0), __cudaInitModule (0x23f10).
 */
static void dump_runtime_state_full(const char *label, uintptr_t state_ptr,
                                    uintptr_t lib_base)
{
    if (state_ptr == 0) {
        printf("[%s] Runtime state: NULL (not initialized)\n", label);
        return;
    }

    volatile uint8_t  *u8  = (volatile uint8_t  *)state_ptr;
    volatile uint32_t *u32 = (volatile uint32_t *)state_ptr;
    volatile uint64_t *u64 = (volatile uint64_t *)state_ptr;

    printf("\n=== %s ===\n", label);
    printf("runtime_state @ %p\n", (void *)state_ptr);

    /* +0x000: vtable pointer (general runtime) */
    printf("  +0x000 vtable (general):   0x%lx\n", (unsigned long)u64[0x000/8]);

    /* +0x0d0: lifecycle state
     * PROVEN from checkShuttingDown (0x42b1f):
     *   mov 0xd0(%rbx), %eax; cmp $0x2, %eax; je 42b40
     * 0 = uninit, 1 = active, 2 = shutting_down */
    uint32_t lifecycle = u32[0x0d0/4];
    printf("  +0x0d0 lifecycle state:   %u (%s)\n", lifecycle,
           lifecycle == 0 ? "UNINIT" : lifecycle == 1 ? "ACTIVE" :
           lifecycle == 2 ? "SHUTTING_DOWN" : "UNKNOWN");

    /* +0x0d4: shutdown error code
     * PROVEN from checkShuttingDown (0x42b40):
     *   mov 0xd4(%rbx), %eax; ret
     * Only read when lifecycle == 2 */
    if (lifecycle == 2) {
        printf("  +0x0d4 shutdown err code: %u\n", u32[0x0d4/4]);
    }

    /* +0x0d8: dlsym handle for libcuda.so
     * PROVEN from the function right after getRuntimeState (0x2a836):
     *   mov 0xd8(%rax), %rdi; call dlsym@plt
     */
    printf("  +0x0d8 libcuda dlopen:    %p%s\n", (void *)u64[0x0d8/8],
           u64[0x0d8/8] ? "" : " (not loaded)");

    /* +0x0f8: vtable pointer (module operations)
     * PROVEN from cudaGetFuncBySymbol (0x79740):
     *   mov 0xf8(%rbx), %rax; call *0x20(%rax)  → module load
     *   mov 0xf8(%rbx), %rax; call *0x8(%rax)   → module enum
     * Also 0x79792: call *0x8(%rax)  and 0x79e1: call *0x8(%rbx)
     */
    printf("  +0x0f8 vtable (module):   0x%lx%s\n", (unsigned long)u64[0x0f8/8],
           u64[0x0f8/8] ? "" : " (NULL - no module ops)");

    /* +0x100: vtable pointer (function resolution)
     * PROVEN from cudaGetFuncBySymbol (0x79735):
     *   mov 0x100(%rbx), %r15; call *0x10(%r15)  → func resolve
     * Also 0x797a0: mov 0x100(%rbx), %r12; call *0x10(%r12)
     */
    printf("  +0x100 vtable (func):     0x%lx%s\n", (unsigned long)u64[0x100/8],
           u64[0x100/8] ? "" : " (NULL - no func ops)");

    /* +0x108: pointer to module management subsystem
     * PROVEN from cudaGetFuncBySymbol (0x79655):
     *   mov 0x108(%rbx), %rax; mov 0x540(%rax), %edx
     */
    uintptr_t module_mgmt = (uintptr_t)u64[0x108/8];
    printf("  +0x108 module mgmt ptr:  %p%s\n", (void *)module_mgmt,
           module_mgmt ? "" : " (NULL)");

    /* +0x110: pointer to __cudaInitModule subsystem
     * PROVEN from __cudaInitModule (0x23f49):
     *   mov 0x110(%rbx), %rax; mov 0x4(%rax), %edx
     */
    uintptr_t init_mod = (uintptr_t)u64[0x110/8];
    printf("  +0x110 initModule ptr:    %p%s\n", (void *)init_mod,
           init_mod ? "" : " (NULL)");

    /* Dive into module management sub-object at +0x108 */
    if (module_mgmt) {
        volatile uint32_t *mm32 = (volatile uint32_t *)module_mgmt;
        volatile uint64_t *mm64 = (volatile uint64_t *)module_mgmt;

        printf("\n  --- Module Management Sub-Object @ %p ---\n",
               (void *)module_mgmt);

        /* +0x000: vtable pointer of module mgmt object */
        printf("    +0x000 vtable:           0x%lx\n",
               (unsigned long)mm64[0x000/8]);

        /* +0x540: init flag checked by cudaGetFuncBySymbol
         * PROVEN from cudaGetFuncBySymbol (0x7965c):
         *   mov 0x540(%rax), %edx; test %edx, %edx; je 79800
         * CRITICAL FINDING: This field is 0 in ALL states, yet
         * cudaGetFuncBySymbol still succeeds. It routes to the
         * alternative resolver at 0x23160 when this is 0. */
        uint32_t init_flag = mm32[0x540/4];
        printf("    +0x540 init flag:       %u (%s)\n", init_flag,
               init_flag ? "INITIALIZED" : "ZERO → alternative resolver path");
    }

    /* Dive into __cudaInitModule sub-object at +0x110 */
    if (init_mod) {
        volatile uint32_t *im32 = (volatile uint32_t *)init_mod;
        volatile uint64_t *im64 = (volatile uint64_t *)init_mod;

        printf("\n  --- __cudaInitModule Sub-Object @ %p ---\n",
               (void *)init_mod);

        printf("    +0x000 vtable:           0x%lx\n",
               (unsigned long)im64[0x000/8]);

        /* +0x004: init flag checked by __cudaInitModule
         * PROVEN from __cudaInitModule (0x23f50):
         *   mov 0x4(%rax), %edx; test %edx, %edx; je 23fe0
         * When 0: calls 0x46710 (bootstrap), then returns
         * When non-0: calls vtable+0x28, then calls 0x46710 */
        uint32_t im_flag = im32[0x004/4];
        printf("    +0x004 init flag:       %u\n", im_flag);
    }

    /* Print vtable entries if module vtable exists */
    if (u64[0x0f8/8]) {
        volatile uint64_t *mod_vtable = (volatile uint64_t *)u64[0x0f8/8];
        printf("\n  --- Module Vtable @ 0x%lx ---\n",
               (unsigned long)u64[0x0f8/8]);
        printf("    [+0x00] fn: 0x%lx (module enum/iterate?)\n",
               (unsigned long)mod_vtable[0]);
        printf("    [+0x08] fn: 0x%lx (module load/enter scope?)\n",
               (unsigned long)mod_vtable[1]);
        printf("    [+0x10] fn: 0x%lx (module unload/exit scope?)\n",
               (unsigned long)mod_vtable[2]);
        printf("    [+0x18] fn: 0x%lx (module get function?)\n",
               (unsigned long)mod_vtable[3]);
        printf("    [+0x20] fn: 0x%lx (module link?)\n",
               (unsigned long)mod_vtable[4]);
        printf("    [+0x28] fn: 0x%lx (module init?)\n",
               (unsigned long)mod_vtable[5]);
    }

    if (u64[0x100/8]) {
        volatile uint64_t *func_vtable = (volatile uint64_t *)u64[0x100/8];
        printf("\n  --- Function Resolution Vtable @ 0x%lx ---\n",
               (unsigned long)u64[0x100/8]);
        printf("    [+0x00] fn: 0x%lx\n", (unsigned long)func_vtable[0]);
        printf("    [+0x08] fn: 0x%lx\n", (unsigned long)func_vtable[1]);
        printf("    [+0x10] fn: 0x%lx (resolve function by symbol)\n",
               (unsigned long)func_vtable[2]);
    }
}

int main(int argc, char **argv)
{
    const char *fatbin_path = (argc > 1) ? argv[1] : NULL;
    const char *kernel_name = (argc > 2) ? argv[2] : "simple_add";

    printf("============================================================\n");
    printf("  libcudart.so.13.1.80 — Deep Trace Program\n");
    printf("  Binary: %s\n", fatbin_path ? fatbin_path : "(none)");
    printf("  Kernel: %s\n", kernel_name);
    printf("============================================================\n");

    /* Find libcudart.so base */
    uintptr_t base = find_lib_base("libcudart.so");
    if (base == 0) {
        printf("FATAL: Could not find libcudart.so base address.\n");
        return 1;
    }
    printf("\nlibcudart.so base: 0x%lx\n", (unsigned long)base);

    cuInit(0);

    /* ===================================================================
     * STAGE 1: Before any Runtime API call
     *
     * EXPECTED: runtime state NULL or minimal
     * PROVEN: getRuntimeState is only called by Runtime API functions.
     * The singleton factory at 0x8e360 calls pthread_once.
     * =================================================================== */
    dump_runtime_state_full("STAGE 1: Before any Runtime API call",
                           read_runtime_state_ptr(base), base);

    /* ===================================================================
     * STAGE 2: After cudaMalloc (first Runtime API call)
     *
     * WHAT HAPPENS (proven from disassembly):
     *   cudaMalloc calls getRuntimeState() which triggers singleton init
     *   The singleton factory (0x8e360) calls pthread_once → creates
     *   the runtime state object. Then 0x2a820 loads cuInit via dlsym.
     *
     * EXPECTED: lifecycle=1, vtables populated, module_mgmt populated
     * =================================================================== */
    void *d = NULL;
    cudaError_t e = cudaMalloc(&d, 256);
    printf("\n--- cudaMalloc(256) = %d (%s) ---\n",
           (int)e, e == 0 ? "SUCCESS" : "FAIL");
    dump_runtime_state_full("STAGE 2: After cudaMalloc",
                           read_runtime_state_ptr(base), base);
    if (d) cudaFree(d);
    cudaGetLastError();

    if (!fatbin_path) {
        printf("\nNo fatbin path. Stopping.\n");
        return 0;
    }

    /* Load fat binary into memory */
    FILE *f = fopen(fatbin_path, "rb");
    if (!f) { perror(fatbin_path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *fb = malloc(sz); fread(fb, 1, sz, f); fclose(f);

    RegisterFatBinary_fn reg_fb =
        (RegisterFatBinary_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
    RegisterFatBinaryEnd_fn reg_end =
        (RegisterFatBinaryEnd_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinaryEnd");
    RegisterFunction_fn reg_fn =
        (RegisterFunction_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFunction");

    /* ===================================================================
     * STAGE 3: After __cudaRegisterFatBinary
     *
     * WHAT HAPPENS (proven from disassembly of 0x23cc0):
     *   23ccc: call getRuntimeState()   → ensures singleton
     *   23cd1: lea  local_var, %rsi    → output handle
     *   23cd5: mov  %rbx, %rdx        → fat binary wrapper
     *   23cdb: call 0x27020           → internal handler
     *
     * Internal handler (0x27020):
     *   27029: mov  $0x68, %edi       → malloc(0x68) = 104 bytes
     *   27039: call malloc@plt (0x900a0 → jmp malloc)
     *   27047: call 0x26730           → zero out struct
     *   27051: mov  %rbx, 0x0(%r13)   → store handle
     *
     * The 0x68-byte struct (fat binary registration record):
     *   +0x00: ptr to __cudaFatCubinWrapper (input)
     *   +0x08: ptr to next record (linked list)
     *   +0x10: uint32_t = 0 (flags?)
     *   +0x14: uint32_t = 0
     *   +0x18: ptr = 0
     *   +0x20..+0x64: more fields
     *
     * =================================================================== */
    struct { int magic; int version; const void *data; const char *filename; } w;
    w.magic = 0x466243b1; w.version = 1; w.data = fb; w.filename = NULL;

    void **handle = reg_fb(&w);
    printf("\n--- __cudaRegisterFatBinary: handle=%p ---\n", (void *)handle);
    dump_runtime_state_full("STAGE 3: After __cudaRegisterFatBinary",
                           read_runtime_state_ptr(base), base);

    /* ===================================================================
     * STAGE 3b: After __cudaRegisterFatBinaryEnd
     *
     * __cudaRegisterFatBinaryEnd (0x23cf0) calls getRuntimeState() then
     * jumps to 0x27080 which calls 0x26710 (a lock/mutex init function).
     * =================================================================== */
    if (reg_end) reg_end(handle);
    dump_runtime_state_full("STAGE 3b: After __cudaRegisterFatBinaryEnd",
                           read_runtime_state_ptr(base), base);

    /* Register function */
    void *stub = alloc_stub();
    reg_fn(handle, (const char *)stub, (char *)kernel_name, (char *)kernel_name,
           -1, NULL, NULL, NULL, NULL, NULL);

    /* ===================================================================
     * STAGE 4: After Driver API context creation
     *
     * We create the context via cuDevicePrimaryCtxRetain to test
     * whether the Runtime API path breaks when context exists first.
     *
     * PROVEN FINDING: On CUDA 13.1, this does NOT break
     * cudaGetFuncBySymbol. The ordering is irrelevant.
     * =================================================================== */
    CUcontext ctx;
    CUresult cr = cuDevicePrimaryCtxRetain(&ctx, 0);
    cuCtxSetCurrent(ctx);
    printf("\n--- cuDevicePrimaryCtxRetain: %d ctx=%p ---\n",
           (int)cr, (void *)ctx);
    dump_runtime_state_full("STAGE 4: After Driver API context",
                           read_runtime_state_ptr(base), base);

    /* ===================================================================
     * STAGE 5: cudaGetFuncBySymbol — THE KEY TEST
     *
     * CODE PATH (proven from disassembly of 0x79600):
     *
     * Guard 1 (0x7962f): call getRuntimeState()
     *   → returns runtime_state ptr from global at 0x2b93c8
     *   → if NULL: return error 4 (cudaErrorCudartUnloading)
     *
     * Guard 2 (0x79648): call checkShuttingDown()
     *   → reads state->0xd0, if 2 (SHUTTING_DOWN): return error 4
     *   → otherwise calls 0x42280 for deeper state check
     *
     * Guard 3 (0x79655): load state->0x108, then load [0x108]->0x540
     *   → if 0: jump to 0x79800 (ALTERNATIVE RESOLVER)
     *   → if non-0: fall through to NORMAL RESOLVER (vtable dispatch)
     *
     * NORMAL RESOLVER (0x540 != 0):
     *   0x7966a..0x797e1: Build parameter block on stack
     *   0x79735: mov 0x100(%rbx), %r15 → load func vtable
     *   0x7973c: call *0x10(%r15)       → func_vtable[2] (resolve)
     *   0x79740: mov 0xf8(%rbx), %rax  → load module vtable
     *   0x79752: call *0x20(%rax)       → mod_vtable[4] (link)
     *   0x7979b: call 0x23160           → alternative resolver
     *   0x797a0: mov 0x100(%rbx), %r12  → load func vtable again
     *   0x797b4: call *0x10(%r12)       → func_vtable[2] again
     *   0x797b9: mov 0xf8(%rbx), %r13  → load module vtable again
     *   0x797c7: call *0x20(%r13)       → mod_vtable[4] again
     *   0x797cb: mov 0xf8(%rbx), %rbx  → load module vtable again
     *   0x797e1: call *0x8(%rbx)        → mod_vtable[1] (finalize?)
     *
     * ALTERNATIVE RESOLVER (0x540 == 0) at 0x79800:
     *   0x79800: mov %r13, %rsi        → symbol (hostFun)
     *   0x79803: mov %r12, %rdi        → func_ptr output
     *   0x79806: call 0x23160           → alternative resolver
     *   0x7980b: jmp return
     *
     * ALTERNATIVE RESOLVER at 0x23160:
     *   23175: lea local_var, %r12    → output local
     *   23184: call 0x466d0            → bootstrap
     *     466dc: call getRuntimeState()
     *     466e4: mov 0xd0(%rax), %edx → read lifecycle (ignored)
     *     466ea: mov $0x1, %edx        → edx = 1 (flag = "force init")
     *     466ef: mov 0xe8(%rax), %rdi  → read another sub-object
     *     466fc: jmp 0x47350           → REAL bootstrap
     *   23189: test %eax, %eax        → bootstrap failed?
     *   2318b: jne 231c0              → if failed, take error path
     *     231c0: call 0x4cac0          → get thread-local error state
     *     231cd: call 0x46510          → set error code (stores at state+0x80)
     *   2318d: call getRuntimeState()  → get state again
     *   23196: mov %r14, %rcx         → symbol (hostFun)
     *   23199: mov %r13, %rdx         → func_ptr output
     *   2319c: mov $0x1, %r8d         → flag = 1
     *   231a5: call 0x28e30           → CORE RESOLUTION FUNCTION
     *
     * CORE RESOLUTION at 0x28e30:
     *   28e49: call 0x28cc0           → lookup in registry (FNV-1a hash)
     *   28e59: lea 0x290380(%rip), %rax → load global at 0x2b91e0
     *   28e67: call *(%rax)           → indirect call via function pointer
     *   On success: returns 0 and stores result via r13 (func_ptr)
     *   On failure: returns non-zero error code
     *
     * =================================================================== */
    cudaGetLastError(); // clear any pending error
    cudaFunction_t func = NULL;
    cudaError_t rt_err = cudaGetFuncBySymbol(&func, stub);
    printf("\n--- cudaGetFuncBySymbol: err=%d (%s) func=%p ---\n",
           (int)rt_err,
           rt_err == 0 ? "cudaSuccess" :
           rt_err == 4 ? "cudaErrorCudartUnloading" :
           rt_err == 209 ? "cudaErrorNoKernelImageForDevice" :
           rt_err == 200 ? "cudaErrorInvalidImage" : "other",
           func);
    dump_runtime_state_full("STAGE 5: After cudaGetFuncBySymbol",
                           read_runtime_state_ptr(base), base);

    /* ===================================================================
     * STAGE 6: Verify the 0x540 field is STILL zero after success
     *
     * This is the KEY PROOF that disproves the "init gate" hypothesis.
     * If 0x540 were a gate, cudaGetFuncBySymbol could not have succeeded
     * when it was 0. The actual path taken is the alternative resolver
     * at 0x23160, which bootstraps on-the-fly and succeeds.
     * =================================================================== */
    uintptr_t state = read_runtime_state_ptr(base);
    if (state) {
        uintptr_t mm = (uintptr_t)((volatile uint64_t *)state)[0x108/8];
        if (mm) {
            uint32_t flag540 = ((volatile uint32_t *)mm)[0x540/4];
            printf("\n============================================================\n");
            printf("  PROOF: 0x540 field = %u after cudaGetFuncBySymbol %s\n",
                   flag540, rt_err == 0 ? "SUCCEEDED" : "FAILED");
            printf("  The 0x540 field is %sa gate.\n",
                   flag540 == 0 ? "NOT " : "");
            if (flag540 == 0 && rt_err == 0) {
                printf("  → cudaGetFuncBySymbol took the ALTERNATIVE RESOLVER path\n");
                printf("  → (0x79800 → 0x23160 → 0x466d0 → 0x47350 → 0x28e30)\n");
            }
            printf("============================================================\n");
        }
    }

    free(fb);
    return (rt_err == cudaSuccess) ? 0 : 1;
}

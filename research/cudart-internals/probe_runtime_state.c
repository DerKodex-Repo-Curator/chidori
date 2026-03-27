/*
 * probe_runtime_state.c
 *
 * Reads internal fields of the CUDA Runtime state object to prove
 * which field gates cudaGetFuncBySymbol.
 *
 * Strategy: We call cudaMalloc to trigger Runtime initialization, then
 * read the runtime state pointer from the known global at offset 0x2b93c8
 * in libcudart.so (relative to library base). We read the module
 * management sub-object at state+0x108 and the init flag at +0x540.
 *
 * We compare these values in three scenarios:
 *   1. After cudaMalloc only (baseline)
 *   2. After __cudaRegisterFatBinary (before context)
 *   3. After Driver API context creation (before registration)
 *
 * Build:
 *   gcc -o probe_runtime_state probe_runtime_state.c \
 *       -I/opt/cuda/include -L/opt/cuda/targets/x86_64-linux/lib \
 *       -lcuda -lcudart -ldl
 */
#include <stdio.h>
#include <stdlib.h>
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

/*
 * We need the base address of libcudart.so to compute the absolute
 * address of the runtime state global. The global is at file offset
 * 0x2b93c8 (discovered from disassembly of getRuntimeState at 0x2a800).
 *
 * In a PIE (position-independent executable) or shared library, the
 * runtime state global is accessed via RIP-relative addressing. We can't
 * just read from a fixed address. Instead, we:
 *
 *   1. Find libcudart.so's load address via /proc/self/maps or dlinfo
 *   2. Compute the absolute address of the global
 *   3. Read the pointer stored there
 *
 * But actually, a simpler approach: since getRuntimeState() returns the
 * pointer, we can call a Runtime API function that we KNOW calls
 * getRuntimeState() internally, and observe the result.
 *
 * The simplest approach: call __cudaRegisterFatBinary with a minimal
 * wrapper and observe what it stores. The handle it returns IS the
 * registered module pointer, not the runtime state.
 *
 * Better approach: We'll trace through the actual code path.
 * From the disassembly, getRuntimeState() at 0x2a800:
 *   1. Calls singleton factory at 0x8e360 with (global_ptr_addr, type_id)
 *   2. Returns *(global_ptr_addr + 4)
 *
 * The global_ptr_addr is computed as RIP + offset in the lea instruction.
 * In the loaded library, this resolves to an absolute address in .bss.
 *
 * We'll find libcudart.so's base, add the .bss offset, and read directly.
 */

/* Find the base load address of a shared library by name.
 * The first LOAD segment has vaddr=0, so the first mapping's
 * start address IS the base. */
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

/* Read the runtime state pointer from the global at offset 0x2b93c8 */
static uintptr_t read_runtime_state_ptr(uintptr_t libcudart_base)
{
    /*
     * From disassembly of getRuntimeState() (0x2a800):
     *   lea 0x28ebb5(%rip), %rdi   → loads address of global at file offset 0x2b93c4
     *   call singleton_factory      → creates/returns singleton
     *   mov 0x28ebaa(%rip), %rax   → loads from file offset 0x2b93c8
     *
     * In the loaded library, file offset 0x2b93c8 maps to:
     *   libcudart_base + 0x2b93c8
     *
     * The value there is a pointer to the runtime state object.
     */
    volatile uintptr_t *global_slot = (volatile uintptr_t *)(libcudart_base + 0x2b93c8);
    return *global_slot;
}

/* Dump interesting fields from the runtime state object */
static void dump_runtime_state(const char *label, uintptr_t state_ptr)
{
    if (state_ptr == 0) {
        printf("[%s] Runtime state pointer: NULL (runtime not initialized)\n", label);
        return;
    }

    volatile uint8_t  *state = (volatile uint8_t  *)state_ptr;
    volatile uint32_t *state32 = (volatile uint32_t *)state_ptr;
    volatile uint64_t *state64 = (volatile uint64_t *)state_ptr;

    printf("[%s] Runtime state @ %p\n", label, (void *)state_ptr);
    printf("  [0x%03x] lifecycle state:    %u\n", 0xd0, state32[0xd0 / 4]);
    printf("  [0x%03x] shutdown err code:  %u\n", 0xd4, state32[0xd4 / 4]);
    printf("  [0x%03x] vtable (module):    0x%lx\n", 0xf8, (unsigned long)state64[0xf8 / 8]);
    printf("  [0x%03x] vtable (func):      0x%lx\n", 0x100, (unsigned long)state64[0x100 / 8]);

    /* state->0x108 = pointer to module management subsystem */
    uintptr_t module_mgmt = (uintptr_t)state64[0x108 / 8];
    printf("  [0x%03x] module mgmt ptr:   %p\n", 0x108, (void *)module_mgmt);

    if (module_mgmt != 0) {
        volatile uint32_t *mm32 = (volatile uint32_t *)module_mgmt;
        /* The critical field: module_mgmt->0x540 */
        uint32_t init_flag = mm32[0x540 / 4];
        printf("    [0x%03x] *** INIT FLAG ***:  %u  (%s)\n", 0x540, init_flag,
               init_flag ? "INITIALIZED" : "NOT INITIALIZED ← cudaGetFuncBySymbol will fail");
    } else {
        printf("    module mgmt ptr is NULL\n");
    }

    /* state->0x110 = pointer to __cudaInitModule subsystem */
    uintptr_t init_mod = (uintptr_t)state64[0x110 / 8];
    printf("  [0x%03x] initModule ptr:     %p\n", 0x110, (void *)init_mod);
    if (init_mod != 0) {
        volatile uint32_t *im32 = (volatile uint32_t *)init_mod;
        printf("    [0x%03x] initModule flag:  %u\n", 0x4, im32[0x4 / 4]);
    }
}

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

int main(int argc, char **argv)
{
    const char *fatbin_path = (argc > 1) ? argv[1] : NULL;
    const char *kernel_name = (argc > 2) ? argv[2] : "simple_add";

    printf("=== CUDA Runtime State Probe ===\n\n");

    /* Find libcudart.so base address */
    uintptr_t base = find_lib_base("libcudart.so");
    if (base == 0) {
        printf("WARNING: Could not find libcudart.so base address.\n");
        printf("Falling back to dlsym-based approach.\n\n");
        base = 0;
    } else {
        printf("libcudart.so base address: 0x%lx\n\n", (unsigned long)base);
    }

    cuInit(0);

    /* --- STATE 1: Before any Runtime API call --- */
    printf("--- STATE 1: Before any Runtime API call ---\n");
    if (base) dump_runtime_state("1a", read_runtime_state_ptr(base));

    /* --- STATE 2: After cudaMalloc (standard Runtime init) --- */
    printf("\n--- STATE 2: After cudaMalloc ---\n");
    void *d = NULL;
    cudaError_t e = cudaMalloc(&d, 256);
    printf("cudaMalloc(256) = %d (%s)\n", (int)e, e == 0 ? "SUCCESS" : "FAIL");
    if (base) dump_runtime_state("2a", read_runtime_state_ptr(base));
    if (d) cudaFree(d);
    cudaGetLastError();

    if (!fatbin_path) {
        printf("\nNo fatbin path provided, stopping here.\n");
        return 0;
    }

    /* Load fat binary */
    FILE *f = fopen(fatbin_path, "rb");
    if (!f) { perror(fatbin_path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *fb = malloc(sz); fread(fb, 1, sz, f); fclose(f);

    RegisterFatBinary_fn reg_fb = (RegisterFatBinary_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
    RegisterFatBinaryEnd_fn reg_end = (RegisterFatBinaryEnd_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinaryEnd");
    RegisterFunction_fn reg_fn = (RegisterFunction_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFunction");

    struct { int magic; int version; const void *data; const char *filename; } w;
    w.magic = 0x466243b1; w.version = 1; w.data = fb; w.filename = NULL;

    /* --- STATE 3: After __cudaRegisterFatBinary (no context yet) --- */
    printf("\n--- STATE 3: After __cudaRegisterFatBinary (no context) ---\n");
    void **handle = reg_fb(&w);
    printf("__cudaRegisterFatBinary: handle=%p\n", (void *)handle);
    if (reg_end) reg_end(handle);
    if (base) dump_runtime_state("3a", read_runtime_state_ptr(base));

    /* Register function */
    void *stub = alloc_stub();
    reg_fn(handle, (const char *)stub, (char *)kernel_name, (char *)kernel_name,
           -1, NULL, NULL, NULL, NULL, NULL);

    /* --- STATE 4: After creating Driver API context --- */
    printf("\n--- STATE 4: After cuDevicePrimaryCtxRetain ---\n");
    CUcontext ctx;
    cuDevicePrimaryCtxRetain(&ctx, 0);
    cuCtxSetCurrent(ctx);
    if (base) dump_runtime_state("4a", read_runtime_state_ptr(base));

    /* --- STATE 5: cudaGetFuncBySymbol result --- */
    printf("\n--- STATE 5: cudaGetFuncBySymbol ---\n");
    cudaFunction_t func = NULL;
    e = cudaGetFuncBySymbol(&func, stub);
    printf("cudaGetFuncBySymbol: err=%d (%s) func=%p\n",
           (int)e,
           e == 0 ? "SUCCESS" : e == 4 ? "cudaErrorCudartUnloading" : "other",
           func);
    if (base) dump_runtime_state("5a", read_runtime_state_ptr(base));

    free(fb);
    return 0;
}

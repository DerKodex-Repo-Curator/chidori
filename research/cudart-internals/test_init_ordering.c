/*
 * test_init_ordering.c — Proof that CUDA Runtime API initialization ordering matters
 *
 * This test demonstrates that cudaGetFuncBySymbol returns cudaErrorCudartUnloading
 * (error 4) when a CUDA context is created via the Driver API BEFORE
 * __cudaRegisterFatBinary is called, but succeeds when the ordering is reversed.
 *
 * Build:
 *   gcc -o test_init_ordering test_init_ordering.c \
 *       -I/opt/cuda/include -L/opt/cuda/targets/x86_64-linux/lib \
 *       -lcuda -lcudart -ldl -lpthread
 *
 * Run:
 *   ./test_init_ordering /path/to/kernel.fatbin simple_add
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

/* ------------------------------------------------------------------ */
/* Types for __cudaRegisterFatBinary / __cudaRegisterFunction         */
/* ------------------------------------------------------------------ */
typedef void **(*RegisterFatBinary_fn)(void *);
typedef void   (*RegisterFatBinaryEnd_fn)(void **);
typedef void   (*RegisterFunction_fn)(void **, const char *, char *,
                                       const char *, int, void *, void *,
                                       void *, void *, int *);

/* ------------------------------------------------------------------ */
/* Stub allocation: mmap executable memory filled with RET (0xC3)     */
/* The Runtime requires hostFun to be a real executable address.      */
/* ------------------------------------------------------------------ */
static void *g_stub_page = NULL;
static int   g_stub_idx  = 0;

static void *alloc_stub(void)
{
    if (!g_stub_page) {
        g_stub_page = mmap(NULL, 65536, PROT_READ | PROT_WRITE | PROT_EXEC,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_stub_page == MAP_FAILED) { perror("mmap"); return NULL; }
        memset(g_stub_page, 0xC3, 65536);
    }
    return (char *)g_stub_page + (g_stub_idx++ * 16);
}

/* ------------------------------------------------------------------ */
/* Load a file into memory                                            */
/* ------------------------------------------------------------------ */
static void *load_file(const char *path, size_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

/* ------------------------------------------------------------------ */
/* Print cudaError_t as string                                        */
/* ------------------------------------------------------------------ */
static const char *cuda_err_str(cudaError_t e)
{
    switch (e) {
    case cudaSuccess:                   return "cudaSuccess (0)";
    case cudaErrorCudartUnloading:      return "cudaErrorCudartUnloading (4)";
    case cudaErrorInvalidImage:         return "cudaErrorInvalidImage (200)";
    case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice (209)";
    case cudaErrorInvalidValue:         return "cudaErrorInvalidValue (1)";
    case cudaErrorInitializationError:  return "cudaErrorInitializationError (3)";
    default: {
        static char buf[64];
        snprintf(buf, sizeof(buf), "cudaError code %d", (int)e);
        return buf;
    }
    }
}

/* ------------------------------------------------------------------ */
/* Test: Register BEFORE creating Driver API context                  */
/* This is the "correct" ordering — what a normal CUDA app does.     */
/* ------------------------------------------------------------------ */
static int test_register_first(RegisterFatBinary_fn reg_fb,
                                RegisterFatBinaryEnd_fn reg_fb_end,
                                RegisterFunction_fn reg_fn,
                                const void *fatbin_data, size_t fatbin_size,
                                const char *kernel_name)
{
    printf("\n========================================\n");
    printf("TEST 1: Register BEFORE Driver API context\n");
    printf("========================================\n");

    /* Step 1: Build the __cudaFatCubinWrapper struct */
    struct {
        int         magic;
        int         version;
        const void *data;
        const char *filename;
    } wrapper;
    wrapper.magic    = 0x466243b1;  /* __cudaFatMAGIC2 */
    wrapper.version  = 1;
    wrapper.data     = fatbin_data;
    wrapper.filename = NULL;

    /* Step 2: Register the fat binary — NO Driver API context exists yet */
    printf("[TEST1] Calling __cudaRegisterFatBinary (no context yet)...\n");
    void **handle = reg_fb(&wrapper);
    if (!handle) {
        printf("[TEST1] __cudaRegisterFatBinary returned NULL\n");
        return 1;
    }
    printf("[TEST1] __cudaRegisterFatBinary succeeded, handle=%p\n", (void *)handle);

    if (reg_fb_end)
        reg_fb_end(handle);
    printf("[TEST1] __cudaRegisterFatBinaryEnd called\n");

    /* Step 3: Register a function with an executable stub */
    void *stub = alloc_stub();
    printf("[TEST1] Calling __cudaRegisterFunction with stub=%p, name='%s'...\n",
           stub, kernel_name);
    reg_fn(handle, (const char *)stub, (char *)kernel_name,
           (char *)kernel_name, -1, NULL, NULL, NULL, NULL, NULL);
    printf("[TEST1] __cudaRegisterFunction succeeded\n");

    /* Step 4: NOW create the Driver API context (after registration) */
    printf("[TEST1] Creating Driver API context AFTER registration...\n");
    CUcontext ctx = NULL;
    CUresult cr = cuDevicePrimaryCtxRetain(&ctx, 0);
    printf("[TEST1] cuDevicePrimaryCtxRetain = %d (%s)\n", (int)cr,
           cr == CUDA_SUCCESS ? "SUCCESS" : "FAIL");
    if (cr != CUDA_SUCCESS) return 1;

    cr = cuCtxSetCurrent(ctx);
    printf("[TEST1] cuCtxSetCurrent = %d\n", (int)cr);

    /* Step 5: Try cudaGetFuncBySymbol */
    printf("[TEST1] Calling cudaGetFuncBySymbol(stub=%p)...\n", stub);
    void *func = NULL;
    cudaError_t err = cudaGetFuncBySymbol(&func, stub);
    printf("[TEST1] cudaGetFuncBySymbol returned: %s  func=%p\n",
           cuda_err_str(err), func);

    if (err == cudaSuccess && func != NULL) {
        printf("[TEST1] RESULT: PASS — function resolved successfully\n");
        return 0;
    } else {
        printf("[TEST1] RESULT: FAIL — %s\n", cuda_err_str(err));
        return 1;
    }
}

/* ------------------------------------------------------------------ */
/* Test: Create Driver API context FIRST, then register              */
/* This is what the Chidori server does — and it fails.              */
/* ------------------------------------------------------------------ */
static int test_context_first(RegisterFatBinary_fn reg_fb,
                               RegisterFatBinaryEnd_fn reg_fb_end,
                               RegisterFunction_fn reg_fn,
                               const void *fatbin_data, size_t fatbin_size,
                               const char *kernel_name)
{
    printf("\n========================================\n");
    printf("TEST 2: Driver API context BEFORE Register\n");
    printf("(This is what the Chidori server does)\n");
    printf("========================================\n");

    /* Step 1: Create Driver API context FIRST */
    printf("[TEST2] Creating Driver API context FIRST...\n");
    CUcontext ctx = NULL;
    CUresult cr = cuDevicePrimaryCtxRetain(&ctx, 0);
    printf("[TEST2] cuDevicePrimaryCtxRetain = %d (%s)\n", (int)cr,
           cr == CUDA_SUCCESS ? "SUCCESS" : "FAIL");
    if (cr != CUDA_SUCCESS) return 1;

    cr = cuCtxSetCurrent(ctx);
    printf("[TEST2] cuCtxSetCurrent = %d\n", (int)cr);

    /* Step 2: Attempt to initialize Runtime API with standard calls */
    printf("[TEST2] Attempting Runtime API init via cudaMalloc/cudaFree...\n");
    cudaError_t rt_err;
    int dev = -1;
    rt_err = cudaGetDevice(&dev);
    printf("[TEST2] cudaGetDevice = %s (dev=%d)\n", cuda_err_str(rt_err), dev);
    rt_err = cudaSetDevice(0);
    printf("[TEST2] cudaSetDevice(0) = %s\n", cuda_err_str(rt_err));
    void *dummy = NULL;
    rt_err = cudaMalloc(&dummy, 1);
    printf("[TEST2] cudaMalloc(1) = %s ptr=%p\n", cuda_err_str(rt_err), dummy);
    if (rt_err == cudaSuccess && dummy) {
        rt_err = cudaFree(dummy);
        printf("[TEST2] cudaFree = %s\n", cuda_err_str(rt_err));
    } else {
        cudaGetLastError(); /* clear error */
    }

    /* Step 3: Register the fat binary */
    struct {
        int         magic;
        int         version;
        const void *data;
        const char *filename;
    } wrapper;
    wrapper.magic    = 0x466243b1;
    wrapper.version  = 1;
    wrapper.data     = fatbin_data;
    wrapper.filename = NULL;

    printf("[TEST2] Calling __cudaRegisterFatBinary (context already exists)...\n");
    void **handle = reg_fb(&wrapper);
    if (!handle) {
        printf("[TEST2] __cudaRegisterFatBinary returned NULL\n");
        return 1;
    }
    printf("[TEST2] __cudaRegisterFatBinary succeeded, handle=%p\n", (void *)handle);

    if (reg_fb_end)
        reg_fb_end(handle);
    printf("[TEST2] __cudaRegisterFatBinaryEnd called\n");

    /* Step 4: Register a function */
    void *stub = alloc_stub();
    printf("[TEST2] Calling __cudaRegisterFunction with stub=%p, name='%s'...\n",
           stub, kernel_name);
    reg_fn(handle, (const char *)stub, (char *)kernel_name,
           (char *)kernel_name, -1, NULL, NULL, NULL, NULL, NULL);
    printf("[TEST2] __cudaRegisterFunction succeeded\n");

    /* Step 5: Try cudaGetFuncBySymbol — THIS IS WHERE IT FAILS */
    printf("[TEST2] Calling cudaGetFuncBySymbol(stub=%p)...\n", stub);
    void *func = NULL;
    cudaError_t err = cudaGetFuncBySymbol(&func, stub);
    printf("[TEST2] cudaGetFuncBySymbol returned: %s  func=%p\n",
           cuda_err_str(err), func);

    if (err == cudaSuccess && func != NULL) {
        printf("[TEST2] RESULT: PASS (unexpected — initialization ordering may not matter)\n");
        return 0;
    } else {
        printf("[TEST2] RESULT: FAIL — %s\n", cuda_err_str(err));
        printf("[TEST2] This proves that creating a Driver API context before\n");
        printf("[TEST2] __cudaRegisterFatBinary breaks cudaGetFuncBySymbol.\n");
        return 1;
    }
}

/* ------------------------------------------------------------------ */
/* Test: Try cudaFree(0) trick + extra Runtime API calls              */
/* This is what the Chidori server currently does as a workaround.   */
/* ------------------------------------------------------------------ */
static int test_context_first_with_cuda_free_zero(RegisterFatBinary_fn reg_fb,
                                                    RegisterFatBinaryEnd_fn reg_fb_end,
                                                    RegisterFunction_fn reg_fn,
                                                    const void *fatbin_data, size_t fatbin_size,
                                                    const char *kernel_name)
{
    printf("\n========================================\n");
    printf("TEST 3: Context first + cudaFree(0) init trick\n");
    printf("========================================\n");

    /* Step 1: Create Driver API context FIRST */
    printf("[TEST3] Creating Driver API context FIRST...\n");
    CUcontext ctx = NULL;
    CUresult cr = cuDevicePrimaryCtxRetain(&ctx, 0);
    printf("[TEST3] cuDevicePrimaryCtxRetain = %d\n", (int)cr);
    if (cr != CUDA_SUCCESS) return 1;
    cr = cuCtxSetCurrent(ctx);
    printf("[TEST3] cuCtxSetCurrent = %d\n", (int)cr);

    /* Step 2: The cudaFree(0) trick */
    printf("[TEST3] Calling cudaFree(0)...\n");
    cudaError_t rt_err = cudaFree(0);
    printf("[TEST3] cudaFree(0) = %s\n", cuda_err_str(rt_err));
    cudaGetLastError();

    /* Step 3: More aggressive init */
    rt_err = cudaSetDevice(0);
    printf("[TEST3] cudaSetDevice(0) = %s\n", cuda_err_str(rt_err));
    rt_err = cudaDeviceSynchronize();
    printf("[TEST3] cudaDeviceSynchronize = %s\n", cuda_err_str(rt_err));
    cudaGetLastError();

    /* Step 4: Register */
    struct {
        int         magic;
        int         version;
        const void *data;
        const char *filename;
    } wrapper;
    wrapper.magic    = 0x466243b1;
    wrapper.version  = 1;
    wrapper.data     = fatbin_data;
    wrapper.filename = NULL;

    printf("[TEST3] Calling __cudaRegisterFatBinary...\n");
    void **handle = reg_fb(&wrapper);
    if (!handle) {
        printf("[TEST3] __cudaRegisterFatBinary returned NULL\n");
        return 1;
    }
    printf("[TEST3] __cudaRegisterFatBinary succeeded, handle=%p\n", (void *)handle);
    if (reg_fb_end) reg_fb_end(handle);

    /* Step 5: Register function */
    void *stub = alloc_stub();
    printf("[TEST3] Calling __cudaRegisterFunction...\n");
    reg_fn(handle, (const char *)stub, (char *)kernel_name,
           (char *)kernel_name, -1, NULL, NULL, NULL, NULL, NULL);

    /* Step 6: Try even MORE init between registration and resolution */
    printf("[TEST3] Attempting post-registration init...\n");
    void *d = NULL;
    rt_err = cudaMalloc(&d, 256);
    printf("[TEST3] cudaMalloc(256) = %s\n", cuda_err_str(rt_err));
    if (rt_err == cudaSuccess && d) {
        rt_err = cudaFree(d);
        printf("[TEST3] cudaFree = %s\n", cuda_err_str(rt_err));
    }
    cudaGetLastError();

    /* Step 7: cudaGetFuncBySymbol */
    printf("[TEST3] Calling cudaGetFuncBySymbol(stub=%p)...\n", stub);
    void *func = NULL;
    cudaError_t err = cudaGetFuncBySymbol(&func, stub);
    printf("[TEST3] cudaGetFuncBySymbol returned: %s  func=%p\n",
           cuda_err_str(err), func);

    if (err == cudaSuccess && func != NULL) {
        printf("[TEST3] RESULT: PASS\n");
        return 0;
    } else {
        printf("[TEST3] RESULT: FAIL — %s\n", cuda_err_str(err));
        return 1;
    }
}

/* ------------------------------------------------------------------ */
/* Test: cudaLaunchKernel to force full bootstrap                    */
/* If a kernel launch forces the module mgmt subsystem to init,     */
/* subsequent cudaGetFuncBySymbol calls might work.                  */
/* ------------------------------------------------------------------ */
static int test_launch_forces_bootstrap(RegisterFatBinary_fn reg_fb,
                                         RegisterFatBinaryEnd_fn reg_fb_end,
                                         RegisterFunction_fn reg_fn,
                                         const void *fatbin_data, size_t fatbin_size,
                                         const char *kernel_name)
{
    printf("\n========================================\n");
    printf("TEST 4: Does a kernel launch bootstrap the module subsystem?\n");
    printf("========================================\n");

    /* Create context first */
    CUcontext ctx = NULL;
    CUresult cr = cuDevicePrimaryCtxRetain(&ctx, 0);
    if (cr != CUDA_SUCCESS) { printf("[TEST4] cuDevicePrimaryCtxRetain failed\n"); return 1; }
    cuCtxSetCurrent(ctx);
    cudaFree(0);
    cudaGetLastError();

    /* Register */
    struct {
        int magic; int version; const void *data; const char *filename;
    } wrapper = { 0x466243b1, 1, fatbin_data, NULL };

    void **handle = reg_fb(&wrapper);
    if (!handle) { printf("[TEST4] RegisterFatBinary failed\n"); return 1; }
    if (reg_fb_end) reg_fb_end(handle);

    void *stub = alloc_stub();
    reg_fn(handle, (const char *)stub, (char *)kernel_name,
           (char *)kernel_name, -1, NULL, NULL, NULL, NULL, NULL);

    /* Try cudaGetFuncBySymbol BEFORE any launch */
    void *func = NULL;
    cudaError_t err = cudaGetFuncBySymbol(&func, stub);
    printf("[TEST4] cudaGetFuncBySymbol BEFORE launch: %s\n", cuda_err_str(err));

    /* Try to launch via the stub pointer directly */
    printf("[TEST4] Attempting kernel launch via stub=%p...\n", stub);
    float dummy_arg = 1.0f;
    void *args[] = { &dummy_arg };
    err = cudaLaunchKernel(stub, 1, 1, args, 0, NULL);
    printf("[TEST4] cudaLaunchKernel via stub: %s\n", cuda_err_str(err));
    cudaDeviceSynchronize();
    cudaGetLastError();

    /* Try cudaGetFuncBySymbol AFTER the launch attempt */
    func = NULL;
    err = cudaGetFuncBySymbol(&func, stub);
    printf("[TEST4] cudaGetFuncBySymbol AFTER launch: %s  func=%p\n",
           cuda_err_str(err), func);

    if (err == cudaSuccess && func != NULL) {
        printf("[TEST4] RESULT: PASS — kernel launch bootstrapped the subsystem\n");
        return 0;
    } else {
        printf("[TEST4] RESULT: FAIL — %s\n", cuda_err_str(err));
        return 1;
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <fatbin_file> <kernel_name>\n", argv[0]);
        fprintf(stderr, "Example: %s simple_sm89.fatbin simple_add\n", argv[0]);
        return 1;
    }

    const char *fatbin_path  = argv[1];
    const char *kernel_name  = argv[2];

    printf("=== CUDA Runtime API Initialization Ordering Test ===\n");
    printf("Fat binary: %s\n", fatbin_path);
    printf("Kernel:     %s\n", kernel_name);
    printf("CUDA driver version: %d.%d\n",
           CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10);

    /* Load fat binary from file */
    size_t fatbin_size = 0;
    void *fatbin_data = load_file(fatbin_path, &fatbin_size);
    if (!fatbin_data) {
        fprintf(stderr, "Failed to load fat binary: %s\n", fatbin_path);
        return 1;
    }
    printf("Fat binary loaded: %zu bytes, magic=0x%08x\n",
           fatbin_size, *(uint32_t *)fatbin_data);

    /* Initialize Driver API */
    CUresult cr = cuInit(0);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", (int)cr);
        return 1;
    }
    printf("cuInit succeeded\n");

    /* Resolve Runtime API registration functions via dlsym */
    RegisterFatBinary_fn reg_fb = (RegisterFatBinary_fn)
        dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
    RegisterFatBinaryEnd_fn reg_fb_end = (RegisterFatBinaryEnd_fn)
        dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinaryEnd");
    RegisterFunction_fn reg_fn = (RegisterFunction_fn)
        dlsym(RTLD_DEFAULT, "__cudaRegisterFunction");

    if (!reg_fb || !reg_fn) {
        fprintf(stderr, "Failed to resolve registration functions:\n");
        if (!reg_fb) fprintf(stderr, "  __cudaRegisterFatBinary: %s\n", dlerror());
        if (!reg_fn) fprintf(stderr, "  __cudaRegisterFunction: %s\n", dlerror());
        return 1;
    }
    printf("Resolved: __cudaRegisterFatBinary=%p __cudaRegisterFunction=%p\n",
           (void *)(uintptr_t)reg_fb, (void *)(uintptr_t)reg_fn);

    /* Run tests */
    int pass = 0, fail = 0;

    /* TEST 1: Register first, then context — should PASS */
    /* NOTE: We can't truly test this in the same process because
       the runtime state persists. But we'll try. */
    int r1 = test_register_first(reg_fb, reg_fb_end, reg_fn,
                                  fatbin_data, fatbin_size, kernel_name);
    if (r1 == 0) pass++; else fail++;

    /* Clean up: release context */
    cuDevicePrimaryCtxRelease(0);
    /* We can't truly reset the runtime state, so TEST 2 might not
       reproduce the bug if TEST 1 already initialized things.
       For a clean test, TEST 2 should be run in a separate process. */

    /* TEST 2: Context first, then register — should FAIL */
    int r2 = test_context_first(reg_fb, reg_fb_end, reg_fn,
                                 fatbin_data, fatbin_size, kernel_name);
    if (r2 == 0) pass++; else fail++;

    /* TEST 3: Context first + cudaFree(0) trick */
    int r3 = test_context_first_with_cuda_free_zero(reg_fb, reg_fb_end, reg_fn,
                                                      fatbin_data, fatbin_size, kernel_name);
    if (r3 == 0) pass++; else fail++;

    /* TEST 4: Does a launch bootstrap the subsystem? */
    int r4 = test_launch_forces_bootstrap(reg_fb, reg_fb_end, reg_fn,
                                           fatbin_data, fatbin_size, kernel_name);
    if (r4 == 0) pass++; else fail++;

    printf("\n========================================\n");
    printf("SUMMARY: %d passed, %d failed\n", pass, fail);
    printf("========================================\n");

    if (r2 != 0 && r1 == 0) {
        printf("\nCONCLUSION: Initialization ordering IS the root cause.\n");
        printf("  - Register BEFORE context creation: WORKS\n");
        printf("  - Context BEFORE registration: FAILS with cudaErrorCudartUnloading\n");
    } else if (r2 == 0 && r1 == 0) {
        printf("\nCONCLUSION: Runtime state persists across tests.\n");
        printf("  Run TEST 2 in isolation (separate process) to confirm.\n");
    }

    free(fatbin_data);
    return (fail > 0) ? 1 : 0;
}

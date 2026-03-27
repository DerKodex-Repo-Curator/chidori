/*
 * test_isolated_register_first.c
 *
 * Run this in ISOLATION to prove that calling __cudaRegisterFatBinary
 * BEFORE creating a Driver API context allows cudaGetFuncBySymbol to succeed.
 *
 * Build:
 *   gcc -o test_isolated_register_first test_isolated_register_first.c \
 *       -I/opt/cuda/include -L/opt/cuda/targets/x86_64-linux/lib \
 *       -lcuda -lcudart -ldl
 *
 * Run:
 *   ./test_isolated_register_first /path/to/kernel.fatbin simple_add
 *
 * Expected: cudaGetFuncBySymbol returns cudaSuccess (0)
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
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <fatbin> <kernel_name>\n", argv[0]);
        return 1;
    }

    /* Load fat binary */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *fb = malloc(sz);
    fread(fb, 1, sz, f);
    fclose(f);

    /* Init driver */
    cuInit(0);

    /* Resolve registration functions */
    RegisterFatBinary_fn reg_fb = (RegisterFatBinary_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
    RegisterFatBinaryEnd_fn reg_end = (RegisterFatBinaryEnd_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinaryEnd");
    RegisterFunction_fn reg_fn = (RegisterFunction_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFunction");

    /* CRITICAL: Register fat binary BEFORE creating any context */
    struct { int magic; int version; const void *data; const char *filename; } w;
    w.magic = 0x466243b1; w.version = 1; w.data = fb; w.filename = NULL;
    void **handle = reg_fb(&w);
    printf("[A] __cudaRegisterFatBinary: handle=%p (NO context yet)\n", (void *)handle);
    if (reg_end) reg_end(handle);

    /* Register function */
    void *stub = alloc_stub();
    reg_fn(handle, (const char *)stub, (char *)argv[2], (char *)argv[2],
           -1, NULL, NULL, NULL, NULL, NULL);
    printf("[B] __cudaRegisterFunction: stub=%p name='%s'\n", stub, argv[2]);

    /* NOW create the Driver API context AFTER registration */
    CUcontext ctx;
    CUresult cr = cuDevicePrimaryCtxRetain(&ctx, 0);
    printf("[C] cuDevicePrimaryCtxRetain: %d ctx=%p\n", (int)cr, (void *)ctx);
    cuCtxSetCurrent(ctx);

    /* THE TEST: cudaGetFuncBySymbol */
    cudaFunction_t func = NULL;
    cudaError_t err = cudaGetFuncBySymbol(&func, stub);
    printf("[D] cudaGetFuncBySymbol: err=%d (%s) func=%p\n",
           (int)err,
           err == 0 ? "cudaSuccess" :
           err == 4 ? "cudaErrorCudartUnloading" :
           err == 209 ? "cudaErrorNoKernelImageForDevice" :
           err == 200 ? "cudaErrorInvalidImage" : "other",
           func);

    if (err == cudaSuccess) {
        printf("\n*** PROVEN: cudaGetFuncBySymbol succeeds ***\n");
        printf("*** when __cudaRegisterFatBinary is called BEFORE context creation. ***\n");
    } else {
        printf("\n*** UNEXPECTED: cudaGetFuncBySymbol failed with error %d ***\n", (int)err);
    }

    free(fb);
    return (err == cudaSuccess) ? 0 : 1;
}

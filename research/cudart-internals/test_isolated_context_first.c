/*
 * test_isolated_context_first.c
 *
 * Run this in ISOLATION to prove that creating a Driver API context
 * before __cudaRegisterFatBinary causes cudaGetFuncBySymbol to fail
 * with cudaErrorCudartUnloading (error 4).
 *
 * Build:
 *   gcc -o test_isolated_context_first test_isolated_context_first.c \
 *       -I/opt/cuda/include -L/opt/cuda/targets/x86_64-linux/lib \
 *       -lcuda -lcudart -ldl
 *
 * Run:
 *   ./test_isolated_context_first /path/to/kernel.fatbin simple_add
 *
 * Expected: cudaGetFuncBySymbol returns cudaErrorCudartUnloading (4)
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

    /* CRITICAL: Create Driver API context BEFORE any Runtime API call */
    CUcontext ctx;
    cuDevicePrimaryCtxRetain(&ctx, 0);
    cuCtxSetCurrent(ctx);
    printf("[A] Driver API context created: ctx=%p\n", (void *)ctx);

    /* Try to "initialize" the Runtime API with standard calls */
    cudaSetDevice(0);
    cudaFree(0);
    cudaGetLastError();
    printf("[B] cudaSetDevice(0) + cudaFree(0) completed\n");

    /* Resolve registration functions */
    RegisterFatBinary_fn reg_fb = (RegisterFatBinary_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinary");
    RegisterFatBinaryEnd_fn reg_end = (RegisterFatBinaryEnd_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFatBinaryEnd");
    RegisterFunction_fn reg_fn = (RegisterFunction_fn)dlsym(RTLD_DEFAULT, "__cudaRegisterFunction");

    /* Register fat binary */
    struct { int magic; int version; const void *data; const char *filename; } w;
    w.magic = 0x466243b1; w.version = 1; w.data = fb; w.filename = NULL;
    void **handle = reg_fb(&w);
    printf("[C] __cudaRegisterFatBinary: handle=%p\n", (void *)handle);
    if (reg_end) reg_end(handle);

    /* Register function */
    void *stub = alloc_stub();
    reg_fn(handle, (const char *)stub, (char *)argv[2], (char *)argv[2],
           -1, NULL, NULL, NULL, NULL, NULL);
    printf("[D] __cudaRegisterFunction: stub=%p name='%s'\n", stub, argv[2]);

    /* THE TEST: cudaGetFuncBySymbol */
    cudaFunction_t func = NULL;
    cudaError_t err = cudaGetFuncBySymbol(&func, stub);
    printf("[E] cudaGetFuncBySymbol: err=%d (%s) func=%p\n",
           (int)err,
           err == 4 ? "cudaErrorCudartUnloading" :
           err == 0 ? "cudaSuccess" :
           err == 209 ? "cudaErrorNoKernelImageForDevice" :
           err == 200 ? "cudaErrorInvalidImage" : "other",
           func);

    if (err == cudaErrorCudartUnloading) {
        printf("\n*** PROVEN: cudaGetFuncBySymbol returns cudaErrorCudartUnloading ***\n");
        printf("*** when Driver API context is created BEFORE registration.  ***\n");
    } else if (err == cudaSuccess) {
        printf("\n*** UNEXPECTED: cudaGetFuncBySymbol succeeded. ***\n");
        printf("*** The initialization ordering may not matter on this CUDA version. ***\n");
    } else {
        printf("\n*** DIFFERENT ERROR: %d ***\n", (int)err);
    }

    free(fb);
    return (err == cudaErrorCudartUnloading) ? 0 : 1;
}

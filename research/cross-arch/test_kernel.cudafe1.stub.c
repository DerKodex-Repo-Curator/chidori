#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "test_kernel.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z7add_onePfi(float *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z7add_onePfi(float *__par0, int __par1){__cudaLaunchPrologue(2);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaLaunch(((char *)((void ( *)(float *, int))add_one)));}
# 2 "/tmp/test_kernel.cu"
void add_one( float *__cuda_0,int __cuda_1)
# 2 "/tmp/test_kernel.cu"
{__device_stub__Z7add_onePfi( __cuda_0,__cuda_1);


}
# 1 "test_kernel.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T4) {  __nv_dummy_param_ref(__T4); __nv_save_fatbinhandle_for_managed_rt(__T4); __cudaRegisterEntry(__T4, ((void ( *)(float *, int))add_one), _Z7add_onePfi, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop

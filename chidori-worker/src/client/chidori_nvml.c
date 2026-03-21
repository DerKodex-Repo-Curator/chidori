/*
 * Chidori NVML Shim — Intercepts nvmlDeviceGetCount for partial GPU offloading.
 *
 * PyTorch calls NVML (libnvidia-ml.so.1) via ctypes to get device count
 * BEFORE calling cudaGetDeviceCount. This thin shim returns the merged
 * count (local + remote GPUs) for partial offloading.
 *
 * Build: gcc -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 -o build/libnvidia-ml.so.1 src/client/chidori_nvml.c -ldl
 * Usage: LD_PRELOAD="build/libcuda.so.1 build/libnvidia-ml.so.1" ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#define NVML_SUCCESS 0

/* Get the merged device count from our main chidori library.
 * We call cudaGetDeviceCount which already handles partial offload. */
static int get_merged_count(void) {
    /* Call our cudaGetDeviceCount (from libchidori.so via LD_PRELOAD) */
    typedef int (*fn_t)(int *);
    fn_t our_fn = (fn_t)dlsym(RTLD_DEFAULT, "cudaGetDeviceCount");
    if (our_fn) {
        int count = 0;
        if (our_fn(&count) == 0)
            return count;
    }
    return 1;
}

/* Forward non-count NVML calls to the real libnvidia-ml.so */
static void *get_real_nvml(void) {
    static void *real = NULL;
    if (!real) {
        /* Try system paths for the real NVML */
        const char *paths[] = {
            "/usr/lib/libnvidia-ml.so.1",
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            NULL
        };
        for (int i = 0; paths[i]; i++) {
            real = dlopen(paths[i], RTLD_LAZY | RTLD_DEEPBIND);
            if (real) break;
        }
    }
    return real;
}

int nvmlInit(void) {
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(void);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlInit");
        if (real_fn) return real_fn();
    }
    return NVML_SUCCESS;
}

int nvmlInit_v2(void) {
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(void);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlInit_v2");
        if (real_fn) return real_fn();
    }
    return NVML_SUCCESS;
}

int nvmlDeviceGetCount(unsigned int *deviceCount) {
    const char *env = getenv("CHIDORI_LOCAL_DEVICES");
    if (env && deviceCount) {
        *deviceCount = (unsigned int)get_merged_count();
        return NVML_SUCCESS;
    }
    /* Non-partial: forward to real NVML */
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetCount");
        if (real_fn) return real_fn(deviceCount);
    }
    if (deviceCount) *deviceCount = 1;
    return NVML_SUCCESS;
}

int nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    return nvmlDeviceGetCount(deviceCount);
}

int nvmlShutdown(void) {
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(void);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlShutdown");
        if (real_fn) return real_fn();
    }
    return NVML_SUCCESS;
}

/* Pass through all other NVML calls via a catchall */
/* PyTorch only calls: nvmlInit, nvmlDeviceGetCount_v2, nvmlShutdown,
 * nvmlDeviceGetHandleByPciBusId_v2, nvmlDeviceGetUUID */
int nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, void *device) {
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(const char *, void *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2");
        if (real_fn) return real_fn(pciBusId, device);
    }
    return 6; /* NVML_ERROR_NOT_FOUND */
}

int nvmlDeviceGetUUID(void *device, char *uuid, unsigned int length) {
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(void *, char *, unsigned int);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetUUID");
        if (real_fn) return real_fn(device, uuid, length);
    }
    return 6;
}

/*
 * Chidori NVML Shim — Intercepts NVML calls for external GPU mode.
 *
 * When running without a local GPU, this shim provides fake NVML
 * responses that describe the remote GPU. This is needed because:
 * - vLLM uses pynvml (ctypes) to detect CUDA availability
 * - PyTorch uses NVML for device enumeration
 * - Both call nvmlInitWithFlags, nvmlDeviceGetCount, nvmlDeviceGetName etc.
 *
 * Build: gcc -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 -o build/libnvidia-ml.so.1 src/client/chidori_nvml.c -ldl
 * Usage: LD_LIBRARY_PATH=build (or LD_PRELOAD=build/libnvidia-ml.so.1)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#define NVML_SUCCESS            0
#define NVML_ERROR_NOT_FOUND    6
#define NVML_ERROR_INVALID_ARG  2

static int g_nvml_initialized = 0;

/* Forward to real NVML if available (for partial offload with local GPU) */
static void *get_real_nvml(void) {
    static void *real = NULL;
    static int tried = 0;
    if (!tried) {
        tried = 1;
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

/* Check if we have a real local GPU (partial offload mode) */
static int has_local_gpu(void) {
    return get_real_nvml() != NULL;
}

/* Get device count from our main chidori library */
static int get_chidori_device_count(void) {
    typedef int (*fn_t)(int *);
    fn_t fn = (fn_t)dlsym(RTLD_DEFAULT, "cudaGetDeviceCount");
    if (fn) {
        int count = 0;
        if (fn(&count) == 0) return count;
    }
    return 1;
}

/* ===== Initialization ===== */

int nvmlInitWithFlags(unsigned int flags) {
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(unsigned int);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlInitWithFlags");
        if (real_fn) return real_fn(flags);
    }
    g_nvml_initialized = 1;
    return NVML_SUCCESS;
}

int nvmlInit(void) {
    return nvmlInitWithFlags(0);
}

int nvmlInit_v2(void) {
    return nvmlInitWithFlags(0);
}

int nvmlShutdown(void) {
    void *nvml = get_real_nvml();
    if (nvml) {
        typedef int (*fn_t)(void);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlShutdown");
        if (real_fn) return real_fn();
    }
    g_nvml_initialized = 0;
    return NVML_SUCCESS;
}

/* ===== Device Enumeration ===== */

int nvmlDeviceGetCount(unsigned int *deviceCount) {
    if (!deviceCount) return NVML_ERROR_INVALID_ARG;
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetCount");
        if (real_fn) return real_fn(deviceCount);
    }
    /* No local GPU — return count from chidori (remote device) */
    *deviceCount = (unsigned int)get_chidori_device_count();
    return NVML_SUCCESS;
}

int nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    return nvmlDeviceGetCount(deviceCount);
}

/* Fake device handle — just use the index as the handle */
typedef struct { unsigned int index; } nvml_device_t;
static nvml_device_t g_devices[16];

int nvmlDeviceGetHandleByIndex(unsigned int index, void *device) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(unsigned int, void *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetHandleByIndex");
        if (real_fn) return real_fn(index, device);
    }
    if (index >= 16 || !device) return NVML_ERROR_INVALID_ARG;
    g_devices[index].index = index;
    memcpy(device, &g_devices[index], sizeof(nvml_device_t));
    return NVML_SUCCESS;
}

int nvmlDeviceGetHandleByIndex_v2(unsigned int index, void *device) {
    return nvmlDeviceGetHandleByIndex(index, device);
}

/* ===== Device Properties ===== */

int nvmlDeviceGetName(void *device, char *name, unsigned int length) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, char *, unsigned int);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetName");
        if (real_fn) return real_fn(device, name, length);
    }
    /* Return the name reported by our remote GPU */
    if (name && length > 0) {
        /* Try to get real name from chidori */
        typedef const char *(*fn_t)(int);
        fn_t fn = (fn_t)dlsym(RTLD_DEFAULT, "cudaGetDeviceName");
        if (fn) {
            const char *devname = fn(0);
            if (devname) {
                strncpy(name, devname, length - 1);
                name[length - 1] = '\0';
                return NVML_SUCCESS;
            }
        }
        strncpy(name, "NVIDIA RTX A6000", length - 1);
        name[length - 1] = '\0';
    }
    return NVML_SUCCESS;
}

int nvmlDeviceGetMemoryInfo(void *device, void *memory) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, void *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetMemoryInfo");
        if (real_fn) return real_fn(device, memory);
    }
    /* Return 48GB for A6000-like */
    if (memory) {
        unsigned long long *info = (unsigned long long *)memory;
        info[0] = 48ULL * 1024 * 1024 * 1024;  /* total */
        info[1] = 0;                             /* free */
        info[2] = 0;                             /* used */
    }
    return NVML_SUCCESS;
}

int nvmlDeviceGetUUID(void *device, char *uuid, unsigned int length) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, char *, unsigned int);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetUUID");
        if (real_fn) return real_fn(device, uuid, length);
    }
    if (uuid && length >= 40) {
        strncpy(uuid, "GPU-chidori-remote-0", length - 1);
        uuid[length - 1] = '\0';
    }
    return NVML_SUCCESS;
}

int nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, void *device) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(const char *, void *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2");
        if (real_fn) return real_fn(pciBusId, device);
    }
    return NVML_ERROR_NOT_FOUND;
}

int nvmlDeviceGetPciInfo(void *device, void *pci) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, void *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetPciInfo");
        if (real_fn) return real_fn(device, pci);
    }
    return NVML_ERROR_NOT_FOUND;
}

int nvmlDeviceGetUtilizationRates(void *device, void *utilization) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, void *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetUtilizationRates");
        if (real_fn) return real_fn(device, utilization);
    }
    if (utilization) {
        memset(utilization, 0, 8);
    }
    return NVML_SUCCESS;
}

int nvmlDeviceGetTemperature(void *device, int sensorType, unsigned int *temp) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, int, unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetTemperature");
        if (real_fn) return real_fn(device, sensorType, temp);
    }
    if (temp) *temp = 40;
    return NVML_SUCCESS;
}

int nvmlDeviceGetPowerUsage(void *device, unsigned int *power) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetPowerUsage");
        if (real_fn) return real_fn(device, power);
    }
    if (power) *power = 100000;
    return NVML_SUCCESS;
}

int nvmlDeviceGetClockInfo(void *device, int type, unsigned int *clock) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, int, unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetClockInfo");
        if (real_fn) return real_fn(device, type, clock);
    }
    if (clock) *clock = 1000;
    return NVML_SUCCESS;
}

int nvmlDeviceGetComputeRunningProcesses(void *device, unsigned int *infoCount, void *infos) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, unsigned int *, void *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetComputeRunningProcesses");
        if (real_fn) return real_fn(device, infoCount, infos);
    }
    if (infoCount) *infoCount = 0;
    return NVML_SUCCESS;
}

int nvmlDeviceGetComputeMode(void *device, int *mode) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetComputeMode");
        if (real_fn) return real_fn(device, mode);
    }
    if (mode) *mode = 0; /* NVML_COMPUTEMODE_DEFAULT */
    return NVML_SUCCESS;
}

int nvmlDeviceGetMaxClockInfo(void *device, int type, unsigned int *clock) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, int, unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetMaxClockInfo");
        if (real_fn) return real_fn(device, type, clock);
    }
    if (clock) *clock = 2000;
    return NVML_SUCCESS;
}

int nvmlDeviceGetNumGpuCores(void *device, unsigned int *numCores) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetNumGpuCores");
        if (real_fn) return real_fn(device, numCores);
    }
    if (numCores) *numCores = 10752; /* A6000 SM count */
    return NVML_SUCCESS;
}

int nvmlDeviceGetBrand(void *device, int *type) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetBrand");
        if (real_fn) return real_fn(device, type);
    }
    if (type) *type = 3; /* NVML_BRAND_QUADRO */
    return NVML_SUCCESS;
}

int nvmlDeviceGetMinorNumber(void *device, unsigned int *minorNumber) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, unsigned int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetMinorNumber");
        if (real_fn) return real_fn(device, minorNumber);
    }
    if (minorNumber) *minorNumber = 0;
    return NVML_SUCCESS;
}

int nvmlDeviceGetPersistenceMode(void *device, int *mode) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetPersistenceMode");
        if (real_fn) return real_fn(device, mode);
    }
    if (mode) *mode = 1;
    return NVML_SUCCESS;
}

int nvmlSystemGetDriverVersion(char *version, unsigned int length) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(char *, unsigned int);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlSystemGetDriverVersion");
        if (real_fn) return real_fn(version, length);
    }
    if (version && length > 0) {
        strncpy(version, "570.195.03", length - 1);
        version[length - 1] = '\0';
    }
    return NVML_SUCCESS;
}

int nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(char *, unsigned int);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlSystemGetNVMLVersion");
        if (real_fn) return real_fn(version, length);
    }
    if (version && length > 0) {
        strncpy(version, "12.8", length - 1);
        version[length - 1] = '\0';
    }
    return NVML_SUCCESS;
}

int nvmlDeviceGetCudaComputeCapability(void *device, int *major, int *minor) {
    if (has_local_gpu()) {
        void *nvml = get_real_nvml();
        typedef int (*fn_t)(void *, int *, int *);
        fn_t real_fn = (fn_t)dlsym(nvml, "nvmlDeviceGetCudaComputeCapability");
        if (real_fn) return real_fn(device, major, minor);
    }
    if (major) *major = 8;
    if (minor) *minor = 6;
    return NVML_SUCCESS;
}

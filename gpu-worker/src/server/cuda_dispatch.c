/*
 * Juice GPU Worker — Real CUDA Dispatch
 *
 * Handles deserialization of wire protocol messages, calls real CUDA
 * Driver API and Runtime API functions, and serializes responses back.
 *
 * This file is included by gpu_worker.c when building with CUDA support.
 * Compile with: -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "juice_protocol.h"
#include "handle_map.h"

/* Maximum number of GPUs we support */
#define JUICE_MAX_DEVICES 16

/* Server-side state */
static CUdevice   g_devices[JUICE_MAX_DEVICES];
static CUcontext  g_contexts[JUICE_MAX_DEVICES]; /* Primary context per device */
static int        g_device_count = 0;
static int        g_initialized = 0;
static CUcontext  g_current_ctx = NULL;          /* Currently active context */
static int        g_current_dev = 0;             /* Device index of current context */

/* Handle maps for opaque CUDA objects */
static handle_map_t g_modules;
static handle_map_t g_functions;
static handle_map_t g_streams;
static handle_map_t g_events;
static handle_map_t g_ctx_to_dev;  /* Maps context handle -> device ordinal */

/* Handle maps for Runtime API opaque objects */
static handle_map_t g_rt_streams;  /* Runtime API cudaStream_t handles */
static handle_map_t g_rt_events;   /* Runtime API cudaEvent_t handles */

static uint64_t g_next_handle = 0x10000;

static uint64_t alloc_handle(void) {
    return g_next_handle++;
}

/* Find device ordinal for a given device handle (CUdevice value) */
static int device_ordinal_for_handle(int32_t dev_handle) {
    for (int i = 0; i < g_device_count; i++) {
        if ((int32_t)g_devices[i] == dev_handle)
            return i;
    }
    return 0; /* fallback to device 0 */
}

/* Set the CUDA context for a given device ordinal, lazily retaining primary ctx */
static CUresult ensure_device_context(int ordinal) {
    if (ordinal < 0 || ordinal >= g_device_count)
        return CUDA_ERROR_INVALID_DEVICE;

    if (g_contexts[ordinal] == NULL) {
        CUresult res = cuDevicePrimaryCtxRetain(&g_contexts[ordinal], g_devices[ordinal]);
        if (res != CUDA_SUCCESS)
            return res;
    }

    if (g_current_ctx != g_contexts[ordinal]) {
        CUresult res = cuCtxSetCurrent(g_contexts[ordinal]);
        if (res != CUDA_SUCCESS)
            return res;
        g_current_ctx = g_contexts[ordinal];
        g_current_dev = ordinal;
    }
    return CUDA_SUCCESS;
}

int cuda_dispatch_init(const char *pcibus) {
    CUresult res;

    hmap_init(&g_modules);
    hmap_init(&g_functions);
    hmap_init(&g_streams);
    hmap_init(&g_events);
    hmap_init(&g_ctx_to_dev);
    hmap_init(&g_rt_streams);
    hmap_init(&g_rt_events);

    memset(g_devices, 0, sizeof(g_devices));
    memset(g_contexts, 0, sizeof(g_contexts));

    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda_dispatch] cuInit failed: %d\n", res);
        return -1;
    }

    /* Detect all GPUs */
    res = cuDeviceGetCount(&g_device_count);
    if (res != CUDA_SUCCESS || g_device_count <= 0) {
        fprintf(stderr, "[cuda_dispatch] cuDeviceGetCount failed: %d (count=%d)\n", res, g_device_count);
        return -1;
    }
    if (g_device_count > JUICE_MAX_DEVICES)
        g_device_count = JUICE_MAX_DEVICES;

    fprintf(stderr, "[cuda_dispatch] Detected %d GPU(s)\n", g_device_count);

    for (int i = 0; i < g_device_count; i++) {
        res = cuDeviceGet(&g_devices[i], i);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[cuda_dispatch] cuDeviceGet(%d) failed: %d\n", i, res);
            return -1;
        }

        char name[256];
        cuDeviceGetName(name, sizeof(name), g_devices[i]);
        size_t mem_total;
        cuDeviceTotalMem(&mem_total, g_devices[i]);
        fprintf(stderr, "[cuda_dispatch]   GPU %d: %s (%.1f GB)\n",
                i, name, (double)mem_total / (1024.0 * 1024.0 * 1024.0));
    }

    /* Initialize primary context for device 0 (default) */
    int default_dev = 0;

    /* If a PCI bus ID was specified, find that device and use it as default */
    if (pcibus != NULL && pcibus[0] != '\0') {
        CUdevice pci_dev;
        res = cuDeviceGetByPCIBusId(&pci_dev, pcibus);
        if (res == CUDA_SUCCESS) {
            for (int i = 0; i < g_device_count; i++) {
                if (g_devices[i] == pci_dev) {
                    default_dev = i;
                    break;
                }
            }
        } else {
            fprintf(stderr, "[cuda_dispatch] cuDeviceGetByPCIBusId(%s) failed: %d, using device 0\n",
                    pcibus, res);
        }
    }

    res = cuDevicePrimaryCtxRetain(&g_contexts[default_dev], g_devices[default_dev]);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda_dispatch] cuDevicePrimaryCtxRetain failed: %d\n", res);
        return -1;
    }

    res = cuCtxSetCurrent(g_contexts[default_dev]);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda_dispatch] cuCtxSetCurrent failed: %d\n", res);
        return -1;
    }

    g_current_ctx = g_contexts[default_dev];
    g_current_dev = default_dev;
    g_initialized = 1;

    return 0;
}

void cuda_dispatch_shutdown(void) {
    if (g_initialized) {
        for (int i = 0; i < g_device_count; i++) {
            if (g_contexts[i] != NULL) {
                cuDevicePrimaryCtxRelease(g_devices[i]);
                g_contexts[i] = NULL;
            }
        }
        g_initialized = 0;
    }
}

int cuda_dispatch(int fd, juice_request_t *req, void *payload) {
    CUresult res;

    switch (req->func_id) {

    /* ================================================================
     * Initialization
     * ================================================================ */
    case JUICE_CU_INIT:
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);

    case JUICE_CU_DRIVER_GET_VERSION: {
        int version;
        res = cuDriverGetVersion(&version);
        return juice_send_response(fd, req->seq_num, res, &version, sizeof(version));
    }

    /* ================================================================
     * Device Management
     * ================================================================ */
    case JUICE_CU_DEVICE_GET_COUNT: {
        int32_t count = (int32_t)g_device_count;
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &count, sizeof(count));
    }

    case JUICE_CU_DEVICE_GET: {
        int32_t ordinal = 0;
        if (req->payload_size >= 4)
            memcpy(&ordinal, payload, 4);
        if (ordinal < 0 || ordinal >= g_device_count) {
            return juice_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_DEVICE, NULL, 0);
        }
        int32_t device = (int32_t)g_devices[ordinal];
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &device, sizeof(device));
    }

    case JUICE_CU_DEVICE_GET_NAME: {
        int32_t len_req = 256, dev_handle = 0;
        if (req->payload_size >= 8) {
            memcpy(&len_req, payload, 4);
            memcpy(&dev_handle, (char *)payload + 4, 4);
        }
        CUdevice dev = (CUdevice)dev_handle;
        char name[256];
        int name_len = len_req < (int)sizeof(name) ? len_req : (int)sizeof(name);
        res = cuDeviceGetName(name, name_len, dev);
        return juice_send_response(fd, req->seq_num, res, name, strlen(name) + 1);
    }

    case JUICE_CU_DEVICE_TOTAL_MEM: {
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        CUdevice dev = (CUdevice)dev_handle;
        size_t bytes;
        res = cuDeviceTotalMem(&bytes, dev);
        uint64_t bytes64 = (uint64_t)bytes;
        return juice_send_response(fd, req->seq_num, res, &bytes64, sizeof(bytes64));
    }

    case JUICE_CU_DEVICE_GET_ATTRIBUTE: {
        int32_t attrib = 0, dev_handle = 0;
        if (req->payload_size >= 8) {
            memcpy(&attrib, payload, 4);
            memcpy(&dev_handle, (char *)payload + 4, 4);
        }
        CUdevice dev = (CUdevice)dev_handle;
        int value;
        res = cuDeviceGetAttribute(&value, (CUdevice_attribute)attrib, dev);
        return juice_send_response(fd, req->seq_num, res, &value, sizeof(value));
    }

    case JUICE_CU_DEVICE_GET_UUID: {
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        CUdevice dev = (CUdevice)dev_handle;
        CUuuid uuid;
        res = cuDeviceGetUuid(&uuid, dev);
        return juice_send_response(fd, req->seq_num, res, &uuid, sizeof(uuid));
    }

    case JUICE_CU_DEVICE_GET_PROPERTIES: {
        /* Deprecated but still used by some apps */
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);
    }

    case JUICE_CU_DEVICE_COMPUTE_CAPABILITY: {
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        CUdevice dev = (CUdevice)dev_handle;
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        struct { int32_t major; int32_t minor; } cc = { major, minor };
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &cc, sizeof(cc));
    }

    /* ================================================================
     * Context Management
     * ================================================================ */
    case JUICE_CU_CTX_CREATE: {
        /* Payload: flags(4) + dev(4) */
        uint32_t flags = 0;
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 8) {
            memcpy(&flags, payload, 4);
            memcpy(&dev_handle, (char *)payload + 4, 4);
        }
        int ordinal = device_ordinal_for_handle(dev_handle);
        res = ensure_device_context(ordinal);
        if (res != CUDA_SUCCESS)
            return juice_send_response(fd, req->seq_num, res, NULL, 0);
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_contexts[ordinal];
        /* Track this context -> device mapping */
        hmap_put(&g_ctx_to_dev, ctx_handle, (uint64_t)ordinal);
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case JUICE_CU_DEVICE_PRIMARY_CTX_RETAIN: {
        /* Payload: dev(4) */
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        int ordinal = device_ordinal_for_handle(dev_handle);
        res = ensure_device_context(ordinal);
        if (res != CUDA_SUCCESS)
            return juice_send_response(fd, req->seq_num, res, NULL, 0);
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_contexts[ordinal];
        hmap_put(&g_ctx_to_dev, ctx_handle, (uint64_t)ordinal);
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case JUICE_CU_CTX_DESTROY:
    case JUICE_CU_DEVICE_PRIMARY_CTX_RELEASE:
    case JUICE_CU_DEVICE_PRIMARY_CTX_RESET:
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);

    case JUICE_CU_CTX_SET_CURRENT: {
        uint64_t ctx_handle = 0;
        if (req->payload_size >= 8)
            memcpy(&ctx_handle, payload, 8);
        CUcontext ctx = (CUcontext)(uintptr_t)ctx_handle;
        res = cuCtxSetCurrent(ctx);
        if (res == CUDA_SUCCESS) {
            g_current_ctx = ctx;
            /* Update current device from our mapping */
            uint64_t dev_ord = hmap_get(&g_ctx_to_dev, ctx_handle);
            g_current_dev = (int)dev_ord;
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_CTX_GET_CURRENT: {
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_current_ctx;
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case JUICE_CU_CTX_PUSH_CURRENT: {
        uint64_t ctx_handle = 0;
        if (req->payload_size >= 8)
            memcpy(&ctx_handle, payload, 8);
        CUcontext ctx = (CUcontext)(uintptr_t)ctx_handle;
        res = cuCtxPushCurrent(ctx);
        if (res == CUDA_SUCCESS) {
            g_current_ctx = ctx;
            uint64_t dev_ord = hmap_get(&g_ctx_to_dev, ctx_handle);
            g_current_dev = (int)dev_ord;
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_CTX_POP_CURRENT: {
        CUcontext ctx;
        res = cuCtxPopCurrent(&ctx);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_CTX_SYNCHRONIZE:
        res = cuCtxSynchronize();
        return juice_send_response(fd, req->seq_num, res, NULL, 0);

    case JUICE_CU_CTX_GET_DEVICE: {
        int32_t dev = (int32_t)g_devices[g_current_dev];
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &dev, sizeof(dev));
    }

    /* ================================================================
     * Module Management
     * ================================================================ */
    case JUICE_CU_MODULE_LOAD_DATA:
    case JUICE_CU_MODULE_LOAD_DATA_EX: {
        /* Payload is raw PTX or CUBIN data */
        CUmodule mod;
        res = cuModuleLoadData(&mod, payload);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_modules, handle, (uint64_t)(uintptr_t)mod);
            return juice_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MODULE_LOAD: {
        /* Payload is a filename string — load from server filesystem */
        CUmodule mod;
        res = cuModuleLoad(&mod, (const char *)payload);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_modules, handle, (uint64_t)(uintptr_t)mod);
            return juice_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MODULE_UNLOAD: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUmodule mod = (CUmodule)(uintptr_t)hmap_get(&g_modules, handle);
        if (mod) {
            res = cuModuleUnload(mod);
            hmap_remove(&g_modules, handle);
        } else {
            res = CUDA_ERROR_INVALID_HANDLE;
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MODULE_GET_FUNCTION: {
        /* Payload: module_handle(8) + function_name (null-terminated) */
        uint64_t mod_handle = 0;
        const char *func_name = "";
        if (req->payload_size >= 9) {
            memcpy(&mod_handle, payload, 8);
            func_name = (const char *)((char *)payload + 8);
        }
        CUmodule mod = (CUmodule)(uintptr_t)hmap_get(&g_modules, mod_handle);
        if (!mod) {
            return juice_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }
        CUfunction func;
        res = cuModuleGetFunction(&func, mod, func_name);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_functions, handle, (uint64_t)(uintptr_t)func);
            return juice_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MODULE_GET_GLOBAL: {
        /* Payload: module_handle(8) + name (null-terminated) */
        uint64_t mod_handle = 0;
        const char *name = "";
        if (req->payload_size >= 9) {
            memcpy(&mod_handle, payload, 8);
            name = (const char *)((char *)payload + 8);
        }
        CUmodule mod = (CUmodule)(uintptr_t)hmap_get(&g_modules, mod_handle);
        if (!mod) {
            return juice_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }
        CUdeviceptr dptr;
        size_t size;
        res = cuModuleGetGlobal(&dptr, &size, mod, name);
        struct { uint64_t dptr; uint64_t size; } result;
        result.dptr = (uint64_t)dptr;
        result.size = (uint64_t)size;
        return juice_send_response(fd, req->seq_num, res, &result, sizeof(result));
    }

    /* ================================================================
     * Memory Management
     * ================================================================ */
    case JUICE_CU_MEM_ALLOC: {
        uint64_t size = 0;
        if (req->payload_size >= 8)
            memcpy(&size, payload, 8);
        CUdeviceptr dptr;
        res = cuMemAlloc(&dptr, (size_t)size);
        uint64_t dptr64 = (uint64_t)dptr;
        return juice_send_response(fd, req->seq_num, res, &dptr64, sizeof(dptr64));
    }

    case JUICE_CU_MEM_FREE: {
        uint64_t ptr = 0;
        if (req->payload_size >= 8)
            memcpy(&ptr, payload, 8);
        res = cuMemFree((CUdeviceptr)ptr);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MEMCPY_H_TO_D:
    case JUICE_CU_MEMCPY_H_TO_D_ASYNC: {
        /* Payload: dst(8) + size(8) + data(size) */
        uint64_t dst = 0, size = 0;
        if (req->payload_size >= 16) {
            memcpy(&dst, payload, 8);
            memcpy(&size, (char *)payload + 8, 8);
        }
        const void *data = (char *)payload + 16;
        res = cuMemcpyHtoD((CUdeviceptr)dst, data, (size_t)size);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MEMCPY_D_TO_H:
    case JUICE_CU_MEMCPY_D_TO_H_ASYNC: {
        /* Payload: src(8) + size(8) */
        uint64_t src = 0, size = 0;
        if (req->payload_size >= 16) {
            memcpy(&src, payload, 8);
            memcpy(&size, (char *)payload + 8, 8);
        }
        void *buf = malloc((size_t)size);
        if (!buf) {
            return juice_send_response(fd, req->seq_num, CUDA_ERROR_OUT_OF_MEMORY, NULL, 0);
        }
        res = cuMemcpyDtoH(buf, (CUdeviceptr)src, (size_t)size);
        int rc = juice_send_response(fd, req->seq_num, res, buf, (size_t)size);
        free(buf);
        return rc;
    }

    case JUICE_CU_MEMCPY_D_TO_D:
    case JUICE_CU_MEMCPY_D_TO_D_ASYNC: {
        /* Payload: dst(8) + src(8) + size(8) — zero data over wire! */
        uint64_t dst = 0, src = 0, size = 0;
        if (req->payload_size >= 24) {
            memcpy(&dst, payload, 8);
            memcpy(&src, (char *)payload + 8, 8);
            memcpy(&size, (char *)payload + 16, 8);
        }
        res = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MEMCPY: {
        /* Generic cuMemcpy: dst(8) + src(8) + size(8) */
        uint64_t dst = 0, src = 0, size = 0;
        if (req->payload_size >= 24) {
            memcpy(&dst, payload, 8);
            memcpy(&src, (char *)payload + 8, 8);
            memcpy(&size, (char *)payload + 16, 8);
        }
        res = cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MEMSET_D8: {
        uint64_t dst = 0; uint8_t val = 0; uint64_t count = 0;
        if (req->payload_size >= 17) {
            memcpy(&dst, payload, 8);
            memcpy(&val, (char *)payload + 8, 1);
            memcpy(&count, (char *)payload + 9, 8);
        }
        res = cuMemsetD8((CUdeviceptr)dst, val, (size_t)count);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MEMSET_D16: {
        uint64_t dst = 0; uint16_t val = 0; uint64_t count = 0;
        if (req->payload_size >= 18) {
            memcpy(&dst, payload, 8);
            memcpy(&val, (char *)payload + 8, 2);
            memcpy(&count, (char *)payload + 10, 8);
        }
        res = cuMemsetD16((CUdeviceptr)dst, val, (size_t)count);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MEMSET_D32: {
        uint64_t dst = 0; uint32_t val = 0; uint64_t count = 0;
        if (req->payload_size >= 20) {
            memcpy(&dst, payload, 8);
            memcpy(&val, (char *)payload + 8, 4);
            memcpy(&count, (char *)payload + 12, 8);
        }
        res = cuMemsetD32((CUdeviceptr)dst, val, (size_t)count);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_MEM_GET_INFO: {
        size_t free_mem, total_mem;
        res = cuMemGetInfo(&free_mem, &total_mem);
        struct { uint64_t free; uint64_t total; } info;
        info.free = (uint64_t)free_mem;
        info.total = (uint64_t)total_mem;
        return juice_send_response(fd, req->seq_num, res, &info, sizeof(info));
    }

    case JUICE_CU_MEM_HOST_ALLOC:
    case JUICE_CU_MEM_FREE_HOST:
    case JUICE_CU_MEM_HOST_GET_DEVICE_PTR:
    case JUICE_CU_POINTER_GET_ATTRIBUTE:
    case JUICE_CU_MEM_ALLOC_MANAGED:
        /* These require more complex handling — stub for now */
        return juice_send_response(fd, req->seq_num, CUDA_ERROR_NOT_SUPPORTED, NULL, 0);

    /* ================================================================
     * Execution Control
     * ================================================================ */
    case JUICE_CU_LAUNCH_KERNEL: {
        /*
         * Payload layout:
         *   func_handle(8) + gridX(4) + gridY(4) + gridZ(4)
         *   + blockX(4) + blockY(4) + blockZ(4)
         *   + sharedMem(4) + stream_handle(8)
         *   + has_params(1):
         *       0 = no kernel parameters
         *       1 = contiguous parameter buffer follows
         *   + [if has_params == 1]: buffer_size(4) + buffer_data(buffer_size)
         */
        if (req->payload_size < 44) {
            return juice_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_VALUE, NULL, 0);
        }

        uint64_t func_handle;
        uint32_t gridX, gridY, gridZ, blockX, blockY, blockZ, shared;
        uint64_t stream_handle;

        char *p = (char *)payload;
        memcpy(&func_handle, p, 8); p += 8;
        memcpy(&gridX, p, 4); p += 4;
        memcpy(&gridY, p, 4); p += 4;
        memcpy(&gridZ, p, 4); p += 4;
        memcpy(&blockX, p, 4); p += 4;
        memcpy(&blockY, p, 4); p += 4;
        memcpy(&blockZ, p, 4); p += 4;
        memcpy(&shared, p, 4); p += 4;
        memcpy(&stream_handle, p, 8); p += 8;

        CUfunction func = (CUfunction)(uintptr_t)hmap_get(&g_functions, func_handle);
        if (!func) {
            return juice_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }

        CUstream stream = NULL;
        if (stream_handle != 0) {
            stream = (CUstream)(uintptr_t)hmap_get(&g_streams, stream_handle);
        }

        /* Parse kernel parameters if present */
        size_t remaining = req->payload_size - 44;

        if (remaining >= 1) {
            uint8_t has_params = 0;
            memcpy(&has_params, p, 1); p += 1;
            remaining -= 1;

            if (has_params && remaining >= 4) {
                uint32_t buf_size = 0;
                memcpy(&buf_size, p, 4); p += 4;
                remaining -= 4;

                if (buf_size > 0 && remaining >= buf_size) {
                    void *param_buffer = p;
                    size_t param_size = (size_t)buf_size;

                    void *extra[] = {
                        CU_LAUNCH_PARAM_BUFFER_POINTER, param_buffer,
                        CU_LAUNCH_PARAM_BUFFER_SIZE, &param_size,
                        CU_LAUNCH_PARAM_END
                    };

                    res = cuLaunchKernel(func,
                                         gridX, gridY, gridZ,
                                         blockX, blockY, blockZ,
                                         shared, stream,
                                         NULL, extra);
                    return juice_send_response(fd, req->seq_num, res, NULL, 0);
                }
            }
        }

        /* No parameters — launch with NULL */
        res = cuLaunchKernel(func,
                             gridX, gridY, gridZ,
                             blockX, blockY, blockZ,
                             shared, stream,
                             NULL, NULL);

        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_FUNC_GET_ATTRIBUTE: {
        int32_t attrib = 0;
        uint64_t func_handle = 0;
        if (req->payload_size >= 12) {
            memcpy(&attrib, payload, 4);
            memcpy(&func_handle, (char *)payload + 4, 8);
        }
        CUfunction func = (CUfunction)(uintptr_t)hmap_get(&g_functions, func_handle);
        int value = 0;
        if (func) {
            res = cuFuncGetAttribute(&value, (CUfunction_attribute)attrib, func);
        } else {
            res = CUDA_ERROR_INVALID_HANDLE;
        }
        return juice_send_response(fd, req->seq_num, res, &value, sizeof(value));
    }

    /* ================================================================
     * Stream Management
     * ================================================================ */
    case JUICE_CU_STREAM_CREATE: {
        CUstream stream;
        uint32_t flags = 0;
        if (req->payload_size >= 4)
            memcpy(&flags, payload, 4);
        res = cuStreamCreate(&stream, flags);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_streams, handle, (uint64_t)(uintptr_t)stream);
            return juice_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_STREAM_CREATE_WITH_PRIORITY: {
        CUstream stream;
        uint32_t flags = 0;
        int32_t priority = 0;
        if (req->payload_size >= 8) {
            memcpy(&flags, payload, 4);
            memcpy(&priority, (char *)payload + 4, 4);
        }
        res = cuStreamCreateWithPriority(&stream, flags, priority);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_streams, handle, (uint64_t)(uintptr_t)stream);
            return juice_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_STREAM_DESTROY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUstream stream = (CUstream)(uintptr_t)hmap_get(&g_streams, handle);
        if (stream) {
            res = cuStreamDestroy(stream);
            hmap_remove(&g_streams, handle);
        } else {
            res = CUDA_SUCCESS; /* Already destroyed or default stream */
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_STREAM_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        if (handle == 0) {
            res = cuCtxSynchronize(); /* Default stream */
        } else {
            CUstream stream = (CUstream)(uintptr_t)hmap_get(&g_streams, handle);
            res = stream ? cuStreamSynchronize(stream) : CUDA_SUCCESS;
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_STREAM_QUERY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUstream stream = (CUstream)(uintptr_t)hmap_get(&g_streams, handle);
        res = stream ? cuStreamQuery(stream) : CUDA_SUCCESS;
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_STREAM_WAIT_EVENT: {
        uint64_t stream_handle = 0, event_handle = 0;
        uint32_t flags = 0;
        if (req->payload_size >= 20) {
            memcpy(&stream_handle, payload, 8);
            memcpy(&event_handle, (char *)payload + 8, 8);
            memcpy(&flags, (char *)payload + 16, 4);
        }
        CUstream stream = stream_handle ? (CUstream)(uintptr_t)hmap_get(&g_streams, stream_handle) : NULL;
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, event_handle);
        res = event ? cuStreamWaitEvent(stream, event, flags) : CUDA_ERROR_INVALID_HANDLE;
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    /* ================================================================
     * Event Management
     * ================================================================ */
    case JUICE_CU_EVENT_CREATE: {
        CUevent event;
        uint32_t flags = 0;
        if (req->payload_size >= 4)
            memcpy(&flags, payload, 4);
        res = cuEventCreate(&event, flags);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_events, handle, (uint64_t)(uintptr_t)event);
            return juice_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_EVENT_DESTROY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, handle);
        if (event) {
            res = cuEventDestroy(event);
            hmap_remove(&g_events, handle);
        } else {
            res = CUDA_SUCCESS;
        }
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_EVENT_RECORD: {
        uint64_t event_handle = 0, stream_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&event_handle, payload, 8);
            memcpy(&stream_handle, (char *)payload + 8, 8);
        }
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, event_handle);
        CUstream stream = stream_handle ? (CUstream)(uintptr_t)hmap_get(&g_streams, stream_handle) : NULL;
        res = event ? cuEventRecord(event, stream) : CUDA_ERROR_INVALID_HANDLE;
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_EVENT_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, handle);
        res = event ? cuEventSynchronize(event) : CUDA_ERROR_INVALID_HANDLE;
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_EVENT_QUERY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, handle);
        res = event ? cuEventQuery(event) : CUDA_ERROR_INVALID_HANDLE;
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_EVENT_ELAPSED_TIME: {
        uint64_t start_handle = 0, end_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&start_handle, payload, 8);
            memcpy(&end_handle, (char *)payload + 8, 8);
        }
        CUevent start = (CUevent)(uintptr_t)hmap_get(&g_events, start_handle);
        CUevent end = (CUevent)(uintptr_t)hmap_get(&g_events, end_handle);
        float ms = 0.0f;
        if (start && end) {
            res = cuEventElapsedTime(&ms, start, end);
        } else {
            res = CUDA_ERROR_INVALID_HANDLE;
        }
        return juice_send_response(fd, req->seq_num, res, &ms, sizeof(ms));
    }

    /* ================================================================
     * CUDA Runtime API (0x0100 - 0x01FF)
     *
     * These dispatch cudaXxx() Runtime API functions used by PyTorch,
     * TensorFlow, and other high-level frameworks.
     * ================================================================ */

    case JUICE_CUDA_GET_DEVICE_COUNT: {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        int32_t count32 = (int32_t)count;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &count32, sizeof(count32));
    }

    case JUICE_CUDA_GET_DEVICE: {
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        int32_t dev32 = (int32_t)device;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &dev32, sizeof(dev32));
    }

    case JUICE_CUDA_SET_DEVICE: {
        int32_t device = 0;
        if (req->payload_size >= 4)
            memcpy(&device, payload, 4);
        cudaError_t err = cudaSetDevice(device);
        /* Update our tracked current device */
        if (err == cudaSuccess && device >= 0 && device < g_device_count) {
            ensure_device_context(device);
        }
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_GET_DEVICE_PROPERTIES: {
        /*
         * cudaDeviceProp is a large struct (~700+ bytes).
         * Serialize the full struct in the response payload.
         * Payload: device ordinal (int32)
         */
        int32_t device = 0;
        if (req->payload_size >= 4)
            memcpy(&device, payload, 4);
        struct cudaDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        return juice_send_response(fd, req->seq_num, (int32_t)err, &prop, sizeof(prop));
    }

    case JUICE_CUDA_DEVICE_GET_ATTRIBUTE: {
        /* Payload: attr(4) + device(4) */
        int32_t attr = 0, device = 0;
        if (req->payload_size >= 8) {
            memcpy(&attr, payload, 4);
            memcpy(&device, (char *)payload + 4, 4);
        }
        int value = 0;
        cudaError_t err = cudaDeviceGetAttribute(&value, (enum cudaDeviceAttr)attr, device);
        int32_t val32 = (int32_t)value;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &val32, sizeof(val32));
    }

    case JUICE_CUDA_MALLOC: {
        /* Payload: size(8) */
        uint64_t size = 0;
        if (req->payload_size >= 8)
            memcpy(&size, payload, 8);
        void *devPtr = NULL;
        cudaError_t err = cudaMalloc(&devPtr, (size_t)size);
        uint64_t ptr64 = (uint64_t)(uintptr_t)devPtr;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &ptr64, sizeof(ptr64));
    }

    case JUICE_CUDA_FREE: {
        /* Payload: ptr(8) */
        uint64_t ptr = 0;
        if (req->payload_size >= 8)
            memcpy(&ptr, payload, 8);
        cudaError_t err = cudaFree((void *)(uintptr_t)ptr);
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_MEMCPY: {
        /*
         * Direction-dependent layout from client:
         *   HtoD: dst_ptr(8) + size(8) + kind(4) + data(size)
         *   DtoH: dst_ptr(8) + src_ptr(8) + size(8) + kind(4)
         *   DtoD: dst_ptr(8) + src_ptr(8) + size(8) + kind(4)
         *
         * Detect direction by reading kind at offset 16.
         * If kind == 1 (HtoD), data follows at offset 20.
         * Otherwise it's the DtoH/DtoD layout with kind at offset 24.
         */
        if (req->payload_size < 20) {
            return juice_send_response(fd, req->seq_num, (int32_t)cudaErrorInvalidValue, NULL, 0);
        }

        /* Peek at kind at offset 16 to determine layout */
        int32_t kind_peek = 0;
        memcpy(&kind_peek, (char *)payload + 16, 4);

        if (kind_peek == cudaMemcpyHostToDevice) {
            /* HtoD: dst_ptr(8) + size(8) + kind(4) + data(size) */
            uint64_t dst_ptr = 0, size = 0;
            memcpy(&dst_ptr, payload, 8);
            memcpy(&size, (char *)payload + 8, 8);
            const void *data = (char *)payload + 20;

            cudaError_t err = cudaMemcpy((void *)(uintptr_t)dst_ptr, data,
                                          (size_t)size, cudaMemcpyHostToDevice);
            return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
        } else {
            /* DtoH or DtoD: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) */
            uint64_t dst_ptr = 0, src_ptr = 0, size = 0;
            int32_t kind = 0;
            memcpy(&dst_ptr, payload, 8);
            memcpy(&src_ptr, (char *)payload + 8, 8);
            memcpy(&size, (char *)payload + 16, 8);
            memcpy(&kind, (char *)payload + 24, 4);

            if (kind == cudaMemcpyDeviceToHost) {
                /* Read from device memory, send data back to client */
                void *buf = malloc((size_t)size);
                if (!buf) {
                    return juice_send_response(fd, req->seq_num,
                                                (int32_t)cudaErrorMemoryAllocation, NULL, 0);
                }
                cudaError_t err = cudaMemcpy(buf, (void *)(uintptr_t)src_ptr,
                                              (size_t)size, cudaMemcpyDeviceToHost);
                int rc = juice_send_response(fd, req->seq_num, (int32_t)err, buf, (size_t)size);
                free(buf);
                return rc;
            } else if (kind == cudaMemcpyDeviceToDevice) {
                /* No data transfer over wire */
                cudaError_t err = cudaMemcpy((void *)(uintptr_t)dst_ptr,
                                              (void *)(uintptr_t)src_ptr,
                                              (size_t)size, cudaMemcpyDeviceToDevice);
                return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
            } else {
                return juice_send_response(fd, req->seq_num,
                                            (int32_t)cudaErrorInvalidMemcpyDirection, NULL, 0);
            }
        }
    }

    case JUICE_CUDA_MEMCPY_ASYNC: {
        /*
         * Same layout as JUICE_CUDA_MEMCPY but with stream handle appended.
         * HtoD: dst_ptr(8) + size(8) + kind(4) + stream(8) + data(size)
         * DtoH: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) + stream(8)
         * DtoD: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) + stream(8)
         */
        if (req->payload_size < 28) {
            return juice_send_response(fd, req->seq_num, (int32_t)cudaErrorInvalidValue, NULL, 0);
        }

        int32_t kind_peek = 0;
        memcpy(&kind_peek, (char *)payload + 16, 4);

        if (kind_peek == cudaMemcpyHostToDevice) {
            /* HtoD: dst_ptr(8) + size(8) + kind(4) + stream(8) + data(size) */
            uint64_t dst_ptr = 0, size = 0, stream_handle = 0;
            memcpy(&dst_ptr, payload, 8);
            memcpy(&size, (char *)payload + 8, 8);
            memcpy(&stream_handle, (char *)payload + 20, 8);
            const void *data = (char *)payload + 28;

            cudaStream_t stream = NULL;
            if (stream_handle != 0)
                stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);

            cudaError_t err = cudaMemcpyAsync((void *)(uintptr_t)dst_ptr, data,
                                               (size_t)size, cudaMemcpyHostToDevice, stream);
            return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
        } else {
            /* DtoH or DtoD: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) + stream(8) */
            uint64_t dst_ptr = 0, src_ptr = 0, size = 0, stream_handle = 0;
            int32_t kind = 0;
            memcpy(&dst_ptr, payload, 8);
            memcpy(&src_ptr, (char *)payload + 8, 8);
            memcpy(&size, (char *)payload + 16, 8);
            memcpy(&kind, (char *)payload + 24, 4);
            memcpy(&stream_handle, (char *)payload + 28, 8);

            cudaStream_t stream = NULL;
            if (stream_handle != 0)
                stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);

            if (kind == cudaMemcpyDeviceToHost) {
                void *buf = malloc((size_t)size);
                if (!buf) {
                    return juice_send_response(fd, req->seq_num,
                                                (int32_t)cudaErrorMemoryAllocation, NULL, 0);
                }
                cudaError_t err = cudaMemcpyAsync(buf, (void *)(uintptr_t)src_ptr,
                                                   (size_t)size, cudaMemcpyDeviceToHost, stream);
                /* Must synchronize to ensure data is ready before sending */
                if (err == cudaSuccess)
                    cudaStreamSynchronize(stream);
                int rc = juice_send_response(fd, req->seq_num, (int32_t)err, buf, (size_t)size);
                free(buf);
                return rc;
            } else if (kind == cudaMemcpyDeviceToDevice) {
                cudaError_t err = cudaMemcpyAsync((void *)(uintptr_t)dst_ptr,
                                                   (void *)(uintptr_t)src_ptr,
                                                   (size_t)size, cudaMemcpyDeviceToDevice, stream);
                return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
            } else {
                return juice_send_response(fd, req->seq_num,
                                            (int32_t)cudaErrorInvalidMemcpyDirection, NULL, 0);
            }
        }
    }

    case JUICE_CUDA_MEMSET: {
        /* Payload: ptr(8) + value(4) + count(8) */
        uint64_t ptr = 0, count = 0;
        int32_t value = 0;
        if (req->payload_size >= 20) {
            memcpy(&ptr, payload, 8);
            memcpy(&value, (char *)payload + 8, 4);
            memcpy(&count, (char *)payload + 12, 8);
        }
        cudaError_t err = cudaMemset((void *)(uintptr_t)ptr, value, (size_t)count);
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_DEVICE_SYNCHRONIZE: {
        cudaError_t err = cudaDeviceSynchronize();
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_STREAM_CREATE: {
        cudaStream_t stream = NULL;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_rt_streams, handle, (uint64_t)(uintptr_t)stream);
            return juice_send_response(fd, req->seq_num, (int32_t)err, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_STREAM_DESTROY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaStream_t stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, handle);
        cudaError_t err = cudaSuccess;
        if (stream) {
            err = cudaStreamDestroy(stream);
            hmap_remove(&g_rt_streams, handle);
        }
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_STREAM_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaError_t err;
        if (handle == 0) {
            err = cudaDeviceSynchronize(); /* Default stream */
        } else {
            cudaStream_t stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, handle);
            err = stream ? cudaStreamSynchronize(stream) : cudaSuccess;
        }
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_EVENT_CREATE: {
        cudaEvent_t event = NULL;
        cudaError_t err = cudaEventCreate(&event);
        if (err == cudaSuccess) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_rt_events, handle, (uint64_t)(uintptr_t)event);
            return juice_send_response(fd, req->seq_num, (int32_t)err, &handle, sizeof(handle));
        }
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_EVENT_RECORD: {
        /* Payload: event_handle(8) + stream_handle(8) */
        uint64_t event_handle = 0, stream_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&event_handle, payload, 8);
            memcpy(&stream_handle, (char *)payload + 8, 8);
        }
        cudaEvent_t event = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, event_handle);
        cudaStream_t stream = NULL;
        if (stream_handle != 0)
            stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);
        cudaError_t err = event ? cudaEventRecord(event, stream) : cudaErrorInvalidResourceHandle;
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_EVENT_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaEvent_t event = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, handle);
        cudaError_t err = event ? cudaEventSynchronize(event) : cudaErrorInvalidResourceHandle;
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_EVENT_DESTROY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaEvent_t event = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, handle);
        cudaError_t err = cudaSuccess;
        if (event) {
            err = cudaEventDestroy(event);
            hmap_remove(&g_rt_events, handle);
        }
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_EVENT_ELAPSED_TIME: {
        uint64_t start_handle = 0, end_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&start_handle, payload, 8);
            memcpy(&end_handle, (char *)payload + 8, 8);
        }
        cudaEvent_t start_ev = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, start_handle);
        cudaEvent_t end_ev = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, end_handle);
        float ms = 0.0f;
        cudaError_t err;
        if (start_ev && end_ev) {
            err = cudaEventElapsedTime(&ms, start_ev, end_ev);
        } else {
            err = cudaErrorInvalidResourceHandle;
        }
        return juice_send_response(fd, req->seq_num, (int32_t)err, &ms, sizeof(ms));
    }

    case JUICE_CUDA_RUNTIME_GET_VERSION: {
        int version = 0;
        cudaError_t err = cudaRuntimeGetVersion(&version);
        int32_t ver32 = (int32_t)version;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &ver32, sizeof(ver32));
    }

    case JUICE_CUDA_DRIVER_GET_VERSION: {
        int version = 0;
        cudaError_t err = cudaDriverGetVersion(&version);
        int32_t ver32 = (int32_t)version;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &ver32, sizeof(ver32));
    }

    case JUICE_CUDA_MEM_GET_INFO: {
        size_t free_mem = 0, total_mem = 0;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        struct { uint64_t free; uint64_t total; } info;
        info.free = (uint64_t)free_mem;
        info.total = (uint64_t)total_mem;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &info, sizeof(info));
    }

    case JUICE_CUDA_DEVICE_RESET: {
        cudaError_t err = cudaDeviceReset();
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_DEVICE_CAN_ACCESS_PEER: {
        /* Payload: device(4) + peerDevice(4) */
        int32_t device = 0, peerDevice = 0;
        if (req->payload_size >= 8) {
            memcpy(&device, payload, 4);
            memcpy(&peerDevice, (char *)payload + 4, 4);
        }
        int canAccess = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(&canAccess, device, peerDevice);
        int32_t ca32 = (int32_t)canAccess;
        return juice_send_response(fd, req->seq_num, (int32_t)err, &ca32, sizeof(ca32));
    }

    case JUICE_CUDA_FUNC_GET_ATTRIBUTES: {
        /* Stub: return zeroed attributes with success */
        char attr_buf[64];
        memset(attr_buf, 0, sizeof(attr_buf));
        return juice_send_response(fd, req->seq_num, (int32_t)cudaSuccess, attr_buf, sizeof(attr_buf));
    }

    case JUICE_CUDA_GET_LAST_ERROR: {
        cudaError_t err = cudaGetLastError();
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case JUICE_CUDA_PEEK_AT_LAST_ERROR: {
        cudaError_t err = cudaPeekAtLastError();
        return juice_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    /* ================================================================
     * Default: unsupported
     * ================================================================ */
    default:
        fprintf(stderr, "[cuda_dispatch] Unknown func_id=0x%04x\n", req->func_id);
        return juice_send_response(fd, req->seq_num, CUDA_ERROR_NOT_SUPPORTED, NULL, 0);
    }
}

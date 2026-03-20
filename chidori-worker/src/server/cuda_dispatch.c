/*
 * Chidori GPU Worker — Real CUDA Dispatch
 *
 * Handles deserialization of wire protocol messages, calls real CUDA
 * Driver API and Runtime API functions, and serializes responses back.
 *
 * This file is included by gpu_worker.c when building with CUDA support.
 * Compile with: -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "chidori_protocol.h"
#include "handle_map.h"

/* Maximum number of GPUs we support */
#define CHIDORI_MAX_DEVICES 16

/* Server-side state */
static CUdevice   g_devices[CHIDORI_MAX_DEVICES];
static CUcontext  g_contexts[CHIDORI_MAX_DEVICES]; /* Primary context per device */
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

/* Handle maps for cuBLAS */
static handle_map_t g_cublas_handles;  /* cublasHandle_t handles */

/* Handle maps for cuDNN */
static handle_map_t g_cudnn_handles;              /* cudnnHandle_t */
static handle_map_t g_cudnn_tensor_descs;          /* cudnnTensorDescriptor_t */
static handle_map_t g_cudnn_filter_descs;          /* cudnnFilterDescriptor_t */
static handle_map_t g_cudnn_conv_descs;            /* cudnnConvolutionDescriptor_t */
static handle_map_t g_cudnn_activation_descs;      /* cudnnActivationDescriptor_t */
static handle_map_t g_cudnn_pooling_descs;         /* cudnnPoolingDescriptor_t */
static handle_map_t g_cudnn_dropout_descs;         /* cudnnDropoutDescriptor_t */
static handle_map_t g_cudnn_rnn_descs;             /* cudnnRNNDescriptor_t */

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
    hmap_init(&g_cublas_handles);
    hmap_init(&g_cudnn_handles);
    hmap_init(&g_cudnn_tensor_descs);
    hmap_init(&g_cudnn_filter_descs);
    hmap_init(&g_cudnn_conv_descs);
    hmap_init(&g_cudnn_activation_descs);
    hmap_init(&g_cudnn_pooling_descs);
    hmap_init(&g_cudnn_dropout_descs);
    hmap_init(&g_cudnn_rnn_descs);

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
    if (g_device_count > CHIDORI_MAX_DEVICES)
        g_device_count = CHIDORI_MAX_DEVICES;

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

int cuda_dispatch(int fd, chidori_request_t *req, void *payload) {
    static int s_verbose = -1;
    if (s_verbose == -1) {
        const char *v = getenv("CHIDORI_SERVER_VERBOSE");
        s_verbose = (v && v[0] == '1') ? 1 : 0;
    }
    if (s_verbose) {
        fprintf(stderr, "[cuda_dispatch] >> func=0x%04x payload=%u seq=%u\n",
                req->func_id, req->payload_size, req->seq_num);
    }

    CUresult res;

    switch (req->func_id) {

    /* ================================================================
     * Initialization
     * ================================================================ */
    case CHIDORI_CU_INIT:
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);

    case CHIDORI_CU_DRIVER_GET_VERSION: {
        int version;
        res = cuDriverGetVersion(&version);
        return chidori_send_response(fd, req->seq_num, res, &version, sizeof(version));
    }

    /* ================================================================
     * Device Management
     * ================================================================ */
    case CHIDORI_CU_DEVICE_GET_COUNT: {
        int32_t count = (int32_t)g_device_count;
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, &count, sizeof(count));
    }

    case CHIDORI_CU_DEVICE_GET: {
        int32_t ordinal = 0;
        if (req->payload_size >= 4)
            memcpy(&ordinal, payload, 4);
        if (ordinal < 0 || ordinal >= g_device_count) {
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_DEVICE, NULL, 0);
        }
        int32_t device = (int32_t)g_devices[ordinal];
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, &device, sizeof(device));
    }

    case CHIDORI_CU_DEVICE_GET_NAME: {
        int32_t len_req = 256, dev_handle = 0;
        if (req->payload_size >= 8) {
            memcpy(&len_req, payload, 4);
            memcpy(&dev_handle, (char *)payload + 4, 4);
        }
        CUdevice dev = (CUdevice)dev_handle;
        char name[256];
        int name_len = len_req < (int)sizeof(name) ? len_req : (int)sizeof(name);
        res = cuDeviceGetName(name, name_len, dev);
        return chidori_send_response(fd, req->seq_num, res, name, strlen(name) + 1);
    }

    case CHIDORI_CU_DEVICE_TOTAL_MEM: {
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        CUdevice dev = (CUdevice)dev_handle;
        size_t bytes;
        res = cuDeviceTotalMem(&bytes, dev);
        uint64_t bytes64 = (uint64_t)bytes;
        return chidori_send_response(fd, req->seq_num, res, &bytes64, sizeof(bytes64));
    }

    case CHIDORI_CU_DEVICE_GET_ATTRIBUTE: {
        int32_t attrib = 0, dev_handle = 0;
        if (req->payload_size >= 8) {
            memcpy(&attrib, payload, 4);
            memcpy(&dev_handle, (char *)payload + 4, 4);
        }
        CUdevice dev = (CUdevice)dev_handle;
        int value;
        res = cuDeviceGetAttribute(&value, (CUdevice_attribute)attrib, dev);
        return chidori_send_response(fd, req->seq_num, res, &value, sizeof(value));
    }

    case CHIDORI_CU_DEVICE_GET_UUID: {
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        CUdevice dev = (CUdevice)dev_handle;
        CUuuid uuid;
        res = cuDeviceGetUuid(&uuid, dev);
        return chidori_send_response(fd, req->seq_num, res, &uuid, sizeof(uuid));
    }

    case CHIDORI_CU_DEVICE_GET_PROPERTIES: {
        /* Deprecated but still used by some apps */
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);
    }

    case CHIDORI_CU_DEVICE_COMPUTE_CAPABILITY: {
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        CUdevice dev = (CUdevice)dev_handle;
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        struct { int32_t major; int32_t minor; } cc = { major, minor };
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, &cc, sizeof(cc));
    }

    /* ================================================================
     * Context Management
     * ================================================================ */
    case CHIDORI_CU_CTX_CREATE: {
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
            return chidori_send_response(fd, req->seq_num, res, NULL, 0);
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_contexts[ordinal];
        /* Track this context -> device mapping */
        hmap_put(&g_ctx_to_dev, ctx_handle, (uint64_t)ordinal);
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case CHIDORI_CU_DEVICE_PRIMARY_CTX_RETAIN: {
        /* Payload: dev(4) */
        int32_t dev_handle = (int32_t)g_devices[g_current_dev];
        if (req->payload_size >= 4)
            memcpy(&dev_handle, payload, 4);
        int ordinal = device_ordinal_for_handle(dev_handle);
        res = ensure_device_context(ordinal);
        if (res != CUDA_SUCCESS)
            return chidori_send_response(fd, req->seq_num, res, NULL, 0);
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_contexts[ordinal];
        hmap_put(&g_ctx_to_dev, ctx_handle, (uint64_t)ordinal);
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case CHIDORI_CU_CTX_DESTROY:
    case CHIDORI_CU_DEVICE_PRIMARY_CTX_RELEASE:
    case CHIDORI_CU_DEVICE_PRIMARY_CTX_RESET:
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);

    case CHIDORI_CU_CTX_SET_CURRENT: {
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
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_CTX_GET_CURRENT: {
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_current_ctx;
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case CHIDORI_CU_CTX_PUSH_CURRENT: {
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
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_CTX_POP_CURRENT: {
        CUcontext ctx;
        res = cuCtxPopCurrent(&ctx);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_CTX_SYNCHRONIZE:
        res = cuCtxSynchronize();
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);

    case CHIDORI_CU_CTX_GET_DEVICE: {
        int32_t dev = (int32_t)g_devices[g_current_dev];
        return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, &dev, sizeof(dev));
    }

    /* ================================================================
     * Module Management
     * ================================================================ */
    case CHIDORI_CU_MODULE_LOAD_DATA:
    case CHIDORI_CU_MODULE_LOAD_DATA_EX: {
        /* Payload is raw PTX or CUBIN data */
        CUmodule mod;
        res = cuModuleLoadData(&mod, payload);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_modules, handle, (uint64_t)(uintptr_t)mod);
            return chidori_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MODULE_LOAD: {
        /* Payload is a filename string — load from server filesystem */
        CUmodule mod;
        res = cuModuleLoad(&mod, (const char *)payload);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_modules, handle, (uint64_t)(uintptr_t)mod);
            return chidori_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MODULE_UNLOAD: {
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
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MODULE_GET_FUNCTION: {
        /* Payload: module_handle(8) + function_name (null-terminated) */
        uint64_t mod_handle = 0;
        const char *func_name = "";
        if (req->payload_size >= 9) {
            memcpy(&mod_handle, payload, 8);
            func_name = (const char *)((char *)payload + 8);
        }
        CUmodule mod = (CUmodule)(uintptr_t)hmap_get(&g_modules, mod_handle);
        if (!mod) {
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }
        CUfunction func;
        res = cuModuleGetFunction(&func, mod, func_name);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_functions, handle, (uint64_t)(uintptr_t)func);
            return chidori_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MODULE_GET_GLOBAL: {
        /* Payload: module_handle(8) + name (null-terminated) */
        uint64_t mod_handle = 0;
        const char *name = "";
        if (req->payload_size >= 9) {
            memcpy(&mod_handle, payload, 8);
            name = (const char *)((char *)payload + 8);
        }
        CUmodule mod = (CUmodule)(uintptr_t)hmap_get(&g_modules, mod_handle);
        if (!mod) {
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }
        CUdeviceptr dptr;
        size_t size;
        res = cuModuleGetGlobal(&dptr, &size, mod, name);
        struct { uint64_t dptr; uint64_t size; } result;
        result.dptr = (uint64_t)dptr;
        result.size = (uint64_t)size;
        return chidori_send_response(fd, req->seq_num, res, &result, sizeof(result));
    }

    /* ================================================================
     * Memory Management
     * ================================================================ */
    case CHIDORI_CU_MEM_ALLOC: {
        uint64_t size = 0;
        if (req->payload_size >= 8)
            memcpy(&size, payload, 8);
        CUdeviceptr dptr;
        res = cuMemAlloc(&dptr, (size_t)size);
        uint64_t dptr64 = (uint64_t)dptr;
        return chidori_send_response(fd, req->seq_num, res, &dptr64, sizeof(dptr64));
    }

    case CHIDORI_CU_MEM_FREE: {
        uint64_t ptr = 0;
        if (req->payload_size >= 8)
            memcpy(&ptr, payload, 8);
        res = cuMemFree((CUdeviceptr)ptr);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MEMCPY_H_TO_D:
    case CHIDORI_CU_MEMCPY_H_TO_D_ASYNC: {
        /* Payload: dst(8) + size(8) + data(size) */
        uint64_t dst = 0, size = 0;
        if (req->payload_size >= 16) {
            memcpy(&dst, payload, 8);
            memcpy(&size, (char *)payload + 8, 8);
        }
        const void *data = (char *)payload + 16;
        res = cuMemcpyHtoD((CUdeviceptr)dst, data, (size_t)size);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MEMCPY_D_TO_H:
    case CHIDORI_CU_MEMCPY_D_TO_H_ASYNC: {
        /* Payload: src(8) + size(8) */
        uint64_t src = 0, size = 0;
        if (req->payload_size >= 16) {
            memcpy(&src, payload, 8);
            memcpy(&size, (char *)payload + 8, 8);
        }
        void *buf = malloc((size_t)size);
        if (!buf) {
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_OUT_OF_MEMORY, NULL, 0);
        }
        res = cuMemcpyDtoH(buf, (CUdeviceptr)src, (size_t)size);
        int rc = chidori_send_response(fd, req->seq_num, res, buf, (size_t)size);
        free(buf);
        return rc;
    }

    case CHIDORI_CU_MEMCPY_D_TO_D:
    case CHIDORI_CU_MEMCPY_D_TO_D_ASYNC: {
        /* Payload: dst(8) + src(8) + size(8) — zero data over wire! */
        uint64_t dst = 0, src = 0, size = 0;
        if (req->payload_size >= 24) {
            memcpy(&dst, payload, 8);
            memcpy(&src, (char *)payload + 8, 8);
            memcpy(&size, (char *)payload + 16, 8);
        }
        res = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MEMCPY: {
        /* Generic cuMemcpy: dst(8) + src(8) + size(8) */
        uint64_t dst = 0, src = 0, size = 0;
        if (req->payload_size >= 24) {
            memcpy(&dst, payload, 8);
            memcpy(&src, (char *)payload + 8, 8);
            memcpy(&size, (char *)payload + 16, 8);
        }
        res = cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MEMSET_D8: {
        uint64_t dst = 0; uint8_t val = 0; uint64_t count = 0;
        if (req->payload_size >= 17) {
            memcpy(&dst, payload, 8);
            memcpy(&val, (char *)payload + 8, 1);
            memcpy(&count, (char *)payload + 9, 8);
        }
        res = cuMemsetD8((CUdeviceptr)dst, val, (size_t)count);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MEMSET_D16: {
        uint64_t dst = 0; uint16_t val = 0; uint64_t count = 0;
        if (req->payload_size >= 18) {
            memcpy(&dst, payload, 8);
            memcpy(&val, (char *)payload + 8, 2);
            memcpy(&count, (char *)payload + 10, 8);
        }
        res = cuMemsetD16((CUdeviceptr)dst, val, (size_t)count);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MEMSET_D32: {
        uint64_t dst = 0; uint32_t val = 0; uint64_t count = 0;
        if (req->payload_size >= 20) {
            memcpy(&dst, payload, 8);
            memcpy(&val, (char *)payload + 8, 4);
            memcpy(&count, (char *)payload + 12, 8);
        }
        res = cuMemsetD32((CUdeviceptr)dst, val, (size_t)count);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_MEM_GET_INFO: {
        size_t free_mem, total_mem;
        res = cuMemGetInfo(&free_mem, &total_mem);
        struct { uint64_t free; uint64_t total; } info;
        info.free = (uint64_t)free_mem;
        info.total = (uint64_t)total_mem;
        return chidori_send_response(fd, req->seq_num, res, &info, sizeof(info));
    }

    case CHIDORI_CU_MEM_HOST_ALLOC:
    case CHIDORI_CU_MEM_FREE_HOST:
    case CHIDORI_CU_MEM_HOST_GET_DEVICE_PTR:
    case CHIDORI_CU_POINTER_GET_ATTRIBUTE:
    case CHIDORI_CU_MEM_ALLOC_MANAGED:
        /* These require more complex handling — stub for now */
        return chidori_send_response(fd, req->seq_num, CUDA_ERROR_NOT_SUPPORTED, NULL, 0);

    /* ================================================================
     * Execution Control
     * ================================================================ */
    case CHIDORI_CU_LAUNCH_KERNEL: {
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
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_VALUE, NULL, 0);
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
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }

        /* Grid with any zero dimension = empty launch (no-op) */
        if (gridX == 0 || gridY == 0 || gridZ == 0) {
            if (s_verbose)
                fprintf(stderr, "[cuda_dispatch] SKIP grid=(%u,%u,%u) zero dim\n", gridX, gridY, gridZ);
            return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);
        }

        CUstream stream = NULL;
        if (stream_handle != 0) {
            stream = (CUstream)(uintptr_t)hmap_get(&g_streams, stream_handle);
            if (!stream) stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);
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

                    /* Use actual kernel param size, not client buffer size */
                    int actual_ps = 0;
                    cuFuncGetAttribute(&actual_ps, (CUfunction_attribute)74, func); /* PARAM_SIZE */
                    size_t param_size = (actual_ps > 0) ? (size_t)actual_ps : (size_t)buf_size;

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
                    return chidori_send_response(fd, req->seq_num, res, NULL, 0);
                }
            }
        }

        /* No parameters — launch with NULL */
        res = cuLaunchKernel(func,
                             gridX, gridY, gridZ,
                             blockX, blockY, blockZ,
                             shared, stream,
                             NULL, NULL);

        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_FUNC_GET_ATTRIBUTE: {
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
        return chidori_send_response(fd, req->seq_num, res, &value, sizeof(value));
    }

    /* ================================================================
     * Stream Management
     * ================================================================ */
    case CHIDORI_CU_STREAM_CREATE: {
        CUstream stream;
        uint32_t flags = 0;
        if (req->payload_size >= 4)
            memcpy(&flags, payload, 4);
        res = cuStreamCreate(&stream, flags);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_streams, handle, (uint64_t)(uintptr_t)stream);
            return chidori_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_STREAM_CREATE_WITH_PRIORITY: {
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
            return chidori_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_STREAM_DESTROY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUstream stream = (CUstream)(uintptr_t)hmap_get(&g_streams, handle);
        if (!stream) stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, handle);
        if (stream) {
            res = cuStreamDestroy(stream);
            hmap_remove(&g_streams, handle);
            hmap_remove(&g_rt_streams, handle);
        } else {
            res = CUDA_SUCCESS; /* Already destroyed or default stream */
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_STREAM_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        if (handle == 0) {
            res = cuCtxSynchronize(); /* Default stream */
        } else {
            CUstream stream = (CUstream)(uintptr_t)hmap_get(&g_streams, handle);
            if (!stream) stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, handle);
            res = stream ? cuStreamSynchronize(stream) : CUDA_SUCCESS;
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_STREAM_QUERY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUstream stream = (CUstream)(uintptr_t)hmap_get(&g_streams, handle);
        if (!stream) stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, handle);
        res = stream ? cuStreamQuery(stream) : CUDA_SUCCESS;
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_STREAM_WAIT_EVENT: {
        uint64_t stream_handle = 0, event_handle = 0;
        uint32_t flags = 0;
        if (req->payload_size >= 20) {
            memcpy(&stream_handle, payload, 8);
            memcpy(&event_handle, (char *)payload + 8, 8);
            memcpy(&flags, (char *)payload + 16, 4);
        }
        CUstream stream = stream_handle ? (CUstream)(uintptr_t)hmap_get(&g_streams, stream_handle) : NULL;
        if (!stream && stream_handle) stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, event_handle);
        res = event ? cuStreamWaitEvent(stream, event, flags) : CUDA_ERROR_INVALID_HANDLE;
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    /* ================================================================
     * Event Management
     * ================================================================ */
    case CHIDORI_CU_EVENT_CREATE: {
        CUevent event;
        uint32_t flags = 0;
        if (req->payload_size >= 4)
            memcpy(&flags, payload, 4);
        res = cuEventCreate(&event, flags);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_events, handle, (uint64_t)(uintptr_t)event);
            return chidori_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_EVENT_DESTROY: {
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
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_EVENT_RECORD: {
        uint64_t event_handle = 0, stream_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&event_handle, payload, 8);
            memcpy(&stream_handle, (char *)payload + 8, 8);
        }
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, event_handle);
        CUstream stream = stream_handle ? (CUstream)(uintptr_t)hmap_get(&g_streams, stream_handle) : NULL;
        if (!stream && stream_handle) stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);
        res = event ? cuEventRecord(event, stream) : CUDA_ERROR_INVALID_HANDLE;
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_EVENT_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, handle);
        res = event ? cuEventSynchronize(event) : CUDA_ERROR_INVALID_HANDLE;
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_EVENT_QUERY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        CUevent event = (CUevent)(uintptr_t)hmap_get(&g_events, handle);
        res = event ? cuEventQuery(event) : CUDA_ERROR_INVALID_HANDLE;
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CU_EVENT_ELAPSED_TIME: {
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
        return chidori_send_response(fd, req->seq_num, res, &ms, sizeof(ms));
    }

    /* ================================================================
     * CUDA Runtime API (0x0100 - 0x01FF)
     *
     * These dispatch cudaXxx() Runtime API functions used by PyTorch,
     * TensorFlow, and other high-level frameworks.
     * ================================================================ */

    case CHIDORI_CUDA_GET_DEVICE_COUNT: {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        int32_t count32 = (int32_t)count;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &count32, sizeof(count32));
    }

    case CHIDORI_CUDA_GET_DEVICE: {
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        int32_t dev32 = (int32_t)device;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &dev32, sizeof(dev32));
    }

    case CHIDORI_CUDA_SET_DEVICE: {
        int32_t device = 0;
        if (req->payload_size >= 4)
            memcpy(&device, payload, 4);
        cudaError_t err = cudaSetDevice(device);
        /* Update our tracked current device */
        if (err == cudaSuccess && device >= 0 && device < g_device_count) {
            ensure_device_context(device);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_GET_DEVICE_PROPERTIES: {
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
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &prop, sizeof(prop));
    }

    case CHIDORI_CUDA_DEVICE_GET_ATTRIBUTE: {
        /* Payload: attr(4) + device(4) */
        int32_t attr = 0, device = 0;
        if (req->payload_size >= 8) {
            memcpy(&attr, payload, 4);
            memcpy(&device, (char *)payload + 4, 4);
        }
        int value = 0;
        cudaError_t err = cudaDeviceGetAttribute(&value, (enum cudaDeviceAttr)attr, device);
        int32_t val32 = (int32_t)value;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &val32, sizeof(val32));
    }

    case CHIDORI_CUDA_MALLOC: {
        /* Payload: size(8) */
        uint64_t size = 0;
        if (req->payload_size >= 8)
            memcpy(&size, payload, 8);
        void *devPtr = NULL;
        cudaError_t err = cudaMalloc(&devPtr, (size_t)size);
        uint64_t ptr64 = (uint64_t)(uintptr_t)devPtr;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &ptr64, sizeof(ptr64));
    }

    case CHIDORI_CUDA_FREE: {
        /* Payload: ptr(8) */
        uint64_t ptr = 0;
        if (req->payload_size >= 8)
            memcpy(&ptr, payload, 8);
        cudaError_t err = cudaFree((void *)(uintptr_t)ptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_MEMCPY: {
        /*
         * Unified layout — kind is always at offset 24:
         *   All: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) [+ data(size) for HtoD]
         *   HtoD src_ptr is 0 (data follows after header at offset 28).
         */
        if (req->payload_size < 28) {
            return chidori_send_response(fd, req->seq_num, (int32_t)cudaErrorInvalidValue, NULL, 0);
        }

        uint64_t dst_ptr = 0, src_ptr = 0, size = 0;
        int32_t kind = 0;
        memcpy(&dst_ptr, payload, 8);
        memcpy(&src_ptr, (char *)payload + 8, 8);
        memcpy(&size, (char *)payload + 16, 8);
        memcpy(&kind, (char *)payload + 24, 4);

        if (kind == cudaMemcpyHostToDevice) {
            const void *data = (char *)payload + 28;
            cudaError_t err = cudaMemcpy((void *)(uintptr_t)dst_ptr, data,
                                          (size_t)size, cudaMemcpyHostToDevice);
            return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
        } else if (kind == cudaMemcpyDeviceToHost) {
            void *buf = malloc((size_t)size);
            if (!buf) {
                return chidori_send_response(fd, req->seq_num,
                                            (int32_t)cudaErrorMemoryAllocation, NULL, 0);
            }
            cudaError_t err = cudaMemcpy(buf, (void *)(uintptr_t)src_ptr,
                                          (size_t)size, cudaMemcpyDeviceToHost);
            int rc = chidori_send_response(fd, req->seq_num, (int32_t)err, buf, (size_t)size);
            free(buf);
            return rc;
        } else if (kind == cudaMemcpyDeviceToDevice) {
            cudaError_t err = cudaMemcpy((void *)(uintptr_t)dst_ptr,
                                          (void *)(uintptr_t)src_ptr,
                                          (size_t)size, cudaMemcpyDeviceToDevice);
            return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
        } else {
            return chidori_send_response(fd, req->seq_num,
                                        (int32_t)cudaErrorInvalidMemcpyDirection, NULL, 0);
        }
    }

    case CHIDORI_CUDA_MEMCPY_ASYNC: {
        /*
         * Unified layout — kind always at offset 24, stream at offset 28:
         *   All: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) + stream(8) [+ data for HtoD]
         *   HtoD src_ptr is 0 (data follows after header at offset 36).
         */
        if (req->payload_size < 36) {
            return chidori_send_response(fd, req->seq_num, (int32_t)cudaErrorInvalidValue, NULL, 0);
        }

        uint64_t dst_ptr = 0, src_ptr = 0, size = 0, stream_handle = 0;
        int32_t kind = 0;
        memcpy(&dst_ptr, payload, 8);
        memcpy(&src_ptr, (char *)payload + 8, 8);
        memcpy(&size, (char *)payload + 16, 8);
        memcpy(&kind, (char *)payload + 24, 4);
        memcpy(&stream_handle, (char *)payload + 28, 8);

        cudaStream_t stream = NULL;
        if (stream_handle != 0) {
            stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);
            if (!stream) stream = (cudaStream_t)(uintptr_t)hmap_get(&g_streams, stream_handle);
        }

        if (kind == cudaMemcpyHostToDevice) {
            const void *data = (char *)payload + 36;
            cudaError_t err = cudaMemcpyAsync((void *)(uintptr_t)dst_ptr, data,
                                               (size_t)size, cudaMemcpyHostToDevice, stream);
            return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
        } else if (kind == cudaMemcpyDeviceToHost) {
            void *buf = malloc((size_t)size);
            if (!buf) {
                return chidori_send_response(fd, req->seq_num,
                                            (int32_t)cudaErrorMemoryAllocation, NULL, 0);
            }
            cudaError_t err = cudaMemcpyAsync(buf, (void *)(uintptr_t)src_ptr,
                                               (size_t)size, cudaMemcpyDeviceToHost, stream);
            if (err == cudaSuccess)
                cudaStreamSynchronize(stream);
            int rc = chidori_send_response(fd, req->seq_num, (int32_t)err, buf, (size_t)size);
            free(buf);
            return rc;
        } else if (kind == cudaMemcpyDeviceToDevice) {
            cudaError_t err = cudaMemcpyAsync((void *)(uintptr_t)dst_ptr,
                                               (void *)(uintptr_t)src_ptr,
                                               (size_t)size, cudaMemcpyDeviceToDevice, stream);
            return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
        } else {
            return chidori_send_response(fd, req->seq_num,
                                        (int32_t)cudaErrorInvalidMemcpyDirection, NULL, 0);
        }
    }

    case CHIDORI_CUDA_MEMSET: {
        /* Payload: ptr(8) + value(4) + count(8) */
        uint64_t ptr = 0, count = 0;
        int32_t value = 0;
        if (req->payload_size >= 20) {
            memcpy(&ptr, payload, 8);
            memcpy(&value, (char *)payload + 8, 4);
            memcpy(&count, (char *)payload + 12, 8);
        }
        cudaError_t err = cudaMemset((void *)(uintptr_t)ptr, value, (size_t)count);
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_DEVICE_SYNCHRONIZE: {
        cudaError_t err = cudaDeviceSynchronize();
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_STREAM_CREATE: {
        cudaStream_t stream = NULL;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_rt_streams, handle, (uint64_t)(uintptr_t)stream);
            return chidori_send_response(fd, req->seq_num, (int32_t)err, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_STREAM_DESTROY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaStream_t stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, handle);
        if (!stream) stream = (cudaStream_t)(uintptr_t)hmap_get(&g_streams, handle);
        cudaError_t err = cudaSuccess;
        if (stream) {
            err = cudaStreamDestroy(stream);
            hmap_remove(&g_rt_streams, handle);
            hmap_remove(&g_streams, handle);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_STREAM_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaError_t err;
        if (handle == 0) {
            err = cudaDeviceSynchronize(); /* Default stream */
        } else {
            cudaStream_t stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, handle);
            if (!stream) stream = (cudaStream_t)(uintptr_t)hmap_get(&g_streams, handle);
            err = stream ? cudaStreamSynchronize(stream) : cudaSuccess;
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_EVENT_CREATE: {
        cudaEvent_t event = NULL;
        cudaError_t err = cudaEventCreate(&event);
        if (err == cudaSuccess) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_rt_events, handle, (uint64_t)(uintptr_t)event);
            return chidori_send_response(fd, req->seq_num, (int32_t)err, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_EVENT_RECORD: {
        /* Payload: event_handle(8) + stream_handle(8) */
        uint64_t event_handle = 0, stream_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&event_handle, payload, 8);
            memcpy(&stream_handle, (char *)payload + 8, 8);
        }
        cudaEvent_t event = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, event_handle);
        cudaStream_t stream = NULL;
        if (stream_handle != 0) {
            stream = (cudaStream_t)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);
            if (!stream) stream = (cudaStream_t)(uintptr_t)hmap_get(&g_streams, stream_handle);
        }
        cudaError_t err = event ? cudaEventRecord(event, stream) : cudaErrorInvalidResourceHandle;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_EVENT_SYNCHRONIZE: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaEvent_t event = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, handle);
        cudaError_t err = event ? cudaEventSynchronize(event) : cudaErrorInvalidResourceHandle;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_EVENT_DESTROY: {
        uint64_t handle = 0;
        if (req->payload_size >= 8)
            memcpy(&handle, payload, 8);
        cudaEvent_t event = (cudaEvent_t)(uintptr_t)hmap_get(&g_rt_events, handle);
        cudaError_t err = cudaSuccess;
        if (event) {
            err = cudaEventDestroy(event);
            hmap_remove(&g_rt_events, handle);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_EVENT_ELAPSED_TIME: {
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
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &ms, sizeof(ms));
    }

    case CHIDORI_CUDA_RUNTIME_GET_VERSION: {
        int version = 0;
        cudaError_t err = cudaRuntimeGetVersion(&version);
        int32_t ver32 = (int32_t)version;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &ver32, sizeof(ver32));
    }

    case CHIDORI_CUDA_DRIVER_GET_VERSION: {
        int version = 0;
        cudaError_t err = cudaDriverGetVersion(&version);
        int32_t ver32 = (int32_t)version;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &ver32, sizeof(ver32));
    }

    case CHIDORI_CUDA_MEM_GET_INFO: {
        size_t free_mem = 0, total_mem = 0;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        struct { uint64_t free; uint64_t total; } info;
        info.free = (uint64_t)free_mem;
        info.total = (uint64_t)total_mem;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &info, sizeof(info));
    }

    case CHIDORI_CUDA_DEVICE_RESET: {
        cudaError_t err = cudaDeviceReset();
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_DEVICE_CAN_ACCESS_PEER: {
        /* Payload: device(4) + peerDevice(4) */
        int32_t device = 0, peerDevice = 0;
        if (req->payload_size >= 8) {
            memcpy(&device, payload, 4);
            memcpy(&peerDevice, (char *)payload + 4, 4);
        }
        int canAccess = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(&canAccess, device, peerDevice);
        int32_t ca32 = (int32_t)canAccess;
        return chidori_send_response(fd, req->seq_num, (int32_t)err, &ca32, sizeof(ca32));
    }

    case CHIDORI_CUDA_FUNC_GET_ATTRIBUTES: {
        /*
         * Client sends: func_handle(8)
         * Server returns: cudaFuncAttributes blob (64 bytes)
         *
         * Layout (matches cudaFuncAttributes struct on x86_64):
         *   offset 0:  sharedSizeBytes (size_t, 8)
         *   offset 8:  constSizeBytes  (size_t, 8)
         *   offset 16: localSizeBytes  (size_t, 8)
         *   offset 24: maxThreadsPerBlock (int, 4)
         *   offset 28: numRegs (int, 4)
         *   offset 32: ptxVersion (int, 4)
         *   offset 36: binaryVersion (int, 4)
         *   offset 40: cacheModeCA (int, 4)
         *   offset 44: maxDynamicSharedSizeBytes (int, 4)
         *   offset 48: preferredShmemCarveout (int, 4)
         */
        uint8_t attr_buf[64];
        memset(attr_buf, 0, sizeof(attr_buf));

        if (req->payload_size < 8) {
            /* No func handle — return defaults */
            int mtp = 1024;
            memcpy(attr_buf + 24, &mtp, 4);
            return chidori_send_response(fd, req->seq_num, (int32_t)cudaSuccess, attr_buf, sizeof(attr_buf));
        }

        uint64_t func_handle = 0;
        memcpy(&func_handle, payload, 8);
        CUfunction func = (CUfunction)(uintptr_t)hmap_get(&g_functions, func_handle);
        if (!func) {
            /* Unknown handle — return defaults */
            int mtp = 1024;
            memcpy(attr_buf + 24, &mtp, 4);
            return chidori_send_response(fd, req->seq_num, (int32_t)cudaSuccess, attr_buf, sizeof(attr_buf));
        }

        /* Query each attribute via cuFuncGetAttribute */
        int val;
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func) == CUDA_SUCCESS) {
            size_t v = (size_t)val; memcpy(attr_buf + 0, &v, 8);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func) == CUDA_SUCCESS) {
            size_t v = (size_t)val; memcpy(attr_buf + 8, &v, 8);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func) == CUDA_SUCCESS) {
            size_t v = (size_t)val; memcpy(attr_buf + 16, &v, 8);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func) == CUDA_SUCCESS) {
            memcpy(attr_buf + 24, &val, 4);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_NUM_REGS, func) == CUDA_SUCCESS) {
            memcpy(attr_buf + 28, &val, 4);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PTX_VERSION, func) == CUDA_SUCCESS) {
            memcpy(attr_buf + 32, &val, 4);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_BINARY_VERSION, func) == CUDA_SUCCESS) {
            memcpy(attr_buf + 36, &val, 4);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, func) == CUDA_SUCCESS) {
            memcpy(attr_buf + 40, &val, 4);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, func) == CUDA_SUCCESS) {
            memcpy(attr_buf + 44, &val, 4);
        }
        if (cuFuncGetAttribute(&val, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, func) == CUDA_SUCCESS) {
            memcpy(attr_buf + 48, &val, 4);
        }

        if (s_verbose) {
            int mtp = 0;
            memcpy(&mtp, attr_buf + 24, 4);
            fprintf(stderr, "[cuda_dispatch] FuncGetAttributes handle=0x%llx maxThreadsPerBlock=%d\n",
                    (unsigned long long)func_handle, mtp);
        }

        return chidori_send_response(fd, req->seq_num, (int32_t)cudaSuccess, attr_buf, sizeof(attr_buf));
    }

    case CHIDORI_CUDA_GET_LAST_ERROR: {
        cudaError_t err = cudaGetLastError();
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_PEEK_AT_LAST_ERROR: {
        cudaError_t err = cudaPeekAtLastError();
        return chidori_send_response(fd, req->seq_num, (int32_t)err, NULL, 0);
    }

    case CHIDORI_CUDA_REGISTER_FAT_BINARY: {
        /* Payload is raw fat binary data. Use cuModuleLoadDataEx with JIT
         * options to enable PTX compilation for cross-architecture support.
         * This allows sm_89 fat binaries (with PTX) to run on sm_120 GPUs. */
        CUmodule mod;
        /* Use __cudaRegisterFatBinary (Runtime API) which properly handles
         * PTX JIT for cross-architecture fat binaries. The Driver API's
         * cuModuleLoadData/cuModuleLoadFatBinary fail with error 209 when
         * SASS doesn't match and PTX needs JIT on Blackwell (sm_120).
         *
         * We reconstruct the __fatBinC_Wrapper struct that the runtime expects. */
        {
            struct { int magic; int version; const void *data; void *filename; } wrapper;
            wrapper.magic = 0x466243b1;
            wrapper.version = 1;
            wrapper.data = payload;
            wrapper.filename = NULL;

            extern void **__cudaRegisterFatBinary(void *);
            extern void __cudaUnregisterFatBinary(void **);
            void **handle = __cudaRegisterFatBinary(&wrapper);
            if (handle) {
                /* Success — the runtime loaded the fat binary.
                 * We need to get a CUmodule from it. Use cuModuleLoadData
                 * which should now find the loaded module in the runtime's cache. */
                res = cuModuleLoadData(&mod, payload);
                if (res != CUDA_SUCCESS) {
                    /* Fallback: try the other loaders */
                    res = cuModuleLoadFatBinary(&mod, payload);
                }
                if (res != CUDA_SUCCESS) {
                    res = cuModuleLoadDataEx(&mod, payload, 0, NULL, NULL);
                }
            } else {
                res = CUDA_ERROR_INVALID_IMAGE;
            }
        }
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[cuda_dispatch] cuModuleLoad FAILED: size=%u res=%d\n",
                    req->payload_size, (int)res);
        }
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_modules, handle, (uint64_t)(uintptr_t)mod);
            return chidori_send_response(fd, req->seq_num, res, &handle, sizeof(handle));
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CUDA_REGISTER_FUNCTION: {
        /* Payload: module_handle(8) + function_name (null-terminated) */
        uint64_t mod_handle = 0;
        const char *func_name = "";
        if (req->payload_size >= 9) {
            memcpy(&mod_handle, payload, 8);
            func_name = (const char *)((char *)payload + 8);
        }
        CUmodule mod = (CUmodule)(uintptr_t)hmap_get(&g_modules, mod_handle);
        if (!mod) {
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }
        CUfunction func;
        res = cuModuleGetFunction(&func, mod, func_name);
        if (res == CUDA_SUCCESS) {
            uint64_t handle = alloc_handle();
            hmap_put(&g_functions, handle, (uint64_t)(uintptr_t)func);

            /*
             * Query EXACT parameter layout using cuFuncGetParamInfo (CUDA 12.3+).
             * Returns offset and size for each kernel parameter.
             * This is the ONLY correct way to serialize kernel args.
             */
            struct { uint32_t offset; uint32_t size; } param_info[64];
            int num_params = 0;
            uint32_t total_param_size = 0;

            for (int i = 0; i < 64; i++) {
                size_t off, sz;
                CUresult pi_res = cuFuncGetParamInfo(func, (size_t)i, &off, &sz);
                if (pi_res != CUDA_SUCCESS) break;
                param_info[num_params].offset = (uint32_t)off;
                param_info[num_params].size = (uint32_t)sz;
                num_params++;
                uint32_t end = (uint32_t)(off + sz);
                if (end > total_param_size) total_param_size = end;
            }

            if (s_verbose)
                fprintf(stderr, "[cuda_dispatch] RegisterFunction '%s' -> handle=0x%llx "
                        "num_params=%d total_param_size=%u\n",
                        func_name, (unsigned long long)handle, num_params, total_param_size);

            /*
             * Response: func_handle(8) + total_param_size(4) + num_params(4)
             *           + [offset(4) + size(4)] × num_params
             */
            size_t resp_size = 8 + 4 + 4 + (num_params * 8);
            uint8_t *resp = (uint8_t *)malloc(resp_size);
            uint8_t *rp = resp;
            memcpy(rp, &handle, 8); rp += 8;
            memcpy(rp, &total_param_size, 4); rp += 4;
            int32_t np = (int32_t)num_params;
            memcpy(rp, &np, 4); rp += 4;
            for (int i = 0; i < num_params; i++) {
                memcpy(rp, &param_info[i].offset, 4); rp += 4;
                memcpy(rp, &param_info[i].size, 4); rp += 4;
            }
            int rc = chidori_send_response(fd, req->seq_num, res, resp, resp_size);
            free(resp);
            return rc;
        }
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case CHIDORI_CUDA_LAUNCH_KERNEL: {
        /*
         * Payload: func_handle(8) + gridX(4) + gridY(4) + gridZ(4)
         *   + blockX(4) + blockY(4) + blockZ(4)
         *   + sharedMem(4) + stream_handle(8) = 44 bytes
         *   + has_params(1) + [buffer_size(4) + buffer_data(buffer_size)]
         *
         * The buffer is a contiguous copy of the client's kernel parameter
         * memory. We pass it directly to cuLaunchKernel via
         * CU_LAUNCH_PARAM_BUFFER_POINTER — the CUDA driver parses it
         * using the kernel's metadata (no need to reconstruct void**).
         */
        if (req->payload_size < 44) {
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_VALUE, NULL, 0);
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
            return chidori_send_response(fd, req->seq_num, CUDA_ERROR_INVALID_HANDLE, NULL, 0);
        }

        /* Grid with any zero dimension = empty launch (no-op) */
        if (gridX == 0 || gridY == 0 || gridZ == 0) {
            if (s_verbose)
                fprintf(stderr, "[cuda_dispatch] SKIP grid=(%u,%u,%u) zero dim\n", gridX, gridY, gridZ);
            return chidori_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);
        }

        CUstream stream = NULL;
        if (stream_handle != 0) {
            stream = (CUstream)(uintptr_t)hmap_get(&g_streams, stream_handle);
            if (!stream) stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, stream_handle);
        }

        /* Parse kernel parameter buffer */
        size_t remaining = req->payload_size - 44;

        if (remaining >= 5) {
            uint8_t has_params = 0;
            memcpy(&has_params, p, 1); p += 1;

            if (has_params) {
                uint32_t buf_size = 0;
                memcpy(&buf_size, p, 4); p += 4;

                if (buf_size > 0) {
                    void *param_buffer = p;
                    size_t param_size = (size_t)buf_size;

                    if (s_verbose) {
                        fprintf(stderr, "[cuda_dispatch] LAUNCH grid=(%u,%u,%u) block=(%u,%u,%u) buf=%u\n",
                                gridX, gridY, gridZ, blockX, blockY, blockZ, buf_size);
                        /* Hex dump of parameter buffer */
                        fprintf(stderr, "[cuda_dispatch] PARAMS: ");
                        for (uint32_t bi = 0; bi < buf_size && bi < 64; bi++)
                            fprintf(stderr, "%02x", ((uint8_t*)param_buffer)[bi]);
                        fprintf(stderr, "\n");
                    }

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

                    if (s_verbose) {
                        fprintf(stderr, "[cuda_dispatch] LAUNCH result=%d\n", res);
                        /* Check for async errors */
                        CUresult sync_res = cuCtxSynchronize();
                        if (sync_res != CUDA_SUCCESS)
                            fprintf(stderr, "[cuda_dispatch] POST-LAUNCH SYNC ERROR=%d\n", sync_res);
                    }

                    return chidori_send_response(fd, req->seq_num, res, NULL, 0);
                }
            }
        }

        /* No parameters */
        res = cuLaunchKernel(func,
                             gridX, gridY, gridZ,
                             blockX, blockY, blockZ,
                             shared, stream,
                             NULL, NULL);
        return chidori_send_response(fd, req->seq_num, res, NULL, 0);
    }

    /* ================================================================
     * cuBLAS API
     * ================================================================ */
    case CHIDORI_CUBLAS_CREATE: {
        cublasHandle_t handle;
        cublasStatus_t st = cublasCreate(&handle);
        if (st == CUBLAS_STATUS_SUCCESS) {
            uint64_t wire_handle = alloc_handle();
            hmap_put(&g_cublas_handles, wire_handle, (uint64_t)(uintptr_t)handle);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire_handle, sizeof(wire_handle));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_DESTROY: {
        uint64_t wire_handle = 0;
        if (req->payload_size >= 8) memcpy(&wire_handle, payload, 8);
        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasDestroy(handle);
        hmap_remove(&g_cublas_handles, wire_handle);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SET_STREAM: {
        uint64_t wire_handle = 0, stream_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&wire_handle, payload, 8);
            memcpy(&stream_handle, (char *)payload + 8, 8);
        }
        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        /* Resolve the stream: try runtime streams first, then driver streams, then raw */
        cudaStream_t stream = NULL;
        if (stream_handle != 0) {
            uint64_t real_stream = hmap_get(&g_rt_streams, stream_handle);
            if (real_stream == 0) real_stream = hmap_get(&g_streams, stream_handle);
            if (real_stream != 0) stream = (cudaStream_t)(uintptr_t)real_stream;
            else stream = (cudaStream_t)(uintptr_t)stream_handle;
        }
        cublasStatus_t st = cublasSetStream(handle, stream);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_GET_STREAM: {
        uint64_t wire_handle = 0;
        if (req->payload_size >= 8) memcpy(&wire_handle, payload, 8);
        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cudaStream_t stream = NULL;
        cublasStatus_t st = cublasGetStream(handle, &stream);
        uint64_t s = (uint64_t)(uintptr_t)stream;
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &s, sizeof(s));
    }

    case CHIDORI_CUBLAS_SET_MATH_MODE: {
        uint64_t wire_handle = 0;
        int32_t mode = 0;
        if (req->payload_size >= 12) {
            memcpy(&wire_handle, payload, 8);
            memcpy(&mode, (char *)payload + 8, 4);
        }
        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasSetMathMode(handle, (cublasMath_t)mode);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_GET_MATH_MODE: {
        uint64_t wire_handle = 0;
        if (req->payload_size >= 8) memcpy(&wire_handle, payload, 8);
        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasMath_t mode;
        cublasStatus_t st = cublasGetMathMode(handle, &mode);
        int32_t m = (int32_t)mode;
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &m, sizeof(m));
    }

    case CHIDORI_CUBLAS_GET_VERSION: {
        uint64_t wire_handle = 0;
        if (req->payload_size >= 8) memcpy(&wire_handle, payload, 8);
        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        int version = 0;
        cublasStatus_t st = cublasGetVersion(handle, &version);
        int32_t v = (int32_t)version;
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &v, sizeof(v));
    }

    case CHIDORI_CUBLAS_SGEMM: {
        /* Payload: handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
         *        + alpha(4) + A(8) + lda(4) + B(8) + ldb(4)
         *        + beta(4) + C(8) + ldc(4) */
        if (req->payload_size < 72) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        float alpha; memcpy(&alpha, p, 4); p += 4;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        uint64_t B; memcpy(&B, p, 8); p += 8;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        float beta; memcpy(&beta, p, 4); p += 4;
        uint64_t C; memcpy(&C, p, 8); p += 8;
        int32_t ldc; memcpy(&ldc, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasSgemm(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k, &alpha,
            (const float *)(uintptr_t)A, lda,
            (const float *)(uintptr_t)B, ldb,
            &beta, (float *)(uintptr_t)C, ldc);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_DGEMM: {
        if (req->payload_size < 80) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        double alpha; memcpy(&alpha, p, 8); p += 8;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        uint64_t B; memcpy(&B, p, 8); p += 8;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        double beta; memcpy(&beta, p, 8); p += 8;
        uint64_t C; memcpy(&C, p, 8); p += 8;
        int32_t ldc; memcpy(&ldc, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasDgemm(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k, &alpha,
            (const double *)(uintptr_t)A, lda,
            (const double *)(uintptr_t)B, ldb,
            &beta, (double *)(uintptr_t)C, ldc);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_HGEMM: {
        if (req->payload_size < 64) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        uint16_t alpha_h; memcpy(&alpha_h, p, 2); p += 2;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        uint64_t B; memcpy(&B, p, 8); p += 8;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        uint16_t beta_h; memcpy(&beta_h, p, 2); p += 2;
        uint64_t C; memcpy(&C, p, 8); p += 8;
        int32_t ldc; memcpy(&ldc, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        /* Use cublasGemmEx for fp16 — avoids __half C type dependency */
        cublasStatus_t st = cublasGemmEx(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k, &alpha_h,
            (const void *)(uintptr_t)A, CUDA_R_16F, lda,
            (const void *)(uintptr_t)B, CUDA_R_16F, ldb,
            &beta_h, (void *)(uintptr_t)C, CUDA_R_16F, ldc,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SGEMM_EX: {
        if (req->payload_size < 80) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        float alpha; memcpy(&alpha, p, 4); p += 4;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t Atype; memcpy(&Atype, p, 4); p += 4;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        uint64_t B; memcpy(&B, p, 8); p += 8;
        int32_t Btype; memcpy(&Btype, p, 4); p += 4;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        float beta; memcpy(&beta, p, 4); p += 4;
        uint64_t C; memcpy(&C, p, 8); p += 8;
        int32_t Ctype; memcpy(&Ctype, p, 4); p += 4;
        int32_t ldc; memcpy(&ldc, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasSgemmEx(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k, &alpha,
            (const void *)(uintptr_t)A, (cudaDataType)Atype, lda,
            (const void *)(uintptr_t)B, (cudaDataType)Btype, ldb,
            &beta, (void *)(uintptr_t)C, (cudaDataType)Ctype, ldc);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_GEMM_EX: {
        if (req->payload_size < 96) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        uint8_t alpha_buf[8]; memcpy(alpha_buf, p, 8); p += 8;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t Atype; memcpy(&Atype, p, 4); p += 4;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        uint64_t B; memcpy(&B, p, 8); p += 8;
        int32_t Btype; memcpy(&Btype, p, 4); p += 4;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        uint8_t beta_buf[8]; memcpy(beta_buf, p, 8); p += 8;
        uint64_t C; memcpy(&C, p, 8); p += 8;
        int32_t Ctype; memcpy(&Ctype, p, 4); p += 4;
        int32_t ldc; memcpy(&ldc, p, 4); p += 4;
        int32_t computeType; memcpy(&computeType, p, 4); p += 4;
        int32_t algo; memcpy(&algo, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasGemmEx(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k,
            alpha_buf, (const void *)(uintptr_t)A, (cudaDataType)Atype, lda,
            (const void *)(uintptr_t)B, (cudaDataType)Btype, ldb,
            beta_buf, (void *)(uintptr_t)C, (cudaDataType)Ctype, ldc,
            (cublasComputeType_t)computeType, (cublasGemmAlgo_t)algo);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SGEMV: {
        if (req->payload_size < 60) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t trans; memcpy(&trans, p, 4); p += 4;
        int32_t m, n;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        float alpha; memcpy(&alpha, p, 4); p += 4;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        float beta; memcpy(&beta, p, 4); p += 4;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasSgemv(handle,
            (cublasOperation_t)trans, m, n, &alpha,
            (const float *)(uintptr_t)A, lda,
            (const float *)(uintptr_t)x, incx,
            &beta, (float *)(uintptr_t)y, incy);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_DGEMV: {
        if (req->payload_size < 68) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t trans; memcpy(&trans, p, 4); p += 4;
        int32_t m, n;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        double alpha; memcpy(&alpha, p, 8); p += 8;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        double beta; memcpy(&beta, p, 8); p += 8;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasDgemv(handle,
            (cublasOperation_t)trans, m, n, &alpha,
            (const double *)(uintptr_t)A, lda,
            (const double *)(uintptr_t)x, incx,
            &beta, (double *)(uintptr_t)y, incy);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SAXPY: {
        if (req->payload_size < 36) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        float alpha; memcpy(&alpha, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasSaxpy(handle, n, &alpha,
            (const float *)(uintptr_t)x, incx,
            (float *)(uintptr_t)y, incy);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_DAXPY: {
        if (req->payload_size < 40) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        double alpha; memcpy(&alpha, p, 8); p += 8;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasDaxpy(handle, n, &alpha,
            (const double *)(uintptr_t)x, incx,
            (double *)(uintptr_t)y, incy);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SNRM2: {
        if (req->payload_size < 24) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        float result = 0.0f;
        cublasStatus_t st = cublasSnrm2(handle, n,
            (const float *)(uintptr_t)x, incx, &result);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &result, sizeof(result));
    }

    case CHIDORI_CUBLAS_DNRM2: {
        if (req->payload_size < 24) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        double result = 0.0;
        cublasStatus_t st = cublasDnrm2(handle, n,
            (const double *)(uintptr_t)x, incx, &result);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &result, sizeof(result));
    }

    case CHIDORI_CUBLAS_SDOT: {
        if (req->payload_size < 40) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        float result = 0.0f;
        cublasStatus_t st = cublasSdot(handle, n,
            (const float *)(uintptr_t)x, incx,
            (const float *)(uintptr_t)y, incy, &result);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &result, sizeof(result));
    }

    case CHIDORI_CUBLAS_DDOT: {
        if (req->payload_size < 40) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        double result = 0.0;
        cublasStatus_t st = cublasDdot(handle, n,
            (const double *)(uintptr_t)x, incx,
            (const double *)(uintptr_t)y, incy, &result);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &result, sizeof(result));
    }

    case CHIDORI_CUBLAS_SSCAL: {
        if (req->payload_size < 28) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        float alpha; memcpy(&alpha, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasSscal(handle, n, &alpha,
            (float *)(uintptr_t)x, incx);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_DSCAL: {
        if (req->payload_size < 32) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        double alpha; memcpy(&alpha, p, 8); p += 8;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasDscal(handle, n, &alpha,
            (double *)(uintptr_t)x, incx);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SCOPY: {
        if (req->payload_size < 36) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasScopy(handle, n,
            (const float *)(uintptr_t)x, incx,
            (float *)(uintptr_t)y, incy);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_DCOPY: {
        if (req->payload_size < 36) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t n; memcpy(&n, p, 4); p += 4;
        uint64_t x; memcpy(&x, p, 8); p += 8;
        int32_t incx; memcpy(&incx, p, 4); p += 4;
        uint64_t y; memcpy(&y, p, 8); p += 8;
        int32_t incy; memcpy(&incy, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasDcopy(handle, n,
            (const double *)(uintptr_t)x, incx,
            (double *)(uintptr_t)y, incy);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SGEMM_BATCHED: {
        /*
         * Payload: handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
         *        + alpha(4) + lda(4) + ldb(4) + beta(4) + ldc(4) + batchCount(4) = 52
         *        + Aarray(8*bc) + Barray(8*bc) + Carray(8*bc)
         */
        if (req->payload_size < 52) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        float alpha; memcpy(&alpha, p, 4); p += 4;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        float beta; memcpy(&beta, p, 4); p += 4;
        int32_t ldc; memcpy(&ldc, p, 4); p += 4;
        int32_t batchCount; memcpy(&batchCount, p, 4); p += 4;

        size_t ptr_array_bytes = (size_t)batchCount * 8;
        if (req->payload_size < 52 + 3 * ptr_array_bytes) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }

        /* Build arrays of real device pointers */
        const float **Aarray = (const float **)malloc((size_t)batchCount * sizeof(float *));
        const float **Barray = (const float **)malloc((size_t)batchCount * sizeof(float *));
        float **Carray = (float **)malloc((size_t)batchCount * sizeof(float *));
        if (!Aarray || !Barray || !Carray) {
            free(Aarray); free(Barray); free(Carray);
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_ALLOC_FAILED, NULL, 0);
        }

        for (int i = 0; i < batchCount; i++) {
            uint64_t ptr; memcpy(&ptr, p, 8); p += 8;
            Aarray[i] = (const float *)(uintptr_t)ptr;
        }
        for (int i = 0; i < batchCount; i++) {
            uint64_t ptr; memcpy(&ptr, p, 8); p += 8;
            Barray[i] = (const float *)(uintptr_t)ptr;
        }
        for (int i = 0; i < batchCount; i++) {
            uint64_t ptr; memcpy(&ptr, p, 8); p += 8;
            Carray[i] = (float *)(uintptr_t)ptr;
        }

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);

        /* cublasSgemmBatched needs device-side pointer arrays. Copy to device. */
        const float **d_Aarray, **d_Barray;
        float **d_Carray;
        size_t arr_sz = (size_t)batchCount * sizeof(void *);
        cudaMalloc((void **)&d_Aarray, arr_sz);
        cudaMalloc((void **)&d_Barray, arr_sz);
        cudaMalloc((void **)&d_Carray, arr_sz);
        cudaMemcpy((void *)d_Aarray, Aarray, arr_sz, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d_Barray, Barray, arr_sz, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)d_Carray, Carray, arr_sz, cudaMemcpyHostToDevice);

        cublasStatus_t st = cublasSgemmBatched(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k, &alpha,
            d_Aarray, lda, d_Barray, ldb,
            &beta, d_Carray, ldc, batchCount);

        cudaFree((void *)d_Aarray);
        cudaFree((void *)d_Barray);
        cudaFree((void *)d_Carray);
        free(Aarray); free(Barray); free(Carray);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_SGEMM_STRIDED_BATCHED: {
        if (req->payload_size < 88) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        float alpha; memcpy(&alpha, p, 4); p += 4;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        int64_t strideA; memcpy(&strideA, p, 8); p += 8;
        uint64_t B; memcpy(&B, p, 8); p += 8;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        int64_t strideB; memcpy(&strideB, p, 8); p += 8;
        float beta; memcpy(&beta, p, 4); p += 4;
        uint64_t C; memcpy(&C, p, 8); p += 8;
        int32_t ldc; memcpy(&ldc, p, 4); p += 4;
        int64_t strideC; memcpy(&strideC, p, 8); p += 8;
        int32_t batchCount; memcpy(&batchCount, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasSgemmStridedBatched(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k, &alpha,
            (const float *)(uintptr_t)A, lda, (long long)strideA,
            (const float *)(uintptr_t)B, ldb, (long long)strideB,
            &beta, (float *)(uintptr_t)C, ldc, (long long)strideC,
            batchCount);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUBLAS_GEMM_STRIDED_BATCHED_EX: {
        if (req->payload_size < 108) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_handle; memcpy(&wire_handle, p, 8); p += 8;
        int32_t transa; memcpy(&transa, p, 4); p += 4;
        int32_t transb; memcpy(&transb, p, 4); p += 4;
        int32_t m, n, k;
        memcpy(&m, p, 4); p += 4;
        memcpy(&n, p, 4); p += 4;
        memcpy(&k, p, 4); p += 4;
        uint8_t alpha_buf[8]; memcpy(alpha_buf, p, 8); p += 8;
        uint64_t A; memcpy(&A, p, 8); p += 8;
        int32_t Atype; memcpy(&Atype, p, 4); p += 4;
        int32_t lda; memcpy(&lda, p, 4); p += 4;
        int64_t strideA; memcpy(&strideA, p, 8); p += 8;
        uint64_t B; memcpy(&B, p, 8); p += 8;
        int32_t Btype; memcpy(&Btype, p, 4); p += 4;
        int32_t ldb; memcpy(&ldb, p, 4); p += 4;
        int64_t strideB; memcpy(&strideB, p, 8); p += 8;
        uint8_t beta_buf[8]; memcpy(beta_buf, p, 8); p += 8;
        uint64_t C; memcpy(&C, p, 8); p += 8;
        int32_t Ctype; memcpy(&Ctype, p, 4); p += 4;
        int32_t ldc; memcpy(&ldc, p, 4); p += 4;
        int64_t strideC; memcpy(&strideC, p, 8); p += 8;
        int32_t batchCount; memcpy(&batchCount, p, 4); p += 4;
        int32_t computeType; memcpy(&computeType, p, 4); p += 4;
        int32_t algo; memcpy(&algo, p, 4);

        cublasHandle_t handle = (cublasHandle_t)(uintptr_t)hmap_get(&g_cublas_handles, wire_handle);
        cublasStatus_t st = cublasGemmStridedBatchedEx(handle,
            (cublasOperation_t)transa, (cublasOperation_t)transb,
            m, n, k,
            alpha_buf, (const void *)(uintptr_t)A, (cudaDataType)Atype, lda, (long long)strideA,
            (const void *)(uintptr_t)B, (cudaDataType)Btype, ldb, (long long)strideB,
            beta_buf, (void *)(uintptr_t)C, (cudaDataType)Ctype, ldc, (long long)strideC,
            batchCount, (cublasComputeType_t)computeType, (cublasGemmAlgo_t)algo);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ================================================================
     * cublasLt API (0x02A0 - 0x02AF)
     * ================================================================ */

    case CHIDORI_CUBLAS_LT_MATMUL_ALGO_HEURISTIC: {
        /*
         * Server-side cublasLt heuristic: receive raw descriptor fields,
         * create real descriptors, call real heuristic, return algo bytes.
         *
         * Payload (176 bytes):
         *   transa(4) transb(4) computeType(4) scaleType(4) = 16
         *   For each of A, B, C, D (x4):
         *     dtype(4) rows(8) cols(8) ld(8) batch(4) stride(8) = 40
         *   Total: 16 + 4*40 = 176
         */
        if (req->payload_size < 188) {
            int32_t zero = 0;
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, &zero, 4);
        }

        int32_t transa, transb, computeType, scaleType, epilogue;
        uint64_t bias_ptr;
        struct { int32_t dtype; uint64_t rows, cols; int64_t ld; int32_t batch; int64_t stride; } mat[4];

        const char *p = (const char *)payload;
        memcpy(&transa, p, 4); p += 4;
        memcpy(&transb, p, 4); p += 4;
        memcpy(&computeType, p, 4); p += 4;
        memcpy(&scaleType, p, 4); p += 4;
        memcpy(&epilogue, p, 4); p += 4;
        memcpy(&bias_ptr, p, 8); p += 8;
        for (int i = 0; i < 4; i++) {
            memcpy(&mat[i].dtype, p, 4); p += 4;
            memcpy(&mat[i].rows, p, 8); p += 8;
            memcpy(&mat[i].cols, p, 8); p += 8;
            memcpy(&mat[i].ld, p, 8); p += 8;
            memcpy(&mat[i].batch, p, 4); p += 4;
            memcpy(&mat[i].stride, p, 8); p += 8;
        }

        fprintf(stderr, "[cublas_lt] heuristic: transa=%d transb=%d compute=%d scale=%d "
                "A(%d,%lu,%lu,ld=%ld) B(%d,%lu,%lu,ld=%ld) C(%d,%lu,%lu,ld=%ld) D(%d,%lu,%lu,ld=%ld) batch=%d\n",
                transa, transb, computeType, scaleType,
                mat[0].dtype, mat[0].rows, mat[0].cols, mat[0].ld,
                mat[1].dtype, mat[1].rows, mat[1].cols, mat[1].ld,
                mat[2].dtype, mat[2].rows, mat[2].cols, mat[2].ld,
                mat[3].dtype, mat[3].rows, mat[3].cols, mat[3].ld, mat[0].batch);

        cublasLtHandle_t ltHandle;
        cublasLtCreate(&ltHandle);

        cublasLtMatmulDesc_t opDesc;
        cublasLtMatmulDescCreate(&opDesc, (cublasComputeType_t)computeType, (cudaDataType_t)scaleType);
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
        if (epilogue)
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        if (bias_ptr) {
            void *bp = (void *)(uintptr_t)bias_ptr;
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp));
        }

        cublasLtMatrixLayout_t descs[4];
        for (int i = 0; i < 4; i++) {
            cublasLtMatrixLayoutCreate(&descs[i], (cudaDataType_t)mat[i].dtype,
                                       mat[i].rows, mat[i].cols, mat[i].ld);
            if (mat[i].batch > 1) {
                cublasLtMatrixLayoutSetAttribute(descs[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &mat[i].batch, sizeof(mat[i].batch));
                cublasLtMatrixLayoutSetAttribute(descs[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &mat[i].stride, sizeof(mat[i].stride));
            }
        }

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);

        cublasLtMatmulHeuristicResult_t result;
        int returnAlgoCount = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
            ltHandle, opDesc, descs[0], descs[1], descs[2], descs[3],
            pref, 1, &result, &returnAlgoCount);

        cublasLtMatmulPreferenceDestroy(pref);
        for (int i = 0; i < 4; i++) cublasLtMatrixLayoutDestroy(descs[i]);
        cublasLtMatmulDescDestroy(opDesc);
        cublasLtDestroy(ltHandle);

        uint8_t resp[4 + sizeof(cublasLtMatmulHeuristicResult_t)];
        memcpy(resp, &returnAlgoCount, 4);
        if (returnAlgoCount > 0)
            memcpy(resp + 4, &result, sizeof(result));
        return chidori_send_response(fd, req->seq_num, (int32_t)st,
                                      resp, 4 + (returnAlgoCount > 0 ? sizeof(result) : 0));
    }

    case CHIDORI_CUBLAS_LT_MATMUL: {
        /*
         * Server-side cublasLtMatmul: same descriptor format as heuristic,
         * plus alpha/beta/pointers/algo.
         *
         * Payload (296 bytes):
         *   176 (descriptors, same as heuristic)
         *   + alpha(8) beta(8) A_ptr(8) B_ptr(8) C_ptr(8) D_ptr(8) stream(8) algo(64)
         */
        if (req->payload_size < 308) {
            return chidori_send_response(fd, req->seq_num, CUBLAS_STATUS_INVALID_VALUE, NULL, 0);
        }

        int32_t transa, transb, computeType, scaleType, epilogue;
        uint64_t bias_ptr;
        struct { int32_t dtype; uint64_t rows, cols; int64_t ld; int32_t batch; int64_t stride; } mat[4];
        uint8_t alpha_bytes[8], beta_bytes[8];
        uint64_t A_ptr, B_ptr, C_ptr, D_ptr, stream_wire;
        uint8_t algo_bytes[64];

        const char *p = (const char *)payload;
        memcpy(&transa, p, 4); p += 4;
        memcpy(&transb, p, 4); p += 4;
        memcpy(&computeType, p, 4); p += 4;
        memcpy(&scaleType, p, 4); p += 4;
        memcpy(&epilogue, p, 4); p += 4;
        memcpy(&bias_ptr, p, 8); p += 8;
        for (int i = 0; i < 4; i++) {
            memcpy(&mat[i].dtype, p, 4); p += 4;
            memcpy(&mat[i].rows, p, 8); p += 8;
            memcpy(&mat[i].cols, p, 8); p += 8;
            memcpy(&mat[i].ld, p, 8); p += 8;
            memcpy(&mat[i].batch, p, 4); p += 4;
            memcpy(&mat[i].stride, p, 8); p += 8;
        }
        memcpy(alpha_bytes, p, 8); p += 8;
        memcpy(beta_bytes, p, 8); p += 8;
        memcpy(&A_ptr, p, 8); p += 8;
        memcpy(&B_ptr, p, 8); p += 8;
        memcpy(&C_ptr, p, 8); p += 8;
        memcpy(&D_ptr, p, 8); p += 8;
        memcpy(&stream_wire, p, 8); p += 8;
        memcpy(algo_bytes, p, 64);

        cublasLtHandle_t ltHandle;
        cublasLtCreate(&ltHandle);

        cublasLtMatmulDesc_t opDesc;
        cublasLtMatmulDescCreate(&opDesc, (cublasComputeType_t)computeType, (cudaDataType_t)scaleType);
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
        if (epilogue)
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        if (bias_ptr) {
            void *bp = (void *)(uintptr_t)bias_ptr;
            cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp));
        }

        cublasLtMatrixLayout_t descs[4];
        for (int i = 0; i < 4; i++) {
            cublasLtMatrixLayoutCreate(&descs[i], (cudaDataType_t)mat[i].dtype,
                                       mat[i].rows, mat[i].cols, mat[i].ld);
            if (mat[i].batch > 1) {
                cublasLtMatrixLayoutSetAttribute(descs[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &mat[i].batch, sizeof(mat[i].batch));
                cublasLtMatrixLayoutSetAttribute(descs[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &mat[i].stride, sizeof(mat[i].stride));
            }
        }

        CUstream stream = (CUstream)(uintptr_t)hmap_get(&g_streams, stream_wire);
        if (!stream && stream_wire)
            stream = (CUstream)(uintptr_t)hmap_get(&g_rt_streams, stream_wire);

        cublasLtMatmulAlgo_t *algo = (cublasLtMatmulAlgo_t *)algo_bytes;

        cublasStatus_t st = cublasLtMatmul(ltHandle, opDesc,
            alpha_bytes, (void *)(uintptr_t)A_ptr, descs[0],
            (void *)(uintptr_t)B_ptr, descs[1],
            beta_bytes, (void *)(uintptr_t)C_ptr, descs[2],
            (void *)(uintptr_t)D_ptr, descs[3],
            algo, NULL, 0, (cudaStream_t)stream);

        for (int i = 0; i < 4; i++) cublasLtMatrixLayoutDestroy(descs[i]);
        cublasLtMatmulDescDestroy(opDesc);
        cublasLtDestroy(ltHandle);

        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ================================================================
     * cuDNN API (0x0300 - 0x03FF)
     * ================================================================ */

    case CHIDORI_CUDNN_CREATE: {
        cudnnHandle_t handle;
        cudnnStatus_t st = cudnnCreate(&handle);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire_handle = alloc_handle();
            hmap_put(&g_cudnn_handles, wire_handle, (uint64_t)(uintptr_t)handle);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire_handle, sizeof(wire_handle));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DESTROY: {
        uint64_t wire_handle = 0;
        if (req->payload_size >= 8) memcpy(&wire_handle, payload, 8);
        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wire_handle);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (handle) {
            st = cudnnDestroy(handle);
            hmap_remove(&g_cudnn_handles, wire_handle);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_STREAM: {
        uint64_t wire_handle = 0, stream_handle = 0;
        if (req->payload_size >= 16) {
            memcpy(&wire_handle, payload, 8);
            memcpy(&stream_handle, (char *)payload + 8, 8);
        }
        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wire_handle);
        cudaStream_t stream = NULL;
        if (stream_handle != 0) {
            uint64_t real_stream = hmap_get(&g_rt_streams, stream_handle);
            if (real_stream == 0) real_stream = hmap_get(&g_streams, stream_handle);
            if (real_stream != 0) stream = (cudaStream_t)(uintptr_t)real_stream;
            else stream = (cudaStream_t)(uintptr_t)stream_handle;
        }
        cudnnStatus_t st = cudnnSetStream(handle, stream);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_GET_VERSION: {
        size_t version = cudnnGetVersion();
        uint64_t v = (uint64_t)version;
        return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_SUCCESS, &v, sizeof(v));
    }

    /* ---- Tensor Descriptor ---- */

    case CHIDORI_CUDNN_CREATE_TENSOR_DESCRIPTOR: {
        cudnnTensorDescriptor_t desc;
        cudnnStatus_t st = cudnnCreateTensorDescriptor(&desc);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire = alloc_handle();
            hmap_put(&g_cudnn_tensor_descs, wire, (uint64_t)(uintptr_t)desc);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire, sizeof(wire));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_TENSOR_4D_DESCRIPTOR: {
        /* Payload: desc(8) + format(4) + dataType(4) + n(4) + c(4) + h(4) + w(4) = 32 */
        if (req->payload_size < 32) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_desc; memcpy(&wire_desc, p, 8); p += 8;
        int32_t format; memcpy(&format, p, 4); p += 4;
        int32_t dataType; memcpy(&dataType, p, 4); p += 4;
        int32_t n, c, h, w;
        memcpy(&n, p, 4); p += 4;
        memcpy(&c, p, 4); p += 4;
        memcpy(&h, p, 4); p += 4;
        memcpy(&w, p, 4);

        cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wire_desc);
        cudnnStatus_t st = cudnnSetTensor4dDescriptor(desc,
            (cudnnTensorFormat_t)format, (cudnnDataType_t)dataType, n, c, h, w);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_TENSOR_ND_DESCRIPTOR: {
        /* Payload: desc(8) + dataType(4) + nbDims(4) + dims(4*n) + strides(4*n) */
        if (req->payload_size < 16) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire_desc; memcpy(&wire_desc, p, 8); p += 8;
        int32_t dataType; memcpy(&dataType, p, 4); p += 4;
        int32_t nbDims; memcpy(&nbDims, p, 4); p += 4;

        if (nbDims > 8) nbDims = 8; /* safety cap */
        int dimA[8], strideA[8];
        for (int i = 0; i < nbDims; i++) { memcpy(&dimA[i], p, 4); p += 4; }
        for (int i = 0; i < nbDims; i++) { memcpy(&strideA[i], p, 4); p += 4; }

        cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wire_desc);
        cudnnStatus_t st = cudnnSetTensorNdDescriptor(desc,
            (cudnnDataType_t)dataType, nbDims, dimA, strideA);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_GET_TENSOR_4D_DESCRIPTOR: {
        uint64_t wire_desc = 0;
        if (req->payload_size >= 8) memcpy(&wire_desc, payload, 8);
        cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wire_desc);
        cudnnDataType_t dataType;
        int n, c, h, w, nStride, cStride, hStride, wStride;
        cudnnStatus_t st = cudnnGetTensor4dDescriptor(desc,
            &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride);
        struct {
            int32_t dataType;
            int32_t n, c, h, w;
            int32_t nStride, cStride, hStride, wStride;
        } out;
        out.dataType = (int32_t)dataType;
        out.n = n; out.c = c; out.h = h; out.w = w;
        out.nStride = nStride; out.cStride = cStride;
        out.hStride = hStride; out.wStride = wStride;
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &out, sizeof(out));
    }

    case CHIDORI_CUDNN_DESTROY_TENSOR_DESCRIPTOR: {
        uint64_t wire = 0;
        if (req->payload_size >= 8) memcpy(&wire, payload, 8);
        cudnnTensorDescriptor_t desc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wire);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (desc) {
            st = cudnnDestroyTensorDescriptor(desc);
            hmap_remove(&g_cudnn_tensor_descs, wire);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Filter Descriptor ---- */

    case CHIDORI_CUDNN_CREATE_FILTER_DESCRIPTOR: {
        cudnnFilterDescriptor_t desc;
        cudnnStatus_t st = cudnnCreateFilterDescriptor(&desc);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire = alloc_handle();
            hmap_put(&g_cudnn_filter_descs, wire, (uint64_t)(uintptr_t)desc);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire, sizeof(wire));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_FILTER_4D_DESCRIPTOR: {
        /* desc(8) + dataType(4) + format(4) + k(4) + c(4) + h(4) + w(4) = 32 */
        if (req->payload_size < 32) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire; memcpy(&wire, p, 8); p += 8;
        int32_t dataType; memcpy(&dataType, p, 4); p += 4;
        int32_t format; memcpy(&format, p, 4); p += 4;
        int32_t k, c, h, w;
        memcpy(&k, p, 4); p += 4;
        memcpy(&c, p, 4); p += 4;
        memcpy(&h, p, 4); p += 4;
        memcpy(&w, p, 4);

        cudnnFilterDescriptor_t desc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, wire);
        cudnnStatus_t st = cudnnSetFilter4dDescriptor(desc,
            (cudnnDataType_t)dataType, (cudnnTensorFormat_t)format, k, c, h, w);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_FILTER_ND_DESCRIPTOR: {
        /* desc(8) + dataType(4) + format(4) + nbDims(4) + dims(4*n) */
        if (req->payload_size < 20) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire; memcpy(&wire, p, 8); p += 8;
        int32_t dataType; memcpy(&dataType, p, 4); p += 4;
        int32_t format; memcpy(&format, p, 4); p += 4;
        int32_t nbDims; memcpy(&nbDims, p, 4); p += 4;
        if (nbDims > 8) nbDims = 8;
        int filterDimA[8];
        for (int i = 0; i < nbDims; i++) { memcpy(&filterDimA[i], p, 4); p += 4; }

        cudnnFilterDescriptor_t desc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, wire);
        cudnnStatus_t st = cudnnSetFilterNdDescriptor(desc,
            (cudnnDataType_t)dataType, (cudnnTensorFormat_t)format, nbDims, filterDimA);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DESTROY_FILTER_DESCRIPTOR: {
        uint64_t wire = 0;
        if (req->payload_size >= 8) memcpy(&wire, payload, 8);
        cudnnFilterDescriptor_t desc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, wire);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (desc) {
            st = cudnnDestroyFilterDescriptor(desc);
            hmap_remove(&g_cudnn_filter_descs, wire);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Convolution Descriptor ---- */

    case CHIDORI_CUDNN_CREATE_CONVOLUTION_DESCRIPTOR: {
        cudnnConvolutionDescriptor_t desc;
        cudnnStatus_t st = cudnnCreateConvolutionDescriptor(&desc);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire = alloc_handle();
            hmap_put(&g_cudnn_conv_descs, wire, (uint64_t)(uintptr_t)desc);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire, sizeof(wire));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR: {
        /* desc(8) + pad_h(4) + pad_w(4) + u(4) + v(4) + dilation_h(4) + dilation_w(4) + mode(4) + computeType(4) = 40 */
        if (req->payload_size < 40) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire; memcpy(&wire, p, 8); p += 8;
        int32_t pad_h, pad_w, u, v, dh, dw, mode, ct;
        memcpy(&pad_h, p, 4); p += 4;
        memcpy(&pad_w, p, 4); p += 4;
        memcpy(&u, p, 4); p += 4;
        memcpy(&v, p, 4); p += 4;
        memcpy(&dh, p, 4); p += 4;
        memcpy(&dw, p, 4); p += 4;
        memcpy(&mode, p, 4); p += 4;
        memcpy(&ct, p, 4);

        cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wire);
        cudnnStatus_t st = cudnnSetConvolution2dDescriptor(desc,
            pad_h, pad_w, u, v, dh, dw,
            (cudnnConvolutionMode_t)mode, (cudnnDataType_t)ct);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR: {
        /* desc(8) + arrayLength(4) + mode(4) + computeType(4) + pads(4*n) + strides(4*n) + dilations(4*n) */
        if (req->payload_size < 20) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire; memcpy(&wire, p, 8); p += 8;
        int32_t arrayLength; memcpy(&arrayLength, p, 4); p += 4;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        int32_t ct; memcpy(&ct, p, 4); p += 4;
        if (arrayLength > 6) arrayLength = 6;
        int padA[6], strideA[6], dilA[6];
        for (int i = 0; i < arrayLength; i++) { memcpy(&padA[i], p, 4); p += 4; }
        for (int i = 0; i < arrayLength; i++) { memcpy(&strideA[i], p, 4); p += 4; }
        for (int i = 0; i < arrayLength; i++) { memcpy(&dilA[i], p, 4); p += 4; }

        cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wire);
        cudnnStatus_t st = cudnnSetConvolutionNdDescriptor(desc,
            arrayLength, padA, strideA, dilA,
            (cudnnConvolutionMode_t)mode, (cudnnDataType_t)ct);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR: {
        uint64_t wire = 0;
        if (req->payload_size >= 8) memcpy(&wire, payload, 8);
        cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wire);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (desc) {
            st = cudnnDestroyConvolutionDescriptor(desc);
            hmap_remove(&g_cudnn_conv_descs, wire);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_CONVOLUTION_MATH_TYPE: {
        uint64_t wire = 0;
        int32_t mathType = 0;
        if (req->payload_size >= 12) {
            memcpy(&wire, payload, 8);
            memcpy(&mathType, (char *)payload + 8, 4);
        }
        cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wire);
        cudnnStatus_t st = cudnnSetConvolutionMathType(desc, (cudnnMathType_t)mathType);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_CONVOLUTION_GROUP_COUNT: {
        uint64_t wire = 0;
        int32_t groupCount = 1;
        if (req->payload_size >= 12) {
            memcpy(&wire, payload, 8);
            memcpy(&groupCount, (char *)payload + 8, 4);
        }
        cudnnConvolutionDescriptor_t desc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wire);
        cudnnStatus_t st = cudnnSetConvolutionGroupCount(desc, groupCount);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Convolution Forward Algorithm ---- */

    case CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM: {
        /* handle(8) + xDesc(8) + wDesc(8) + convDesc(8) + yDesc(8) + requestedAlgoCount(4) = 44 */
        if (req->payload_size < 44) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh, wx, ww, wc, wy;
        int32_t requestedCount;
        memcpy(&wh, p, 8); p += 8;
        memcpy(&wx, p, 8); p += 8;
        memcpy(&ww, p, 8); p += 8;
        memcpy(&wc, p, 8); p += 8;
        memcpy(&wy, p, 8); p += 8;
        memcpy(&requestedCount, p, 4);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wx);
        cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, ww);
        cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wc);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wy);

        if (requestedCount > 8) requestedCount = 8;
        cudnnConvolutionFwdAlgoPerf_t perfResults[8];
        int returnedCount = 0;

        cudnnStatus_t st = cudnnGetConvolutionForwardAlgorithm_v7(handle,
            xDesc, wDesc, convDesc, yDesc,
            requestedCount, &returnedCount, perfResults);

        /* Response: returnedCount(4) + perfResults array */
        size_t perf_sz = (size_t)returnedCount * sizeof(cudnnConvolutionFwdAlgoPerf_t);
        size_t out_len = 4 + perf_sz;
        void *out = malloc(out_len);
        if (!out) return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_ALLOC_FAILED, NULL, 0);
        int32_t rc32 = (int32_t)returnedCount;
        memcpy(out, &rc32, 4);
        memcpy((char *)out + 4, perfResults, perf_sz);
        int r = chidori_send_response(fd, req->seq_num, (int32_t)st, out, out_len);
        free(out);
        return r;
    }

    /* ---- Convolution Workspace Size ---- */

    case CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE: {
        /* handle(8) + xDesc(8) + wDesc(8) + convDesc(8) + yDesc(8) + algo(4) = 44 */
        if (req->payload_size < 44) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh, wx, ww, wc, wy;
        int32_t algo;
        memcpy(&wh, p, 8); p += 8;
        memcpy(&wx, p, 8); p += 8;
        memcpy(&ww, p, 8); p += 8;
        memcpy(&wc, p, 8); p += 8;
        memcpy(&wy, p, 8); p += 8;
        memcpy(&algo, p, 4);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wx);
        cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, ww);
        cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wc);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wy);

        size_t sizeInBytes = 0;
        cudnnStatus_t st = cudnnGetConvolutionForwardWorkspaceSize(handle,
            xDesc, wDesc, convDesc, yDesc,
            (cudnnConvolutionFwdAlgo_t)algo, &sizeInBytes);
        uint64_t sz = (uint64_t)sizeInBytes;
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &sz, sizeof(sz));
    }

    case CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE: {
        if (req->payload_size < 44) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh, ww, wdy, wc, wdx;
        int32_t algo;
        memcpy(&wh, p, 8); p += 8;
        memcpy(&ww, p, 8); p += 8;
        memcpy(&wdy, p, 8); p += 8;
        memcpy(&wc, p, 8); p += 8;
        memcpy(&wdx, p, 8); p += 8;
        memcpy(&algo, p, 4);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, ww);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdy);
        cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wc);
        cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdx);

        size_t sizeInBytes = 0;
        cudnnStatus_t st = cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
            wDesc, dyDesc, convDesc, dxDesc,
            (cudnnConvolutionBwdDataAlgo_t)algo, &sizeInBytes);
        uint64_t sz = (uint64_t)sizeInBytes;
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &sz, sizeof(sz));
    }

    case CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE: {
        if (req->payload_size < 44) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh, wx, wdy, wc, wgrad;
        int32_t algo;
        memcpy(&wh, p, 8); p += 8;
        memcpy(&wx, p, 8); p += 8;
        memcpy(&wdy, p, 8); p += 8;
        memcpy(&wc, p, 8); p += 8;
        memcpy(&wgrad, p, 8); p += 8;
        memcpy(&algo, p, 4);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wx);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdy);
        cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wc);
        cudnnFilterDescriptor_t gradDesc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, wgrad);

        size_t sizeInBytes = 0;
        cudnnStatus_t st = cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,
            xDesc, dyDesc, convDesc, gradDesc,
            (cudnnConvolutionBwdFilterAlgo_t)algo, &sizeInBytes);
        uint64_t sz = (uint64_t)sizeInBytes;
        return chidori_send_response(fd, req->seq_num, (int32_t)st, &sz, sizeof(sz));
    }

    /* ---- Convolution Forward ---- */

    case CHIDORI_CUDNN_CONVOLUTION_FORWARD: {
        /*
         * handle(8) + alpha(8) + xDesc(8) + x(8) + wDesc(8) + w(8)
         * + convDesc(8) + algo(4) + workspace(8) + workspaceSize(8)
         * + beta(8) + yDesc(8) + y(8) = 100
         */
        if (req->payload_size < 100) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint64_t wwd; memcpy(&wwd, p, 8); p += 8;
        uint64_t wptr; memcpy(&wptr, p, 8); p += 8;
        uint64_t wcd; memcpy(&wcd, p, 8); p += 8;
        int32_t algo; memcpy(&algo, p, 4); p += 4;
        uint64_t ws_ptr; memcpy(&ws_ptr, p, 8); p += 8;
        uint64_t ws_size; memcpy(&ws_size, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, wwd);
        cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wcd);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);

        cudnnStatus_t st = cudnnConvolutionForward(handle,
            alpha, xDesc, (const void *)(uintptr_t)xptr,
            wDesc, (const void *)(uintptr_t)wptr,
            convDesc, (cudnnConvolutionFwdAlgo_t)algo,
            (void *)(uintptr_t)ws_ptr, (size_t)ws_size,
            beta, yDesc, (void *)(uintptr_t)yptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Convolution Backward Data ---- */

    case CHIDORI_CUDNN_CONVOLUTION_BACKWARD_DATA: {
        if (req->payload_size < 100) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wwd; memcpy(&wwd, p, 8); p += 8;
        uint64_t wptr; memcpy(&wptr, p, 8); p += 8;
        uint64_t wdyd; memcpy(&wdyd, p, 8); p += 8;
        uint64_t dyptr; memcpy(&dyptr, p, 8); p += 8;
        uint64_t wcd; memcpy(&wcd, p, 8); p += 8;
        int32_t algo; memcpy(&algo, p, 4); p += 4;
        uint64_t ws_ptr; memcpy(&ws_ptr, p, 8); p += 8;
        uint64_t ws_size; memcpy(&ws_size, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wdxd; memcpy(&wdxd, p, 8); p += 8;
        uint64_t dxptr; memcpy(&dxptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, wwd);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdyd);
        cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wcd);
        cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdxd);

        cudnnStatus_t st = cudnnConvolutionBackwardData(handle,
            alpha, wDesc, (const void *)(uintptr_t)wptr,
            dyDesc, (const void *)(uintptr_t)dyptr,
            convDesc, (cudnnConvolutionBwdDataAlgo_t)algo,
            (void *)(uintptr_t)ws_ptr, (size_t)ws_size,
            beta, dxDesc, (void *)(uintptr_t)dxptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Convolution Backward Filter ---- */

    case CHIDORI_CUDNN_CONVOLUTION_BACKWARD_FILTER: {
        if (req->payload_size < 100) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint64_t wdyd; memcpy(&wdyd, p, 8); p += 8;
        uint64_t dyptr; memcpy(&dyptr, p, 8); p += 8;
        uint64_t wcd; memcpy(&wcd, p, 8); p += 8;
        int32_t algo; memcpy(&algo, p, 4); p += 4;
        uint64_t ws_ptr; memcpy(&ws_ptr, p, 8); p += 8;
        uint64_t ws_size; memcpy(&ws_size, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wdwd; memcpy(&wdwd, p, 8); p += 8;
        uint64_t dwptr; memcpy(&dwptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdyd);
        cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_conv_descs, wcd);
        cudnnFilterDescriptor_t dwDesc = (cudnnFilterDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_filter_descs, wdwd);

        cudnnStatus_t st = cudnnConvolutionBackwardFilter(handle,
            alpha, xDesc, (const void *)(uintptr_t)xptr,
            dyDesc, (const void *)(uintptr_t)dyptr,
            convDesc, (cudnnConvolutionBwdFilterAlgo_t)algo,
            (void *)(uintptr_t)ws_ptr, (size_t)ws_size,
            beta, dwDesc, (void *)(uintptr_t)dwptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Activation Descriptor ---- */

    case CHIDORI_CUDNN_CREATE_ACTIVATION_DESCRIPTOR: {
        cudnnActivationDescriptor_t desc;
        cudnnStatus_t st = cudnnCreateActivationDescriptor(&desc);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire = alloc_handle();
            hmap_put(&g_cudnn_activation_descs, wire, (uint64_t)(uintptr_t)desc);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire, sizeof(wire));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_ACTIVATION_DESCRIPTOR: {
        /* desc(8) + mode(4) + reluNanOpt(4) + coef(8) = 24 */
        if (req->payload_size < 24) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire; memcpy(&wire, p, 8); p += 8;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        int32_t nanOpt; memcpy(&nanOpt, p, 4); p += 4;
        double coef; memcpy(&coef, p, 8);

        cudnnActivationDescriptor_t desc = (cudnnActivationDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_activation_descs, wire);
        cudnnStatus_t st = cudnnSetActivationDescriptor(desc,
            (cudnnActivationMode_t)mode, (cudnnNanPropagation_t)nanOpt, coef);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DESTROY_ACTIVATION_DESCRIPTOR: {
        uint64_t wire = 0;
        if (req->payload_size >= 8) memcpy(&wire, payload, 8);
        cudnnActivationDescriptor_t desc = (cudnnActivationDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_activation_descs, wire);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (desc) {
            st = cudnnDestroyActivationDescriptor(desc);
            hmap_remove(&g_cudnn_activation_descs, wire);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_ACTIVATION_FORWARD: {
        /* handle(8) + activationDesc(8) + alpha(8) + xDesc(8) + x(8) + beta(8) + yDesc(8) + y(8) = 64 */
        if (req->payload_size < 64) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        uint64_t wad; memcpy(&wad, p, 8); p += 8;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnActivationDescriptor_t actDesc = (cudnnActivationDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_activation_descs, wad);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);

        cudnnStatus_t st = cudnnActivationForward(handle, actDesc,
            alpha, xDesc, (const void *)(uintptr_t)xptr,
            beta, yDesc, (void *)(uintptr_t)yptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_ACTIVATION_BACKWARD: {
        /* handle(8) + actDesc(8) + alpha(8) + yDesc(8) + y(8) + dyDesc(8) + dy(8) + xDesc(8) + x(8) + beta(8) + dxDesc(8) + dx(8) = 96 */
        if (req->payload_size < 96) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        uint64_t wad; memcpy(&wad, p, 8); p += 8;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8); p += 8;
        uint64_t wdyd; memcpy(&wdyd, p, 8); p += 8;
        uint64_t dyptr; memcpy(&dyptr, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wdxd; memcpy(&wdxd, p, 8); p += 8;
        uint64_t dxptr; memcpy(&dxptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnActivationDescriptor_t actDesc = (cudnnActivationDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_activation_descs, wad);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdyd);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdxd);

        cudnnStatus_t st = cudnnActivationBackward(handle, actDesc,
            alpha, yDesc, (const void *)(uintptr_t)yptr,
            dyDesc, (const void *)(uintptr_t)dyptr,
            xDesc, (const void *)(uintptr_t)xptr,
            beta, dxDesc, (void *)(uintptr_t)dxptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Pooling Descriptor ---- */

    case CHIDORI_CUDNN_CREATE_POOLING_DESCRIPTOR: {
        cudnnPoolingDescriptor_t desc;
        cudnnStatus_t st = cudnnCreatePoolingDescriptor(&desc);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire = alloc_handle();
            hmap_put(&g_cudnn_pooling_descs, wire, (uint64_t)(uintptr_t)desc);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire, sizeof(wire));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_POOLING_2D_DESCRIPTOR: {
        /* desc(8) + mode(4) + nanOpt(4) + wH(4) + wW(4) + vP(4) + hP(4) + vS(4) + hS(4) = 40 */
        if (req->payload_size < 40) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire; memcpy(&wire, p, 8); p += 8;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        int32_t nanOpt; memcpy(&nanOpt, p, 4); p += 4;
        int32_t wH, wW, vP, hP, vS, hS;
        memcpy(&wH, p, 4); p += 4;
        memcpy(&wW, p, 4); p += 4;
        memcpy(&vP, p, 4); p += 4;
        memcpy(&hP, p, 4); p += 4;
        memcpy(&vS, p, 4); p += 4;
        memcpy(&hS, p, 4);

        cudnnPoolingDescriptor_t desc = (cudnnPoolingDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_pooling_descs, wire);
        cudnnStatus_t st = cudnnSetPooling2dDescriptor(desc,
            (cudnnPoolingMode_t)mode, (cudnnNanPropagation_t)nanOpt,
            wH, wW, vP, hP, vS, hS);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DESTROY_POOLING_DESCRIPTOR: {
        uint64_t wire = 0;
        if (req->payload_size >= 8) memcpy(&wire, payload, 8);
        cudnnPoolingDescriptor_t desc = (cudnnPoolingDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_pooling_descs, wire);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (desc) {
            st = cudnnDestroyPoolingDescriptor(desc);
            hmap_remove(&g_cudnn_pooling_descs, wire);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_POOLING_FORWARD: {
        /* handle(8) + poolingDesc(8) + alpha(8) + xDesc(8) + x(8) + beta(8) + yDesc(8) + y(8) = 64 */
        if (req->payload_size < 64) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        uint64_t wpd; memcpy(&wpd, p, 8); p += 8;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnPoolingDescriptor_t poolDesc = (cudnnPoolingDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_pooling_descs, wpd);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);

        cudnnStatus_t st = cudnnPoolingForward(handle, poolDesc,
            alpha, xDesc, (const void *)(uintptr_t)xptr,
            beta, yDesc, (void *)(uintptr_t)yptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_POOLING_BACKWARD: {
        /* handle(8) + poolDesc(8) + alpha(8) + yDesc(8) + y(8) + dyDesc(8) + dy(8) + xDesc(8) + x(8) + beta(8) + dxDesc(8) + dx(8) = 96 */
        if (req->payload_size < 96) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        uint64_t wpd; memcpy(&wpd, p, 8); p += 8;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8); p += 8;
        uint64_t wdyd; memcpy(&wdyd, p, 8); p += 8;
        uint64_t dyptr; memcpy(&dyptr, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wdxd; memcpy(&wdxd, p, 8); p += 8;
        uint64_t dxptr; memcpy(&dxptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnPoolingDescriptor_t poolDesc = (cudnnPoolingDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_pooling_descs, wpd);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdyd);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdxd);

        cudnnStatus_t st = cudnnPoolingBackward(handle, poolDesc,
            alpha, yDesc, (const void *)(uintptr_t)yptr,
            dyDesc, (const void *)(uintptr_t)dyptr,
            xDesc, (const void *)(uintptr_t)xptr,
            beta, dxDesc, (void *)(uintptr_t)dxptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Batch Normalization ---- */

    case CHIDORI_CUDNN_BATCH_NORM_FORWARD_TRAINING: {
        /*
         * handle(8) + mode(4) + alpha(8) + beta(8) + xDesc(8) + x(8)
         * + yDesc(8) + y(8) + bnDesc(8) + bnScale(8) + bnBias(8)
         * + expAvgFactor(8) + runMean(8) + runVar(8)
         * + epsilon(8) + saveMean(8) + saveInvVar(8) = 140
         */
        if (req->payload_size < 140) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8); p += 8;
        uint64_t wbn; memcpy(&wbn, p, 8); p += 8;
        uint64_t scaleptr; memcpy(&scaleptr, p, 8); p += 8;
        uint64_t biasptr; memcpy(&biasptr, p, 8); p += 8;
        double expAvg; memcpy(&expAvg, p, 8); p += 8;
        uint64_t runMean; memcpy(&runMean, p, 8); p += 8;
        uint64_t runVar; memcpy(&runVar, p, 8); p += 8;
        double epsilon; memcpy(&epsilon, p, 8); p += 8;
        uint64_t saveMean; memcpy(&saveMean, p, 8); p += 8;
        uint64_t saveInvVar; memcpy(&saveInvVar, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);
        cudnnTensorDescriptor_t bnDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wbn);

        cudnnStatus_t st = cudnnBatchNormalizationForwardTraining(handle,
            (cudnnBatchNormMode_t)mode, alpha, beta,
            xDesc, (const void *)(uintptr_t)xptr,
            yDesc, (void *)(uintptr_t)yptr,
            bnDesc,
            (const void *)(uintptr_t)scaleptr,
            (const void *)(uintptr_t)biasptr,
            expAvg,
            (void *)(uintptr_t)runMean,
            (void *)(uintptr_t)runVar,
            epsilon,
            (void *)(uintptr_t)saveMean,
            (void *)(uintptr_t)saveInvVar);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_BATCH_NORM_FORWARD_INFERENCE: {
        /*
         * handle(8) + mode(4) + alpha(8) + beta(8) + xDesc(8) + x(8)
         * + yDesc(8) + y(8) + bnDesc(8) + bnScale(8) + bnBias(8)
         * + estMean(8) + estVar(8) + epsilon(8) = 116
         */
        if (req->payload_size < 116) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8); p += 8;
        uint64_t wbn; memcpy(&wbn, p, 8); p += 8;
        uint64_t scaleptr; memcpy(&scaleptr, p, 8); p += 8;
        uint64_t biasptr; memcpy(&biasptr, p, 8); p += 8;
        uint64_t estMean; memcpy(&estMean, p, 8); p += 8;
        uint64_t estVar; memcpy(&estVar, p, 8); p += 8;
        double epsilon; memcpy(&epsilon, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);
        cudnnTensorDescriptor_t bnDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wbn);

        cudnnStatus_t st = cudnnBatchNormalizationForwardInference(handle,
            (cudnnBatchNormMode_t)mode, alpha, beta,
            xDesc, (const void *)(uintptr_t)xptr,
            yDesc, (void *)(uintptr_t)yptr,
            bnDesc,
            (const void *)(uintptr_t)scaleptr,
            (const void *)(uintptr_t)biasptr,
            (const void *)(uintptr_t)estMean,
            (const void *)(uintptr_t)estVar,
            epsilon);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_BATCH_NORM_BACKWARD: {
        /*
         * handle(8) + mode(4) + alphaDD(8) + betaDD(8) + alphaPD(8) + betaPD(8)
         * + xDesc(8) + x(8) + dyDesc(8) + dy(8) + dxDesc(8) + dx(8)
         * + dBnDesc(8) + bnScale(8) + dScale(8) + dBias(8)
         * + epsilon(8) + savedMean(8) + savedInvVar(8) = 164
         */
        if (req->payload_size < 164) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        uint8_t alphaDD[8]; memcpy(alphaDD, p, 8); p += 8;
        uint8_t betaDD[8]; memcpy(betaDD, p, 8); p += 8;
        uint8_t alphaPD[8]; memcpy(alphaPD, p, 8); p += 8;
        uint8_t betaPD[8]; memcpy(betaPD, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint64_t wdyd; memcpy(&wdyd, p, 8); p += 8;
        uint64_t dyptr; memcpy(&dyptr, p, 8); p += 8;
        uint64_t wdxd; memcpy(&wdxd, p, 8); p += 8;
        uint64_t dxptr; memcpy(&dxptr, p, 8); p += 8;
        uint64_t wdbn; memcpy(&wdbn, p, 8); p += 8;
        uint64_t scaleptr; memcpy(&scaleptr, p, 8); p += 8;
        uint64_t dscaleptr; memcpy(&dscaleptr, p, 8); p += 8;
        uint64_t dbiasptr; memcpy(&dbiasptr, p, 8); p += 8;
        double epsilon; memcpy(&epsilon, p, 8); p += 8;
        uint64_t savedMean; memcpy(&savedMean, p, 8); p += 8;
        uint64_t savedInvVar; memcpy(&savedInvVar, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdyd);
        cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdxd);
        cudnnTensorDescriptor_t dBnDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdbn);

        cudnnStatus_t st = cudnnBatchNormalizationBackward(handle,
            (cudnnBatchNormMode_t)mode,
            alphaDD, betaDD, alphaPD, betaPD,
            xDesc, (const void *)(uintptr_t)xptr,
            dyDesc, (const void *)(uintptr_t)dyptr,
            dxDesc, (void *)(uintptr_t)dxptr,
            dBnDesc,
            (const void *)(uintptr_t)scaleptr,
            (void *)(uintptr_t)dscaleptr,
            (void *)(uintptr_t)dbiasptr,
            epsilon,
            (const void *)(uintptr_t)savedMean,
            (const void *)(uintptr_t)savedInvVar);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Softmax ---- */

    case CHIDORI_CUDNN_SOFTMAX_FORWARD: {
        /* handle(8) + algo(4) + mode(4) + alpha(8) + xDesc(8) + x(8) + beta(8) + yDesc(8) + y(8) = 64 */
        if (req->payload_size < 64) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        int32_t algo; memcpy(&algo, p, 4); p += 4;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wxd; memcpy(&wxd, p, 8); p += 8;
        uint64_t xptr; memcpy(&xptr, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wxd);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);

        cudnnStatus_t st = cudnnSoftmaxForward(handle,
            (cudnnSoftmaxAlgorithm_t)algo, (cudnnSoftmaxMode_t)mode,
            alpha, xDesc, (const void *)(uintptr_t)xptr,
            beta, yDesc, (void *)(uintptr_t)yptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SOFTMAX_BACKWARD: {
        /* handle(8) + algo(4) + mode(4) + alpha(8) + yDesc(8) + y(8) + dyDesc(8) + dy(8) + beta(8) + dxDesc(8) + dx(8) = 80 */
        if (req->payload_size < 80) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        int32_t algo; memcpy(&algo, p, 4); p += 4;
        int32_t mode; memcpy(&mode, p, 4); p += 4;
        uint8_t alpha[8]; memcpy(alpha, p, 8); p += 8;
        uint64_t wyd; memcpy(&wyd, p, 8); p += 8;
        uint64_t yptr; memcpy(&yptr, p, 8); p += 8;
        uint64_t wdyd; memcpy(&wdyd, p, 8); p += 8;
        uint64_t dyptr; memcpy(&dyptr, p, 8); p += 8;
        uint8_t beta[8]; memcpy(beta, p, 8); p += 8;
        uint64_t wdxd; memcpy(&wdxd, p, 8); p += 8;
        uint64_t dxptr; memcpy(&dxptr, p, 8);

        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wyd);
        cudnnTensorDescriptor_t dyDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdyd);
        cudnnTensorDescriptor_t dxDesc = (cudnnTensorDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_tensor_descs, wdxd);

        cudnnStatus_t st = cudnnSoftmaxBackward(handle,
            (cudnnSoftmaxAlgorithm_t)algo, (cudnnSoftmaxMode_t)mode,
            alpha, yDesc, (const void *)(uintptr_t)yptr,
            dyDesc, (const void *)(uintptr_t)dyptr,
            beta, dxDesc, (void *)(uintptr_t)dxptr);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ---- Dropout ---- */

    case CHIDORI_CUDNN_CREATE_DROPOUT_DESCRIPTOR: {
        cudnnDropoutDescriptor_t desc;
        cudnnStatus_t st = cudnnCreateDropoutDescriptor(&desc);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire = alloc_handle();
            hmap_put(&g_cudnn_dropout_descs, wire, (uint64_t)(uintptr_t)desc);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire, sizeof(wire));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_SET_DROPOUT_DESCRIPTOR: {
        /* desc(8) + handle(8) + dropout(4) + states(8) + stateSize(8) + seed(8) = 44 */
        if (req->payload_size < 44) {
            return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_BAD_PARAM, NULL, 0);
        }
        char *p = (char *)payload;
        uint64_t wire; memcpy(&wire, p, 8); p += 8;
        uint64_t wh; memcpy(&wh, p, 8); p += 8;
        float dropout; memcpy(&dropout, p, 4); p += 4;
        uint64_t states; memcpy(&states, p, 8); p += 8;
        uint64_t stateSize; memcpy(&stateSize, p, 8); p += 8;
        uint64_t seed; memcpy(&seed, p, 8);

        cudnnDropoutDescriptor_t desc = (cudnnDropoutDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_dropout_descs, wire);
        cudnnHandle_t handle = (cudnnHandle_t)(uintptr_t)hmap_get(&g_cudnn_handles, wh);
        cudnnStatus_t st = cudnnSetDropoutDescriptor(desc, handle, dropout,
            (void *)(uintptr_t)states, (size_t)stateSize, (unsigned long long)seed);
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DESTROY_DROPOUT_DESCRIPTOR: {
        uint64_t wire = 0;
        if (req->payload_size >= 8) memcpy(&wire, payload, 8);
        cudnnDropoutDescriptor_t desc = (cudnnDropoutDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_dropout_descs, wire);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (desc) {
            st = cudnnDestroyDropoutDescriptor(desc);
            hmap_remove(&g_cudnn_dropout_descs, wire);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DROPOUT_FORWARD: {
        /* Stub: dropout forward requires reserveSpace management; return NOT_SUPPORTED for now */
        return chidori_send_response(fd, req->seq_num, (int32_t)CUDNN_STATUS_NOT_SUPPORTED, NULL, 0);
    }

    /* ---- RNN ---- */

    case CHIDORI_CUDNN_CREATE_RNN_DESCRIPTOR: {
        cudnnRNNDescriptor_t desc;
        cudnnStatus_t st = cudnnCreateRNNDescriptor(&desc);
        if (st == CUDNN_STATUS_SUCCESS) {
            uint64_t wire = alloc_handle();
            hmap_put(&g_cudnn_rnn_descs, wire, (uint64_t)(uintptr_t)desc);
            return chidori_send_response(fd, req->seq_num, (int32_t)st, &wire, sizeof(wire));
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    case CHIDORI_CUDNN_DESTROY_RNN_DESCRIPTOR: {
        uint64_t wire = 0;
        if (req->payload_size >= 8) memcpy(&wire, payload, 8);
        cudnnRNNDescriptor_t desc = (cudnnRNNDescriptor_t)(uintptr_t)hmap_get(&g_cudnn_rnn_descs, wire);
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        if (desc) {
            st = cudnnDestroyRNNDescriptor(desc);
            hmap_remove(&g_cudnn_rnn_descs, wire);
        }
        return chidori_send_response(fd, req->seq_num, (int32_t)st, NULL, 0);
    }

    /* ================================================================
     * Default: unsupported
     * ================================================================ */
    default:
        fprintf(stderr, "[cuda_dispatch] Unknown func_id=0x%04x\n", req->func_id);
        return chidori_send_response(fd, req->seq_num, CUDA_ERROR_NOT_SUPPORTED, NULL, 0);
    }
}

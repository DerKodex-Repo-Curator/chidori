/*
 * Juice GPU Worker — Real CUDA Dispatch
 *
 * Handles deserialization of wire protocol messages, calls real CUDA
 * Driver API functions, and serializes responses back.
 *
 * This file is included by gpu_worker.c when building with CUDA support.
 * Compile with: -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
 */

#include <cuda.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "juice_protocol.h"
#include "handle_map.h"

/* Server-side state */
static CUdevice   g_device;
static CUcontext  g_context;
static int        g_initialized = 0;

/* Handle maps for opaque CUDA objects */
static handle_map_t g_modules;
static handle_map_t g_functions;
static handle_map_t g_streams;
static handle_map_t g_events;

static uint64_t g_next_handle = 0x10000;

static uint64_t alloc_handle(void) {
    return g_next_handle++;
}

int cuda_dispatch_init(const char *pcibus) {
    CUresult res;

    hmap_init(&g_modules);
    hmap_init(&g_functions);
    hmap_init(&g_streams);
    hmap_init(&g_events);

    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda_dispatch] cuInit failed: %d\n", res);
        return -1;
    }

    if (pcibus != NULL && pcibus[0] != '\0') {
        res = cuDeviceGetByPCIBusId(&g_device, pcibus);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[cuda_dispatch] cuDeviceGetByPCIBusId(%s) failed: %d, falling back to device 0\n",
                    pcibus, res);
            res = cuDeviceGet(&g_device, 0);
        }
    } else {
        res = cuDeviceGet(&g_device, 0);
    }

    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda_dispatch] cuDeviceGet failed: %d\n", res);
        return -1;
    }

    res = cuDevicePrimaryCtxRetain(&g_context, g_device);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda_dispatch] cuDevicePrimaryCtxRetain failed: %d\n", res);
        return -1;
    }

    res = cuCtxSetCurrent(g_context);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[cuda_dispatch] cuCtxSetCurrent failed: %d\n", res);
        return -1;
    }

    g_initialized = 1;

    char name[256];
    cuDeviceGetName(name, sizeof(name), g_device);
    size_t mem_total;
    cuDeviceTotalMem(&mem_total, g_device);
    fprintf(stderr, "[cuda_dispatch] Initialized: %s (%.1f GB)\n",
            name, (double)mem_total / (1024.0 * 1024.0 * 1024.0));

    return 0;
}

void cuda_dispatch_shutdown(void) {
    if (g_initialized) {
        cuDevicePrimaryCtxRelease(g_device);
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
        /* Report 1 GPU per session (the assigned one) */
        int32_t count = 1;
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &count, sizeof(count));
    }

    case JUICE_CU_DEVICE_GET: {
        int32_t device = (int32_t)g_device;
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &device, sizeof(device));
    }

    case JUICE_CU_DEVICE_GET_NAME: {
        char name[256];
        res = cuDeviceGetName(name, sizeof(name), g_device);
        return juice_send_response(fd, req->seq_num, res, name, strlen(name) + 1);
    }

    case JUICE_CU_DEVICE_TOTAL_MEM: {
        size_t bytes;
        res = cuDeviceTotalMem(&bytes, g_device);
        uint64_t bytes64 = (uint64_t)bytes;
        return juice_send_response(fd, req->seq_num, res, &bytes64, sizeof(bytes64));
    }

    case JUICE_CU_DEVICE_GET_ATTRIBUTE: {
        int32_t attrib = 0, dev = 0;
        if (req->payload_size >= 8) {
            memcpy(&attrib, payload, 4);
            memcpy(&dev, (char *)payload + 4, 4);
        }
        int value;
        res = cuDeviceGetAttribute(&value, (CUdevice_attribute)attrib, g_device);
        return juice_send_response(fd, req->seq_num, res, &value, sizeof(value));
    }

    case JUICE_CU_DEVICE_GET_UUID: {
        CUuuid uuid;
        res = cuDeviceGetUuid(&uuid, g_device);
        return juice_send_response(fd, req->seq_num, res, &uuid, sizeof(uuid));
    }

    case JUICE_CU_DEVICE_GET_PROPERTIES: {
        /* Deprecated but still used by some apps */
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);
    }

    case JUICE_CU_DEVICE_COMPUTE_CAPABILITY: {
        int major = 0, minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_device);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_device);
        struct { int32_t major; int32_t minor; } cc = { major, minor };
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &cc, sizeof(cc));
    }

    /* ================================================================
     * Context Management
     * ================================================================ */
    case JUICE_CU_CTX_CREATE:
    case JUICE_CU_DEVICE_PRIMARY_CTX_RETAIN: {
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_context;
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case JUICE_CU_CTX_DESTROY:
    case JUICE_CU_DEVICE_PRIMARY_CTX_RELEASE:
    case JUICE_CU_DEVICE_PRIMARY_CTX_RESET:
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, NULL, 0);

    case JUICE_CU_CTX_SET_CURRENT:
        res = cuCtxSetCurrent(g_context);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);

    case JUICE_CU_CTX_GET_CURRENT: {
        uint64_t ctx_handle = (uint64_t)(uintptr_t)g_context;
        return juice_send_response(fd, req->seq_num, CUDA_SUCCESS, &ctx_handle, sizeof(ctx_handle));
    }

    case JUICE_CU_CTX_PUSH_CURRENT:
        res = cuCtxPushCurrent(g_context);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);

    case JUICE_CU_CTX_POP_CURRENT: {
        CUcontext ctx;
        res = cuCtxPopCurrent(&ctx);
        return juice_send_response(fd, req->seq_num, res, NULL, 0);
    }

    case JUICE_CU_CTX_SYNCHRONIZE:
        res = cuCtxSynchronize();
        return juice_send_response(fd, req->seq_num, res, NULL, 0);

    case JUICE_CU_CTX_GET_DEVICE: {
        int32_t dev = (int32_t)g_device;
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
         *   + num_params(4) + param_sizes[num_params](4 each)
         *   + param_data (concatenated)
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
        void **kernel_params = NULL;
        int num_params = 0;

        size_t remaining = req->payload_size - 44;
        if (remaining >= 4) {
            memcpy(&num_params, p, 4); p += 4;
            remaining -= 4;

            if (num_params > 0 && num_params < 256) {
                uint32_t *param_sizes = (uint32_t *)malloc(num_params * sizeof(uint32_t));
                kernel_params = (void **)malloc(num_params * sizeof(void *));

                for (int i = 0; i < num_params && remaining >= 4; i++) {
                    memcpy(&param_sizes[i], p, 4); p += 4;
                    remaining -= 4;
                }

                for (int i = 0; i < num_params; i++) {
                    kernel_params[i] = p;
                    p += param_sizes[i];
                }

                free(param_sizes);
            }
        }

        res = cuLaunchKernel(func,
                             gridX, gridY, gridZ,
                             blockX, blockY, blockZ,
                             shared, stream,
                             kernel_params, NULL);

        free(kernel_params);
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
     * Default: unsupported
     * ================================================================ */
    default:
        fprintf(stderr, "[cuda_dispatch] Unknown func_id=0x%04x\n", req->func_id);
        return juice_send_response(fd, req->seq_num, CUDA_ERROR_NOT_SUPPORTED, NULL, 0);
    }
}

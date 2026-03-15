/*
 * Juice CUDA Client Interception Library (libjuice-cuda.so)
 *
 * This shared library is loaded via LD_PRELOAD to intercept CUDA Driver API
 * calls. Each intercepted call is serialized and sent over TCP to the
 * juice-gpu-worker running on the remote GPU host.
 *
 * The library reads connection info from JUICE_CFG_OVERRIDE env var
 * (set by juicify) to know which server to connect to.
 *
 * Phase 0: Stub implementation with basic protocol integration.
 * Phase 2: Full CUDA Driver API interception via codegen.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>

#include "juice_protocol.h"

/* CUDA types (minimal definitions to avoid requiring cuda.h on client) */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_INVALID_VALUE 1

/* Client state */
static int g_server_fd = -1;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_seq_num = 0;
static int g_initialized = 0;

/*
 * Connect to the Juice GPU worker.
 * Parses JUICE_CFG_OVERRIDE for server address, or falls back to
 * connecting to the Agent's address directly.
 */
static int juice_connect(void) {
    if (g_server_fd >= 0)
        return 0;

    /* Read configuration from environment */
    const char *cfg = getenv("JUICE_CFG_OVERRIDE");
    const char *host = "127.0.0.1";
    int port = 43210;

    if (cfg != NULL) {
        /*
         * JUICE_CFG_OVERRIDE is a JSON string like:
         * {"id":"...","servers":["host:port"],...}
         *
         * For now, do minimal parsing to extract the first server address.
         * TODO: Use a proper JSON parser.
         */
        const char *servers = strstr(cfg, "\"servers\":[\"");
        if (servers) {
            servers += strlen("\"servers\":[\"");
            char addr_buf[256];
            int i = 0;
            while (*servers && *servers != '"' && i < (int)sizeof(addr_buf) - 1) {
                addr_buf[i++] = *servers++;
            }
            addr_buf[i] = '\0';

            /* Split host:port */
            char *colon = strrchr(addr_buf, ':');
            if (colon) {
                *colon = '\0';
                host = strdup(addr_buf);
                port = atoi(colon + 1);
            }
        }
    }

    fprintf(stderr, "[juice-cuda] Connecting to %s:%d\n", host, port);

    /* Resolve and connect */
    struct addrinfo hints = {0}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    int rc = getaddrinfo(host, port_str, &hints, &res);
    if (rc != 0) {
        fprintf(stderr, "[juice-cuda] getaddrinfo: %s\n", gai_strerror(rc));
        return -1;
    }

    g_server_fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (g_server_fd < 0) {
        perror("[juice-cuda] socket");
        freeaddrinfo(res);
        return -1;
    }

    /* TCP_NODELAY for low latency */
    int flag = 1;
    setsockopt(g_server_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    if (connect(g_server_fd, res->ai_addr, res->ai_addrlen) < 0) {
        perror("[juice-cuda] connect");
        close(g_server_fd);
        g_server_fd = -1;
        freeaddrinfo(res);
        return -1;
    }

    freeaddrinfo(res);
    fprintf(stderr, "[juice-cuda] Connected to GPU worker\n");
    return 0;
}

/*
 * Send an RPC and receive the response.
 * Thread-safe via mutex (simple approach; could use per-thread connections later).
 */
static CUresult juice_rpc(uint16_t func_id, const void *payload, size_t payload_len,
                          void *out_payload, size_t *out_payload_len) {
    pthread_mutex_lock(&g_lock);

    if (g_server_fd < 0) {
        if (juice_connect() != 0) {
            pthread_mutex_unlock(&g_lock);
            return CUDA_ERROR_NOT_INITIALIZED;
        }
    }

    /* Build and send request */
    juice_request_t req = {0};
    req.func_id = func_id;
    req.flags = JUICE_FLAG_SYNC;
    req.seq_num = ++g_seq_num;

    if (juice_send_request(g_server_fd, &req, payload, payload_len) != 0) {
        fprintf(stderr, "[juice-cuda] Failed to send request for func=0x%04x\n", func_id);
        close(g_server_fd);
        g_server_fd = -1;
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    /* Receive response */
    juice_response_t resp;
    if (juice_recv_response(g_server_fd, &resp) != 0) {
        fprintf(stderr, "[juice-cuda] Failed to receive response for func=0x%04x\n", func_id);
        close(g_server_fd);
        g_server_fd = -1;
        pthread_mutex_unlock(&g_lock);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    /* Read response payload if present */
    size_t resp_payload_len = resp.msg_len - JUICE_RESPONSE_HEADER_SIZE;
    if (resp_payload_len > 0 && out_payload != NULL) {
        size_t to_read = resp_payload_len;
        if (out_payload_len && *out_payload_len < to_read)
            to_read = *out_payload_len;

        if (juice_read_full(g_server_fd, out_payload, to_read) != 0) {
            close(g_server_fd);
            g_server_fd = -1;
            pthread_mutex_unlock(&g_lock);
            return CUDA_ERROR_NOT_INITIALIZED;
        }

        /* Drain any extra bytes we didn't consume */
        if (to_read < resp_payload_len) {
            char drain[4096];
            size_t remaining = resp_payload_len - to_read;
            while (remaining > 0) {
                size_t chunk = remaining < sizeof(drain) ? remaining : sizeof(drain);
                juice_read_full(g_server_fd, drain, chunk);
                remaining -= chunk;
            }
        }

        if (out_payload_len)
            *out_payload_len = to_read;
    } else if (resp_payload_len > 0) {
        /* Caller doesn't want payload, drain it */
        char drain[4096];
        size_t remaining = resp_payload_len;
        while (remaining > 0) {
            size_t chunk = remaining < sizeof(drain) ? remaining : sizeof(drain);
            juice_read_full(g_server_fd, drain, chunk);
            remaining -= chunk;
        }
    }

    pthread_mutex_unlock(&g_lock);
    return (CUresult)resp.result;
}

/*
 * ============================================================================
 * CUDA Driver API Interceptions
 *
 * These are the symbols that applications (and cuBLAS, cuDNN, etc.) link against.
 * By providing them in our LD_PRELOAD library, we intercept all CUDA calls.
 *
 * Phase 0: Core functions implemented below.
 * Phase 2: Codegen will auto-generate the full set (~100+ functions).
 * ============================================================================
 */

CUresult cuInit(unsigned int flags) {
    (void)flags;
    g_initialized = 1;
    return juice_rpc(JUICE_CU_INIT, &flags, sizeof(flags), NULL, NULL);
}

CUresult cuDriverGetVersion(int *version) {
    size_t len = sizeof(int32_t);
    CUresult r = juice_rpc(JUICE_CU_DRIVER_GET_VERSION, NULL, 0, version, &len);
    return r;
}

CUresult cuDeviceGetCount(int *count) {
    size_t len = sizeof(int32_t);
    return juice_rpc(JUICE_CU_DEVICE_GET_COUNT, NULL, 0, count, &len);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    size_t len = sizeof(int32_t);
    return juice_rpc(JUICE_CU_DEVICE_GET, &ordinal, sizeof(ordinal), device, &len);
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    struct { int32_t len; int32_t dev; } req = { len, dev };
    size_t out_len = (size_t)len;
    return juice_rpc(JUICE_CU_DEVICE_GET_NAME, &req, sizeof(req), name, &out_len);
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    size_t len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_DEVICE_TOTAL_MEM, &dev, sizeof(dev), bytes, &len);
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
    struct { int32_t attrib; int32_t dev; } req = { attrib, dev };
    size_t len = sizeof(int32_t);
    return juice_rpc(JUICE_CU_DEVICE_GET_ATTRIBUTE, &req, sizeof(req), pi, &len);
}

/* Context */
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    struct { uint32_t flags; int32_t dev; } req = { flags, dev };
    size_t len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_CTX_CREATE, &req, sizeof(req), pctx, &len);
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
    uint64_t handle = (uint64_t)(uintptr_t)ctx;
    return juice_rpc(JUICE_CU_CTX_DESTROY, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
    uint64_t handle = (uint64_t)(uintptr_t)ctx;
    return juice_rpc(JUICE_CU_CTX_SET_CURRENT, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
    size_t len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_CTX_GET_CURRENT, NULL, 0, pctx, &len);
}

CUresult cuCtxSynchronize(void) {
    return juice_rpc(JUICE_CU_CTX_SYNCHRONIZE, NULL, 0, NULL, NULL);
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    size_t len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_DEVICE_PRIMARY_CTX_RETAIN, &dev, sizeof(dev), pctx, &len);
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    return juice_rpc(JUICE_CU_DEVICE_PRIMARY_CTX_RELEASE, &dev, sizeof(dev), NULL, NULL);
}

/* Memory */
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    uint64_t size = (uint64_t)bytesize;
    size_t len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_MEM_ALLOC, &size, sizeof(size), dptr, &len);
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    uint64_t ptr = (uint64_t)dptr;
    return juice_rpc(JUICE_CU_MEM_FREE, &ptr, sizeof(ptr), NULL, NULL);
}

CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t byteCount) {
    /* Payload: dst_ptr(8) + size(8) + data(byteCount) */
    size_t payload_len = 16 + byteCount;
    void *payload = malloc(payload_len);
    if (!payload) return CUDA_ERROR_INVALID_VALUE;

    memcpy(payload, &dstDevice, 8);
    uint64_t size64 = (uint64_t)byteCount;
    memcpy((char *)payload + 8, &size64, 8);
    memcpy((char *)payload + 16, srcHost, byteCount);

    CUresult r = juice_rpc(JUICE_CU_MEMCPY_H_TO_D, payload, payload_len, NULL, NULL);
    free(payload);
    return r;
}

CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t byteCount) {
    /* Payload: src_ptr(8) + size(8) */
    struct { uint64_t src; uint64_t size; } req;
    req.src = (uint64_t)srcDevice;
    req.size = (uint64_t)byteCount;

    size_t out_len = byteCount;
    return juice_rpc(JUICE_CU_MEMCPY_D_TO_H, &req, sizeof(req), dstHost, &out_len);
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount) {
    struct { uint64_t dst; uint64_t src; uint64_t size; } req;
    req.dst = (uint64_t)dstDevice;
    req.src = (uint64_t)srcDevice;
    req.size = (uint64_t)byteCount;
    return juice_rpc(JUICE_CU_MEMCPY_D_TO_D, &req, sizeof(req), NULL, NULL);
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    struct { uint64_t free; uint64_t total; } info;
    size_t len = sizeof(info);
    CUresult r = juice_rpc(JUICE_CU_MEM_GET_INFO, NULL, 0, &info, &len);
    if (r == CUDA_SUCCESS) {
        *free = (size_t)info.free;
        *total = (size_t)info.total;
    }
    return r;
}

/* Module */
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    /* Send the PTX/CUBIN data to the server */
    size_t image_len = strlen((const char *)image) + 1; /* For PTX; CUBIN needs header parsing */
    size_t out_len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_MODULE_LOAD_DATA, image, image_len, module, &out_len);
}

CUresult cuModuleUnload(CUmodule mod) {
    uint64_t handle = (uint64_t)(uintptr_t)mod;
    return juice_rpc(JUICE_CU_MODULE_UNLOAD, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    /* Payload: module_handle(8) + function_name (null-terminated) */
    uint64_t mod_handle = (uint64_t)(uintptr_t)hmod;
    size_t name_len = strlen(name) + 1;
    size_t payload_len = 8 + name_len;
    void *payload = malloc(payload_len);
    if (!payload) return CUDA_ERROR_INVALID_VALUE;

    memcpy(payload, &mod_handle, 8);
    memcpy((char *)payload + 8, name, name_len);

    size_t out_len = sizeof(uint64_t);
    CUresult r = juice_rpc(JUICE_CU_MODULE_GET_FUNCTION, payload, payload_len, hfunc, &out_len);
    free(payload);
    return r;
}

/* Execution */
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
    /*
     * Kernel launch serialization:
     *   func_handle(8) + grid(12) + block(12) + shared(4) + stream(8) = 44 bytes header
     *   + serialized kernel parameters
     *
     * TODO Phase 2: Properly serialize kernelParams and extra.
     * For now, send the header only (stub mode).
     */
    (void)kernelParams;
    (void)extra;

    struct __attribute__((packed)) {
        uint64_t func;
        uint32_t gridX, gridY, gridZ;
        uint32_t blockX, blockY, blockZ;
        uint32_t shared;
        uint64_t stream;
    } req;

    req.func = (uint64_t)(uintptr_t)f;
    req.gridX = gridDimX;  req.gridY = gridDimY;  req.gridZ = gridDimZ;
    req.blockX = blockDimX; req.blockY = blockDimY; req.blockZ = blockDimZ;
    req.shared = sharedMemBytes;
    req.stream = (uint64_t)(uintptr_t)hStream;

    return juice_rpc(JUICE_CU_LAUNCH_KERNEL, &req, sizeof(req), NULL, NULL);
}

/* Stream */
CUresult cuStreamCreate(CUstream *phStream, unsigned int flags) {
    size_t len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_STREAM_CREATE, &flags, sizeof(flags), phStream, &len);
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
    uint64_t handle = (uint64_t)(uintptr_t)hStream;
    return juice_rpc(JUICE_CU_STREAM_DESTROY, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuStreamSynchronize(CUstream hStream) {
    uint64_t handle = (uint64_t)(uintptr_t)hStream;
    return juice_rpc(JUICE_CU_STREAM_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
}

/* Event */
CUresult cuEventCreate(CUevent *phEvent, unsigned int flags) {
    size_t len = sizeof(uint64_t);
    return juice_rpc(JUICE_CU_EVENT_CREATE, &flags, sizeof(flags), phEvent, &len);
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
    uint64_t handle = (uint64_t)(uintptr_t)hEvent;
    return juice_rpc(JUICE_CU_EVENT_DESTROY, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    struct { uint64_t event; uint64_t stream; } req;
    req.event = (uint64_t)(uintptr_t)hEvent;
    req.stream = (uint64_t)(uintptr_t)hStream;
    return juice_rpc(JUICE_CU_EVENT_RECORD, &req, sizeof(req), NULL, NULL);
}

CUresult cuEventSynchronize(CUevent hEvent) {
    uint64_t handle = (uint64_t)(uintptr_t)hEvent;
    return juice_rpc(JUICE_CU_EVENT_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    struct { uint64_t start; uint64_t end; } req;
    req.start = (uint64_t)(uintptr_t)hStart;
    req.end = (uint64_t)(uintptr_t)hEnd;
    size_t len = sizeof(float);
    return juice_rpc(JUICE_CU_EVENT_ELAPSED_TIME, &req, sizeof(req), pMilliseconds, &len);
}

/*
 * Cleanup on library unload
 */
__attribute__((destructor))
static void juice_cleanup(void) {
    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }
}

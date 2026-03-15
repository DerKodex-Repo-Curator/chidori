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
#include <sys/time.h>
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
/*
 * Extract a JSON string value by key (minimal parser).
 * Returns pointer to malloc'd string or NULL.
 */
static char *json_get_string(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *start = strstr(json, search);
    if (!start) return NULL;
    start += strlen(search);
    const char *end = strchr(start, '"');
    if (!end) return NULL;
    size_t len = (size_t)(end - start);
    char *val = (char *)malloc(len + 1);
    memcpy(val, start, len);
    val[len] = '\0';
    return val;
}

static char *json_get_array_first(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":[\"", key);
    const char *start = strstr(json, search);
    if (!start) return NULL;
    start += strlen(search);
    const char *end = strchr(start, '"');
    if (!end) return NULL;
    size_t len = (size_t)(end - start);
    char *val = (char *)malloc(len + 1);
    memcpy(val, start, len);
    val[len] = '\0';
    return val;
}

static int tcp_connect(const char *host, int port) {
    struct addrinfo hints = {0}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    int rc = getaddrinfo(host, port_str, &hints, &res);
    if (rc != 0) {
        fprintf(stderr, "[juice-cuda] getaddrinfo(%s): %s\n", host, gai_strerror(rc));
        return -1;
    }

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
        perror("[juice-cuda] socket");
        freeaddrinfo(res);
        return -1;
    }

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        perror("[juice-cuda] connect");
        close(fd);
        freeaddrinfo(res);
        return -1;
    }

    freeaddrinfo(res);
    return fd;
}

static int juice_connect(void) {
    if (g_server_fd >= 0)
        return 0;

    const char *cfg = getenv("JUICE_CFG_OVERRIDE");
    if (!cfg) {
        fprintf(stderr, "[juice-cuda] JUICE_CFG_OVERRIDE not set\n");
        return -1;
    }

    /* Parse session ID and server address from config */
    char *session_id = json_get_string(cfg, "id");
    char *server = json_get_array_first(cfg, "servers");

    if (!session_id || !server) {
        fprintf(stderr, "[juice-cuda] Missing id or servers in config\n");
        free(session_id);
        free(server);
        return -1;
    }

    /* Split host:port */
    char host[256] = "127.0.0.1";
    int port = 43210;
    char *colon = strrchr(server, ':');
    if (colon) {
        *colon = '\0';
        strncpy(host, server, sizeof(host) - 1);
        port = atoi(colon + 1);
    } else {
        strncpy(host, server, sizeof(host) - 1);
    }

    /*
     * Check if JUICE_DIRECT_CONNECT is set — if so, connect directly to
     * juice-gpu-worker's standalone TCP port (no HTTP/agent middleman).
     */
    const char *direct = getenv("JUICE_DIRECT_CONNECT");

    if (direct && direct[0] == '1') {
        /* Direct TCP connection to gpu-worker standalone mode */
        fprintf(stderr, "[juice-cuda] Direct connect to gpu-worker at %s:%d\n", host, port);

        g_server_fd = tcp_connect(host, port);
        if (g_server_fd < 0) {
            free(session_id);
            free(server);
            return -1;
        }

        fprintf(stderr, "[juice-cuda] Connected directly to GPU worker\n");
    } else {
        /* Agent-mediated connection via HTTP session protocol */
        fprintf(stderr, "[juice-cuda] Connecting to agent at %s:%d (session %s)\n",
                host, port, session_id);

        g_server_fd = tcp_connect(host, port);
        if (g_server_fd < 0) {
            free(session_id);
            free(server);
            return -1;
        }

        char http_req[2048];
        const char *body = "{\"id\":\"conn-1\",\"pid\":\"0\",\"processName\":\"juice-client\"}";
        int body_len = (int)strlen(body);

        int req_len = snprintf(http_req, sizeof(http_req),
            "POST /v1/connect/session/%s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %d\r\n"
            "Connection: Upgrade\r\n"
            "\r\n"
            "%s",
            session_id, host, port, body_len, body);

        if (juice_write_full(g_server_fd, http_req, (size_t)req_len) != 0) {
            fprintf(stderr, "[juice-cuda] Failed to send HTTP connect request\n");
            close(g_server_fd);
            g_server_fd = -1;
            free(session_id);
            free(server);
            return -1;
        }

        /*
         * Read the HTTP 101 Switching Protocols response from the agent.
         * This confirms the agent has hijacked the connection and forwarded
         * the socket to gpu-worker.  We read byte-by-byte looking for the
         * end-of-headers marker "\r\n\r\n".  After this point the TCP
         * stream carries raw Juice CUDA protocol.
         *
         * Timeout: we give the agent up to 30 seconds to respond (CUDA
         * init on the worker side can take a while on first launch).
         */
        {
            struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
            setsockopt(g_server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

            char resp_buf[1024];
            int resp_len = 0;
            int found_end = 0;

            while (resp_len < (int)sizeof(resp_buf) - 1) {
                ssize_t n = read(g_server_fd, resp_buf + resp_len, 1);
                if (n <= 0) {
                    fprintf(stderr, "[juice-cuda] Failed to read HTTP response from agent (n=%zd errno=%d)\n", n, errno);
                    close(g_server_fd);
                    g_server_fd = -1;
                    free(session_id);
                    free(server);
                    return -1;
                }
                resp_len += (int)n;
                /* Look for \r\n\r\n end-of-headers */
                if (resp_len >= 4 &&
                    resp_buf[resp_len - 4] == '\r' &&
                    resp_buf[resp_len - 3] == '\n' &&
                    resp_buf[resp_len - 2] == '\r' &&
                    resp_buf[resp_len - 1] == '\n') {
                    found_end = 1;
                    break;
                }
            }

            /* Clear the receive timeout for subsequent CUDA RPC traffic */
            tv.tv_sec = 0;
            tv.tv_usec = 0;
            setsockopt(g_server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

            if (!found_end) {
                fprintf(stderr, "[juice-cuda] Invalid HTTP response from agent\n");
                close(g_server_fd);
                g_server_fd = -1;
                free(session_id);
                free(server);
                return -1;
            }

            resp_buf[resp_len] = '\0';
            if (strstr(resp_buf, "101") == NULL) {
                fprintf(stderr, "[juice-cuda] Agent did not return 101: %s\n", resp_buf);
                close(g_server_fd);
                g_server_fd = -1;
                free(session_id);
                free(server);
                return -1;
            }
        }

        fprintf(stderr, "[juice-cuda] Connected to remote GPU (session %s)\n", session_id);
    }

    free(session_id);
    free(server);
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

/*
 * CUDA extra parameter sentinel values.
 * Defined as ((void*)0x01) etc. in cuda.h; redefined here to avoid
 * requiring cuda.h on the client side.
 */
#define JUICE_CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define JUICE_CU_LAUNCH_PARAM_BUFFER_SIZE    ((void *)0x02)
#define JUICE_CU_LAUNCH_PARAM_END            ((void *)0x00)

CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    struct { int32_t attrib; uint64_t func; } req;
    req.attrib = attrib;
    req.func = (uint64_t)(uintptr_t)hfunc;
    size_t len = sizeof(int32_t);
    return juice_rpc(JUICE_CU_FUNC_GET_ATTRIBUTE, &req, sizeof(req), pi, &len);
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
    /*
     * Kernel launch serialization — wire format:
     *   func_handle(8) + gridX(4) + gridY(4) + gridZ(4)
     *   + blockX(4) + blockY(4) + blockZ(4)
     *   + sharedMem(4) + stream_handle(8) = 44 bytes header
     *   + has_params(1):
     *       0 = no kernel parameters
     *       1 = contiguous parameter buffer follows
     *   + [if has_params == 1]: buffer_size(4) + buffer_data(buffer_size)
     *
     * On the server side, the buffer is passed via CU_LAUNCH_PARAM_BUFFER_POINTER
     * and CU_LAUNCH_PARAM_BUFFER_SIZE through cuLaunchKernel's extra parameter.
     */

    /* --- Extract kernel parameter buffer --- */
    void *param_buffer = NULL;
    size_t param_buffer_size = 0;
    int param_buffer_allocated = 0; /* 1 if we malloc'd param_buffer */

    if (extra != NULL) {
        /*
         * The extra array contains tagged key-value pairs:
         *   CU_LAUNCH_PARAM_BUFFER_POINTER, <ptr>,
         *   CU_LAUNCH_PARAM_BUFFER_SIZE, <size_t*>,
         *   CU_LAUNCH_PARAM_END
         */
        for (int i = 0; extra[i] != JUICE_CU_LAUNCH_PARAM_END; i++) {
            if (extra[i] == JUICE_CU_LAUNCH_PARAM_BUFFER_POINTER) {
                param_buffer = extra[++i];
            } else if (extra[i] == JUICE_CU_LAUNCH_PARAM_BUFFER_SIZE) {
                param_buffer_size = *(size_t *)extra[++i];
            }
        }
    } else if (kernelParams != NULL) {
        /*
         * kernelParams is an array of pointers to individual kernel arguments.
         * We query CU_FUNC_ATTRIBUTE_PARAM_SIZE (enum value 74) to get the
         * total parameter buffer size, then pack arguments contiguously.
         *
         * CUDA kernel parameters are laid out in declaration order with each
         * argument aligned to its natural alignment (max 8 bytes). Since all
         * CUDA kernel arguments are scalar types (1/2/4/8 bytes) or pointers
         * (8 bytes), we iterate and copy each as an 8-byte-aligned value.
         *
         * Note: this issues an extra RPC to query param size. Applications
         * that care about launch latency should use the extra parameter approach.
         */
        int total_param_size = 0;
        CUresult attr_res = cuFuncGetAttribute(&total_param_size,
                                                74 /* CU_FUNC_ATTRIBUTE_PARAM_SIZE */, f);
        if (attr_res == CUDA_SUCCESS && total_param_size > 0) {
            param_buffer_size = (size_t)total_param_size;
            param_buffer = malloc(param_buffer_size);
            if (!param_buffer) return CUDA_ERROR_INVALID_VALUE;
            param_buffer_allocated = 1;
            memset(param_buffer, 0, param_buffer_size);

            /*
             * Pack each kernelParams[i] into the contiguous buffer.
             * We copy 8 bytes per parameter (the largest scalar/pointer size)
             * with 8-byte alignment. If the total param buffer is smaller
             * than expected, we stop early. This handles the common case
             * where all args are pointers or 64-bit values.
             *
             * For kernels with smaller argument types (int, float), the CUDA
             * driver zero-extends them in the parameter buffer, so copying
             * 8 bytes from the caller's pointer is safe as long as the
             * caller's storage is at least 8 bytes (which is typical for
             * stack variables on 64-bit systems).
             */
            size_t offset = 0;
            for (int i = 0; offset < param_buffer_size && kernelParams[i] != NULL; i++) {
                /* Align to 8-byte boundary */
                offset = (offset + 7) & ~(size_t)7;
                if (offset >= param_buffer_size) break;
                size_t remaining = param_buffer_size - offset;
                size_t copy_size = remaining >= 8 ? 8 : remaining;
                memcpy((char *)param_buffer + offset, kernelParams[i], copy_size);
                offset += copy_size;
            }
        }
    }

    /* --- Build the wire payload --- */
    uint8_t has_params = (param_buffer != NULL && param_buffer_size > 0) ? 1 : 0;
    size_t header_size = 44; /* func(8) + grid(12) + block(12) + shared(4) + stream(8) */
    size_t payload_len = header_size + 1; /* + has_params(1) */
    if (has_params) {
        payload_len += 4 + param_buffer_size; /* buffer_size(4) + buffer_data */
    }

    void *payload = malloc(payload_len);
    if (!payload) {
        if (param_buffer_allocated) free(param_buffer);
        return CUDA_ERROR_INVALID_VALUE;
    }

    char *p = (char *)payload;

    /* Header: func_handle(8) + grid(12) + block(12) + shared(4) + stream(8) */
    uint64_t func_handle = (uint64_t)(uintptr_t)f;
    uint64_t stream_handle = (uint64_t)(uintptr_t)hStream;
    memcpy(p, &func_handle, 8); p += 8;
    memcpy(p, &gridDimX, 4); p += 4;
    memcpy(p, &gridDimY, 4); p += 4;
    memcpy(p, &gridDimZ, 4); p += 4;
    memcpy(p, &blockDimX, 4); p += 4;
    memcpy(p, &blockDimY, 4); p += 4;
    memcpy(p, &blockDimZ, 4); p += 4;
    memcpy(p, &sharedMemBytes, 4); p += 4;
    memcpy(p, &stream_handle, 8); p += 8;

    /* Kernel parameter buffer */
    memcpy(p, &has_params, 1); p += 1;
    if (has_params) {
        uint32_t buf_size = (uint32_t)param_buffer_size;
        memcpy(p, &buf_size, 4); p += 4;
        memcpy(p, param_buffer, param_buffer_size);
    }

    CUresult r = juice_rpc(JUICE_CU_LAUNCH_KERNEL, payload, payload_len, NULL, NULL);

    free(payload);
    if (param_buffer_allocated) free(param_buffer);

    return r;
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
 * ============================================================================
 * CUDA Runtime API Interceptions
 *
 * These intercept the cudaXxx() functions that PyTorch, TensorFlow, and other
 * frameworks call. cudaError_t is typedef'd as int, same wire format as CUresult.
 * ============================================================================
 */

typedef int cudaError_t;
#define cudaSuccess 0

/* cudaMemcpyKind enum values */
#define cudaMemcpyHostToHost     0
#define cudaMemcpyHostToDevice   1
#define cudaMemcpyDeviceToHost   2
#define cudaMemcpyDeviceToDevice 3

/* Track last error locally */
static cudaError_t g_last_error = 0;

cudaError_t cudaGetDeviceCount(int *count) {
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_GET_DEVICE_COUNT, NULL, 0, count, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaGetDevice(int *device) {
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_GET_DEVICE, NULL, 0, device, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaSetDevice(int device) {
    int32_t dev = device;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_SET_DEVICE, &dev, sizeof(dev), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaGetDeviceProperties(void *prop, int device) {
    /*
     * cudaDeviceProp is ~700 bytes. We send the device ordinal and receive
     * the full serialized struct from the server.
     */
    int32_t dev = device;
    size_t len = 1024; /* cudaDeviceProp is large; allow generous buffer */
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_GET_DEVICE_PROPERTIES,
                                            &dev, sizeof(dev), prop, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
    struct { int32_t attr; int32_t device; } req = { attr, device };
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_DEVICE_GET_ATTRIBUTE,
                                            &req, sizeof(req), value, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    uint64_t size64 = (uint64_t)size;
    uint64_t ptr64 = 0;
    size_t len = sizeof(uint64_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_MALLOC, &size64, sizeof(size64), &ptr64, &len);
    if (r == cudaSuccess)
        *devPtr = (void *)(uintptr_t)ptr64;
    g_last_error = r;
    return r;
}

cudaError_t cudaFree(void *devPtr) {
    uint64_t ptr = (uint64_t)(uintptr_t)devPtr;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_FREE, &ptr, sizeof(ptr), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
    cudaError_t r;

    switch (kind) {
    case cudaMemcpyHostToDevice: {
        /* Payload: dst_ptr(8) + size(8) + kind(4) + data(count) */
        size_t payload_len = 8 + 8 + 4 + count;
        void *payload = malloc(payload_len);
        if (!payload) { g_last_error = CUDA_ERROR_INVALID_VALUE; return CUDA_ERROR_INVALID_VALUE; }

        uint64_t dst64 = (uint64_t)(uintptr_t)dst;
        uint64_t size64 = (uint64_t)count;
        int32_t kind32 = kind;
        memcpy(payload, &dst64, 8);
        memcpy((char *)payload + 8, &size64, 8);
        memcpy((char *)payload + 16, &kind32, 4);
        memcpy((char *)payload + 20, src, count);

        r = (cudaError_t)juice_rpc(JUICE_CUDA_MEMCPY, payload, payload_len, NULL, NULL);
        free(payload);
        break;
    }
    case cudaMemcpyDeviceToHost: {
        /* Payload: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) */
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;

        size_t out_len = count;
        r = (cudaError_t)juice_rpc(JUICE_CUDA_MEMCPY, &req, sizeof(req), dst, &out_len);
        break;
    }
    case cudaMemcpyDeviceToDevice: {
        /* Payload: dst_ptr(8) + src_ptr(8) + size(8) + kind(4) */
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;

        r = (cudaError_t)juice_rpc(JUICE_CUDA_MEMCPY, &req, sizeof(req), NULL, NULL);
        break;
    }
    case cudaMemcpyHostToHost:
    default:
        /* HtoH: just memcpy locally */
        memcpy(dst, src, count);
        r = cudaSuccess;
        break;
    }

    g_last_error = r;
    return r;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind, void *stream) {
    cudaError_t r;

    switch (kind) {
    case cudaMemcpyHostToDevice: {
        /* Payload: dst_ptr(8) + size(8) + kind(4) + stream(8) + data(count) */
        size_t payload_len = 8 + 8 + 4 + 8 + count;
        void *payload = malloc(payload_len);
        if (!payload) { g_last_error = CUDA_ERROR_INVALID_VALUE; return CUDA_ERROR_INVALID_VALUE; }

        uint64_t dst64 = (uint64_t)(uintptr_t)dst;
        uint64_t size64 = (uint64_t)count;
        int32_t kind32 = kind;
        uint64_t stream64 = (uint64_t)(uintptr_t)stream;
        char *p = (char *)payload;
        memcpy(p, &dst64, 8); p += 8;
        memcpy(p, &size64, 8); p += 8;
        memcpy(p, &kind32, 4); p += 4;
        memcpy(p, &stream64, 8); p += 8;
        memcpy(p, src, count);

        r = (cudaError_t)juice_rpc(JUICE_CUDA_MEMCPY_ASYNC, payload, payload_len, NULL, NULL);
        free(payload);
        break;
    }
    case cudaMemcpyDeviceToHost: {
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
            uint64_t stream;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;
        req.stream = (uint64_t)(uintptr_t)stream;

        size_t out_len = count;
        r = (cudaError_t)juice_rpc(JUICE_CUDA_MEMCPY_ASYNC, &req, sizeof(req), dst, &out_len);
        break;
    }
    case cudaMemcpyDeviceToDevice: {
        struct __attribute__((packed)) {
            uint64_t dst;
            uint64_t src;
            uint64_t size;
            int32_t  kind;
            uint64_t stream;
        } req;
        req.dst = (uint64_t)(uintptr_t)dst;
        req.src = (uint64_t)(uintptr_t)src;
        req.size = (uint64_t)count;
        req.kind = kind;
        req.stream = (uint64_t)(uintptr_t)stream;

        r = (cudaError_t)juice_rpc(JUICE_CUDA_MEMCPY_ASYNC, &req, sizeof(req), NULL, NULL);
        break;
    }
    case cudaMemcpyHostToHost:
    default:
        memcpy(dst, src, count);
        r = cudaSuccess;
        break;
    }

    g_last_error = r;
    return r;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    struct __attribute__((packed)) {
        uint64_t ptr;
        int32_t  value;
        uint64_t count;
    } req;
    req.ptr = (uint64_t)(uintptr_t)devPtr;
    req.value = value;
    req.count = (uint64_t)count;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_MEMSET, &req, sizeof(req), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaDeviceSynchronize(void) {
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_DEVICE_SYNCHRONIZE, NULL, 0, NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaStreamCreate(void **pStream) {
    uint64_t handle = 0;
    size_t len = sizeof(uint64_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_STREAM_CREATE, NULL, 0, &handle, &len);
    if (r == cudaSuccess)
        *pStream = (void *)(uintptr_t)handle;
    g_last_error = r;
    return r;
}

cudaError_t cudaStreamDestroy(void *stream) {
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_STREAM_DESTROY, &handle, sizeof(handle), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaStreamSynchronize(void *stream) {
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_STREAM_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaEventCreate(void **pEvent) {
    uint64_t handle = 0;
    size_t len = sizeof(uint64_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_EVENT_CREATE, NULL, 0, &handle, &len);
    if (r == cudaSuccess)
        *pEvent = (void *)(uintptr_t)handle;
    g_last_error = r;
    return r;
}

cudaError_t cudaEventRecord(void *event, void *stream) {
    struct { uint64_t event; uint64_t stream; } req;
    req.event = (uint64_t)(uintptr_t)event;
    req.stream = (uint64_t)(uintptr_t)stream;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_EVENT_RECORD, &req, sizeof(req), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaEventSynchronize(void *event) {
    uint64_t handle = (uint64_t)(uintptr_t)event;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_EVENT_SYNCHRONIZE, &handle, sizeof(handle), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaEventDestroy(void *event) {
    uint64_t handle = (uint64_t)(uintptr_t)event;
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_EVENT_DESTROY, &handle, sizeof(handle), NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaRuntimeGetVersion(int *version) {
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_RUNTIME_GET_VERSION, NULL, 0, version, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaDriverGetVersion(int *version) {
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_DRIVER_GET_VERSION, NULL, 0, version, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaMemGetInfo(size_t *free_mem, size_t *total_mem) {
    struct { uint64_t free; uint64_t total; } info;
    size_t len = sizeof(info);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_MEM_GET_INFO, NULL, 0, &info, &len);
    if (r == cudaSuccess) {
        *free_mem = (size_t)info.free;
        *total_mem = (size_t)info.total;
    }
    g_last_error = r;
    return r;
}

cudaError_t cudaDeviceReset(void) {
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_DEVICE_RESET, NULL, 0, NULL, NULL);
    g_last_error = r;
    return r;
}

cudaError_t cudaGetLastError(void) {
    cudaError_t r = g_last_error;
    g_last_error = cudaSuccess;
    return r;
}

cudaError_t cudaPeekAtLastError(void) {
    return g_last_error;
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccess, int device, int peerDevice) {
    struct { int32_t device; int32_t peerDevice; } req = { device, peerDevice };
    size_t len = sizeof(int32_t);
    cudaError_t r = (cudaError_t)juice_rpc(JUICE_CUDA_DEVICE_CAN_ACCESS_PEER,
                                            &req, sizeof(req), canAccess, &len);
    g_last_error = r;
    return r;
}

cudaError_t cudaFuncGetAttributes(void *attr, const void *func) {
    /*
     * Stub: cudaFuncAttributes is ~60 bytes. Most callers can tolerate
     * zeroed-out attributes. A full implementation would require the server
     * to look up the function pointer, which is non-trivial for host-side
     * function pointers in the Runtime API.
     */
    (void)func;
    if (attr)
        memset(attr, 0, 64); /* sizeof(cudaFuncAttributes) ~= 56-64 bytes */
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

/*
 * ============================================================================
 * cuBLAS API Interceptions
 *
 * cuBLAS uses cublasHandle_t (opaque pointer) and cublasStatus_t (int).
 * alpha/beta parameters are HOST pointers to scalars, not device pointers.
 * Matrix pointers (A, B, C) are device pointers sent as uint64_t handles.
 * ============================================================================
 */

typedef int cublasStatus_t;
typedef void *cublasHandle_t;
typedef int cublasOperation_t;
typedef int cublasMath_t;

#define CUBLAS_STATUS_SUCCESS          0
#define CUBLAS_STATUS_NOT_INITIALIZED  1
#define CUBLAS_STATUS_ALLOC_FAILED     3
#define CUBLAS_STATUS_INVALID_VALUE    7
#define CUBLAS_STATUS_NOT_SUPPORTED    15

/* Helper to determine scalar size from cudaDataType / cublasComputeType */
static size_t juice_cublas_scalar_size(int dtype) {
    switch (dtype) {
    case 0:  /* CUDA_R_16F / CUBLAS_COMPUTE_16F */
        return 2;
    case 2:  /* CUDA_R_32F / CUBLAS_COMPUTE_32F */
    case 6:  /* CUBLAS_COMPUTE_32F_FAST_16F etc */
    case 7:
    case 8:
        return 4;
    case 1:  /* CUDA_R_64F / CUBLAS_COMPUTE_64F */
        return 8;
    default:
        return 4; /* safe default: float-sized */
    }
}

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    uint64_t h = 0;
    size_t len = sizeof(uint64_t);
    cublasStatus_t r = (cublasStatus_t)juice_rpc(JUICE_CUBLAS_CREATE, NULL, 0, &h, &len);
    if (r == CUBLAS_STATUS_SUCCESS)
        *handle = (cublasHandle_t)(uintptr_t)h;
    return r;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    uint64_t h = (uint64_t)(uintptr_t)handle;
    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DESTROY, &h, sizeof(h), NULL, NULL);
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, CUstream stream) {
    struct { uint64_t handle; uint64_t stream; } req;
    req.handle = (uint64_t)(uintptr_t)handle;
    req.stream = (uint64_t)(uintptr_t)stream;
    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SET_STREAM, &req, sizeof(req), NULL, NULL);
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, CUstream *stream) {
    uint64_t h = (uint64_t)(uintptr_t)handle;
    uint64_t s = 0;
    size_t len = sizeof(uint64_t);
    cublasStatus_t r = (cublasStatus_t)juice_rpc(JUICE_CUBLAS_GET_STREAM, &h, sizeof(h), &s, &len);
    if (r == CUBLAS_STATUS_SUCCESS)
        *stream = (CUstream)(uintptr_t)s;
    return r;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    struct { uint64_t handle; int32_t mode; } req;
    req.handle = (uint64_t)(uintptr_t)handle;
    req.mode = (int32_t)mode;
    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SET_MATH_MODE, &req, sizeof(req), NULL, NULL);
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    uint64_t h = (uint64_t)(uintptr_t)handle;
    int32_t m = 0;
    size_t len = sizeof(int32_t);
    cublasStatus_t r = (cublasStatus_t)juice_rpc(JUICE_CUBLAS_GET_MATH_MODE, &h, sizeof(h), &m, &len);
    if (r == CUBLAS_STATUS_SUCCESS)
        *mode = (cublasMath_t)m;
    return r;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    uint64_t h = (uint64_t)(uintptr_t)handle;
    int32_t v = 0;
    size_t len = sizeof(int32_t);
    cublasStatus_t r = (cublasStatus_t)juice_rpc(JUICE_CUBLAS_GET_VERSION, &h, sizeof(h), &v, &len);
    if (r == CUBLAS_STATUS_SUCCESS)
        *version = (int)v;
    return r;
}

/*
 * cublasSgemm_v2 — single-precision GEMM
 * C = alpha * op(A) * op(B) + beta * C
 *
 * Wire format:
 *   handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
 *   + alpha(4) + A(8) + lda(4) + B(8) + ldb(4)
 *   + beta(4) + C(8) + ldc(4) = 72 bytes
 */
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const float *beta,
    void *C, int ldc) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        float    alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t B;
        int32_t  ldb;
        float    beta;
        uint64_t C;
        int32_t  ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SGEMM, &req, sizeof(req), NULL, NULL);
}

/*
 * cublasDgemm_v2 — double-precision GEMM
 * Same layout as Sgemm but with double alpha/beta.
 */
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const double *beta,
    void *C, int ldc) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        double   alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t B;
        int32_t  ldb;
        double   beta;
        uint64_t C;
        int32_t  ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DGEMM, &req, sizeof(req), NULL, NULL);
}

/*
 * cublasHgemm — half-precision GEMM
 * alpha/beta are __half (2 bytes each), sent as uint16_t.
 */
cublasStatus_t cublasHgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int lda,
    const void *B, int ldb,
    const void *beta,
    void *C, int ldc) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        uint16_t alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t B;
        int32_t  ldb;
        uint16_t beta;
        uint64_t C;
        int32_t  ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    memcpy(&req.alpha, alpha, 2);
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    memcpy(&req.beta, beta, 2);
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_HGEMM, &req, sizeof(req), NULL, NULL);
}

/*
 * cublasSgemmEx — mixed-precision Sgemm with type descriptors
 */
cublasStatus_t cublasSgemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const float *beta,
    void *C, int Ctype, int ldc) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        float    alpha;
        uint64_t A;
        int32_t  Atype, lda;
        uint64_t B;
        int32_t  Btype, ldb;
        float    beta;
        uint64_t C;
        int32_t  Ctype, ldc;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.Atype = Atype; req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.Btype = Btype; req.ldb = ldb;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.Ctype = Ctype; req.ldc = ldc;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SGEMM_EX, &req, sizeof(req), NULL, NULL);
}

/*
 * cublasGemmEx — the critical mixed-precision GEMM used by PyTorch
 *
 * Wire format:
 *   handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
 *   + alpha_scalar(8) + A(8) + Atype(4) + lda(4) + B(8) + Btype(4) + ldb(4)
 *   + beta_scalar(8) + C(8) + Ctype(4) + ldc(4)
 *   + computeType(4) + algo(4)
 *
 * alpha/beta are sent as 8-byte values (max scalar size) with the actual
 * type determined by computeType.
 */
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta,
    void *C, int Ctype, int ldc,
    int computeType, int algo) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        uint8_t  alpha[8]; /* up to double-precision scalar */
        uint64_t A;
        int32_t  Atype, lda;
        uint64_t B;
        int32_t  Btype, ldb;
        uint8_t  beta[8];
        uint64_t C;
        int32_t  Ctype, ldc;
        int32_t  computeType;
        int32_t  algo;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;

    size_t scalar_sz = juice_cublas_scalar_size(computeType);
    memcpy(req.alpha, alpha, scalar_sz);
    req.A = (uint64_t)(uintptr_t)A;
    req.Atype = Atype; req.lda = lda;
    req.B = (uint64_t)(uintptr_t)B;
    req.Btype = Btype; req.ldb = ldb;
    memcpy(req.beta, beta, scalar_sz);
    req.C = (uint64_t)(uintptr_t)C;
    req.Ctype = Ctype; req.ldc = ldc;
    req.computeType = computeType;
    req.algo = algo;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_GEMM_EX, &req, sizeof(req), NULL, NULL);
}

/*
 * cublasSgemv_v2 — single-precision matrix-vector multiply
 * y = alpha * op(A) * x + beta * y
 */
cublasStatus_t cublasSgemv_v2(cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float *alpha,
    const void *A, int lda,
    const void *x, int incx,
    const float *beta,
    void *y, int incy) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  trans;
        int32_t  m, n;
        float    alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t x;
        int32_t  incx;
        float    beta;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.trans = (int32_t)trans;
    req.m = m; req.n = n;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.beta = *beta;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SGEMV, &req, sizeof(req), NULL, NULL);
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const double *alpha,
    const void *A, int lda,
    const void *x, int incx,
    const double *beta,
    void *y, int incy) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  trans;
        int32_t  m, n;
        double   alpha;
        uint64_t A;
        int32_t  lda;
        uint64_t x;
        int32_t  incx;
        double   beta;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.trans = (int32_t)trans;
    req.m = m; req.n = n;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.beta = *beta;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DGEMV, &req, sizeof(req), NULL, NULL);
}

/*
 * AXPY: y = alpha * x + y
 */
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle,
    int n, const float *alpha,
    const void *x, int incx,
    void *y, int incy) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        float    alpha;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SAXPY, &req, sizeof(req), NULL, NULL);
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle,
    int n, const double *alpha,
    const void *x, int incx,
    void *y, int incy) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        double   alpha;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DAXPY, &req, sizeof(req), NULL, NULL);
}

/*
 * NRM2: result = ||x||_2
 * Result scalar is returned in the response payload.
 */
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle,
    int n, const void *x, int incx, float *result) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    size_t len = sizeof(float);
    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SNRM2, &req, sizeof(req), result, &len);
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle,
    int n, const void *x, int incx, double *result) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    size_t len = sizeof(double);
    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DNRM2, &req, sizeof(req), result, &len);
}

/*
 * DOT: result = x . y
 */
cublasStatus_t cublasSdot_v2(cublasHandle_t handle,
    int n, const void *x, int incx,
    const void *y, int incy, float *result) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    size_t len = sizeof(float);
    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SDOT, &req, sizeof(req), result, &len);
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle,
    int n, const void *x, int incx,
    const void *y, int incy, double *result) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    size_t len = sizeof(double);
    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DDOT, &req, sizeof(req), result, &len);
}

/*
 * SCAL: x = alpha * x
 */
cublasStatus_t cublasSscal_v2(cublasHandle_t handle,
    int n, const float *alpha, void *x, int incx) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        float    alpha;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SSCAL, &req, sizeof(req), NULL, NULL);
}

cublasStatus_t cublasDscal_v2(cublasHandle_t handle,
    int n, const double *alpha, void *x, int incx) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        double   alpha;
        uint64_t x;
        int32_t  incx;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.alpha = *alpha;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DSCAL, &req, sizeof(req), NULL, NULL);
}

/*
 * COPY: y = x
 */
cublasStatus_t cublasScopy_v2(cublasHandle_t handle,
    int n, const void *x, int incx, void *y, int incy) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SCOPY, &req, sizeof(req), NULL, NULL);
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t handle,
    int n, const void *x, int incx, void *y, int incy) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  n;
        uint64_t x;
        int32_t  incx;
        uint64_t y;
        int32_t  incy;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.n = n;
    req.x = (uint64_t)(uintptr_t)x;
    req.incx = incx;
    req.y = (uint64_t)(uintptr_t)y;
    req.incy = incy;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_DCOPY, &req, sizeof(req), NULL, NULL);
}

/*
 * cublasSgemmBatched — batched Sgemm with array of device pointers
 *
 * Aarray, Barray, Carray are HOST arrays of device pointers.
 * We send all batchCount pointers inline.
 */
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void * const *Aarray, int lda,
    const void * const *Barray, int ldb,
    const float *beta,
    void * const *Carray, int ldc,
    int batchCount) {

    /*
     * Wire format:
     *   handle(8) + transa(4) + transb(4) + m(4) + n(4) + k(4)
     *   + alpha(4) + lda(4) + ldb(4) + beta(4) + ldc(4) + batchCount(4)
     *   = 52 bytes header
     *   + Aarray(8*batchCount) + Barray(8*batchCount) + Carray(8*batchCount)
     */
    size_t ptr_array_size = (size_t)batchCount * sizeof(uint64_t);
    size_t payload_len = 52 + 3 * ptr_array_size;
    void *payload = malloc(payload_len);
    if (!payload) return CUBLAS_STATUS_ALLOC_FAILED;

    char *p = (char *)payload;
    uint64_t h = (uint64_t)(uintptr_t)handle;
    int32_t ta = (int32_t)transa, tb = (int32_t)transb;
    int32_t m32 = m, n32 = n, k32 = k;
    float a = *alpha, b = *beta;
    int32_t lda32 = lda, ldb32 = ldb, ldc32 = ldc, bc32 = batchCount;

    memcpy(p, &h, 8); p += 8;
    memcpy(p, &ta, 4); p += 4;
    memcpy(p, &tb, 4); p += 4;
    memcpy(p, &m32, 4); p += 4;
    memcpy(p, &n32, 4); p += 4;
    memcpy(p, &k32, 4); p += 4;
    memcpy(p, &a, 4); p += 4;
    memcpy(p, &lda32, 4); p += 4;
    memcpy(p, &ldb32, 4); p += 4;
    memcpy(p, &b, 4); p += 4;
    memcpy(p, &ldc32, 4); p += 4;
    memcpy(p, &bc32, 4); p += 4;

    /* Copy device pointer arrays */
    for (int i = 0; i < batchCount; i++) {
        uint64_t ptr = (uint64_t)(uintptr_t)Aarray[i];
        memcpy(p, &ptr, 8); p += 8;
    }
    for (int i = 0; i < batchCount; i++) {
        uint64_t ptr = (uint64_t)(uintptr_t)Barray[i];
        memcpy(p, &ptr, 8); p += 8;
    }
    for (int i = 0; i < batchCount; i++) {
        uint64_t ptr = (uint64_t)(uintptr_t)Carray[i];
        memcpy(p, &ptr, 8); p += 8;
    }

    cublasStatus_t r = (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SGEMM_BATCHED, payload, payload_len, NULL, NULL);
    free(payload);
    return r;
}

/*
 * cublasSgemmStridedBatched — strided batched Sgemm
 */
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const void *A, int lda, long long int strideA,
    const void *B, int ldb, long long int strideB,
    const float *beta,
    void *C, int ldc, long long int strideC,
    int batchCount) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        float    alpha;
        uint64_t A;
        int32_t  lda;
        int64_t  strideA;
        uint64_t B;
        int32_t  ldb;
        int64_t  strideB;
        float    beta;
        uint64_t C;
        int32_t  ldc;
        int64_t  strideC;
        int32_t  batchCount;
    } req;

    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;
    req.alpha = *alpha;
    req.A = (uint64_t)(uintptr_t)A;
    req.lda = lda;
    req.strideA = (int64_t)strideA;
    req.B = (uint64_t)(uintptr_t)B;
    req.ldb = ldb;
    req.strideB = (int64_t)strideB;
    req.beta = *beta;
    req.C = (uint64_t)(uintptr_t)C;
    req.ldc = ldc;
    req.strideC = (int64_t)strideC;
    req.batchCount = batchCount;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_SGEMM_STRIDED_BATCHED, &req, sizeof(req), NULL, NULL);
}

/*
 * cublasGemmStridedBatchedEx — mixed-precision strided batched GEMM
 */
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long int strideA,
    const void *B, int Btype, int ldb, long long int strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long int strideC,
    int batchCount, int computeType, int algo) {

    struct __attribute__((packed)) {
        uint64_t handle;
        int32_t  transa, transb;
        int32_t  m, n, k;
        uint8_t  alpha[8];
        uint64_t A;
        int32_t  Atype, lda;
        int64_t  strideA;
        uint64_t B;
        int32_t  Btype, ldb;
        int64_t  strideB;
        uint8_t  beta[8];
        uint64_t C;
        int32_t  Ctype, ldc;
        int64_t  strideC;
        int32_t  batchCount;
        int32_t  computeType;
        int32_t  algo;
    } req;

    memset(&req, 0, sizeof(req));
    req.handle = (uint64_t)(uintptr_t)handle;
    req.transa = (int32_t)transa;
    req.transb = (int32_t)transb;
    req.m = m; req.n = n; req.k = k;

    size_t scalar_sz = juice_cublas_scalar_size(computeType);
    memcpy(req.alpha, alpha, scalar_sz);
    req.A = (uint64_t)(uintptr_t)A;
    req.Atype = Atype; req.lda = lda;
    req.strideA = (int64_t)strideA;
    req.B = (uint64_t)(uintptr_t)B;
    req.Btype = Btype; req.ldb = ldb;
    req.strideB = (int64_t)strideB;
    memcpy(req.beta, beta, scalar_sz);
    req.C = (uint64_t)(uintptr_t)C;
    req.Ctype = Ctype; req.ldc = ldc;
    req.strideC = (int64_t)strideC;
    req.batchCount = batchCount;
    req.computeType = computeType;
    req.algo = algo;

    return (cublasStatus_t)juice_rpc(JUICE_CUBLAS_GEMM_STRIDED_BATCHED_EX, &req, sizeof(req), NULL, NULL);
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

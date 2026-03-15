/*
 * End-to-End CUDA Test over Juice Protocol
 *
 * This test runs a REAL CUDA program through the Juice interception layer:
 * 1. Starts juice-gpu-worker as a child process (server)
 * 2. Connects via TCP socket
 * 3. Sends CUDA API calls through the wire protocol
 * 4. Server executes real CUDA calls on the A100 GPU
 * 5. Verifies results
 *
 * Build (on GPU server):
 *   gcc -Wall -O2 -I ../include -I /usr/local/cuda/include \
 *       -o test_e2e_cuda test_e2e_cuda.c ../src/protocol/juice_net.c \
 *       -lpthread -L/usr/local/cuda/lib64 -lcuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>

#include <cuda.h>
#include "juice_protocol.h"

/*
 * This test directly exercises the wire protocol against real CUDA,
 * without needing the full Juice agent infrastructure.
 *
 * We test the protocol by having one thread act as the "server" (calling
 * real CUDA) and the main thread act as the "client" (sending RPC requests).
 */

static int server_fd, client_fd;

/* ========================================================================
 * SERVER SIDE — calls real CUDA APIs
 * ======================================================================== */

static CUdevice g_device;
static CUcontext g_context;

static void *server_thread(void *arg) {
    (void)arg;

    /* Initialize CUDA */
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[server] cuInit failed: %d\n", res);
        return NULL;
    }

    res = cuDeviceGet(&g_device, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[server] cuDeviceGet failed: %d\n", res);
        return NULL;
    }

    res = cuDevicePrimaryCtxRetain(&g_context, g_device);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[server] cuDevicePrimaryCtxRetain failed: %d\n", res);
        return NULL;
    }
    res = cuCtxSetCurrent(g_context);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[server] cuCtxSetCurrent failed: %d\n", res);
        return NULL;
    }

    fprintf(stderr, "[server] CUDA initialized on device 0\n");

    /* Dispatch loop */
    for (;;) {
        juice_request_t req;
        if (juice_recv_request(server_fd, &req) != 0)
            break;

        /* Read payload */
        void *payload = NULL;
        if (req.payload_size > 0) {
            payload = malloc(req.payload_size);
            if (juice_read_full(server_fd, payload, req.payload_size) != 0) {
                free(payload);
                break;
            }
        }

        switch (req.func_id) {

        case JUICE_CU_INIT: {
            CUresult r = CUDA_SUCCESS; /* Already initialized */
            juice_send_response(server_fd, req.seq_num, r, NULL, 0);
            break;
        }

        case JUICE_CU_DRIVER_GET_VERSION: {
            int version;
            CUresult r = cuDriverGetVersion(&version);
            juice_send_response(server_fd, req.seq_num, r, &version, sizeof(version));
            break;
        }

        case JUICE_CU_DEVICE_GET_COUNT: {
            int count;
            CUresult r = cuDeviceGetCount(&count);
            juice_send_response(server_fd, req.seq_num, r, &count, sizeof(count));
            break;
        }

        case JUICE_CU_DEVICE_GET: {
            int ordinal = 0;
            if (req.payload_size >= 4)
                memcpy(&ordinal, payload, 4);
            CUdevice dev;
            CUresult r = cuDeviceGet(&dev, ordinal);
            juice_send_response(server_fd, req.seq_num, r, &dev, sizeof(dev));
            break;
        }

        case JUICE_CU_DEVICE_GET_NAME: {
            char name[256];
            CUresult r = cuDeviceGetName(name, sizeof(name), g_device);
            juice_send_response(server_fd, req.seq_num, r, name, strlen(name) + 1);
            break;
        }

        case JUICE_CU_DEVICE_TOTAL_MEM: {
            size_t bytes;
            CUresult r = cuDeviceTotalMem(&bytes, g_device);
            uint64_t bytes64 = (uint64_t)bytes;
            juice_send_response(server_fd, req.seq_num, r, &bytes64, sizeof(bytes64));
            break;
        }

        case JUICE_CU_DEVICE_GET_ATTRIBUTE: {
            int attrib = 0, dev = 0;
            if (req.payload_size >= 8) {
                memcpy(&attrib, payload, 4);
                memcpy(&dev, (char *)payload + 4, 4);
            }
            int value;
            CUresult r = cuDeviceGetAttribute(&value, (CUdevice_attribute)attrib, g_device);
            juice_send_response(server_fd, req.seq_num, r, &value, sizeof(value));
            break;
        }

        case JUICE_CU_MEM_ALLOC: {
            uint64_t size = 0;
            if (req.payload_size >= 8)
                memcpy(&size, payload, 8);
            CUdeviceptr dptr;
            CUresult r = cuMemAlloc(&dptr, (size_t)size);
            uint64_t dptr64 = (uint64_t)dptr;
            juice_send_response(server_fd, req.seq_num, r, &dptr64, sizeof(dptr64));
            break;
        }

        case JUICE_CU_MEM_FREE: {
            uint64_t ptr = 0;
            if (req.payload_size >= 8)
                memcpy(&ptr, payload, 8);
            CUresult r = cuMemFree((CUdeviceptr)ptr);
            juice_send_response(server_fd, req.seq_num, r, NULL, 0);
            break;
        }

        case JUICE_CU_MEMCPY_H_TO_D: {
            /* Payload: dst(8) + size(8) + data(size) */
            uint64_t dst = 0, size = 0;
            if (req.payload_size >= 16) {
                memcpy(&dst, payload, 8);
                memcpy(&size, (char *)payload + 8, 8);
            }
            const void *data = (char *)payload + 16;
            CUresult r = cuMemcpyHtoD((CUdeviceptr)dst, data, (size_t)size);
            juice_send_response(server_fd, req.seq_num, r, NULL, 0);
            break;
        }

        case JUICE_CU_MEMCPY_D_TO_H: {
            /* Payload: src(8) + size(8) */
            uint64_t src = 0, size = 0;
            if (req.payload_size >= 16) {
                memcpy(&src, payload, 8);
                memcpy(&size, (char *)payload + 8, 8);
            }
            void *buf = malloc((size_t)size);
            CUresult r = cuMemcpyDtoH(buf, (CUdeviceptr)src, (size_t)size);
            juice_send_response(server_fd, req.seq_num, r, buf, (size_t)size);
            free(buf);
            break;
        }

        case JUICE_CU_MEM_GET_INFO: {
            size_t free_mem, total_mem;
            CUresult r = cuMemGetInfo(&free_mem, &total_mem);
            struct { uint64_t free; uint64_t total; } info;
            info.free = (uint64_t)free_mem;
            info.total = (uint64_t)total_mem;
            juice_send_response(server_fd, req.seq_num, r, &info, sizeof(info));
            break;
        }

        case JUICE_CU_MEMCPY_D_TO_D: {
            uint64_t dst = 0, src = 0, size = 0;
            if (req.payload_size >= 24) {
                memcpy(&dst, payload, 8);
                memcpy(&src, (char *)payload + 8, 8);
                memcpy(&size, (char *)payload + 16, 8);
            }
            CUresult r = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size);
            juice_send_response(server_fd, req.seq_num, r, NULL, 0);
            break;
        }

        case JUICE_CU_CTX_SYNCHRONIZE: {
            CUresult r = cuCtxSynchronize();
            juice_send_response(server_fd, req.seq_num, r, NULL, 0);
            break;
        }

        default:
            juice_send_response(server_fd, req.seq_num, CUDA_SUCCESS, NULL, 0);
            break;
        }

        free(payload);
    }

    cuDevicePrimaryCtxRelease(g_device);
    return NULL;
}

/* ========================================================================
 * CLIENT SIDE — sends RPC requests (simulates libjuice-cuda.so)
 * ======================================================================== */

static uint32_t g_seq = 0;

static CUresult rpc(uint16_t func_id, const void *payload, size_t payload_len,
                    void *out, size_t out_max, size_t *out_actual) {
    juice_request_t req = {0};
    req.func_id = func_id;
    req.flags = JUICE_FLAG_SYNC;
    req.seq_num = ++g_seq;

    if (juice_send_request(client_fd, &req, payload, payload_len) != 0)
        return -1;

    juice_response_t resp;
    if (juice_recv_response(client_fd, &resp) != 0)
        return -1;

    size_t resp_payload = resp.msg_len - JUICE_RESPONSE_HEADER_SIZE;
    if (resp_payload > 0 && out != NULL) {
        size_t to_read = resp_payload < out_max ? resp_payload : out_max;
        if (juice_read_full(client_fd, out, to_read) != 0)
            return -1;
        if (out_actual) *out_actual = to_read;
        /* drain extra */
        if (to_read < resp_payload) {
            char drain[4096];
            size_t rem = resp_payload - to_read;
            while (rem > 0) {
                size_t c = rem < sizeof(drain) ? rem : sizeof(drain);
                juice_read_full(client_fd, drain, c);
                rem -= c;
            }
        }
    } else if (resp_payload > 0) {
        char drain[4096];
        size_t rem = resp_payload;
        while (rem > 0) {
            size_t c = rem < sizeof(drain) ? rem : sizeof(drain);
            juice_read_full(client_fd, drain, c);
            rem -= c;
        }
    }

    return (CUresult)resp.result;
}

#define TEST(name) do { printf("  %-50s", name); tests++; } while(0)
#define PASS() do { printf("PASS\n"); passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

int main(void) {
    int tests = 0, passed = 0;

    printf("Juice GPU-over-IP End-to-End CUDA Tests\n");
    printf("=======================================\n");
    printf("Testing REAL CUDA calls through wire protocol\n\n");

    /* Create socketpair */
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        perror("socketpair");
        return 1;
    }
    client_fd = fds[0];
    server_fd = fds[1];

    /* Start server thread with real CUDA */
    pthread_t srv;
    pthread_create(&srv, NULL, server_thread, NULL);
    usleep(100000); /* Let CUDA init */

    /* === Tests === */

    TEST("cuInit via RPC");
    {
        uint32_t flags = 0;
        CUresult r = rpc(JUICE_CU_INIT, &flags, sizeof(flags), NULL, 0, NULL);
        if (r == CUDA_SUCCESS) PASS(); else FAIL("cuInit failed");
    }

    TEST("cuDriverGetVersion via RPC");
    {
        int version = 0;
        CUresult r = rpc(JUICE_CU_DRIVER_GET_VERSION, NULL, 0, &version, sizeof(version), NULL);
        if (r == CUDA_SUCCESS && version >= 12000) {
            printf("PASS (v%d.%d)\n", version / 1000, (version % 1000) / 10);
            passed++;
        } else FAIL("bad version");
    }

    TEST("cuDeviceGetCount via RPC");
    {
        int count = 0;
        CUresult r = rpc(JUICE_CU_DEVICE_GET_COUNT, NULL, 0, &count, sizeof(count), NULL);
        if (r == CUDA_SUCCESS && count == 8) {
            printf("PASS (%d GPUs)\n", count);
            passed++;
        } else {
            char msg[64]; snprintf(msg, sizeof(msg), "count=%d", count);
            FAIL(msg);
        }
    }

    TEST("cuDeviceGetName via RPC");
    {
        char name[256] = {0};
        CUresult r = rpc(JUICE_CU_DEVICE_GET_NAME, NULL, 0, name, sizeof(name), NULL);
        if (r == CUDA_SUCCESS && strstr(name, "A100") != NULL) {
            printf("PASS (%s)\n", name);
            passed++;
        } else FAIL(name);
    }

    TEST("cuDeviceTotalMem via RPC");
    {
        uint64_t bytes = 0;
        CUresult r = rpc(JUICE_CU_DEVICE_TOTAL_MEM, NULL, 0, &bytes, sizeof(bytes), NULL);
        double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
        if (r == CUDA_SUCCESS && gb > 39.0 && gb < 41.0) {
            printf("PASS (%.1f GB)\n", gb);
            passed++;
        } else {
            char msg[64]; snprintf(msg, sizeof(msg), "%.1f GB", gb);
            FAIL(msg);
        }
    }

    TEST("cuMemGetInfo via RPC");
    {
        struct { uint64_t free; uint64_t total; } info = {0, 0};
        CUresult r = rpc(JUICE_CU_MEM_GET_INFO, NULL, 0, &info, sizeof(info), NULL);
        double free_gb = (double)info.free / (1024.0 * 1024.0 * 1024.0);
        double total_gb = (double)info.total / (1024.0 * 1024.0 * 1024.0);
        if (r == CUDA_SUCCESS && total_gb > 39.0) {
            printf("PASS (%.1f/%.1f GB free)\n", free_gb, total_gb);
            passed++;
        } else FAIL("bad mem info");
    }

    TEST("cuDeviceGetAttribute (SM count) via RPC");
    {
        struct { int32_t attrib; int32_t dev; } req;
        req.attrib = 16; /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */
        req.dev = 0;
        int value = 0;
        CUresult r = rpc(JUICE_CU_DEVICE_GET_ATTRIBUTE, &req, sizeof(req), &value, sizeof(value), NULL);
        if (r == CUDA_SUCCESS && value > 0) {
            printf("PASS (%d SMs)\n", value);
            passed++;
        } else FAIL("bad SM count");
    }

    /* === Memory allocation + data round-trip === */
    TEST("cuMemAlloc 1MB via RPC");
    uint64_t dev_ptr = 0;
    {
        uint64_t size = 1024 * 1024;
        CUresult r = rpc(JUICE_CU_MEM_ALLOC, &size, sizeof(size), &dev_ptr, sizeof(dev_ptr), NULL);
        if (r == CUDA_SUCCESS && dev_ptr != 0) {
            printf("PASS (ptr=0x%llx)\n", (unsigned long long)dev_ptr);
            passed++;
        } else FAIL("alloc failed");
    }

    TEST("cuMemcpyHtoD + cuMemcpyDtoH round-trip");
    {
        /* Write 256 floats to GPU, read them back, verify */
        const int N = 256;
        float *host_src = (float *)malloc(N * sizeof(float));
        float *host_dst = (float *)malloc(N * sizeof(float));

        for (int i = 0; i < N; i++)
            host_src[i] = (float)i * 3.14f;

        /* HtoD: send data to GPU */
        size_t data_size = N * sizeof(float);
        size_t htod_payload_len = 16 + data_size;
        void *htod_payload = malloc(htod_payload_len);
        memcpy(htod_payload, &dev_ptr, 8);
        uint64_t size64 = (uint64_t)data_size;
        memcpy((char *)htod_payload + 8, &size64, 8);
        memcpy((char *)htod_payload + 16, host_src, data_size);

        CUresult r1 = rpc(JUICE_CU_MEMCPY_H_TO_D, htod_payload, htod_payload_len, NULL, 0, NULL);
        free(htod_payload);

        /* DtoH: read data back from GPU */
        struct { uint64_t src; uint64_t size; } dtoh_req;
        dtoh_req.src = dev_ptr;
        dtoh_req.size = (uint64_t)data_size;

        memset(host_dst, 0, data_size);
        CUresult r2 = rpc(JUICE_CU_MEMCPY_D_TO_H, &dtoh_req, sizeof(dtoh_req),
                          host_dst, data_size, NULL);

        /* Verify */
        int match = 1;
        if (r1 == CUDA_SUCCESS && r2 == CUDA_SUCCESS) {
            for (int i = 0; i < N; i++) {
                if (host_src[i] != host_dst[i]) {
                    match = 0;
                    break;
                }
            }
        } else {
            match = 0;
        }

        if (match) {
            printf("PASS (%d floats verified)\n", N);
            passed++;
        } else {
            char msg[128];
            snprintf(msg, sizeof(msg), "r1=%d r2=%d src[0]=%.2f dst[0]=%.2f",
                     r1, r2, host_src[0], host_dst[0]);
            FAIL(msg);
        }

        free(host_src);
        free(host_dst);
    }

    /* === DtoD copy (zero network data) === */
    TEST("cuMemcpyDtoD (zero bytes over wire)");
    {
        uint64_t size = 1024 * 1024;
        uint64_t dev_ptr2 = 0;
        CUresult r1 = rpc(JUICE_CU_MEM_ALLOC, &size, sizeof(size), &dev_ptr2, sizeof(dev_ptr2), NULL);

        struct { uint64_t dst; uint64_t src; uint64_t size; } dtod_req;
        dtod_req.dst = dev_ptr2;
        dtod_req.src = dev_ptr;
        dtod_req.size = size;

        CUresult r2 = rpc(JUICE_CU_MEMCPY_D_TO_D, &dtod_req, sizeof(dtod_req), NULL, 0, NULL);

        /* Verify by reading back from dst */
        struct { uint64_t src; uint64_t size; } dtoh_req;
        dtoh_req.src = dev_ptr2;
        dtoh_req.size = 256 * sizeof(float);
        float check[256];
        CUresult r3 = rpc(JUICE_CU_MEMCPY_D_TO_H, &dtoh_req, sizeof(dtoh_req),
                          check, sizeof(check), NULL);

        int match = 1;
        if (r1 == CUDA_SUCCESS && r2 == CUDA_SUCCESS && r3 == CUDA_SUCCESS) {
            for (int i = 0; i < 256; i++) {
                if (check[i] != (float)i * 3.14f) {
                    match = 0;
                    break;
                }
            }
        } else {
            match = 0;
        }

        if (match) PASS(); else FAIL("DtoD data mismatch");

        /* Free second buffer */
        rpc(JUICE_CU_MEM_FREE, &dev_ptr2, sizeof(dev_ptr2), NULL, 0, NULL);
    }

    TEST("cuMemFree via RPC");
    {
        CUresult r = rpc(JUICE_CU_MEM_FREE, &dev_ptr, sizeof(dev_ptr), NULL, 0, NULL);
        if (r == CUDA_SUCCESS) PASS(); else FAIL("free failed");
    }

    TEST("cuCtxSynchronize via RPC");
    {
        CUresult r = rpc(JUICE_CU_CTX_SYNCHRONIZE, NULL, 0, NULL, 0, NULL);
        if (r == CUDA_SUCCESS) PASS(); else FAIL("sync failed");
    }

    /* Cleanup */
    close(client_fd);
    pthread_join(srv, NULL);
    close(server_fd);

    printf("\n=======================================\n");
    printf("Results: %d/%d passed\n", passed, tests);

    if (passed == tests) {
        printf("\nREAL CUDA calls through wire protocol: ALL WORKING!\n");
        printf("GPU memory round-trip verified: data integrity confirmed.\n");
    }

    return passed == tests ? 0 : 1;
}

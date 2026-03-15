/*
 * Juice Protocol Unit Test
 *
 * Tests the wire protocol by connecting a client and server over a local
 * socketpair. No GPU or network required.
 *
 * Build: gcc -Wall -O2 -I ../include -o test_protocol test_protocol.c ../src/protocol/chidori_net.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <pthread.h>
#include <assert.h>

#include "chidori_protocol.h"

static int server_fd, client_fd;

/* Simulate a minimal GPU worker on the server side */
static void *server_thread(void *arg) {
    (void)arg;

    for (;;) {
        chidori_request_t req;
        if (chidori_recv_request(server_fd, &req) != 0)
            break;

        /* Read and discard payload */
        if (req.payload_size > 0) {
            void *buf = malloc(req.payload_size);
            chidori_read_full(server_fd, buf, req.payload_size);
            free(buf);
        }

        /* Dispatch based on func_id */
        switch (req.func_id) {
        case CHIDORI_CU_INIT:
            chidori_send_response(server_fd, req.seq_num, 0, NULL, 0);
            break;

        case CHIDORI_CU_DEVICE_GET_COUNT: {
            int32_t count = 2;
            chidori_send_response(server_fd, req.seq_num, 0, &count, sizeof(count));
            break;
        }

        case CHIDORI_CU_DEVICE_GET_NAME: {
            const char *name = "Test GPU 4090";
            chidori_send_response(server_fd, req.seq_num, 0, name, strlen(name) + 1);
            break;
        }

        case CHIDORI_CU_MEM_ALLOC: {
            uint64_t fake_ptr = 0xDEADBEEF0000ULL;
            chidori_send_response(server_fd, req.seq_num, 0, &fake_ptr, sizeof(fake_ptr));
            break;
        }

        case CHIDORI_CU_MEM_GET_INFO: {
            struct { uint64_t free; uint64_t total; } info;
            info.total = 24ULL * 1024 * 1024 * 1024;
            info.free  = 20ULL * 1024 * 1024 * 1024;
            chidori_send_response(server_fd, req.seq_num, 0, &info, sizeof(info));
            break;
        }

        default:
            /* Unknown: return success */
            chidori_send_response(server_fd, req.seq_num, 0, NULL, 0);
            break;
        }
    }

    return NULL;
}

static int rpc(uint16_t func_id, const void *payload, size_t payload_len,
               void *out, size_t out_len) {
    static uint32_t seq = 0;

    chidori_request_t req = {0};
    req.func_id = func_id;
    req.flags = CHIDORI_FLAG_SYNC;
    req.seq_num = ++seq;

    if (chidori_send_request(client_fd, &req, payload, payload_len) != 0)
        return -1;

    chidori_response_t resp;
    if (chidori_recv_response(client_fd, &resp) != 0)
        return -1;

    assert(resp.seq_num == seq);

    size_t resp_payload = resp.msg_len - CHIDORI_RESPONSE_HEADER_SIZE;
    if (resp_payload > 0 && out != NULL) {
        size_t to_read = resp_payload < out_len ? resp_payload : out_len;
        if (chidori_read_full(client_fd, out, to_read) != 0)
            return -1;
    }

    return resp.result;
}

#define TEST(name) do { printf("  %-40s", name); } while(0)
#define PASS() do { printf("PASS\n"); passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); failed++; } while(0)

int main(void) {
    int passed = 0, failed = 0;

    printf("Chidori GPU-over-IP Protocol Tests\n");
    printf("================================\n\n");

    /* Create socketpair for local testing */
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        perror("socketpair");
        return 1;
    }
    client_fd = fds[0];
    server_fd = fds[1];

    /* Start server thread */
    pthread_t srv;
    pthread_create(&srv, NULL, server_thread, NULL);

    /* Test 1: cuInit */
    TEST("cuInit");
    {
        uint32_t flags = 0;
        int result = rpc(CHIDORI_CU_INIT, &flags, sizeof(flags), NULL, 0);
        if (result == 0) PASS(); else FAIL("expected CUDA_SUCCESS");
    }

    /* Test 2: cuDeviceGetCount */
    TEST("cuDeviceGetCount");
    {
        int32_t count = 0;
        int result = rpc(CHIDORI_CU_DEVICE_GET_COUNT, NULL, 0, &count, sizeof(count));
        if (result == 0 && count == 2) PASS();
        else { char msg[64]; snprintf(msg, sizeof(msg), "result=%d count=%d", result, count); FAIL(msg); }
    }

    /* Test 3: cuDeviceGetName */
    TEST("cuDeviceGetName");
    {
        char name[256] = {0};
        int result = rpc(CHIDORI_CU_DEVICE_GET_NAME, NULL, 0, name, sizeof(name));
        if (result == 0 && strcmp(name, "Test GPU 4090") == 0) PASS();
        else { char msg[256]; snprintf(msg, sizeof(msg), "result=%d name='%s'", result, name); FAIL(msg); }
    }

    /* Test 4: cuMemAlloc */
    TEST("cuMemAlloc");
    {
        uint64_t size = 1024 * 1024;
        uint64_t dptr = 0;
        int result = rpc(CHIDORI_CU_MEM_ALLOC, &size, sizeof(size), &dptr, sizeof(dptr));
        if (result == 0 && dptr == 0xDEADBEEF0000ULL) PASS();
        else { char msg[128]; snprintf(msg, sizeof(msg), "result=%d dptr=0x%llx", result, (unsigned long long)dptr); FAIL(msg); }
    }

    /* Test 5: cuMemGetInfo */
    TEST("cuMemGetInfo");
    {
        struct { uint64_t free; uint64_t total; } info = {0, 0};
        int result = rpc(CHIDORI_CU_MEM_GET_INFO, NULL, 0, &info, sizeof(info));
        if (result == 0 &&
            info.total == 24ULL * 1024 * 1024 * 1024 &&
            info.free == 20ULL * 1024 * 1024 * 1024) PASS();
        else FAIL("memory info mismatch");
    }

    /* Test 6: Header size verification */
    TEST("Protocol header sizes");
    {
        if (CHIDORI_REQUEST_HEADER_SIZE == 16 && CHIDORI_RESPONSE_HEADER_SIZE == 12) PASS();
        else {
            char msg[128];
            snprintf(msg, sizeof(msg), "req=%zu resp=%zu",
                     CHIDORI_REQUEST_HEADER_SIZE, CHIDORI_RESPONSE_HEADER_SIZE);
            FAIL(msg);
        }
    }

    /* Test 7: cuMemFree (void return) */
    TEST("cuMemFree");
    {
        uint64_t ptr = 0xDEADBEEF0000ULL;
        int result = rpc(CHIDORI_CU_MEM_FREE, &ptr, sizeof(ptr), NULL, 0);
        if (result == 0) PASS(); else FAIL("expected CUDA_SUCCESS");
    }

    /* Test 8: Unknown function */
    TEST("Unknown function (graceful)");
    {
        int result = rpc(0xFFEE, NULL, 0, NULL, 0);
        if (result == 0) PASS(); else FAIL("server should handle gracefully");
    }

    /* Cleanup */
    close(client_fd);
    pthread_join(srv, NULL);
    close(server_fd);

    printf("\n================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}

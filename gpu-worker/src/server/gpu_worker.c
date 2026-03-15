/*
 * Juice GPU Worker — Server-side CUDA execution engine
 *
 * Replaces the proprietary "Renderer_Win" from the original Juice Labs.
 * Receives forwarded TCP sockets from the Juice Agent via IPC (SCM_RIGHTS),
 * then serves CUDA API calls over the wire protocol.
 *
 * CLI interface (compatible with original Renderer_Win):
 *   juice-gpu-worker --id <session_id> --ipc_write <fd> --ipc_read <fd> --pcibus <pci>
 *
 * Linux only.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "juice_protocol.h"

/* From cuda_dispatch.c */
extern int  cuda_dispatch_init(const char *pcibus);
extern void cuda_dispatch_shutdown(void);
extern int  cuda_dispatch(int fd, juice_request_t *req, void *payload);

typedef struct {
    char session_id[128];
    char pcibus[64];
    char log_file[512];
    int  ipc_write_fd;
    int  ipc_read_fd;
} worker_config_t;

static void parse_args(int argc, char **argv, worker_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    static struct option long_options[] = {
        {"id",        required_argument, 0, 'i'},
        {"log_file",  required_argument, 0, 'l'},
        {"ipc_write", required_argument, 0, 'w'},
        {"ipc_read",  required_argument, 0, 'r'},
        {"pcibus",    required_argument, 0, 'p'},
        {"log_group", required_argument, 0, 'g'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch (opt) {
        case 'i': strncpy(cfg->session_id, optarg, sizeof(cfg->session_id) - 1); break;
        case 'l': strncpy(cfg->log_file, optarg, sizeof(cfg->log_file) - 1); break;
        case 'w': cfg->ipc_write_fd = atoi(optarg); break;
        case 'r': cfg->ipc_read_fd = atoi(optarg); break;
        case 'p': strncpy(cfg->pcibus, optarg, sizeof(cfg->pcibus) - 1); break;
        case 'g': break; /* log_group: ignored */
        }
    }
}

/*
 * Receive a file descriptor from the Agent via Unix domain socket (SCM_RIGHTS).
 */
static int receive_socket_from_agent(int ipc_read_fd) {
    char buf[1];
    struct iovec iov = { .iov_base = buf, .iov_len = sizeof(buf) };

    char cmsg_buf[CMSG_SPACE(sizeof(int))];
    memset(cmsg_buf, 0, sizeof(cmsg_buf));

    struct msghdr msg = {
        .msg_iov = &iov,
        .msg_iovlen = 1,
        .msg_control = cmsg_buf,
        .msg_controllen = sizeof(cmsg_buf),
    };

    ssize_t n = recvmsg(ipc_read_fd, &msg, 0);
    if (n < 0) {
        perror("[juice-gpu-worker] recvmsg");
        return -1;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        fprintf(stderr, "[juice-gpu-worker] No SCM_RIGHTS in control message\n");
        return -1;
    }

    int client_fd;
    memcpy(&client_fd, CMSG_DATA(cmsg), sizeof(int));
    return client_fd;
}

/*
 * Main dispatch loop: read CUDA API requests, execute them via real CUDA, send responses.
 */
static int dispatch_loop(int client_fd) {
    void *payload_buf = NULL;
    size_t payload_buf_size = 0;

    for (;;) {
        juice_request_t req;
        if (juice_recv_request(client_fd, &req) != 0) {
            fprintf(stderr, "[juice-gpu-worker] Client disconnected\n");
            break;
        }

        /* Read payload if present */
        if (req.payload_size > 0) {
            if (req.payload_size > payload_buf_size) {
                free(payload_buf);
                payload_buf_size = req.payload_size;
                payload_buf = malloc(payload_buf_size);
                if (!payload_buf) {
                    fprintf(stderr, "[juice-gpu-worker] OOM allocating %u bytes\n", req.payload_size);
                    break;
                }
            }
            if (juice_read_full(client_fd, payload_buf, req.payload_size) != 0) {
                fprintf(stderr, "[juice-gpu-worker] Failed to read payload\n");
                break;
            }
        }

        /* Dispatch to real CUDA */
        if (cuda_dispatch(client_fd, &req, payload_buf) != 0) {
            fprintf(stderr, "[juice-gpu-worker] Error dispatching func_id=0x%04x\n", req.func_id);
            break;
        }
    }

    free(payload_buf);
    close(client_fd);
    return 0;
}

int main(int argc, char **argv) {
    worker_config_t cfg;
    parse_args(argc, argv, &cfg);

    fprintf(stderr, "[juice-gpu-worker] Starting session=%s pcibus=%s\n",
            cfg.session_id, cfg.pcibus);
    fprintf(stderr, "[juice-gpu-worker] IPC write_fd=%d read_fd=%d\n",
            cfg.ipc_write_fd, cfg.ipc_read_fd);

    /* Initialize real CUDA on the assigned GPU */
    if (cuda_dispatch_init(cfg.pcibus) != 0) {
        fprintf(stderr, "[juice-gpu-worker] Failed to initialize CUDA\n");
        return 1;
    }

    /* Serve client connections forwarded by the Agent */
    for (;;) {
        int client_fd = receive_socket_from_agent(cfg.ipc_read_fd);
        if (client_fd < 0) {
            fprintf(stderr, "[juice-gpu-worker] Failed to receive client socket\n");
            break;
        }

        fprintf(stderr, "[juice-gpu-worker] Received client socket fd=%d\n", client_fd);

        /* Signal readiness to Agent */
        char ready = 1;
        if (write(cfg.ipc_write_fd, &ready, 1) != 1) {
            perror("[juice-gpu-worker] write ready");
            close(client_fd);
            break;
        }

        /* Wait for Agent's confirmation */
        char confirm;
        if (read(cfg.ipc_read_fd, &confirm, 1) != 1) {
            fprintf(stderr, "[juice-gpu-worker] Agent confirmation read failed\n");
        }

        dispatch_loop(client_fd);
    }

    cuda_dispatch_shutdown();
    fprintf(stderr, "[juice-gpu-worker] Shutting down\n");
    return 0;
}

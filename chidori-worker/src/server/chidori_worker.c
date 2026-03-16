/*
 * Chidori GPU Worker — Server-side CUDA execution engine
 *
 * Replaces the proprietary "Renderer_Win" from the original Juice Labs.
 * Receives forwarded TCP sockets from the Juice Agent via IPC (SCM_RIGHTS),
 * then serves CUDA API calls over the wire protocol.
 *
 * CLI interface (compatible with original Renderer_Win):
 *   chidori-worker --id <session_id> --ipc_write <fd> --ipc_read <fd> --pcibus <pci>
 *
 * Linux only.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <getopt.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include "chidori_protocol.h"

/* From cuda_dispatch.c */
extern int  cuda_dispatch_init(const char *pcibus);
extern void cuda_dispatch_shutdown(void);
extern int  cuda_dispatch(int fd, chidori_request_t *req, void *payload);

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

    /*
     * Go's exec.Cmd.ExtraFiles maps files to fd 3, 4, 5, ... in the child.
     * The --ipc_write and --ipc_read args contain the PARENT's fd numbers,
     * but in the child they're always fd 3 (first ExtraFile) and fd 4 (second).
     * Override with the correct child fd numbers.
     */
    cfg->ipc_write_fd = 3;  /* ch1Write = first ExtraFile */
    cfg->ipc_read_fd = 4;   /* ch2Read = second ExtraFile */
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
        perror("[chidori-worker] recvmsg");
        return -1;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        fprintf(stderr, "[chidori-worker] No SCM_RIGHTS in control message\n");
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
    static int s_verbose = -1;
    if (s_verbose == -1) {
        const char *v = getenv("CHIDORI_SERVER_VERBOSE");
        s_verbose = (v && v[0] == '1') ? 1 : 0;
    }

    if (s_verbose)
        fprintf(stderr, "[dispatch_loop] Entering dispatch loop for client fd=%d\n", client_fd);

    void *payload_buf = NULL;
    size_t payload_buf_size = 0;

    for (;;) {
        chidori_request_t req;
        if (chidori_recv_request(client_fd, &req) != 0) {
            fprintf(stderr, "[chidori-worker] Client disconnected (EOF or read error on fd=%d)\n", client_fd);
            break;
        }

        if (s_verbose)
            fprintf(stderr, "[dispatch_loop] Received func_id=0x%04x payload_size=%u seq=%u\n",
                    req.func_id, req.payload_size, req.seq_num);

        /* Read payload if present */
        if (req.payload_size > 0) {
            if (req.payload_size > payload_buf_size) {
                free(payload_buf);
                payload_buf_size = req.payload_size;
                payload_buf = malloc(payload_buf_size);
                if (!payload_buf) {
                    fprintf(stderr, "[chidori-worker] OOM allocating %u bytes for fd=%d\n",
                            req.payload_size, client_fd);
                    break;
                }
            }
            if (chidori_read_full(client_fd, payload_buf, req.payload_size) != 0) {
                fprintf(stderr, "[chidori-worker] Failed to read payload (%u bytes) from fd=%d\n",
                        req.payload_size, client_fd);
                break;
            }
        }

        /*
         * Dispatch to real CUDA.
         * For async requests (CHIDORI_FLAG_ASYNC), use a dummy fd (-1) so
         * cuda_dispatch skips sending the response. The client doesn't wait
         * for it — this prevents TCP buffer deadlock and lets the GPU
         * process a queue of operations without network round-trips.
         */
        int dispatch_fd = (req.flags & CHIDORI_FLAG_ASYNC) ? -1 : client_fd;
        if (cuda_dispatch(dispatch_fd, &req, payload_buf) != 0) {
            /* dispatch_fd == -1 means send_response was skipped, which returns 0.
             * If it failed with the real fd, the connection is dead. */
            if (dispatch_fd != -1) {
                fprintf(stderr, "[chidori-worker] Error dispatching func_id=0x%04x on fd=%d\n",
                        req.func_id, client_fd);
                break;
            }
        }
    }

    if (s_verbose)
        fprintf(stderr, "[dispatch_loop] Exiting dispatch loop for client fd=%d\n", client_fd);

    free(payload_buf);
    close(client_fd);
    return 0;
}

/*
 * Standalone TCP server mode: listen on a port and accept direct connections.
 * Used for testing without the Juice agent.
 */
static int standalone_mode(const char *pcibus, int port) {
    fprintf(stderr, "[chidori-worker] Standalone mode on port %d\n", port);

    if (cuda_dispatch_init(pcibus) != 0) {
        fprintf(stderr, "[chidori-worker] Failed to initialize CUDA\n");
        return 1;
    }

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    if (listen(server_fd, 5) < 0) { perror("listen"); return 1; }

    fprintf(stderr, "[chidori-worker] Listening on port %d\n", port);

    for (;;) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) { perror("accept"); break; }

        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        fprintf(stderr, "[chidori-worker] Client connected (fd=%d)\n", client_fd);
        dispatch_loop(client_fd);
        fprintf(stderr, "[chidori-worker] Client disconnected\n");
    }

    close(server_fd);
    cuda_dispatch_shutdown();
    return 0;
}

int main(int argc, char **argv) {
    worker_config_t cfg;
    parse_args(argc, argv, &cfg);

    /* Check for standalone mode: --standalone <port> */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--standalone") == 0 && i + 1 < argc) {
            int port = atoi(argv[i + 1]);
            return standalone_mode(cfg.pcibus, port > 0 ? port : 43211);
        }
    }

    fprintf(stderr, "[chidori-worker] Starting session=%s pcibus=%s\n",
            cfg.session_id, cfg.pcibus);
    fprintf(stderr, "[chidori-worker] IPC write_fd=%d read_fd=%d\n",
            cfg.ipc_write_fd, cfg.ipc_read_fd);

    /* Serve client connections forwarded by the Agent.
     * CUDA initialization is deferred until after the first socket is received,
     * because cuInit() can take several seconds and the agent's context
     * would time out if we blocked here.
     */
    int cuda_initialized = 0;

    for (;;) {
        int client_fd = receive_socket_from_agent(cfg.ipc_read_fd);
        if (client_fd < 0) {
            fprintf(stderr, "[chidori-worker] Failed to receive client socket\n");
            break;
        }

        fprintf(stderr, "[chidori-worker] Received client socket fd=%d\n", client_fd);

        /*
         * The Go agent sets sockets to non-blocking mode (required by Go's
         * runtime netpoller).  When the fd is forwarded via SCM_RIGHTS, it
         * retains the O_NONBLOCK flag.  We must clear it so that our
         * blocking read/write calls in dispatch_loop work correctly.
         */
        {
            int flags = fcntl(client_fd, F_GETFL, 0);
            if (flags != -1 && (flags & O_NONBLOCK)) {
                fcntl(client_fd, F_SETFL, flags & ~O_NONBLOCK);
                fprintf(stderr, "[chidori-worker] Cleared O_NONBLOCK on fd=%d\n", client_fd);
            }
        }

        /* Signal readiness to Agent IMMEDIATELY (before CUDA init) */
        char ready = 1;
        if (write(cfg.ipc_write_fd, &ready, 1) != 1) {
            perror("[chidori-worker] write ready");
            close(client_fd);
            break;
        }

        /* Wait for Agent's confirmation */
        char confirm;
        if (read(cfg.ipc_read_fd, &confirm, 1) != 1) {
            fprintf(stderr, "[chidori-worker] Agent confirmation read failed\n");
        }

        /* Lazy CUDA init on first connection */
        if (!cuda_initialized) {
            if (cuda_dispatch_init(cfg.pcibus) != 0) {
                fprintf(stderr, "[chidori-worker] Failed to initialize CUDA\n");
                close(client_fd);
                break;
            }
            cuda_initialized = 1;
        }

        dispatch_loop(client_fd);
    }

    cuda_dispatch_shutdown();
    fprintf(stderr, "[chidori-worker] Shutting down\n");
    return 0;
}

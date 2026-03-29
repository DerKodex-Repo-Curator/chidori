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
#include <errno.h>
#include <getopt.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
/* Thread to handle multiple simultaneous connections in standalone mode. */
static void *client_thread(void *arg_ptr {
    int fd = *arg;
    dispatch_loop(arg fd);
    close(fd);
    fprintf(stderr, "[chidori-worker] Client disconnected ( fd=%d)\n", client_fd);
    free(arg);
    return NULL;
}

client_thread(void *arg) {
    int *arg = (int *)fd;
    dispatch_loop(fd);
        close(fd);
        fprintf(stderr, "[chidori-worker] Client disconnected (fd=%d)\n", client_fd);
    free(arg);
    }

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
 * Batch-read dispatch loop.
 *
 * Instead of 2 blocking read() syscalls per request (header + payload),
 * we recv() ALL available TCP data into a 256KB buffer, then parse and
 * dispatch every complete request in a tight loop. This lets the GPU
 * receive bursts of N requests with zero syscall overhead between them.
 *
 * Oversized payloads (e.g. HtoD memcpy of a 64MB matrix) that exceed
 * the buffer are handled by a fallback path that reads directly from
 * the socket into a separate allocation.
 */
static int dispatch_loop(int client_fd) {
    static int s_verbose = -1;
    if (s_verbose == -1) {
        const char *v = getenv("CHIDORI_SERVER_VERBOSE");
        s_verbose = (v && v[0] == '1') ? 1 : 0;
    }

    if (s_verbose)
        fprintf(stderr, "[dispatch] Entering batch-read loop for fd=%d\n", client_fd);

    #define BATCH_BUF_SIZE (4 * 1024 * 1024)
    char *buf = malloc(BATCH_BUF_SIZE);
    if (!buf) { fprintf(stderr, "[dispatch] malloc failed\n"); return -1; }
    size_t buf_len = 0;
    int request_count = 0;
    int batch_count = 0;

    /* Separate buffer for oversized payloads */
    void *big_payload = NULL;
    size_t big_payload_cap = 0;

    for (;;) {
        /* Step 1: Read ALL available data from socket */
        ssize_t n = recv(client_fd, buf + buf_len, BATCH_BUF_SIZE - buf_len, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            fprintf(stderr, "[chidori-worker] Client disconnected (fd=%d)\n", client_fd);
            break;
        }
        buf_len += (size_t)n;

        int batch_dispatched = 0;

        /* Step 2: Parse and dispatch ALL complete requests in buffer */
        size_t offset = 0;
        while (offset + CHIDORI_REQUEST_HEADER_SIZE <= buf_len) {
            chidori_request_t *req = (chidori_request_t *)(buf + offset);
            size_t total_msg = CHIDORI_REQUEST_HEADER_SIZE + req->payload_size;

            if (offset + total_msg <= buf_len) {
                /*
                 * Common path: entire request fits in buffer.
                 * Dispatch directly from buffer — zero copy.
                 */
                void *payload = (req->payload_size > 0)
                    ? buf + offset + CHIDORI_REQUEST_HEADER_SIZE
                    : NULL;

                if (s_verbose)
                    fprintf(stderr, "[dispatch] func=0x%04x pl=%u seq=%u\n",
                            req->func_id, req->payload_size, req->seq_num);

                int dispatch_fd = (req->flags & CHIDORI_FLAG_ASYNC) ? -1 : client_fd;
                if (cuda_dispatch(dispatch_fd, req, payload) != 0 && dispatch_fd != -1) {
                    fprintf(stderr, "[chidori-worker] Dispatch error func=0x%04x fd=%d\n",
                            req->func_id, client_fd);
                    goto done;
                }
                request_count++;
                batch_dispatched++;
                offset += total_msg;
            } else if (total_msg > BATCH_BUF_SIZE) {
                /*
                 * Oversized payload — will never fit in buffer.
                 * Copy header, read payload via socket into separate alloc.
                 */
                chidori_request_t req_copy = *req;
                size_t in_buf = buf_len - offset - CHIDORI_REQUEST_HEADER_SIZE;

                if (req_copy.payload_size > big_payload_cap) {
                    free(big_payload);
                    big_payload_cap = req_copy.payload_size;
                    big_payload = malloc(big_payload_cap);
                    if (!big_payload) {
                        fprintf(stderr, "[chidori-worker] OOM for %u byte payload\n",
                                req_copy.payload_size);
                        goto done;
                    }
                }

                /* Copy payload bytes already in buffer */
                if (in_buf > 0)
                    memcpy(big_payload, buf + offset + CHIDORI_REQUEST_HEADER_SIZE, in_buf);

                /* Read the rest directly from socket */
                if (chidori_read_full(client_fd, (char *)big_payload + in_buf,
                                      req_copy.payload_size - in_buf) != 0) {
                    fprintf(stderr, "[chidori-worker] Failed reading oversized payload (%u bytes)\n",
                            req_copy.payload_size);
                    goto done;
                }

                if (s_verbose)
                    fprintf(stderr, "[dispatch] func=0x%04x pl=%u seq=%u (oversized)\n",
                            req_copy.func_id, req_copy.payload_size, req_copy.seq_num);

                int dispatch_fd = (req_copy.flags & CHIDORI_FLAG_ASYNC) ? -1 : client_fd;
                if (cuda_dispatch(dispatch_fd, &req_copy, big_payload) != 0 && dispatch_fd != -1) {
                    fprintf(stderr, "[chidori-worker] Dispatch error func=0x%04x fd=%d\n",
                            req_copy.func_id, client_fd);
                    goto done;
                }
                request_count++;
                batch_dispatched++;

                /* Buffer is consumed — this request ate everything */
                buf_len = 0;
                offset = 0;
                break;
            } else {
                /* Payload partially in buffer but fits after compaction + more recv */
                break;
            }
        }

        /* Step 3: Compact buffer — move remaining partial data to front */
        if (offset > 0) {
            buf_len -= offset;
            if (buf_len > 0)
                memmove(buf, buf + offset, buf_len);
        }

        if (batch_dispatched > 0)
            batch_count++;

        if (s_verbose && batch_dispatched > 1)
            fprintf(stderr, "[dispatch] Batch #%d: %d requests dispatched at once\n",
                    batch_count, batch_dispatched);
    }

done:
    free(buf);
    free(big_payload);
    fprintf(stderr, "[chidori-worker] Processed %d requests in %d recv batches (fd=%d)\n",
            request_count, batch_count, client_fd);
    close(client_fd);
    return 0;
    #undef BATCH_BUF_SIZE
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

        /* Large socket buffers for WAN throughput */
        int bufsize = 16 * 1024 * 1024;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

        fprintf(stderr, "[chidori-worker] Client connected (fd=%d)\n", client_fd);

        /* Fork a thread per connection so vLLM can reconnect from forked child */
        {
            int *arg = malloc(sizeof(int));
            *arg = client_fd;
            pthread_t tid;
            if (pthread_create(&tid, NULL, client_thread, arg) == 0) {
                fprintf(stderr, "[chidori-worker] Failed to create thread: %s\n", strerror(errno));
                close(client_fd);
            } else {
                pthread_detach(tid);
            }
        }
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

/*
 * Chidori GPU-over-IP — Network I/O primitives
 *
 * Reliable read/write over TCP sockets with proper partial-send handling.
 */
#include "chidori_protocol.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>

int chidori_read_full(int fd, void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, (char *)buf + total, len - total);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return -1; /* EOF or error */
        }
        total += (size_t)n;
    }
    return 0;
}

int chidori_write_full(int fd, const void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = write(fd, (const char *)buf + total, len - total);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return -1;
        }
        total += (size_t)n;
    }
    return 0;
}

int chidori_send_request(int fd, chidori_request_t *hdr, const void *payload, size_t payload_len) {
    hdr->payload_size = (uint32_t)payload_len;
    hdr->msg_len = (uint32_t)(CHIDORI_REQUEST_HEADER_SIZE + payload_len);

    if (chidori_write_full(fd, hdr, CHIDORI_REQUEST_HEADER_SIZE) != 0)
        return -1;

    if (payload_len > 0 && payload != NULL) {
        if (chidori_write_full(fd, payload, payload_len) != 0)
            return -1;
    }

    return 0;
}

int chidori_recv_request(int fd, chidori_request_t *hdr) {
    return chidori_read_full(fd, hdr, CHIDORI_REQUEST_HEADER_SIZE);
}

int chidori_send_response(int fd, uint32_t seq_num, int32_t result,
                        const void *payload, size_t payload_len) {
    /* fd == -1 means async request — skip response */
    if (fd < 0) return 0;

    chidori_response_t resp;
    resp.msg_len = (uint32_t)(CHIDORI_RESPONSE_HEADER_SIZE + payload_len);
    resp.seq_num = seq_num;
    resp.result = result;

    if (chidori_write_full(fd, &resp, CHIDORI_RESPONSE_HEADER_SIZE) != 0)
        return -1;

    if (payload_len > 0 && payload != NULL) {
        if (chidori_write_full(fd, payload, payload_len) != 0)
            return -1;
    }

    return 0;
}

int chidori_recv_response(int fd, chidori_response_t *hdr) {
    return chidori_read_full(fd, hdr, CHIDORI_RESPONSE_HEADER_SIZE);
}

/*
 * Juice GPU-over-IP Wire Protocol
 *
 * Binary protocol for CUDA API remoting over TCP.
 * All values are native byte order (both sides assumed x86_64).
 */
#ifndef JUICE_PROTOCOL_H
#define JUICE_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Protocol version */
#define JUICE_PROTOCOL_VERSION 1

/* Message flags */
#define JUICE_FLAG_SYNC          0x0001  /* Synchronous call, client blocks for response */
#define JUICE_FLAG_ASYNC         0x0002  /* Async call, no immediate response needed */
#define JUICE_FLAG_BATCH_MORE    0x0004  /* More messages follow in this batch */
#define JUICE_FLAG_ERROR         0x0008  /* Response indicates error */

/*
 * Request header (16 bytes)
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                       Message Length                          |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |         Function ID           |            Flags              |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                      Sequence Number                         |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                       Payload Size                           |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
typedef struct __attribute__((packed)) {
    uint32_t msg_len;       /* Total message length including header */
    uint16_t func_id;       /* CUDA function identifier */
    uint16_t flags;         /* JUICE_FLAG_* */
    uint32_t seq_num;       /* Sequence number for request/response matching */
    uint32_t payload_size;  /* Size of payload following this header */
} juice_request_t;

/*
 * Response header (12 bytes + optional payload)
 */
typedef struct __attribute__((packed)) {
    uint32_t msg_len;       /* Total message length including header */
    uint32_t seq_num;       /* Matching request sequence number */
    int32_t  result;        /* CUresult / cudaError_t */
    /* Followed by output payload if any */
} juice_response_t;

#define JUICE_REQUEST_HEADER_SIZE  sizeof(juice_request_t)
#define JUICE_RESPONSE_HEADER_SIZE sizeof(juice_response_t)

/*
 * CUDA Driver API function IDs
 * Organized by functional group with gaps for future additions.
 */
typedef enum {
    /* Initialization (0x0000 - 0x000F) */
    JUICE_CU_INIT                       = 0x0001,
    JUICE_CU_DRIVER_GET_VERSION         = 0x0002,

    /* Device Management (0x0010 - 0x002F) */
    JUICE_CU_DEVICE_GET                 = 0x0010,
    JUICE_CU_DEVICE_GET_COUNT           = 0x0011,
    JUICE_CU_DEVICE_GET_NAME            = 0x0012,
    JUICE_CU_DEVICE_TOTAL_MEM           = 0x0013,
    JUICE_CU_DEVICE_GET_ATTRIBUTE       = 0x0014,
    JUICE_CU_DEVICE_GET_UUID            = 0x0015,
    JUICE_CU_DEVICE_GET_PROPERTIES      = 0x0016,
    JUICE_CU_DEVICE_COMPUTE_CAPABILITY  = 0x0017,

    /* Context Management (0x0030 - 0x004F) */
    JUICE_CU_CTX_CREATE                 = 0x0030,
    JUICE_CU_CTX_DESTROY                = 0x0031,
    JUICE_CU_CTX_SET_CURRENT            = 0x0032,
    JUICE_CU_CTX_GET_CURRENT            = 0x0033,
    JUICE_CU_CTX_PUSH_CURRENT           = 0x0034,
    JUICE_CU_CTX_POP_CURRENT            = 0x0035,
    JUICE_CU_CTX_SYNCHRONIZE            = 0x0036,
    JUICE_CU_DEVICE_PRIMARY_CTX_RETAIN  = 0x0037,
    JUICE_CU_DEVICE_PRIMARY_CTX_RELEASE = 0x0038,
    JUICE_CU_DEVICE_PRIMARY_CTX_RESET   = 0x0039,
    JUICE_CU_CTX_GET_DEVICE             = 0x003A,

    /* Module Management (0x0050 - 0x006F) */
    JUICE_CU_MODULE_LOAD               = 0x0050,
    JUICE_CU_MODULE_LOAD_DATA          = 0x0051,
    JUICE_CU_MODULE_LOAD_DATA_EX       = 0x0052,
    JUICE_CU_MODULE_UNLOAD             = 0x0053,
    JUICE_CU_MODULE_GET_FUNCTION       = 0x0054,
    JUICE_CU_MODULE_GET_GLOBAL         = 0x0055,

    /* Memory Management (0x0070 - 0x009F) */
    JUICE_CU_MEM_ALLOC                 = 0x0070,
    JUICE_CU_MEM_FREE                  = 0x0071,
    JUICE_CU_MEMCPY_H_TO_D            = 0x0072,
    JUICE_CU_MEMCPY_D_TO_H            = 0x0073,
    JUICE_CU_MEMCPY_D_TO_D            = 0x0074,
    JUICE_CU_MEMCPY_H_TO_D_ASYNC      = 0x0075,
    JUICE_CU_MEMCPY_D_TO_H_ASYNC      = 0x0076,
    JUICE_CU_MEMCPY_D_TO_D_ASYNC      = 0x0077,
    JUICE_CU_MEMSET_D8                 = 0x0078,
    JUICE_CU_MEMSET_D16                = 0x0079,
    JUICE_CU_MEMSET_D32                = 0x007A,
    JUICE_CU_MEM_GET_INFO              = 0x007B,
    JUICE_CU_MEM_HOST_ALLOC            = 0x007C,
    JUICE_CU_MEM_FREE_HOST             = 0x007D,
    JUICE_CU_MEM_HOST_GET_DEVICE_PTR   = 0x007E,
    JUICE_CU_POINTER_GET_ATTRIBUTE     = 0x007F,
    JUICE_CU_MEM_ALLOC_MANAGED         = 0x0080,
    JUICE_CU_MEMCPY                    = 0x0081,

    /* Execution Control (0x00A0 - 0x00BF) */
    JUICE_CU_LAUNCH_KERNEL             = 0x00A0,
    JUICE_CU_LAUNCH_COOPERATIVE_KERNEL = 0x00A1,
    JUICE_CU_FUNC_GET_ATTRIBUTE        = 0x00A2,
    JUICE_CU_FUNC_SET_ATTRIBUTE        = 0x00A3,
    JUICE_CU_FUNC_SET_CACHE_CONFIG     = 0x00A4,

    /* Stream Management (0x00C0 - 0x00DF) */
    JUICE_CU_STREAM_CREATE             = 0x00C0,
    JUICE_CU_STREAM_CREATE_WITH_PRIORITY = 0x00C1,
    JUICE_CU_STREAM_DESTROY            = 0x00C2,
    JUICE_CU_STREAM_SYNCHRONIZE        = 0x00C3,
    JUICE_CU_STREAM_QUERY              = 0x00C4,
    JUICE_CU_STREAM_WAIT_EVENT         = 0x00C5,

    /* Event Management (0x00E0 - 0x00FF) */
    JUICE_CU_EVENT_CREATE              = 0x00E0,
    JUICE_CU_EVENT_DESTROY             = 0x00E1,
    JUICE_CU_EVENT_RECORD              = 0x00E2,
    JUICE_CU_EVENT_SYNCHRONIZE         = 0x00E3,
    JUICE_CU_EVENT_QUERY               = 0x00E4,
    JUICE_CU_EVENT_ELAPSED_TIME        = 0x00E5,

    /* CUDA Runtime API (0x0100 - 0x01FF) */
    JUICE_CUDA_MALLOC                  = 0x0100,
    JUICE_CUDA_FREE                    = 0x0101,
    JUICE_CUDA_MEMCPY                  = 0x0102,
    JUICE_CUDA_MEMCPY_ASYNC            = 0x0103,
    JUICE_CUDA_MEMSET                  = 0x0104,
    JUICE_CUDA_MEMSET_ASYNC            = 0x0105,
    JUICE_CUDA_DEVICE_SYNCHRONIZE      = 0x0106,
    JUICE_CUDA_STREAM_SYNCHRONIZE      = 0x0107,
    JUICE_CUDA_GET_DEVICE              = 0x0108,
    JUICE_CUDA_SET_DEVICE              = 0x0109,
    JUICE_CUDA_GET_DEVICE_COUNT        = 0x010A,
    JUICE_CUDA_GET_DEVICE_PROPERTIES   = 0x010B,
    JUICE_CUDA_DEVICE_GET_ATTRIBUTE    = 0x010C,
    JUICE_CUDA_STREAM_CREATE           = 0x010D,
    JUICE_CUDA_STREAM_DESTROY          = 0x010E,
    JUICE_CUDA_EVENT_CREATE            = 0x010F,
    JUICE_CUDA_EVENT_DESTROY           = 0x0110,
    JUICE_CUDA_EVENT_RECORD            = 0x0111,
    JUICE_CUDA_EVENT_SYNCHRONIZE       = 0x0112,
    JUICE_CUDA_EVENT_ELAPSED_TIME      = 0x0113,
    JUICE_CUDA_LAUNCH_KERNEL           = 0x0114,
    JUICE_CUDA_MEM_GET_INFO            = 0x0115,
    JUICE_CUDA_DEVICE_RESET            = 0x0116,
    JUICE_CUDA_RUNTIME_GET_VERSION     = 0x0117,
    JUICE_CUDA_DRIVER_GET_VERSION      = 0x0118,
    JUICE_CUDA_DEVICE_CAN_ACCESS_PEER  = 0x0119,
    JUICE_CUDA_FUNC_GET_ATTRIBUTES     = 0x011A,
    JUICE_CUDA_GET_LAST_ERROR          = 0x011B,
    JUICE_CUDA_PEEK_AT_LAST_ERROR      = 0x011C,

    /* cuBLAS (0x0200 - 0x02FF) — Phase 4 */
    JUICE_CUBLAS_CREATE                = 0x0200,
    JUICE_CUBLAS_DESTROY               = 0x0201,
    JUICE_CUBLAS_SET_STREAM            = 0x0202,
    JUICE_CUBLAS_SGEMM                 = 0x0210,
    JUICE_CUBLAS_DGEMM                 = 0x0211,
    JUICE_CUBLAS_HGEMM                 = 0x0212,
    JUICE_CUBLAS_SGEMM_EX             = 0x0213,
    JUICE_CUBLAS_GEMM_EX              = 0x0214,

    /* Sentinel */
    JUICE_FUNC_MAX                     = 0xFFFF,
} juice_func_id_t;

/*
 * Network I/O helpers
 */

/* Read exactly `len` bytes from fd. Returns 0 on success, -1 on error/EOF. */
int juice_read_full(int fd, void *buf, size_t len);

/* Write exactly `len` bytes to fd. Returns 0 on success, -1 on error. */
int juice_write_full(int fd, const void *buf, size_t len);

/* Send a request (header + payload). Returns 0 on success. */
int juice_send_request(int fd, juice_request_t *hdr, const void *payload, size_t payload_len);

/* Receive a request header. Caller then reads payload_size bytes. Returns 0 on success. */
int juice_recv_request(int fd, juice_request_t *hdr);

/* Send a response (header + optional payload). Returns 0 on success. */
int juice_send_response(int fd, uint32_t seq_num, int32_t result, const void *payload, size_t payload_len);

/* Receive a response header. Caller then reads remaining payload. Returns 0 on success. */
int juice_recv_response(int fd, juice_response_t *hdr);

#ifdef __cplusplus
}
#endif

#endif /* JUICE_PROTOCOL_H */

/*
 * Chidori GPU-over-IP Wire Protocol
 *
 * Binary protocol for CUDA API remoting over TCP.
 * All values are native byte order (both sides assumed x86_64).
 */
#ifndef CHIDORI_PROTOCOL_H
#define CHIDORI_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Protocol version */
#define CHIDORI_PROTOCOL_VERSION 1

/* Message flags */
#define CHIDORI_FLAG_SYNC          0x0001  /* Synchronous call, client blocks for response */
#define CHIDORI_FLAG_ASYNC         0x0002  /* Async call, no immediate response needed */
#define CHIDORI_FLAG_BATCH_MORE    0x0004  /* More messages follow in this batch */
#define CHIDORI_FLAG_ERROR         0x0008  /* Response indicates error */

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
    uint16_t flags;         /* CHIDORI_FLAG_* */
    uint32_t seq_num;       /* Sequence number for request/response matching */
    uint32_t payload_size;  /* Size of payload following this header */
} chidori_request_t;

/*
 * Response header (12 bytes + optional payload)
 */
typedef struct __attribute__((packed)) {
    uint32_t msg_len;       /* Total message length including header */
    uint32_t seq_num;       /* Matching request sequence number */
    int32_t  result;        /* CUresult / cudaError_t */
    /* Followed by output payload if any */
} chidori_response_t;

#define CHIDORI_REQUEST_HEADER_SIZE  sizeof(chidori_request_t)
#define CHIDORI_RESPONSE_HEADER_SIZE sizeof(chidori_response_t)

/*
 * CUDA Driver API function IDs
 * Organized by functional group with gaps for future additions.
 */
typedef enum {
    /* Initialization (0x0000 - 0x000F) */
    CHIDORI_CU_INIT                       = 0x0001,
    CHIDORI_CU_DRIVER_GET_VERSION         = 0x0002,

    /* Device Management (0x0010 - 0x002F) */
    CHIDORI_CU_DEVICE_GET                 = 0x0010,
    CHIDORI_CU_DEVICE_GET_COUNT           = 0x0011,
    CHIDORI_CU_DEVICE_GET_NAME            = 0x0012,
    CHIDORI_CU_DEVICE_TOTAL_MEM           = 0x0013,
    CHIDORI_CU_DEVICE_GET_ATTRIBUTE       = 0x0014,
    CHIDORI_CU_DEVICE_GET_UUID            = 0x0015,
    CHIDORI_CU_DEVICE_GET_PROPERTIES      = 0x0016,
    CHIDORI_CU_DEVICE_COMPUTE_CAPABILITY  = 0x0017,

    /* Context Management (0x0030 - 0x004F) */
    CHIDORI_CU_CTX_CREATE                 = 0x0030,
    CHIDORI_CU_CTX_DESTROY                = 0x0031,
    CHIDORI_CU_CTX_SET_CURRENT            = 0x0032,
    CHIDORI_CU_CTX_GET_CURRENT            = 0x0033,
    CHIDORI_CU_CTX_PUSH_CURRENT           = 0x0034,
    CHIDORI_CU_CTX_POP_CURRENT            = 0x0035,
    CHIDORI_CU_CTX_SYNCHRONIZE            = 0x0036,
    CHIDORI_CU_CTX_SET_LIMIT              = 0x003B,
    CHIDORI_CU_CTX_GET_LIMIT              = 0x003C,
    CHIDORI_CU_DEVICE_PRIMARY_CTX_RETAIN  = 0x0037,
    CHIDORI_CU_DEVICE_PRIMARY_CTX_RELEASE = 0x0038,
    CHIDORI_CU_DEVICE_PRIMARY_CTX_RESET   = 0x0039,
    CHIDORI_CU_CTX_GET_DEVICE             = 0x003A,

    /* Module Management (0x0050 - 0x006F) */
    CHIDORI_CU_MODULE_LOAD               = 0x0050,
    CHIDORI_CU_MODULE_LOAD_DATA          = 0x0051,
    CHIDORI_CU_MODULE_LOAD_DATA_EX       = 0x0052,
    CHIDORI_CU_MODULE_UNLOAD             = 0x0053,
    CHIDORI_CU_MODULE_GET_FUNCTION       = 0x0054,
    CHIDORI_CU_MODULE_GET_GLOBAL         = 0x0055,

    /* Memory Management (0x0070 - 0x009F) */
    CHIDORI_CU_MEM_ALLOC                 = 0x0070,
    CHIDORI_CU_MEM_FREE                  = 0x0071,
    CHIDORI_CU_MEMCPY_H_TO_D            = 0x0072,
    CHIDORI_CU_MEMCPY_D_TO_H            = 0x0073,
    CHIDORI_CU_MEMCPY_D_TO_D            = 0x0074,
    CHIDORI_CU_MEMCPY_H_TO_D_ASYNC      = 0x0075,
    CHIDORI_CU_MEMCPY_D_TO_H_ASYNC      = 0x0076,
    CHIDORI_CU_MEMCPY_D_TO_D_ASYNC      = 0x0077,
    CHIDORI_CU_MEMSET_D8                 = 0x0078,
    CHIDORI_CU_MEMSET_D16                = 0x0079,
    CHIDORI_CU_MEMSET_D32                = 0x007A,
    CHIDORI_CU_MEM_GET_INFO              = 0x007B,
    CHIDORI_CU_MEM_HOST_ALLOC            = 0x007C,
    CHIDORI_CU_MEM_FREE_HOST             = 0x007D,
    CHIDORI_CU_MEM_HOST_GET_DEVICE_PTR   = 0x007E,
    CHIDORI_CU_POINTER_GET_ATTRIBUTE     = 0x007F,
    CHIDORI_CU_MEM_ALLOC_MANAGED         = 0x0080,
    CHIDORI_CU_MEMCPY                    = 0x0081,

    /* Execution Control (0x00A0 - 0x00BF) */
    CHIDORI_CU_LAUNCH_KERNEL             = 0x00A0,
    CHIDORI_CU_LAUNCH_COOPERATIVE_KERNEL = 0x00A1,
    CHIDORI_CU_FUNC_GET_ATTRIBUTE        = 0x00A2,
    CHIDORI_CU_FUNC_SET_ATTRIBUTE        = 0x00A3,
    CHIDORI_CU_FUNC_SET_CACHE_CONFIG     = 0x00A4,

    /* Stream Management (0x00C0 - 0x00DF) */
    CHIDORI_CU_STREAM_CREATE             = 0x00C0,
    CHIDORI_CU_STREAM_CREATE_WITH_PRIORITY = 0x00C1,
    CHIDORI_CU_STREAM_DESTROY            = 0x00C2,
    CHIDORI_CU_STREAM_SYNCHRONIZE        = 0x00C3,
    CHIDORI_CU_STREAM_QUERY              = 0x00C4,
    CHIDORI_CU_STREAM_WAIT_EVENT         = 0x00C5,

    /* Event Management (0x00E0 - 0x00FF) */
    CHIDORI_CU_EVENT_CREATE              = 0x00E0,
    CHIDORI_CU_EVENT_DESTROY             = 0x00E1,
    CHIDORI_CU_EVENT_RECORD              = 0x00E2,
    CHIDORI_CU_EVENT_SYNCHRONIZE         = 0x00E3,
    CHIDORI_CU_EVENT_QUERY               = 0x00E4,
    CHIDORI_CU_EVENT_ELAPSED_TIME        = 0x00E5,

    /* CUDA Runtime API (0x0100 - 0x01FF) */
    CHIDORI_CUDA_MALLOC                  = 0x0100,
    CHIDORI_CUDA_FREE                    = 0x0101,
    CHIDORI_CUDA_MEMCPY                  = 0x0102,
    CHIDORI_CUDA_MEMCPY_ASYNC            = 0x0103,
    CHIDORI_CUDA_MEMSET                  = 0x0104,
    CHIDORI_CUDA_MEMSET_ASYNC            = 0x0105,
    CHIDORI_CUDA_DEVICE_SYNCHRONIZE      = 0x0106,
    CHIDORI_CUDA_STREAM_SYNCHRONIZE      = 0x0107,
    CHIDORI_CUDA_GET_DEVICE              = 0x0108,
    CHIDORI_CUDA_SET_DEVICE              = 0x0109,
    CHIDORI_CUDA_GET_DEVICE_COUNT        = 0x010A,
    CHIDORI_CUDA_GET_DEVICE_PROPERTIES   = 0x010B,
    CHIDORI_CUDA_DEVICE_GET_ATTRIBUTE    = 0x010C,
    CHIDORI_CUDA_STREAM_CREATE           = 0x010D,
    CHIDORI_CUDA_STREAM_DESTROY          = 0x010E,
    CHIDORI_CUDA_EVENT_CREATE            = 0x010F,
    CHIDORI_CUDA_EVENT_DESTROY           = 0x0110,
    CHIDORI_CUDA_EVENT_RECORD            = 0x0111,
    CHIDORI_CUDA_EVENT_SYNCHRONIZE       = 0x0112,
    CHIDORI_CUDA_EVENT_ELAPSED_TIME      = 0x0113,
    CHIDORI_CUDA_LAUNCH_KERNEL           = 0x0114,
    CHIDORI_CUDA_MEM_GET_INFO            = 0x0115,
    CHIDORI_CUDA_DEVICE_RESET            = 0x0116,
    CHIDORI_CUDA_RUNTIME_GET_VERSION     = 0x0117,
    CHIDORI_CUDA_DRIVER_GET_VERSION      = 0x0118,
    CHIDORI_CUDA_DEVICE_CAN_ACCESS_PEER  = 0x0119,
    CHIDORI_CUDA_FUNC_GET_ATTRIBUTES     = 0x011A,
    CHIDORI_CUDA_GET_LAST_ERROR          = 0x011B,
    CHIDORI_CUDA_PEEK_AT_LAST_ERROR      = 0x011C,
    CHIDORI_CUDA_REGISTER_FAT_BINARY     = 0x011D,
    CHIDORI_CUDA_REGISTER_FUNCTION       = 0x011E,

    /* CUDA Graph API (0x0120 - 0x012F) */
    CHIDORI_CU_STREAM_BEGIN_CAPTURE      = 0x0120,
    CHIDORI_CU_STREAM_END_CAPTURE        = 0x0121,
    CHIDORI_CU_STREAM_IS_CAPTURING       = 0x0122,
    CHIDORI_CU_GRAPH_CREATE              = 0x0123,
    CHIDORI_CU_GRAPH_DESTROY             = 0x0124,
    CHIDORI_CU_GRAPH_INSTANTIATE         = 0x0125,
    CHIDORI_CU_GRAPH_EXEC_DESTROY        = 0x0126,
    CHIDORI_CU_GRAPH_LAUNCH             = 0x0127,
    CHIDORI_CU_GRAPH_EXEC_UPDATE         = 0x0128,
    CHIDORI_CU_GRAPH_UPLOAD              = 0x0129,

    /* cuBLAS (0x0200 - 0x02FF) — Phase 4 */
    CHIDORI_CUBLAS_CREATE                = 0x0200,
    CHIDORI_CUBLAS_DESTROY               = 0x0201,
    CHIDORI_CUBLAS_SET_STREAM            = 0x0202,
    CHIDORI_CUBLAS_SET_MATH_MODE         = 0x0203,
    CHIDORI_CUBLAS_GET_MATH_MODE         = 0x0204,
    CHIDORI_CUBLAS_GET_STREAM            = 0x0205,

    /* GEMM family (0x0210 - 0x021F) */
    CHIDORI_CUBLAS_SGEMM                 = 0x0210,
    CHIDORI_CUBLAS_DGEMM                 = 0x0211,
    CHIDORI_CUBLAS_HGEMM                 = 0x0212,
    CHIDORI_CUBLAS_SGEMM_EX             = 0x0213,
    CHIDORI_CUBLAS_GEMM_EX              = 0x0214,

    /* GEMV family (0x0220 - 0x022F) */
    CHIDORI_CUBLAS_SGEMV                 = 0x0220,
    CHIDORI_CUBLAS_DGEMV                 = 0x0221,

    /* AXPY family (0x0230 - 0x023F) */
    CHIDORI_CUBLAS_SAXPY                 = 0x0230,
    CHIDORI_CUBLAS_DAXPY                 = 0x0231,

    /* NRM2 family (0x0240 - 0x024F) */
    CHIDORI_CUBLAS_SNRM2                 = 0x0240,
    CHIDORI_CUBLAS_DNRM2                 = 0x0241,

    /* DOT family (0x0250 - 0x025F) */
    CHIDORI_CUBLAS_SDOT                  = 0x0250,
    CHIDORI_CUBLAS_DDOT                  = 0x0251,

    /* SCAL family (0x0260 - 0x026F) */
    CHIDORI_CUBLAS_SSCAL                 = 0x0260,
    CHIDORI_CUBLAS_DSCAL                 = 0x0261,

    /* COPY family (0x0270 - 0x027F) */
    CHIDORI_CUBLAS_SCOPY                 = 0x0270,
    CHIDORI_CUBLAS_DCOPY                 = 0x0271,

    /* Batched GEMM (0x0280 - 0x028F) */
    CHIDORI_CUBLAS_SGEMM_BATCHED         = 0x0280,
    CHIDORI_CUBLAS_SGEMM_STRIDED_BATCHED = 0x0281,
    CHIDORI_CUBLAS_GEMM_STRIDED_BATCHED_EX = 0x0282,

    /* Utility (0x0290 - 0x029F) */
    CHIDORI_CUBLAS_GET_VERSION           = 0x0290,

    /* cublasLt (0x02A0 - 0x02AF) */
    CHIDORI_CUBLAS_LT_CREATE                 = 0x02A0,
    CHIDORI_CUBLAS_LT_MATMUL                 = 0x02A1,
    CHIDORI_CUBLAS_LT_MATMUL_ALGO_HEURISTIC  = 0x02A2,

    /* cuDNN (0x0300 - 0x03FF) — Phase 5 */
    CHIDORI_CUDNN_CREATE                              = 0x0300,
    CHIDORI_CUDNN_DESTROY                             = 0x0301,
    CHIDORI_CUDNN_SET_STREAM                          = 0x0302,
    CHIDORI_CUDNN_GET_VERSION                         = 0x0303,

    /* Tensor descriptor (0x0310) */
    CHIDORI_CUDNN_CREATE_TENSOR_DESCRIPTOR            = 0x0310,
    CHIDORI_CUDNN_SET_TENSOR_4D_DESCRIPTOR            = 0x0311,
    CHIDORI_CUDNN_SET_TENSOR_ND_DESCRIPTOR            = 0x0312,
    CHIDORI_CUDNN_DESTROY_TENSOR_DESCRIPTOR           = 0x0313,
    CHIDORI_CUDNN_GET_TENSOR_4D_DESCRIPTOR            = 0x0314,

    /* Filter descriptor (0x0320) */
    CHIDORI_CUDNN_CREATE_FILTER_DESCRIPTOR            = 0x0320,
    CHIDORI_CUDNN_SET_FILTER_4D_DESCRIPTOR            = 0x0321,
    CHIDORI_CUDNN_SET_FILTER_ND_DESCRIPTOR            = 0x0322,
    CHIDORI_CUDNN_DESTROY_FILTER_DESCRIPTOR           = 0x0323,

    /* Convolution (0x0330) */
    CHIDORI_CUDNN_CREATE_CONVOLUTION_DESCRIPTOR       = 0x0330,
    CHIDORI_CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR       = 0x0331,
    CHIDORI_CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR       = 0x0332,
    CHIDORI_CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR      = 0x0333,
    CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM   = 0x0334,
    CHIDORI_CUDNN_CONVOLUTION_FORWARD                 = 0x0335,
    CHIDORI_CUDNN_CONVOLUTION_BACKWARD_DATA           = 0x0336,
    CHIDORI_CUDNN_CONVOLUTION_BACKWARD_FILTER         = 0x0337,
    CHIDORI_CUDNN_SET_CONVOLUTION_MATH_TYPE           = 0x0338,
    CHIDORI_CUDNN_SET_CONVOLUTION_GROUP_COUNT         = 0x0339,

    /* Activation (0x0340) */
    CHIDORI_CUDNN_CREATE_ACTIVATION_DESCRIPTOR        = 0x0340,
    CHIDORI_CUDNN_SET_ACTIVATION_DESCRIPTOR           = 0x0341,
    CHIDORI_CUDNN_DESTROY_ACTIVATION_DESCRIPTOR       = 0x0342,
    CHIDORI_CUDNN_ACTIVATION_FORWARD                  = 0x0343,
    CHIDORI_CUDNN_ACTIVATION_BACKWARD                 = 0x0344,

    /* Pooling (0x0350) */
    CHIDORI_CUDNN_CREATE_POOLING_DESCRIPTOR           = 0x0350,
    CHIDORI_CUDNN_SET_POOLING_2D_DESCRIPTOR           = 0x0351,
    CHIDORI_CUDNN_DESTROY_POOLING_DESCRIPTOR          = 0x0352,
    CHIDORI_CUDNN_POOLING_FORWARD                     = 0x0353,
    CHIDORI_CUDNN_POOLING_BACKWARD                    = 0x0354,

    /* Batch Normalization (0x0360) */
    CHIDORI_CUDNN_BATCH_NORM_FORWARD_TRAINING         = 0x0360,
    CHIDORI_CUDNN_BATCH_NORM_FORWARD_INFERENCE        = 0x0361,
    CHIDORI_CUDNN_BATCH_NORM_BACKWARD                 = 0x0362,

    /* Softmax (0x0370) */
    CHIDORI_CUDNN_SOFTMAX_FORWARD                     = 0x0370,
    CHIDORI_CUDNN_SOFTMAX_BACKWARD                    = 0x0371,

    /* Dropout (0x0380) */
    CHIDORI_CUDNN_CREATE_DROPOUT_DESCRIPTOR           = 0x0380,
    CHIDORI_CUDNN_SET_DROPOUT_DESCRIPTOR              = 0x0381,
    CHIDORI_CUDNN_DESTROY_DROPOUT_DESCRIPTOR          = 0x0382,
    CHIDORI_CUDNN_DROPOUT_FORWARD                     = 0x0383,

    /* RNN (0x0390) */
    CHIDORI_CUDNN_CREATE_RNN_DESCRIPTOR               = 0x0390,
    CHIDORI_CUDNN_DESTROY_RNN_DESCRIPTOR              = 0x0391,

    /* Workspace (0x03A0) */
    CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE      = 0x03A0,
    CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE  = 0x03A1,
    CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE = 0x03A2,

    /* ================================================================
     * Replay Control (0x0400 - 0x04FF)
     *
     * Server-side call sequence recording and replay. Records one
     * training step, replays it on subsequent steps with only the
     * changed data (input batch) sent over the wire. Eliminates
     * per-call RPC overhead for repeated training steps.
     * ================================================================ */
    CHIDORI_REPLAY_BEGIN_RECORD          = 0x0400,  /* Start recording a call sequence */
    CHIDORI_REPLAY_END_RECORD            = 0x0401,  /* Finalize recording, get fingerprint */
    CHIDORI_REPLAY_EXECUTE               = 0x0402,  /* Replay with new variable data */
    CHIDORI_REPLAY_INVALIDATE            = 0x0403,  /* Discard a recorded sequence */
    CHIDORI_REPLAY_STATUS                = 0x0404,  /* Query replay state/stats */

    /* Bulk Transfer (0x0500) — combined alloc + HtoD in one RPC */
    CHIDORI_ALLOC_AND_TRANSFER           = 0x0500,

    /* Sentinel */
    CHIDORI_FUNC_MAX                     = 0xFFFF,
} chidori_func_id_t;

/*
 * Network I/O helpers
 */

/* Read exactly `len` bytes from fd. Returns 0 on success, -1 on error/EOF. */
int chidori_read_full(int fd, void *buf, size_t len);

/* Write exactly `len` bytes to fd. Returns 0 on success, -1 on error. */
int chidori_write_full(int fd, const void *buf, size_t len);

/* Send a request (header + payload). Returns 0 on success. */
int chidori_send_request(int fd, chidori_request_t *hdr, const void *payload, size_t payload_len);

/* Receive a request header. Caller then reads payload_size bytes. Returns 0 on success. */
int chidori_recv_request(int fd, chidori_request_t *hdr);

/* Send a response (header + optional payload). Returns 0 on success. */
int chidori_send_response(int fd, uint32_t seq_num, int32_t result, const void *payload, size_t payload_len);

/* Receive a response header. Caller then reads remaining payload. Returns 0 on success. */
int chidori_recv_response(int fd, chidori_response_t *hdr);

#ifdef __cplusplus
}
#endif

#endif /* CHIDORI_PROTOCOL_H */

# Chidori Wire Protocol

## Protocol Overview

- **Byte order:** Native byte order (x86_64 only), no endian conversion
- **Protocol version:** 1 (`CHIDORI_PROTOCOL_VERSION`)
- **Packing:** All structs are `__attribute__((packed))`
- **Transport:** TCP, blocking I/O (no select/poll/epoll)
- **Message framing:** Length-prefixed (msg_len includes the header itself)

## Request Header (16 bytes)

```
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                       Message Length                          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |         Function ID           |            Flags              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                      Sequence Number                         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                       Payload Size                           |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | uint32_t | msg_len | Total message length including this header |
| 4 | 2 | uint16_t | func_id | CUDA function identifier (see Function ID Table) |
| 6 | 2 | uint16_t | flags | Bitfield of CHIDORI_FLAG_* values |
| 8 | 4 | uint32_t | seq_num | Sequence number for request/response matching |
| 12 | 4 | uint32_t | payload_size | Size of variable-length payload following header |

C struct definition:
```c
typedef struct __attribute__((packed)) {
    uint32_t msg_len;       /* Total message length including header */
    uint16_t func_id;       /* CUDA function identifier */
    uint16_t flags;         /* CHIDORI_FLAG_* */
    uint32_t seq_num;       /* Sequence number for request/response matching */
    uint32_t payload_size;  /* Size of payload following this header */
} chidori_request_t;
```

## Response Header (12 bytes + optional payload)

```
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                       Message Length                          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                      Sequence Number                         |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                          Result                              |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                                                               |
 |                   Optional Payload Data                       |
 |                                                               |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | uint32_t | msg_len | Total message length including this header |
| 4 | 4 | uint32_t | seq_num | Matching request sequence number |
| 8 | 4 | int32_t | result | CUresult / cudaError_t status code |
| 12 | ? | bytes | payload | Optional output payload (msg_len - 12 bytes) |

C struct definition:
```c
typedef struct __attribute__((packed)) {
    uint32_t msg_len;       /* Total message length including header */
    uint32_t seq_num;       /* Matching request sequence number */
    int32_t  result;        /* CUresult / cudaError_t */
    /* Followed by output payload if any */
} chidori_response_t;
```

## Flag Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CHIDORI_FLAG_SYNC` | 0x0001 | Synchronous call -- client blocks for response |
| `CHIDORI_FLAG_ASYNC` | 0x0002 | Async call -- fire-and-forget, no response sent by server |
| `CHIDORI_FLAG_BATCH_MORE` | 0x0004 | More messages follow in this batch |
| `CHIDORI_FLAG_ERROR` | 0x0008 | Response indicates error |

The server uses `CHIDORI_FLAG_ASYNC` to determine whether to send a response. When dispatching, if the request has `CHIDORI_FLAG_ASYNC`, the server passes `fd=-1` to `cuda_dispatch()`, and `chidori_send_response()` returns immediately without writing.

## Complete Function ID Table

### Driver API Initialization (0x0001-0x000F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0001 | `CHIDORI_CU_INIT` | cuInit |
| 0x0002 | `CHIDORI_CU_DRIVER_GET_VERSION` | cuDriverGetVersion |

### Driver API Device Management (0x0010-0x001F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0010 | `CHIDORI_CU_DEVICE_GET` | cuDeviceGet |
| 0x0011 | `CHIDORI_CU_DEVICE_GET_COUNT` | cuDeviceGetCount |
| 0x0012 | `CHIDORI_CU_DEVICE_GET_NAME` | cuDeviceGetName |
| 0x0013 | `CHIDORI_CU_DEVICE_TOTAL_MEM` | cuDeviceTotalMem_v2 |
| 0x0014 | `CHIDORI_CU_DEVICE_GET_ATTRIBUTE` | cuDeviceGetAttribute |
| 0x0015 | `CHIDORI_CU_DEVICE_GET_UUID` | cuDeviceGetUuid |
| 0x0016 | `CHIDORI_CU_DEVICE_GET_PROPERTIES` | cuDeviceGetProperties |
| 0x0017 | `CHIDORI_CU_DEVICE_COMPUTE_CAPABILITY` | Compute capability query |

### Driver API Context Management (0x0030-0x003F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0030 | `CHIDORI_CU_CTX_CREATE` | cuCtxCreate_v2 |
| 0x0031 | `CHIDORI_CU_CTX_DESTROY` | cuCtxDestroy_v2 |
| 0x0032 | `CHIDORI_CU_CTX_SET_CURRENT` | cuCtxSetCurrent |
| 0x0033 | `CHIDORI_CU_CTX_GET_CURRENT` | cuCtxGetCurrent |
| 0x0034 | `CHIDORI_CU_CTX_PUSH_CURRENT` | cuCtxPushCurrent |
| 0x0035 | `CHIDORI_CU_CTX_POP_CURRENT` | cuCtxPopCurrent |
| 0x0036 | `CHIDORI_CU_CTX_SYNCHRONIZE` | cuCtxSynchronize |
| 0x0037 | `CHIDORI_CU_DEVICE_PRIMARY_CTX_RETAIN` | cuDevicePrimaryCtxRetain |
| 0x0038 | `CHIDORI_CU_DEVICE_PRIMARY_CTX_RELEASE` | cuDevicePrimaryCtxRelease_v2 |
| 0x0039 | `CHIDORI_CU_DEVICE_PRIMARY_CTX_RESET` | cuDevicePrimaryCtxReset |
| 0x003A | `CHIDORI_CU_CTX_GET_DEVICE` | cuCtxGetDevice |
| 0x003B | `CHIDORI_CU_CTX_SET_LIMIT` | cuCtxSetLimit |
| 0x003C | `CHIDORI_CU_CTX_GET_LIMIT` | cuCtxGetLimit |

### Driver API Module Management (0x0050-0x005F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0050 | `CHIDORI_CU_MODULE_LOAD` | cuModuleLoad |
| 0x0051 | `CHIDORI_CU_MODULE_LOAD_DATA` | cuModuleLoadData |
| 0x0052 | `CHIDORI_CU_MODULE_LOAD_DATA_EX` | cuModuleLoadDataEx |
| 0x0053 | `CHIDORI_CU_MODULE_UNLOAD` | cuModuleUnload |
| 0x0054 | `CHIDORI_CU_MODULE_GET_FUNCTION` | cuModuleGetFunction |
| 0x0055 | `CHIDORI_CU_MODULE_GET_GLOBAL` | cuModuleGetGlobal |

### Driver API Memory Management (0x0070-0x008F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0070 | `CHIDORI_CU_MEM_ALLOC` | cuMemAlloc_v2 |
| 0x0071 | `CHIDORI_CU_MEM_FREE` | cuMemFree_v2 |
| 0x0072 | `CHIDORI_CU_MEMCPY_H_TO_D` | cuMemcpyHtoD_v2 |
| 0x0073 | `CHIDORI_CU_MEMCPY_D_TO_H` | cuMemcpyDtoH_v2 |
| 0x0074 | `CHIDORI_CU_MEMCPY_D_TO_D` | cuMemcpyDtoD_v2 |
| 0x0075 | `CHIDORI_CU_MEMCPY_H_TO_D_ASYNC` | cuMemcpyHtoDAsync |
| 0x0076 | `CHIDORI_CU_MEMCPY_D_TO_H_ASYNC` | cuMemcpyDtoHAsync |
| 0x0077 | `CHIDORI_CU_MEMCPY_D_TO_D_ASYNC` | cuMemcpyDtoDAsync |
| 0x0078 | `CHIDORI_CU_MEMSET_D8` | cuMemsetD8 |
| 0x0079 | `CHIDORI_CU_MEMSET_D16` | cuMemsetD16 |
| 0x007A | `CHIDORI_CU_MEMSET_D32` | cuMemsetD32 |
| 0x007B | `CHIDORI_CU_MEM_GET_INFO` | cuMemGetInfo_v2 |
| 0x007C | `CHIDORI_CU_MEM_HOST_ALLOC` | cuMemHostAlloc |
| 0x007D | `CHIDORI_CU_MEM_FREE_HOST` | cuMemFreeHost |
| 0x007E | `CHIDORI_CU_MEM_HOST_GET_DEVICE_PTR` | cuMemHostGetDevicePointer_v2 |
| 0x007F | `CHIDORI_CU_POINTER_GET_ATTRIBUTE` | cuPointerGetAttribute |
| 0x0080 | `CHIDORI_CU_MEM_ALLOC_MANAGED` | cuMemAllocManaged |
| 0x0081 | `CHIDORI_CU_MEMCPY` | cuMemcpy |

### Driver API Execution Control (0x00A0-0x00AF)

| ID | Constant | Function |
|----|----------|----------|
| 0x00A0 | `CHIDORI_CU_LAUNCH_KERNEL` | cuLaunchKernel |
| 0x00A1 | `CHIDORI_CU_LAUNCH_COOPERATIVE_KERNEL` | cuLaunchCooperativeKernel |
| 0x00A2 | `CHIDORI_CU_FUNC_GET_ATTRIBUTE` | cuFuncGetAttribute |
| 0x00A3 | `CHIDORI_CU_FUNC_SET_ATTRIBUTE` | cuFuncSetAttribute |
| 0x00A4 | `CHIDORI_CU_FUNC_SET_CACHE_CONFIG` | cuFuncSetCacheConfig |

### Driver API Stream Management (0x00C0-0x00CF)

| ID | Constant | Function |
|----|----------|----------|
| 0x00C0 | `CHIDORI_CU_STREAM_CREATE` | cuStreamCreate |
| 0x00C1 | `CHIDORI_CU_STREAM_CREATE_WITH_PRIORITY` | cuStreamCreateWithPriority |
| 0x00C2 | `CHIDORI_CU_STREAM_DESTROY` | cuStreamDestroy_v2 |
| 0x00C3 | `CHIDORI_CU_STREAM_SYNCHRONIZE` | cuStreamSynchronize |
| 0x00C4 | `CHIDORI_CU_STREAM_QUERY` | cuStreamQuery |
| 0x00C5 | `CHIDORI_CU_STREAM_WAIT_EVENT` | cuStreamWaitEvent |

### Driver API Event Management (0x00E0-0x00EF)

| ID | Constant | Function |
|----|----------|----------|
| 0x00E0 | `CHIDORI_CU_EVENT_CREATE` | cuEventCreate |
| 0x00E1 | `CHIDORI_CU_EVENT_DESTROY` | cuEventDestroy_v2 |
| 0x00E2 | `CHIDORI_CU_EVENT_RECORD` | cuEventRecord |
| 0x00E3 | `CHIDORI_CU_EVENT_SYNCHRONIZE` | cuEventSynchronize |
| 0x00E4 | `CHIDORI_CU_EVENT_QUERY` | cuEventQuery |
| 0x00E5 | `CHIDORI_CU_EVENT_ELAPSED_TIME` | cuEventElapsedTime |

### CUDA Runtime API (0x0100-0x011F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0100 | `CHIDORI_CUDA_MALLOC` | cudaMalloc |
| 0x0101 | `CHIDORI_CUDA_FREE` | cudaFree |
| 0x0102 | `CHIDORI_CUDA_MEMCPY` | cudaMemcpy |
| 0x0103 | `CHIDORI_CUDA_MEMCPY_ASYNC` | cudaMemcpyAsync |
| 0x0104 | `CHIDORI_CUDA_MEMSET` | cudaMemset |
| 0x0105 | `CHIDORI_CUDA_MEMSET_ASYNC` | cudaMemsetAsync |
| 0x0106 | `CHIDORI_CUDA_DEVICE_SYNCHRONIZE` | cudaDeviceSynchronize |
| 0x0107 | `CHIDORI_CUDA_STREAM_SYNCHRONIZE` | cudaStreamSynchronize |
| 0x0108 | `CHIDORI_CUDA_GET_DEVICE` | cudaGetDevice |
| 0x0109 | `CHIDORI_CUDA_SET_DEVICE` | cudaSetDevice |
| 0x010A | `CHIDORI_CUDA_GET_DEVICE_COUNT` | cudaGetDeviceCount |
| 0x010B | `CHIDORI_CUDA_GET_DEVICE_PROPERTIES` | cudaGetDeviceProperties |
| 0x010C | `CHIDORI_CUDA_DEVICE_GET_ATTRIBUTE` | cudaDeviceGetAttribute |
| 0x010D | `CHIDORI_CUDA_STREAM_CREATE` | cudaStreamCreate |
| 0x010E | `CHIDORI_CUDA_STREAM_DESTROY` | cudaStreamDestroy |
| 0x010F | `CHIDORI_CUDA_EVENT_CREATE` | cudaEventCreate |
| 0x0110 | `CHIDORI_CUDA_EVENT_DESTROY` | cudaEventDestroy |
| 0x0111 | `CHIDORI_CUDA_EVENT_RECORD` | cudaEventRecord |
| 0x0112 | `CHIDORI_CUDA_EVENT_SYNCHRONIZE` | cudaEventSynchronize |
| 0x0113 | `CHIDORI_CUDA_EVENT_ELAPSED_TIME` | cudaEventElapsedTime |
| 0x0114 | `CHIDORI_CUDA_LAUNCH_KERNEL` | cudaLaunchKernel |
| 0x0115 | `CHIDORI_CUDA_MEM_GET_INFO` | cudaMemGetInfo |
| 0x0116 | `CHIDORI_CUDA_DEVICE_RESET` | cudaDeviceReset |
| 0x0117 | `CHIDORI_CUDA_RUNTIME_GET_VERSION` | cudaRuntimeGetVersion |
| 0x0118 | `CHIDORI_CUDA_DRIVER_GET_VERSION` | cudaDriverGetVersion |
| 0x0119 | `CHIDORI_CUDA_DEVICE_CAN_ACCESS_PEER` | cudaDeviceCanAccessPeer |
| 0x011A | `CHIDORI_CUDA_FUNC_GET_ATTRIBUTES` | cudaFuncGetAttributes |
| 0x011B | `CHIDORI_CUDA_GET_LAST_ERROR` | cudaGetLastError |
| 0x011C | `CHIDORI_CUDA_PEEK_AT_LAST_ERROR` | cudaPeekAtLastError |
| 0x011D | `CHIDORI_CUDA_REGISTER_FAT_BINARY` | __cudaRegisterFatBinary |
| 0x011E | `CHIDORI_CUDA_REGISTER_FUNCTION` | __cudaRegisterFunction |

### CUDA Graph API (0x0120-0x012F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0120 | `CHIDORI_CU_STREAM_BEGIN_CAPTURE` | cuStreamBeginCapture |
| 0x0121 | `CHIDORI_CU_STREAM_END_CAPTURE` | cuStreamEndCapture |
| 0x0122 | `CHIDORI_CU_STREAM_IS_CAPTURING` | cuStreamIsCapturing |
| 0x0123 | `CHIDORI_CU_GRAPH_CREATE` | cuGraphCreate |
| 0x0124 | `CHIDORI_CU_GRAPH_DESTROY` | cuGraphDestroy |
| 0x0125 | `CHIDORI_CU_GRAPH_INSTANTIATE` | cuGraphInstantiate |
| 0x0126 | `CHIDORI_CU_GRAPH_EXEC_DESTROY` | cuGraphExecDestroy |
| 0x0127 | `CHIDORI_CU_GRAPH_LAUNCH` | cuGraphLaunch |
| 0x0128 | `CHIDORI_CU_GRAPH_EXEC_UPDATE` | cuGraphExecUpdate |
| 0x0129 | `CHIDORI_CU_GRAPH_UPLOAD` | cuGraphUpload |

### cuBLAS (0x0200-0x029F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0200 | `CHIDORI_CUBLAS_CREATE` | cublasCreate_v2 |
| 0x0201 | `CHIDORI_CUBLAS_DESTROY` | cublasDestroy_v2 |
| 0x0202 | `CHIDORI_CUBLAS_SET_STREAM` | cublasSetStream_v2 |
| 0x0203 | `CHIDORI_CUBLAS_SET_MATH_MODE` | cublasSetMathMode |
| 0x0204 | `CHIDORI_CUBLAS_GET_MATH_MODE` | cublasGetMathMode |
| 0x0205 | `CHIDORI_CUBLAS_GET_STREAM` | cublasGetStream_v2 |
| 0x0210 | `CHIDORI_CUBLAS_SGEMM` | cublasSgemm_v2 |
| 0x0211 | `CHIDORI_CUBLAS_DGEMM` | cublasDgemm_v2 |
| 0x0212 | `CHIDORI_CUBLAS_HGEMM` | cublasHgemm |
| 0x0213 | `CHIDORI_CUBLAS_SGEMM_EX` | cublasSgemmEx |
| 0x0214 | `CHIDORI_CUBLAS_GEMM_EX` | cublasGemmEx |
| 0x0220 | `CHIDORI_CUBLAS_SGEMV` | cublasSgemv_v2 |
| 0x0221 | `CHIDORI_CUBLAS_DGEMV` | cublasDgemv_v2 |
| 0x0230 | `CHIDORI_CUBLAS_SAXPY` | cublasSaxpy_v2 |
| 0x0231 | `CHIDORI_CUBLAS_DAXPY` | cublasDaxpy_v2 |
| 0x0240 | `CHIDORI_CUBLAS_SNRM2` | cublasSnrm2_v2 |
| 0x0241 | `CHIDORI_CUBLAS_DNRM2` | cublasDnrm2_v2 |
| 0x0250 | `CHIDORI_CUBLAS_SDOT` | cublasSdot_v2 |
| 0x0251 | `CHIDORI_CUBLAS_DDOT` | cublasDdot_v2 |
| 0x0260 | `CHIDORI_CUBLAS_SSCAL` | cublasSscal_v2 |
| 0x0261 | `CHIDORI_CUBLAS_DSCAL` | cublasDscal_v2 |
| 0x0270 | `CHIDORI_CUBLAS_SCOPY` | cublasScopy_v2 |
| 0x0271 | `CHIDORI_CUBLAS_DCOPY` | cublasDcopy_v2 |
| 0x0280 | `CHIDORI_CUBLAS_SGEMM_BATCHED` | cublasSgemmBatched |
| 0x0281 | `CHIDORI_CUBLAS_SGEMM_STRIDED_BATCHED` | cublasSgemmStridedBatched |
| 0x0282 | `CHIDORI_CUBLAS_GEMM_STRIDED_BATCHED_EX` | cublasGemmStridedBatchedEx |
| 0x0290 | `CHIDORI_CUBLAS_GET_VERSION` | cublasGetVersion_v2 |

### cublasLt (0x02A0-0x02AF)

| ID | Constant | Function |
|----|----------|----------|
| 0x02A0 | `CHIDORI_CUBLAS_LT_CREATE` | cublasLtCreate |
| 0x02A1 | `CHIDORI_CUBLAS_LT_MATMUL` | cublasLtMatmul |
| 0x02A2 | `CHIDORI_CUBLAS_LT_MATMUL_ALGO_HEURISTIC` | cublasLtMatmulAlgoGetHeuristic |

### cuDNN (0x0300-0x03A2)

| ID | Constant | Function |
|----|----------|----------|
| 0x0300 | `CHIDORI_CUDNN_CREATE` | cudnnCreate |
| 0x0301 | `CHIDORI_CUDNN_DESTROY` | cudnnDestroy |
| 0x0302 | `CHIDORI_CUDNN_SET_STREAM` | cudnnSetStream |
| 0x0303 | `CHIDORI_CUDNN_GET_VERSION` | cudnnGetVersion |
| 0x0310 | `CHIDORI_CUDNN_CREATE_TENSOR_DESCRIPTOR` | cudnnCreateTensorDescriptor |
| 0x0311 | `CHIDORI_CUDNN_SET_TENSOR_4D_DESCRIPTOR` | cudnnSetTensor4dDescriptor |
| 0x0312 | `CHIDORI_CUDNN_SET_TENSOR_ND_DESCRIPTOR` | cudnnSetTensorNdDescriptor |
| 0x0313 | `CHIDORI_CUDNN_DESTROY_TENSOR_DESCRIPTOR` | cudnnDestroyTensorDescriptor |
| 0x0314 | `CHIDORI_CUDNN_GET_TENSOR_4D_DESCRIPTOR` | cudnnGetTensor4dDescriptor |
| 0x0320 | `CHIDORI_CUDNN_CREATE_FILTER_DESCRIPTOR` | cudnnCreateFilterDescriptor |
| 0x0321 | `CHIDORI_CUDNN_SET_FILTER_4D_DESCRIPTOR` | cudnnSetFilter4dDescriptor |
| 0x0322 | `CHIDORI_CUDNN_SET_FILTER_ND_DESCRIPTOR` | cudnnSetFilterNdDescriptor |
| 0x0323 | `CHIDORI_CUDNN_DESTROY_FILTER_DESCRIPTOR` | cudnnDestroyFilterDescriptor |
| 0x0330 | `CHIDORI_CUDNN_CREATE_CONVOLUTION_DESCRIPTOR` | cudnnCreateConvolutionDescriptor |
| 0x0331 | `CHIDORI_CUDNN_SET_CONVOLUTION_2D_DESCRIPTOR` | cudnnSetConvolution2dDescriptor |
| 0x0332 | `CHIDORI_CUDNN_SET_CONVOLUTION_ND_DESCRIPTOR` | cudnnSetConvolutionNdDescriptor |
| 0x0333 | `CHIDORI_CUDNN_DESTROY_CONVOLUTION_DESCRIPTOR` | cudnnDestroyConvolutionDescriptor |
| 0x0334 | `CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_ALGORITHM` | cudnnGetConvolutionForwardAlgorithm |
| 0x0335 | `CHIDORI_CUDNN_CONVOLUTION_FORWARD` | cudnnConvolutionForward |
| 0x0336 | `CHIDORI_CUDNN_CONVOLUTION_BACKWARD_DATA` | cudnnConvolutionBackwardData |
| 0x0337 | `CHIDORI_CUDNN_CONVOLUTION_BACKWARD_FILTER` | cudnnConvolutionBackwardFilter |
| 0x0338 | `CHIDORI_CUDNN_SET_CONVOLUTION_MATH_TYPE` | cudnnSetConvolutionMathType |
| 0x0339 | `CHIDORI_CUDNN_SET_CONVOLUTION_GROUP_COUNT` | cudnnSetConvolutionGroupCount |
| 0x0340 | `CHIDORI_CUDNN_CREATE_ACTIVATION_DESCRIPTOR` | cudnnCreateActivationDescriptor |
| 0x0341 | `CHIDORI_CUDNN_SET_ACTIVATION_DESCRIPTOR` | cudnnSetActivationDescriptor |
| 0x0342 | `CHIDORI_CUDNN_DESTROY_ACTIVATION_DESCRIPTOR` | cudnnDestroyActivationDescriptor |
| 0x0343 | `CHIDORI_CUDNN_ACTIVATION_FORWARD` | cudnnActivationForward |
| 0x0344 | `CHIDORI_CUDNN_ACTIVATION_BACKWARD` | cudnnActivationBackward |
| 0x0350 | `CHIDORI_CUDNN_CREATE_POOLING_DESCRIPTOR` | cudnnCreatePoolingDescriptor |
| 0x0351 | `CHIDORI_CUDNN_SET_POOLING_2D_DESCRIPTOR` | cudnnSetPooling2dDescriptor |
| 0x0352 | `CHIDORI_CUDNN_DESTROY_POOLING_DESCRIPTOR` | cudnnDestroyPoolingDescriptor |
| 0x0353 | `CHIDORI_CUDNN_POOLING_FORWARD` | cudnnPoolingForward |
| 0x0354 | `CHIDORI_CUDNN_POOLING_BACKWARD` | cudnnPoolingBackward |
| 0x0360 | `CHIDORI_CUDNN_BATCH_NORM_FORWARD_TRAINING` | cudnnBatchNormalizationForwardTraining |
| 0x0361 | `CHIDORI_CUDNN_BATCH_NORM_FORWARD_INFERENCE` | cudnnBatchNormalizationForwardInference |
| 0x0362 | `CHIDORI_CUDNN_BATCH_NORM_BACKWARD` | cudnnBatchNormalizationBackward |
| 0x0370 | `CHIDORI_CUDNN_SOFTMAX_FORWARD` | cudnnSoftmaxForward |
| 0x0371 | `CHIDORI_CUDNN_SOFTMAX_BACKWARD` | cudnnSoftmaxBackward |
| 0x0380 | `CHIDORI_CUDNN_CREATE_DROPOUT_DESCRIPTOR` | cudnnCreateDropoutDescriptor |
| 0x0381 | `CHIDORI_CUDNN_SET_DROPOUT_DESCRIPTOR` | cudnnSetDropoutDescriptor |
| 0x0382 | `CHIDORI_CUDNN_DESTROY_DROPOUT_DESCRIPTOR` | cudnnDestroyDropoutDescriptor |
| 0x0383 | `CHIDORI_CUDNN_DROPOUT_FORWARD` | cudnnDropoutForward |
| 0x0390 | `CHIDORI_CUDNN_CREATE_RNN_DESCRIPTOR` | cudnnCreateRNNDescriptor |
| 0x0391 | `CHIDORI_CUDNN_DESTROY_RNN_DESCRIPTOR` | cudnnDestroyRNNDescriptor |
| 0x03A0 | `CHIDORI_CUDNN_GET_CONVOLUTION_FORWARD_WORKSPACE_SIZE` | cudnnGetConvolutionForwardWorkspaceSize |
| 0x03A1 | `CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_DATA_WORKSPACE_SIZE` | cudnnGetConvolutionBackwardDataWorkspaceSize |
| 0x03A2 | `CHIDORI_CUDNN_GET_CONVOLUTION_BACKWARD_FILTER_WORKSPACE_SIZE` | cudnnGetConvolutionBackwardFilterWorkspaceSize |

### Replay Control (0x0400-0x040F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0400 | `CHIDORI_REPLAY_BEGIN_RECORD` | Start recording a call sequence |
| 0x0401 | `CHIDORI_REPLAY_END_RECORD` | Finalize recording, get fingerprint |
| 0x0402 | `CHIDORI_REPLAY_EXECUTE` | Replay with new variable data |
| 0x0403 | `CHIDORI_REPLAY_INVALIDATE` | Discard a recorded sequence |
| 0x0404 | `CHIDORI_REPLAY_STATUS` | Query replay state/stats |

### Bulk Transfer and Decode (0x0500-0x050F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0500 | `CHIDORI_ALLOC_AND_TRANSFER` | Combined alloc + HtoD in one RPC |
| 0x0501 | `CHIDORI_GRAPH_DECODE` | Server-side autoregressive decode (graph + argmax loop) |
| 0x0502 | `CHIDORI_GRAPH_DECODE_SPLIT` | Split GPU decode (pipeline-parallel inference) |

### Virtual Memory Management (0x0600-0x060F)

| ID | Constant | Function |
|----|----------|----------|
| 0x0600 | `CHIDORI_CU_MEM_ADDRESS_RESERVE` | cuMemAddressReserve |
| 0x0601 | `CHIDORI_CU_MEM_ADDRESS_FREE` | cuMemAddressFree |
| 0x0602 | `CHIDORI_CU_MEM_CREATE` | cuMemCreate |
| 0x0603 | `CHIDORI_CU_MEM_RELEASE` | cuMemRelease |
| 0x0604 | `CHIDORI_CU_MEM_MAP` | cuMemMap |
| 0x0605 | `CHIDORI_CU_MEM_UNMAP` | cuMemUnmap |
| 0x0606 | `CHIDORI_CU_MEM_SET_ACCESS` | cuMemSetAccess |
| 0x0607 | `CHIDORI_CU_MEM_GET_ALLOC_GRANULARITY` | cuMemGetAllocationGranularity |

## TCP I/O Implementation (chidori_net.c)

All I/O is blocking. No select/poll/epoll is used. The server handles multiple requests via the batch-read loop in the dispatch layer, not via I/O multiplexing.

### chidori_read_full(fd, buf, len)
Reads exactly `len` bytes from fd. Loops on `read()`, handling `EINTR` by retrying. Returns 0 on success, -1 on EOF or error.

### chidori_write_full(fd, buf, len)
Writes exactly `len` bytes to fd. Loops on `write()`, handling `EINTR` by retrying. Returns 0 on success, -1 on error.

### chidori_send_request(fd, hdr, payload, payload_len)
Sets `hdr->payload_size` and `hdr->msg_len` (header size + payload size), then writes the 16-byte header followed by the payload as two separate `write_full` calls. Returns 0 on success.

### chidori_recv_request(fd, hdr)
Reads exactly 16 bytes (one request header) from fd into hdr. Caller then reads `hdr->payload_size` additional bytes for the payload.

### chidori_send_response(fd, seq_num, result, payload, payload_len)
If `fd < 0`, returns immediately (async optimization -- no response needed). Otherwise builds a response header and writes it followed by the payload.

### chidori_recv_response(fd, hdr)
Reads exactly 12 bytes (one response header) from fd into hdr. Caller then reads `hdr->msg_len - 12` additional bytes for the response payload.

## Payload Format Examples

### cuMemAlloc (0x0070)
Request payload: 8 bytes
```
Offset  Size  Field
0       8     size (uint64_t) -- allocation size in bytes
```
Response payload: 8 bytes
```
Offset  Size  Field
0       8     dptr (uint64_t) -- device pointer
```

### cuMemcpyHtoD (0x0072)
Request payload: 16 + data bytes
```
Offset  Size    Field
0       8       dst (uint64_t) -- destination device pointer
8       8       size (uint64_t) -- number of bytes to copy
16      size    data (bytes) -- host data to copy to device
```
Response: no payload (result code only)

### cuMemcpyDtoD (0x0074)
Request payload: 24 bytes -- **ZERO data over wire**
```
Offset  Size  Field
0       8     dst (uint64_t) -- destination device pointer
8       8     src (uint64_t) -- source device pointer
16      8     size (uint64_t) -- number of bytes to copy
```
Response: no payload. The data stays on the GPU; only the pointers and size traverse the network.

### cuMemcpyDtoH (0x0073)
Request payload: 16 bytes
```
Offset  Size  Field
0       8     dst (uint64_t) -- destination host pointer (ignored by server)
8       8     size (uint64_t) -- number of bytes to copy
```
Response payload: `size` bytes of device data

### cuLaunchKernel (0x00A0)
Request payload: 44 + variable bytes
```
Offset  Size  Field
0       8     func_handle (uint64_t) -- kernel function handle
8       4     gridDimX (uint32_t)
12      4     gridDimY (uint32_t)
16      4     gridDimZ (uint32_t)
20      4     blockDimX (uint32_t)
24      4     blockDimY (uint32_t)
28      4     blockDimZ (uint32_t)
32      4     sharedMemBytes (uint32_t)
36      8     stream (uint64_t)
44      1     has_params (uint8_t) -- 1 if kernel has parameters
45      4     buf_size (uint32_t) -- size of parameter buffer (if has_params)
49      var   buffer (bytes) -- kernel parameter buffer
```
Response: no payload (async)

### cublasSgemm (0x0210)
Request payload: 72 bytes
```
Offset  Size  Field
0       8     handle (uint64_t) -- cublas handle
8       4     transa (int32_t) -- CUBLAS_OP_N/T
12      4     transb (int32_t) -- CUBLAS_OP_N/T
16      4     m (int32_t) -- rows of A/C
20      4     n (int32_t) -- cols of B/C
24      4     k (int32_t) -- cols of A / rows of B
28      4     alpha (float) -- scalar multiplier
32      8     A (uint64_t) -- device pointer to matrix A
40      4     lda (int32_t) -- leading dimension of A
44      8     B (uint64_t) -- device pointer to matrix B
52      4     ldb (int32_t) -- leading dimension of B
56      4     beta (float) -- scalar multiplier
60      8     C (uint64_t) -- device pointer to matrix C
68      4     ldc (int32_t) -- leading dimension of C
```
Response: no payload (async)

### cublasLtMatmul (0x02A1)
Request payload: 308 bytes total
```
Offset  Size  Field
0       4     transa (int32_t)
4       4     transb (int32_t)
8       4     computeType (int32_t)
12      4     scaleType (int32_t)
16      4     epilogue (int32_t)
20      8     bias_pointer (uint64_t)
28      4     A_dtype (int32_t)
32      8     A_rows (uint64_t)
40      8     A_cols (uint64_t)
48      8     A_ld (int64_t)
56      4     A_batch_count (int32_t)
60      8     A_stride (int64_t)
68      4     B_dtype (int32_t)
72      8     B_rows (uint64_t)
80      8     B_cols (uint64_t)
88      8     B_ld (int64_t)
96      4     B_batch_count (int32_t)
100     8     B_stride (int64_t)
108     4     C_dtype (int32_t)
112     8     C_rows (uint64_t)
120     8     C_cols (uint64_t)
128     8     C_ld (int64_t)
136     4     C_batch_count (int32_t)
140     8     C_stride (int64_t)
148     4     D_dtype (int32_t)
152     8     D_rows (uint64_t)
160     8     D_cols (uint64_t)
168     8     D_ld (int64_t)
176     4     D_batch_count (int32_t)
180     8     D_stride (int64_t)
188     8     alpha (double/float, 8 bytes)
196     8     beta (double/float, 8 bytes)
204     8     A pointer (uint64_t)
212     8     B pointer (uint64_t)
220     8     C pointer (uint64_t)
228     8     D pointer (uint64_t)
236     8     stream (uint64_t)
244     64    algo (bytes) -- cublasLtMatmulAlgo_t opaque data
```
Response: no payload (async)

### cudaMemcpy (0x0102)
Request payload: 28 + optional data bytes
```
Offset  Size    Field
0       8       dst (uint64_t) -- destination pointer
8       8       src (uint64_t) -- source pointer
16      8       size (uint64_t) -- bytes to copy
24      4       kind (int32_t) -- cudaMemcpyKind (1=HtoD, 2=DtoH, 3=DtoD)
28      size    data (bytes) -- present only for HtoD (kind=1)
```
For DtoH (kind=2), data appears in the response payload instead.
For DtoD (kind=3), no data is transferred over the wire.

### ALLOC_AND_TRANSFER (0x0500)
Combined alloc + HtoD in a single RPC, used for model loading optimization.
```
Request: size(8) + data(size)
Response: dptr(8) -- device pointer for the allocation
```

### GRAPH_DECODE (0x0501)
Server-side autoregressive decode loop.
```
Request: graph_exec(8) + stream(8) + input_ptr(8) + position_ptr(8) +
         logits_ptr(8) + vocab_size(4) + num_tokens(4) + start_position(4) = 52 bytes
Response: num_generated(4) + token_ids(4 x N) bytes
```

### GRAPH_DECODE_SPLIT (0x0502)
Split GPU decode for pipeline-parallel inference.
```
Request: graph_exec(8) + stream(8) + activation_dst(8) + position_ptr(8) +
         logits_ptr(8) + vocab_size(4) + position(4) + activation_size(4) +
         activation_data(activation_size) = 52 + activation_size bytes
Response: token_id(4) bytes
```

## Server-Side Batch-Read Dispatch

The server does not use I/O multiplexing. Instead, it reads ALL available data from the socket into a 256KB buffer in a single `recv()` call, then parses and dispatches every complete request found in the buffer. This is the batch-read dispatch loop:

```
recv() -> fill 256KB buffer
  while (complete request in buffer):
    dispatch(cuda_dispatch)
    advance offset
  compact buffer (memmove remaining bytes to front)
  loop
```

Oversized payloads (larger than 256KB) are handled by a fallback path that reads directly from the socket into a separately allocated buffer.

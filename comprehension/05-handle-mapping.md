# Handle Mapping

## Overview

CUDA uses opaque handles (pointers) for all resources: contexts, modules, functions, streams, events, cuBLAS handles, cuDNN descriptors, and more. These handles are memory addresses or indices that are only meaningful on the machine that created them. A `CUmodule` handle created on the server is a pointer into the server's address space -- it means nothing on the client.

Chidori must translate between client-side **wire handles** (arbitrary IDs sent over the network) and server-side **real CUDA handles** (actual pointers returned by CUDA driver/runtime calls). This translation is the job of the handle mapping system defined in `handle_map.h`.

## Data Structure (`handle_map.h`)

```c
#define HANDLE_MAP_MAX 8192

typedef struct {
    uint64_t keys[HANDLE_MAP_MAX];  // Wire handle (server-chosen ID)
    uint64_t vals[HANDLE_MAP_MAX];  // Real CUDA pointer (cast to uint64_t)
    int count;                       // Number of active entries
} handle_map_t;
```

Each handle map is a flat array of key-value pairs with a count of active entries. Keys are the wire handles (assigned by the server) and values are the real CUDA handles (cast from pointers to `uint64_t`).

## Operations

All functions are `static inline` to avoid function call overhead.

**`hmap_init(handle_map_t *m)`:**
- `memset(m, 0, sizeof(*m))` -- zeros the entire structure.

**`hmap_put(handle_map_t *m, uint64_t key, uint64_t val)`:**
- Appends the key-value pair at index `m->count`.
- Increments `m->count`.
- Returns 0 on success, -1 if the map is full (`count >= HANDLE_MAP_MAX`).
- O(1) -- simple array append.

**`hmap_get(handle_map_t *m, uint64_t key)`:**
- Linear scan from index 0 to `m->count - 1`.
- Returns the corresponding value if found, or 0 if not found.
- O(n) -- linear search, but acceptable because n is bounded by 8192.

**`hmap_remove(handle_map_t *m, uint64_t key)`:**
- Linear scan to find the entry.
- Swap-with-last deletion: copies the last entry into the deleted slot, then decrements `m->count`.
- O(n) for the scan, but the deletion itself is O(1).

## Design Tradeoffs

- **Append-only with swap-delete**: Fast insertion (O(1)), acceptable lookup cost (O(n) with n bounded at 8192). Swap-delete avoids shifting the entire array.
- **No hashing**: Eliminates collision handling complexity. With a maximum of 8192 entries, a linear scan is fast enough (at worst ~64K memory comparisons, well within CPU cache).
- **All functions static inline**: Zero function call overhead. The compiler inlines the scan loops directly at each call site.
- **Fixed 8192 limit**: Each map can hold at most 8192 entries. If a workload creates more handles of any single type, new allocations will fail. In practice this limit is far above what typical workloads require.

## Handle Allocation on Server

Wire handles are assigned by the server using a monotonic counter:

```c
static uint64_t g_next_handle = 0x10000;

static uint64_t alloc_handle(void) {
    return g_next_handle++;
}
```

The starting value `0x10000` is chosen to avoid collision with real CUDA pointer values. Real CUDA handles are typically memory addresses in the range of valid user-space pointers (e.g., 0x7f...), so starting at a low address avoids any ambiguity.

Handle IDs are never reused. Once allocated, a handle ID persists for the lifetime of the session. This eliminates any risk of stale references.

## Handle Maps Used (18 total)

The server maintains 18 separate handle maps, one for each type of CUDA resource:

| Map | Resource Type | Description |
|---|---|---|
| `g_modules` | `CUmodule` | Loaded CUDA module handles (PTX/CUBIN binaries) |
| `g_functions` | `CUfunction` | Kernel function handles extracted from modules |
| `g_streams` | `CUstream` | Driver API stream handles |
| `g_events` | `CUevent` | Driver API event handles |
| `g_ctx_to_dev` | Context-to-device | Maps context handle to device ordinal |
| `g_rt_streams` | `cudaStream_t` | Runtime API stream handles (separate from driver) |
| `g_rt_events` | `cudaEvent_t` | Runtime API event handles (separate from driver) |
| `g_cublas_handles` | `cublasHandle_t` | cuBLAS library handles |
| `g_graphs` | `CUgraph` | CUDA graph handles |
| `g_graph_execs` | `CUgraphExec` | CUDA graph executable handles |
| `g_cudnn_handles` | `cudnnHandle_t` | cuDNN library handles |
| `g_cudnn_tensor_descs` | `cudnnTensorDescriptor_t` | Tensor descriptors |
| `g_cudnn_filter_descs` | `cudnnFilterDescriptor_t` | Convolution filter descriptors |
| `g_cudnn_conv_descs` | `cudnnConvolutionDescriptor_t` | Convolution descriptors |
| `g_cudnn_activation_descs` | `cudnnActivationDescriptor_t` | Activation function descriptors |
| `g_cudnn_pooling_descs` | `cudnnPoolingDescriptor_t` | Pooling layer descriptors |
| `g_cudnn_dropout_descs` | `cudnnDropoutDescriptor_t` | Dropout layer descriptors |
| `g_cudnn_rnn_descs` | `cudnnRNNDescriptor_t` | RNN layer descriptors |

## Dual Stream/Event Maps

CUDA provides two APIs that create streams and events: the Driver API (`cuStreamCreate`, `cuEventCreate`) and the Runtime API (`cudaStreamCreate`, `cudaEventCreate`). In practice, PyTorch and other frameworks may use either or both. The same wire handle might be created through either API, but it must be found regardless of which API is used for subsequent operations.

The server handles this with **dual-map lookups**. For any operation that takes a stream or event handle:

1. First look in the Driver API map (`g_streams` / `g_events`)
2. If not found, look in the Runtime API map (`g_rt_streams` / `g_rt_events`)

For destroy operations, the server removes the handle from **both** maps to ensure cleanup regardless of which API created it.

## Device Pointers: No Mapping Needed

GPU memory addresses (device pointers returned by `cuMemAlloc`, `cudaMalloc`, etc.) are passed directly as `uint64_t` values over the wire. No translation table is needed because:

- The server allocates GPU memory and returns the device address to the client
- The client stores this address and sends it back in subsequent operations (memcpy, kernel launch, etc.)
- The server receives the same device address and passes it directly to CUDA API calls

Both sides agree on the same GPU address values because the server is the only machine that actually has GPU memory. The client never dereferences these addresses -- they are opaque tokens that happen to be valid on the server.

## Fat Binary Handle Special Case

Fat binaries (containing PTX for multiple GPU architectures) use a separate caching mechanism distinct from the standard handle maps:

**`g_fat_cache[1024]`**: A cache array that stores:
- FNV-1a hash of the binary payload (for deduplication)
- Payload size
- Wire handle (the handle assigned to this fat binary)

This cache prevents the server from loading the same fat binary multiple times when PyTorch registers it repeatedly (which it does during module initialization).

**Bit 63 tagging**: Runtime API handles are stored with bit 63 set by OR-ing with `(1ULL << 63)`. This single bit distinguishes Runtime API handles from Driver API handles in the fat binary cache. When `RegisterFunction` is called, it checks bit 63 to decide whether to use the Runtime API path (`cudaLaunchKernel`) or the Driver API path (`cuLaunchKernel`).

This distinction matters because the same kernel function might be registered through different API paths depending on how PyTorch loaded the module, and the execution path must match.

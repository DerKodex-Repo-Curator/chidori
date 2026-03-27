# Chidori System Overview

## What Chidori Is

Chidori is a **GPU-over-IP system** that provides transparent remote GPU access for CUDA workloads over TCP/IP. A client-side interception library (`libchidori.so`) is loaded via `LD_PRELOAD` before the application starts. This library intercepts CUDA API calls (Driver API, Runtime API, cuBLAS, cuDNN), serializes them into a binary wire protocol, and sends them over a TCP socket to a server-side worker (`chidori-worker`) running on a machine with real GPUs. The worker dispatches the calls to the actual CUDA driver and sends responses back.

Applications require **zero code changes**. The entire CUDA API surface is intercepted transparently.

## Architecture

Chidori consists of two main components:

### 1. chidori-worker/ (C) -- Core Interception and Dispatch Layer

| File | Lines | Purpose |
|------|-------|---------|
| `src/client/chidori_client.c` | ~9510 | LD_PRELOAD library intercepting ~280 CUDA symbols |
| `src/server/chidori_worker.c` | ~391 | Server main loop, socket management, dispatch entry point |
| `src/server/cuda_dispatch.c` | ~4618 | Giant switch dispatching 100+ function IDs to real CUDA calls |
| `src/server/handle_map.h` | ~56 | Wire handle to real CUDA handle mapping (append-only arrays, 8192 entry limit) |
| `src/server/replay.c` | ~472 | Server-side call sequence recording and replay for training steps |
| `src/protocol/chidori_net.c` | ~90 | TCP I/O primitives (blocking read/write with EINTR handling) |
| `include/chidori_protocol.h` | ~387 | Wire protocol definitions (headers, function ID enum, flags) |
| `src/client/chidori_client.map` | ~76 | GNU linker version script for symbol versioning |
| `src/client/chidori_nvml.c` | ~50 | NVML proxy library source |
| `codegen/` | -- | Tool to auto-generate CUDA stubs from headers |

### 2. control-plane/ (Go) -- Enterprise Orchestration

| Component | Purpose |
|-----------|---------|
| `cmd/agent/` | Runs on GPU hosts, manages worker processes, reports GPU metrics via NVML |
| `cmd/controller/` | Central scheduler with REST API, database backend (PostgreSQL/SQLite) |
| `cmd/juicify/` | Client CLI launcher that sets environment variables and LD_PRELOAD |

## End-to-End Data Flow

The following diagram shows how a CUDA call like `cudaMalloc` travels from application to GPU and back:

```
  CLIENT (no GPU)                           SERVER (GPU host)
  ================                          ==================

  +------------------+     TCP/IP      +------------------+
  | Application      |                 | chidori-worker   |
  |                  |                 |                  |
  | cudaMalloc(1024) |                 |                  |
  |       |          |                 |                  |
  |       v          |                 |                  |
  | libchidori.so    |                 |                  |
  | (LD_PRELOAD)     |   16-byte hdr   |                  |
  |       |          | + payload       |                  |
  |  serialize       | --------------->|  dispatch loop   |
  |       |          |                 |       |          |
  |  send TCP        |                 |       v          |
  |       |          |                 |  cuMemAlloc()    |
  |       |          |                 |       |          |
  |       |          |  12-byte resp   |       v          |
  |  recv TCP        |<--------------- |  real CUDA       |
  |       |          |  + dev_ptr      |  driver          |
  |       v          |                 |                  |
  | return ptr       |                 |                  |
  +------------------+                 +------------------+
```

**Step by step:**

1. Application starts with `LD_PRELOAD=build/libchidori.so`
2. Application calls e.g. `cudaMalloc(1024)`
3. `libchidori.so` intercepts the call (its `cudaMalloc` symbol takes priority)
4. Client serializes parameters into wire format (16-byte request header + payload)
5. Sends over TCP socket to `chidori-worker`
6. Server receives data in 256KB batch buffer
7. Parses request header, extracts function ID and payload
8. Dispatches to `cuda_dispatch()` switch, which calls real `cuMemAlloc()` on the GPU
9. Sends response back (device pointer as uint64 in response payload)
10. Client receives response, returns pointer to application
11. Application uses the pointer for subsequent operations (kernels, memcpy, etc.)

## Connection Modes

### Standalone Mode (testing)

```
  Client                          Server
  ------                          ------
  LD_PRELOAD=libchidori.so        chidori-worker --standalone 43211
       |                                  |
       +-------- TCP connect ------------>+
       |                                  |
       +<--- CUDA protocol over TCP ----->+
       |                                  |
       +-------- close ------------------>+
```

Direct TCP listener on a port. One client at a time. Used for testing and development.

### Agent-Mediated Mode (production)

```
  Client              Go Agent (GPU host)           chidori-worker
  ------              ==================           =============
  LD_PRELOAD=libchidori.so    Agent process         Worker process
       |                         |                       |
       +-- HTTP POST ------------>+                       |
       |   /v1/connect/session/   |                       |
       |   {id}                   |                       |
       |                          +-- SCM_RIGHTS -------->+
       |                          |   (pass TCP socket)   |
       |<-- 101 Switching --------+                       |
       |                          |                       |
       +<--- CUDA protocol ------+<----------------------+--> GPU
       +---> CUDA protocol ------>+------------------------->---+
```

The Go agent runs on the GPU host, manages worker lifecycle, and forwards client TCP sockets to workers via Unix domain socket `SCM_RIGHTS`. The client connects to the agent via HTTP upgrade, which then hijacks the connection and passes the socket fd to the worker.

## Key Performance Optimizations

### Batch-Read Dispatch
The server uses a 256KB receive buffer and processes all complete requests in a tight inner loop. Instead of two blocking `read()` syscalls per request (header + payload), a single `recv()` fills the buffer, and the parser dispatches every complete message it finds. Oversized payloads (e.g. large HtoD copies) fall back to direct socket reads into a separate allocation.

### TCP_CORK Batching
The client enables `TCP_CORK` during sequences of async RPC calls (compute operations like kernel launches, cuBLAS calls). This delays TCP segment transmission until either the cork is disabled (at the next sync call) or the segment is full. Multiple small async requests are packed into a single TCP segment, reducing per-segment overhead.

### DtoD Zero-Network
Device-to-device memory copies (`cuMemcpyDtoD`) between two remote pointers send only 24 bytes over the wire (dst, src, size). The actual data stays on the GPU -- no round-trip to the host. This is critical for operations like tensor permutations that rearrange data entirely on the GPU.

### Async RPC Path
Compute-phase calls (kernel launches, cuBLAS operations, memory sets) use `CHIDORI_FLAG_ASYNC`. No response is sent by the server, allowing hundreds of operations to be queued without waiting for acknowledgment. The client flushes pending responses at synchronization points.

### Replay System
After `CHIDORI_REPLAY_WARMUP` training steps (default 3), the client tells the server to record one complete training step. The server records the full call sequence, classifying each call as:
- **H2D** (data that changes every step)
- **D2H** (output to return)
- **KERNEL** (compute, fixed structure)
- **SYNC** (must execute)
- **SKIP** (one-time setup)
- **FIXED** (replay as-is)

Subsequent steps send only the changed H2D data in a single `REPLAY_EXECUTE` RPC, eliminating per-call overhead for the repeated training loop.

### Server-Side Graph Decode
For autoregressive inference, `GRAPH_DECODE` (0x0501) runs the entire graph launch + argmax loop on the server. The client sends graph parameters and receives all generated tokens at once -- one RPC per N tokens instead of N RPCs.

### Memory Pool (Bump Allocator)
`cudaMalloc` uses a bump allocator with 512MB+ chunks allocated from the server. Successive allocations within a chunk are satisfied locally with zero network round-trips. New chunks are allocated from the server when the current one is exhausted.

### Bulk Transfer
During model loading, PyTorch makes ~200 individual `cudaMalloc` + `cudaMemcpy(HtoD)` pairs. The client buffers these pairs and sends them as `ALLOC_AND_TRANSFER` RPCs, converting 200 round-trips into 1 round-trip plus bandwidth-limited streaming.

### Event Pool
`cudaEventCreate/Destroy` uses a pool of 256 pre-allocated events on the server. Creation returns a pre-allocated handle instantly (zero network cost). Destruction returns the handle to the pool. The pool is refilled in batches of 64 using pipelined sync RPCs when it runs low.

## Build Outputs

| Output | Description |
|--------|-------------|
| `chidori-worker` | GPU host server binary |
| `libchidori.so` | Client interception library (linked as `libcuda.so.1` via symlink) |
| `libcublas.so.12` | cuBLAS proxy library (intercepts cuBLAS symbols) |
| `libcublasLt.so.12` | cuBLASLt proxy library (intercepts cublasLt symbols) |
| `libcublas.so.13` | cuBLAS proxy library for CUDA 13 |
| `libcublasLt.so.13` | cuBLASLt proxy library for CUDA 13 |
| `libnvidia-ml.so.1` | NVML proxy library |

All build outputs go to `chidori-worker/build/`.

## CUDA API Coverage

| API Category | Implemented Functions | Notes |
|-------------|----------------------|-------|
| Driver API | ~30 | Init, device, context, module, memory, execution, stream, event |
| Runtime API | ~26 | Memory, device, stream, event, graph, launch, register |
| cuBLAS | ~27 | Create/destroy, GEMM family, GEMV, BLAS1, batched |
| cublasLt | ~20+ | Matmul, descriptors, layouts, preferences, algorithms |
| cuDNN | ~47 | Convolution, pooling, activation, batchnorm, softmax, dropout, RNN |
| Symbol stubs | ~150 | Stubs for library compatibility |
| NVML | subset | Via separate `libnvidia-ml.so.1` proxy |
| NCCL | Not implemented | Planned for future |

## Adding a New CUDA Function

1. Add function ID constant to `include/chidori_protocol.h`
2. Add client-side serialization in `src/client/chidori_client.c`
3. Add server-side dispatch case in `src/server/cuda_dispatch.c`
4. If the function returns handles, add handle mapping in the dispatch case using `handle_map.h`

## Architecture Diagram

```
+------------------------------------------------------------------+
|                        CLIENT MACHINE                             |
|                                                                   |
|  +-------------+    +------------------+    +------------------+  |
|  | Application | -> |  libchidori.so   | -> |   TCP Socket     |  |
|  | (PyTorch,   |    |  (LD_PRELOAD)    |    |   (fd >= 500)    |  |
|  |  CUDA apps) |    |                  |    |                  |  |
|  +-------------+    +------------------+    +------------------+  |
|       |                     |                        |            |
|       |  cuBLAS/cuBLASLt    |  cuDNN/NVML            |            |
|       |  proxy libs         |  proxy libs            |            |
|       +----------->---------+----------->------------+            |
|                                                                   |
+-----------------------------------|-------------------------------+
                                    |
                              TCP/IP (WAN/LAN)
                                    |
+-----------------------------------|-------------------------------+
|                                   v                            |
|                        SERVER MACHINE (GPU host)                |
|                                                                   |
|  +------------------+    +------------------+    +-------------+ |
|  | chidori-worker   | -> | cuda_dispatch()  | -> | Real CUDA   | |
|  | (dispatch loop)  |    | (switch on ID)   |    | Driver      | |
|  +------------------+    +------------------+    | GPU Hardware| |
|                                                    +-------------+ |
+------------------------------------------------------------------+
```

## Partial Offloading

When `CHIDORI_LOCAL_DEVICES` is set (e.g., `CHIDORI_LOCAL_DEVICES=0`), the client machine's local GPU handles some operations directly while overflow work goes to the remote server. This is the "one client with a local GPU + one remote server" architecture:

- Device 0 (local): uses the real CUDA driver loaded via `dlopen`
- Device 1 (remote): routes through `chidori_rpc()` to the server

The library loads the real NVIDIA driver (`libcuda.so.1`) using `realpath()` to resolve symlinks and avoid loading itself. cuBLAS/cublasLt calls on the local device are forwarded to the real libraries, bypassing interception entirely.

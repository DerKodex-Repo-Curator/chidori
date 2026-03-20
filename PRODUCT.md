# Chidori — Final Product Vision

## What It Is

Chidori is a transparent GPU-over-IP system. Any machine can use any remote GPU over the network without changing a single line of code. No GPU drivers needed on the client. No PyTorch or framework dependencies on the server. Just TCP.

```
YOUR LAPTOP (no GPU)          CLOUD GPU (H100, A100, 5090, whatever)
   python train.py     ──TCP──>   chidori-worker
   (thinks it has a GPU)           (has the real GPU)
```

## What It Does

1. **CPU-to-GPU training/inference over the network.** A machine with zero GPU runs PyTorch training or inference. All CUDA calls go over TCP to a remote GPU. The application doesn't know the difference.

2. **Cross-architecture support.** Client compiled for sm_89 (RTX 4090) works with a server running sm_120 (Blackwell) or sm_70 (V100). Fat binaries are JIT-compiled on the server via PTX.

3. **Partial GPU offloading.** A machine with a small GPU (16GB) trains a model that doesn't fit. Layers that fit stay local. Overflow layers go to a bigger remote GPU (32GB+). PyTorch's `device_map` splits automatically. One process, two GPUs.

4. **Zero server dependencies.** The server runs a single binary (`chidori-worker`) that needs only CUDA toolkit. No Python, no PyTorch, no frameworks. It's a dumb CUDA relay.

5. **Zero client code changes.** `LD_PRELOAD=libchidori.so python train.py` — that's it. Works with PyTorch, TensorFlow, JAX, custom CUDA programs, anything that calls CUDA.

## Performance Targets

| Workload | Target | Current |
|----------|--------|---------|
| Matmul throughput (loopback) | 100% of native | 100% (15 TFLOPS RTX 4090) |
| Training throughput (loopback) | >95% of native | 97% (321K tok/s vs 330K native) |
| Inference tok/s (loopback) | >50% of native | 56% (30 tok/s vs 54 native) |
| Training throughput (WAN, 40ms) | >10K tok/s | 14.2K tok/s |
| Cross-arch overhead vs same-arch | <5% | TBD (not yet working) |
| Partial offload vs single GPU | model fits, trains | not yet working |

## Architecture

```
Client (any machine)                Server (GPU machine)
┌─────────────────┐                ┌──────────────────────┐
│  Application     │                │  chidori-worker      │
│  (PyTorch etc)   │                │                      │
│       │          │                │  ┌────────────────┐  │
│  LD_PRELOAD      │                │  │ cuda_dispatch   │  │
│  libchidori.so   │──── TCP ──────>│  │ (real CUDA)     │  │
│                  │                │  └────────────────┘  │
│  Intercepts ALL  │                │         │            │
│  CUDA API calls  │                │    ┌────┴────┐       │
└─────────────────┘                │    │  GPU    │       │
                                   │    └─────────┘       │
                                   └──────────────────────┘
```

### Wire Protocol

16-byte request header, 12-byte response header. Native byte order. Async RPCs for compute (fire-and-forget), sync RPCs only for data reads and synchronization.

### Client Library (libchidori.so)

- Intercepts ~280 CUDA symbols via LD_PRELOAD
- Serializes arguments into binary wire protocol
- TCP_CORK batching: N async calls → 1 TCP segment
- Client-side caching: device attributes, cuBLAS state, version queries
- cublasLt → cublasGemmEx translation for server-side heuristic RPC

### Server (chidori-worker)

- Single C binary, ~3500 lines
- Batch-read dispatch loop: 256KB recv buffer, zero-copy for common messages
- Handle mapping: wire handles ↔ real CUDA handles
- Links against libcuda, libcudart, libcublas, libcublasLt, libcudnn
- Standalone mode (direct TCP) or agent-mediated (Go control plane)

### Partial Offloading (in progress)

- Thread-local `g_in_real_cuda` reentrance guard prevents LD_PRELOAD recursion
- `CHIDORI_LOCAL_DEVICES=0` routes device 0 to real CUDA, device 1+ to Chidori
- `RTLD_NEXT` finds real CUDA functions behind LD_PRELOAD
- Fat binary registration split: real CUDA + Chidori lazy registration
- Cross-device memory copies staged through host RAM

## What Needs to Finish

### Cross-Architecture (branch: feat/runtime-api-kernel-loading)

Fat binary loading via `__cudaRegisterFatBinary` (Runtime API) works for sm_89 → sm_120. Function registration works. Kernel launch needs `void** args` reconstruction from the contiguous parameter buffer. Once the launch handler is updated, cross-arch is done.

### Partial GPU Offloading

The `g_in_real_cuda` reentrance guard is in place for 32 Driver API functions. Fat binary registration split forwards to both real CUDA and Chidori. The blocker was `cuDeviceGetCount` — PyTorch uses `dlsym` to resolve it, bypassing LD_PRELOAD. Need to intercept `dlsym` or find another way to inject the merged device count.

### Test Suite

`./tests/run_all.sh <server_ip> <port>` runs 5 single-server tests + 5 partial offload tests. All single-server tests pass. Partial offload tests are TDD stubs awaiting implementation.

## Non-Goals

- NCCL / multi-GPU collective communication
- TLS / encryption (use SSH tunnel)
- GPU isolation / multi-tenant
- Windows support

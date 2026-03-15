# Chidori ⚡ — GPU-over-IP

Transparent remote GPU access for CUDA compute workloads over TCP/IP.

Run any CUDA application on a machine with **no GPU** — Chidori intercepts CUDA API calls via `LD_PRELOAD` and forwards them to a remote GPU server over the network. The application doesn't know the GPU is remote.

```
CLIENT (no GPU)                         SERVER (has GPU)
===============                         ================

  Your App (unmodified)                  juice-gpu-worker
       |                                      |
  [LD_PRELOAD]                           [Real CUDA API]
       |                                      |
  libjuice-cuda.so  ──── TCP/IP ────>   A100 / H100 / etc.
```

## Proven Working

Tested across two physical machines — a GPU-less client executing CUDA on a remote NVIDIA A100-SXM4-40GB:

- Device queries return real GPU info (108 SMs, 39.5 GB VRAM, CUDA 13.0)
- 16MB memory round-trips with full data integrity verification
- Device-to-device copies transfer zero bytes over the network
- 100x 1MB sustained throughput benchmarks
- 20 concurrent GPU memory allocations (210 MB)
- Stream and event operations

## Why This Works

CUDA applications never touch GPU memory directly — everything goes through API calls that return opaque handles. A `CUdeviceptr` is just a 64-bit number your CPU code never dereferences. Replace the bottom layer (`libcuda.so`) with a network proxy and the application can't tell the difference.

ML workloads are especially network-friendly: once model weights are uploaded to the GPU (one-time cost), forward/backward passes are hundreds of kernel launches that move **zero data over the network** — just tiny RPC messages saying "run this kernel with these pointers."

## Quick Start

### One-command test across two machines

```bash
cd gpu-worker/tests
cp request_example.sh request.sh
# Edit request.sh with your server/client IPs
./request.sh
```

This deploys, builds, and runs the full test suite automatically.

### Manual setup

**GPU Server** (has NVIDIA GPU + CUDA):
```bash
cd gpu-worker
make server
./build/juice-gpu-worker --standalone 43211
```

**Client** (no GPU needed, just GCC):
```bash
cd gpu-worker
make client

JUICE_DIRECT_CONNECT=1 \
JUICE_CFG_OVERRIDE='{"id":"test","servers":["GPU_SERVER_IP:43211"]}' \
LD_PRELOAD=./build/libjuice-cuda.so \
your_cuda_application
```

## Project Structure

```
juice/
├── gpu-worker/              ← CUDA interception layer (C)
│   ├── src/server/          ← juice-gpu-worker binary (runs on GPU host)
│   ├── src/client/          ← libjuice-cuda.so (LD_PRELOAD on client)
│   ├── src/protocol/        ← Wire protocol (16B req / 12B resp headers)
│   ├── codegen/             ← Auto-generates stubs from cuda.h
│   └── tests/               ← Protocol + e2e + remote GPU tests
│
├── control-plane/           ← Session orchestration (Go)
│   ├── cmd/agent/           ← Runs on GPU hosts, manages sessions
│   ├── cmd/controller/      ← Central scheduler and web UI
│   └── cmd/juicify/         ← Client CLI launcher
│
├── deploy/                  ← Docker, Helm charts, installer
├── integrations/slurm/      ← SLURM HPC integration (SPANK plugin)
└── docs/                    ← Documentation
```

## How It Works

### Wire Protocol

Binary protocol optimized for low latency:

```
Request  [16B]: msg_len(4) | func_id(2) | flags(2) | seq_num(4) | payload_size(4)
Response [12B]: msg_len(4) | seq_num(4) | result(4)
```

### Interception

`libjuice-cuda.so` exports all CUDA Driver API symbols (`cuInit`, `cuMemAlloc`, `cuLaunchKernel`, etc.). When loaded via `LD_PRELOAD`, it intercepts every CUDA call, serializes the arguments, sends them over TCP, and returns the result.

### Server

`juice-gpu-worker` receives serialized CUDA calls, executes them on the real GPU via the actual CUDA Driver API, and sends back results. Handles for contexts, modules, functions, streams, and events are mapped between wire protocol IDs and real CUDA handles.

### Key Optimization: DtoD Copies

`cuMemcpyDtoD(dst, src, 16MB)` sends only 24 bytes over the network (three 8-byte values). The actual 16MB copy happens entirely on the remote GPU. This is why ML workloads perform well — most data movement is GPU-internal.

## Status

| Component | Status |
|-----------|--------|
| Wire protocol | Working |
| CUDA Driver API interception | Working (30+ functions) |
| Real CUDA dispatch on server | Working (all function groups) |
| Cross-machine GPU access | Working (tested on A100) |
| Standalone mode | Working |
| Agent-mediated mode | In progress (socket forwarding) |
| CUDA Runtime API | Not yet |
| cuBLAS / cuDNN | Not yet |
| PyTorch / TensorFlow | Not yet |

## Background

This project revives [Juice Labs](https://github.com/Juice-Labs/Juice-Labs), which built a GPU-over-IP system that was abandoned and partially closed-sourced. The proprietary GPU interception binary (`Renderer_Win`) has been replaced with an open-source implementation. The Go control plane has been updated to use NVML directly and target Linux only.

## License

MIT

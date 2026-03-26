# Chidori Architecture — Changes & Current State

## What Chidori Does

Chidori is a GPU-over-IP system. A client-side library (`libchidori.so`) loaded via `LD_PRELOAD` intercepts CUDA API calls, serializes them over TCP, and a server-side worker dispatches them to real GPUs. Applications require zero code changes.

## Core Components

```
┌─────────────────────┐         TCP          ┌──────────────────────┐
│  Client (Lenovo)    │◄────────────────────►│  Server (A100)       │
│                     │      71ms RTT         │                      │
│  libchidori.so      │                      │  chidori-worker      │
│  (LD_PRELOAD)       │                      │  (standalone mode)   │
│                     │                      │                      │
│  Intercepts ~280    │  Wire Protocol:      │  Dispatches to real  │
│  CUDA symbols       │  16-byte header      │  CUDA driver         │
│  Serializes → TCP   │  + payload           │                      │
└─────────────────────┘                      └──────────────────────┘
```

## Key Files

| File | What it does |
|------|-------------|
| `src/client/chidori_client.c` | LD_PRELOAD library (~9000 lines). Intercepts all CUDA calls. |
| `src/server/cuda_dispatch.c` | Server dispatch (~4500 lines). Switch on func_id → real CUDA call. |
| `src/server/chidori_worker.c` | Server main loop. TCP accept, batch-read dispatch. |
| `src/protocol/chidori_net.c` | TCP I/O primitives (blocking read/write). |
| `include/chidori_protocol.h` | Wire protocol definitions. Function IDs, headers, flags. |
| `src/server/handle_map.h` | Maps wire handles ↔ real CUDA handles. |

## Wire Protocol

```
Request:  msg_len(4) | func_id(2) | flags(2) | seq_num(4) | payload_size(4) | payload
Response: msg_len(4) | seq_num(4)  | result(4) | payload
```

- `CHIDORI_FLAG_SYNC (0x01)`: Client blocks for response
- `CHIDORI_FLAG_ASYNC (0x02)`: Fire-and-forget, server sends no response

## Changes Made This Session

### 1. Model Loading: 510s → 20s

**Problem:** Loading a 2.2GB model sent ~462 fat binary registrations as individual sync RPCs, each waiting for server response.

**Fixes:**
- **Lazy fat binary registration** (`chidori_client.c:6342`): Only register the ~25 modules actually used during inference, not all 462.
- **Pipelined fat binary send** (`chidori_client.c:6206`): Send all requests at once, read all responses after. Turns N×RTT into 1×RTT.
- **Server fat binary cache** (`cuda_dispatch.c:68`): Hash-based. Skip `cuModuleLoadData` for identical modules.
- **Client dedup** (`chidori_client.c:6345`): Skip RPC for duplicate fat binaries.
- **Skip sync during loading** (`chidori_client.c:3074`): `cudaStreamSynchronize` returns immediately before any kernel launch. CUDA stream ordering guarantees HtoD completes before kernels.
- **TCP buffer fix**: Server's `wmem_max` was 212KB. Set to 128MB via `/etc/sysctl.d/99-chidori-tcp.conf`.

### 2. Async Sync RPCs: Eliminate per-token RTT

**Problem:** `cudaStreamSynchronize` and `cudaDeviceSynchronize` were sync RPCs (57ms each). Called hundreds of times during inference.

**Fix:** Convert to `chidori_rpc_async` (`chidori_client.c:3082,3004`). Server still syncs the GPU, but client doesn't wait. The next sync operation (DtoH for `.item()`) naturally orders via TCP — server processes requests sequentially.

### 3. Client-Side Optimizations

| Optimization | Location | What |
|-------------|----------|------|
| Capture tracking | `chidori_client.c:6035` | `cudaStreamIsCapturing` returns locally — no RPC unless capture is active |
| Event pool | `chidori_client.c:3186` | Pre-allocate 256 CUDA events via pipelined RPCs. Recycle on destroy. |
| Thread-local device | `chidori_client.c:309` | `g_current_device` is `__thread` for multi-threaded model loading |
| Skip cudaSetDevice | `chidori_client.c:2370` | In partial offload, remote server has 1 GPU — skip the RPC entirely |
| FP32 accumulation | `cuda_dispatch.c:2730,2829` | Force `CUBLAS_COMPUTE_32F` for FP16 matmuls. Prevents NaN on Qwen. |

### 4. Server-Side Decode (GRAPH_DECODE, 0x0501)

**The big feature.** Instead of sending ~100 CUDA RPCs per token, capture the entire forward pass as a CUDA Graph and run the autoregressive loop on the server.

```
Client:                          Server:
  Prefill → graph capture
  Send GRAPH_DECODE RPC    ──►   for N tokens:
    (graph_exec, stream,           cuMemcpyHtoD(position)
     input_ptr, position_ptr,      cuGraphLaunch(exec, stream)
     logits_ptr, vocab_size,       cuStreamSynchronize(stream)
     num_tokens, start_pos)        cuMemcpyDtoH(logits)
                                   CPU argmax (FP16→float)
                              ◄──  cuMemcpyHtoD(next_token)
  Receive all token IDs           Return: [token_ids]
```

**Result:** 1 network round-trip for N tokens. TinyLlama: 213 tok/s. Llama 8B: 59 tok/s. Llama 70B INT4: 13 tok/s.

**Files:**
- Protocol: `chidori_protocol.h:335` — `CHIDORI_GRAPH_DECODE = 0x0501`
- Server: `cuda_dispatch.c:4002` — decode loop with pinned DtoH + fast FP16 argmax
- Client: `chidori_client.c:9120` — `chidori_graph_decode()` public function callable from Python via ctypes

### 5. Split GPU Decode (GRAPH_DECODE_SPLIT, 0x0502)

For pipeline-parallel inference across local + remote GPU. Client sends activations from local GPU along with remote graph launch in one RPC.

**Protocol:** `chidori_protocol.h:339` — `CHIDORI_GRAPH_DECODE_SPLIT = 0x0502`
**Server:** `cuda_dispatch.c:4133` — writes activations to GPU, launches graph, returns token
**Client:** `chidori_client.c:9165` — `chidori_graph_decode_split()`

### 6. Ansible Deployment

`deploy/setup-gpu-server.yml` — one command to set up a new GPU server:
```bash
ansible-playbook -i "IP," -u ubuntu deploy/setup-gpu-server.yml
```

Installs CUDA deps, builds chidori-worker, sets up Python venv + PyTorch, configures TCP buffers, starts server.

## Test Suite

| Test | What it tests |
|------|--------------|
| `test_train.py` | FP32 matmul training (2000 matmuls) |
| `test_training_fp16.py` | FP16 mixed precision training (50 steps, loss decrease) |
| `test_training.py` | Full forward + backward + optimizer (50 steps) |
| `test_inference.py` | TinyLlama baseline inference |
| `test_inference_cached.py` | Baseline + StaticCache + server-side decode |
| `test_inference_graph.py` | CUDA Graph capture + client replay + server decode |
| `test_inference_multi.py` | 5 prompts via server decode (factual, code, creative) |
| `test_inference_flash.py` | SDPA attention inference |
| `test_split_inference.py` | Split model across 2 GPUs via dispatch_model |
| `test_llama8b.py` | Llama 3.1 8B FP16 server decode |

## Performance

| Workload | Speed |
|----------|-------|
| FP16 training (23M model) | 50K tok/s |
| TinyLlama 1.1B server decode | 157-213 tok/s |
| Llama 3.1 8B FP16 server decode | 56-62 tok/s |
| Llama 3.1 70B INT4 server decode | 13 tok/s |
| Model loading (2.2GB) | 20s |
| Model loading (16GB) | ~5 min |

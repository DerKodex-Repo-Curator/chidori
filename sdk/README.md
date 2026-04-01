<p align="center">
  <img src="https://pypi-camo.freetls.fastly.net/23cb0d4d28bff1d1a0717d16354ddc5f2d10f6d5/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f32346336313137666163643863656130316365386438363666656466613461303f73697a653d323235" width="300" alt="Chidori">
</p>

<h1 align="center">Chidori GPU</h1>

<p align="center">
  <strong>Use any GPU, from anywhere. Zero code changes.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/chidori-gpu/"><img src="https://img.shields.io/pypi/v/chidori-gpu" alt="PyPI"></a>
  <a href="https://pypi.org/project/chidori-gpu/"><img src="https://img.shields.io/pypi/pyversions/chidori-gpu" alt="Python"></a>
</p>

---

Chidori makes remote GPUs feel local. Your PyTorch training scripts, HuggingFace models, and CUDA applications run on remote GPU servers without changing a single line of code. Just `pip install` and go.

```bash
pip install chidori-gpu
```

## 30-Second Demo

**On a machine with a GPU:**
```bash
pip install chidori-gpu
chidori serve
```

**On your laptop (no GPU needed):**
```bash
pip install chidori-gpu
chidori run --server gpu-box:43211 python train.py
```

That's it. Your `train.py` runs on the remote GPU. No Docker, no SSH tunneling, no code changes.

## Why Chidori?

| Without Chidori | With Chidori |
|----------------|-------------|
| SSH into GPU server, set up env, scp files | `chidori run -s host python train.py` |
| Manage CUDA versions across machines | Auto-detects and translates between versions |
| One user per machine | One user per GPU, 8 clients on 8 GPUs |
| Port-forward Jupyter, sync files | Run locally, compute remotely |

## Performance

Tested over WAN with 70ms round-trip latency:

| Workload | GPU | Throughput |
|----------|-----|-----------|
| Matrix multiply (4096x4096) | A100 80GB | **96 TFLOPS** |
| Training (23M params, FP32) | A100 80GB | **32,600 tok/s** |
| Training (23M params, FP16) | A100 80GB | **59,000 tok/s** |
| Inference (server-side decode) | A100 80GB | **100+ tok/s** |

4 parallel clients on 4 A100s: **384 TFLOPS combined.**

## Getting Started

### 1. Install

```bash
pip install chidori-gpu
```

Both client and server. One package, two roles.

### 2. Start a GPU Server

On any machine with an NVIDIA GPU:

```bash
chidori serve
```

First run auto-compiles the CUDA worker from bundled source (needs `gcc` + CUDA toolkit). Subsequent runs use the cached binary.

### 3. Run Your Code Remotely

```bash
chidori run --server 10.0.0.5:43211 python train.py
```

Your script sees a GPU. All CUDA calls are transparently routed over TCP to the remote server.

## Multi-GPU

Got a server with 8 GPUs? Start one server per GPU:

```bash
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i chidori serve --port $((43211 + i)) &
done
```

Each client connects to its own GPU:

```bash
chidori run -s gpu-box:43211 python job_a.py &   # GPU 0
chidori run -s gpu-box:43212 python job_b.py &   # GPU 1
chidori run -s gpu-box:43213 python job_c.py &   # GPU 2
# ...
```

8 researchers, 8 GPUs, zero contention.

## Fast Inference

Standard `model.generate()` works out of the box at ~1 tok/s (limited by per-token RPC overhead). For 100x speedup, enable server-side graph decode:

```python
import chidori
chidori.optimize()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype=torch.float16
).cuda()

# 100+ tok/s — server runs entire decode loop in a single RPC
output = model.generate(
    tokenizer("Hello", return_tensors="pt").input_ids.cuda(),
    max_new_tokens=200,
    do_sample=False,
    cache_implementation="static",
)
print(tokenizer.decode(output[0]))
```

Or use the direct API:

```python
from chidori import server_decode
text = server_decode(model, tokenizer, "Explain quantum computing:", max_new_tokens=200)
# Returns generated text at 100+ tok/s
```

## How It Works

```
┌──────────────────┐          TCP/IP          ┌──────────────────┐
│   Your Laptop    │  ◄──────────────────────► │   GPU Server     │
│                  │                           │                  │
│  python train.py │                           │  chidori serve   │
│       │          │                           │       │          │
│  LD_PRELOAD      │                           │  Real CUDA       │
│  libchidori.so   │   CUDA calls over wire    │  Driver + GPU    │
│       │          │  ─────────────────────►   │       │          │
│  intercepts      │                           │  executes on     │
│  280+ CUDA calls │  ◄─────────────────────   │  real hardware   │
│                  │   results back             │                  │
└──────────────────┘                           └──────────────────┘
```

Chidori uses `LD_PRELOAD` to intercept CUDA API calls at the driver level before they reach the (non-existent) local GPU. Each call is serialized into a compact wire protocol and sent over TCP. The remote server executes it on a real GPU and returns the result.

**Key technical details:**

- **280+ CUDA functions** intercepted — Driver API, Runtime API, cuBLAS, cuBLASLt, cuDNN
- **Async RPC batching** — kernel launches are fire-and-forget, batched via TCP_CORK into single TCP segments
- **Server-side graph decode** — captures a CUDA graph of the forward pass, then runs the entire autoregressive loop server-side (one RPC instead of hundreds)
- **Version-aware struct mapping** — automatically translates `cudaDeviceProp` between CUDA 11.8 through 13.1
- **Zero-copy device-to-device** — D2D copies send only 24 bytes over the wire (data stays on GPU)

## Supported CUDA Versions

| PyTorch Build | CUDA Version | Status |
|--------------|-------------|--------|
| cu118 | CUDA 11.8 | Supported |
| cu121 | CUDA 12.1 | Supported |
| cu126 | CUDA 12.6 | Supported |
| cu128 | CUDA 12.8 | Supported |
| cu130 | CUDA 13.0 | Supported |
| cu131 | CUDA 13.1 | Supported |

Client and server can run different CUDA versions. Chidori handles the translation.

## CLI Reference

```bash
# Start a GPU server (auto-builds on first run)
chidori serve [--port 43211]

# Run a command on a remote GPU
chidori run --server HOST[:PORT] [--verbose] [--trace] COMMAND...

# Examples
chidori run -s 10.0.0.5 python train.py
chidori run -s gpu1:43211 python -c "import torch; print(torch.cuda.get_device_name(0))"
chidori run -s gpu1:43211 --trace python benchmark.py   # show RPC trace
```

## Requirements

**Client (your machine):**
- Linux x86_64
- Python 3.9+
- PyTorch (any cu11x/cu12x/cu13x build)

**Server (GPU machine):**
- Linux x86_64
- NVIDIA GPU with drivers installed
- CUDA toolkit (`nvcc`, headers, libraries)
- `gcc` and `libzstd-dev`

## Contributing

Interested in contributing? Reach out to us.

## License

Proprietary. All rights reserved.

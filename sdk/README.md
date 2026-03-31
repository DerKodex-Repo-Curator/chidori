# Chidori GPU

Run any PyTorch script on a remote GPU. Zero code changes.

```bash
pip install chidori-gpu
```

## Quick Start

**GPU Server:**
```bash
chidori serve
```

**Client (any machine, no GPU needed):**
```bash
chidori run --server GPU_HOST:43211 python train.py
```

Your `train.py` is completely unmodified. Chidori intercepts CUDA calls and routes them to the remote GPU over TCP.

## Features

- **Zero code changes** — works with any PyTorch/HuggingFace script
- **Training** — full forward + backward + optimizer over the wire
- **Inference** — 100+ tok/s with server-side graph decode
- **Multi-GPU** — one client per GPU, scale to 8+ GPUs
- **All PyTorch CUDA versions** — cu118, cu121, cu126, cu128, cu130, cu131

## Performance

Tested over WAN (70ms RTT):

| Workload | Hardware | Speed |
|----------|----------|-------|
| Matmul 4096x4096 | A100-40GB | 96 TFLOPS |
| FP32 Training | A100-40GB | 35K tok/s |
| FP16 Training | A100-40GB | 59K tok/s |
| Inference (graph decode) | A100-40GB | 100+ tok/s |

## Fast Inference

For optimized inference (100+ tok/s instead of ~1 tok/s):

```python
import chidori
chidori.optimize()

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("model", dtype=torch.float16).cuda()
out = model.generate(inputs, max_new_tokens=100, cache_implementation="static")
```

Or use the direct API:

```python
from chidori import server_decode
text = server_decode(model, tokenizer, "Hello world", max_new_tokens=100)
```

## Multi-GPU

Start one server per GPU on different ports:

```bash
# On an 8-GPU server
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i chidori serve --port $((43211 + i)) &
done
```

Each client connects to a specific GPU:

```bash
chidori run -s gpu-server:43211 python job1.py  # GPU 0
chidori run -s gpu-server:43212 python job2.py  # GPU 1
chidori run -s gpu-server:43213 python job3.py  # GPU 2
```

## How It Works

Chidori uses `LD_PRELOAD` to intercept CUDA API calls at the driver level. Each call is serialized over TCP to a remote GPU server that executes it on a real GPU. The application sees a virtual GPU — no code changes needed.

- **280+ CUDA functions** intercepted (Driver API, Runtime API, cuBLAS, cuBLASLt, cuDNN)
- **Async RPC batching** — kernel launches are fire-and-forget, batched via TCP_CORK
- **Server-side graph decode** — captures a CUDA graph and runs the entire inference loop server-side
- **Version-aware** — translates struct layouts between CUDA versions automatically

## License

Apache 2.0

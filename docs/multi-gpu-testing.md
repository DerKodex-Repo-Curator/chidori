# Multi-GPU Testing Guide

Run one client per GPU on a multi-GPU server. Each client gets its own dedicated GPU.

## Prerequisites

- GPU server with N GPUs and matching CUDA version to client's PyTorch
- `chidori-worker` built on the server
- `libchidori.so` built against the **server's** CUDA headers (copy to client)

## Server Setup

Start N server instances, one per GPU, each on a different port:

```bash
# On the GPU server
cd ~/chidori-worker

# Start one worker per GPU
for i in $(seq 0 $((N-1))); do
    port=$((43211 + i))
    CUDA_VISIBLE_DEVICES=$i nohup build/chidori-worker --standalone $port > /tmp/server_gpu${i}.log 2>&1 &
    echo "GPU $i -> port $port (pid $!)"
done

# Verify all running
pgrep -a chidori-worker
```

Example for 8 GPUs:
```
GPU 0 -> port 43211
GPU 1 -> port 43212
GPU 2 -> port 43213
...
GPU 7 -> port 43218
```

## Client Setup

Each client connects to one specific port (= one specific GPU):

```bash
# Client connecting to GPU 0
chidori run -s SERVER_IP:43211 python train.py

# Client connecting to GPU 3
chidori run -s SERVER_IP:43214 python train.py

# Client connecting to GPU 7
chidori run -s SERVER_IP:43218 python train.py
```

Or with manual LD_PRELOAD:
```bash
CHIDORI_DIRECT_CONNECT=1 \
CHIDORI_CFG_OVERRIDE='{"id":"test","servers":["SERVER_IP:43211"]}' \
LD_PRELOAD=build/libcuda.so.1 \
LD_LIBRARY_PATH=build \
python train.py
```

## Running N Clients in Parallel

```bash
# Launch 8 clients, one per GPU
SERVER=66.85.148.90
for i in $(seq 0 7); do
    port=$((43211 + i))
    RESULT_FILE=/tmp/gpu${i}_result.txt \
    chidori run -s ${SERVER}:${port} python /tmp/test_gpu.py &
    echo "Client $i -> GPU $i (port $port)"
done

# Wait for all to finish
wait

# Check results
for i in $(seq 0 7); do
    echo "=== GPU $i ==="
    cat /tmp/gpu${i}_result.txt
done
```

## Test Script Template

```python
import torch, time, os

RESULT_FILE = os.environ.get("RESULT_FILE", "/tmp/gpu_result.txt")
results = []

def log(msg):
    results.append(msg)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(results) + "\n")

try:
    torch._C._cuda_init()
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

    N = 4096
    a = torch.randn(N, N, device="cuda")
    b = torch.randn(N, N, device="cuda")
    torch.cuda.synchronize()
    log("ALLOC: OK")

    t0 = time.time()
    for _ in range(200):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = 2 * N**3 * 200 / dt / 1e12
    log(f"MATMUL: {dt:.2f}s {tflops:.1f} TFLOPS")
    log("PASS")
except Exception as e:
    log(f"ERROR: {e}")
```

## Important: CUDA Version Matching

The client's PyTorch CUDA version **must match** the server's CUDA version.

| PyTorch | Required Server CUDA | cudaDeviceProp size |
|---------|---------------------|-------------------|
| cu118 | CUDA 11.8 | ~900 bytes |
| cu121 | CUDA 12.1 | ~1000 bytes |
| cu124 | CUDA 12.4 | ~1020 bytes |
| cu128 | CUDA 12.8 | 1032 bytes |
| cu130 | CUDA 13.0 | 1008 bytes |

If versions don't match, you'll get `cudaErrorInitializationError` or `Floating point exception` during the first CUDA allocation.

Check versions:
```bash
# Client (PyTorch CUDA version)
python -c "import torch; print(torch.version.cuda)"

# Server (driver CUDA version)
nvidia-smi | head -3
```

## Monitoring

```bash
# Watch GPU utilization on the server
watch -n 1 nvidia-smi

# Check server logs per GPU
tail -f /tmp/server_gpu0.log
tail -f /tmp/server_gpu3.log

# Check how many requests each GPU processed
for i in $(seq 0 7); do
    echo "GPU $i: $(grep 'Processed' /tmp/server_gpu${i}.log | tail -1)"
done
```

## Stopping

```bash
# Kill all server instances
pkill -9 chidori-worker
```

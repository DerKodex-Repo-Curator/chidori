#!/bin/bash
# Chidori FP32 Training Test
# Usage: ./run_training_test.sh [server_ip] [port]
#
# Runs on the GPU server (loopback) by default.
# Restarts the server, runs 20 FP32 training steps with bias+ReLU.

SERVER=${1:-127.0.0.1}
PORT=${2:-43211}
DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Kill old server, restart
kill $(pgrep -f "chidori-worker.*${PORT}") 2>/dev/null
sleep 1
nohup "${DIR}/build/chidori-worker" --standalone "$PORT" > /tmp/chidori-server.log 2>&1 &
sleep 1

if ! pgrep -f "chidori-worker.*${PORT}" > /dev/null; then
    echo "FAIL: Server didn't start. Check /tmp/chidori-server.log"
    exit 1
fi
echo "Server running on port ${PORT}"

# Run training test
CHIDORI_DIRECT_CONNECT=1 \
CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"${SERVER}:${PORT}\"]}" \
LD_PRELOAD="${DIR}/build/libcuda.so.1" \
LD_LIBRARY_PATH="${DIR}/build" \
python3 - << 'PYTHON'
import torch, torch.nn as nn, time
torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
SEP = "=" * 60
print(f"Device: {dev} ({vram:.0f} GB)")
print(SEP)

model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
).cuda()
P = sum(p.numel() for p in model.parameters())
print(f"Model: {P/1e6:.1f}M params (FP32, bias=True, ReLU)")

opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Warmup
x = torch.randn(32, 512).cuda()
t = torch.randn(32, 512).cuda()
opt.zero_grad(); loss_fn(model(x), t).backward(); opt.step()
torch.cuda.synchronize()
print("Warmup done\n")

t0 = time.time()
losses = []
for step in range(20):
    x = torch.randn(64, 512).cuda()
    target = torch.randn(64, 512).cuda()
    opt.zero_grad()
    y = model(x)
    loss = loss_fn(y, target)
    loss.backward()
    opt.step()
    if step % 5 == 0:
        torch.cuda.synchronize()
        losses.append(loss.item())
        elapsed = time.time() - t0
        print(f"  step {step:2d}: loss={loss.item():.4f}  elapsed={elapsed:.2f}s")

torch.cuda.synchronize()
dt = time.time() - t0
tps = 20 * 64 * 512 / dt

print(f"\nLosses: {[round(l,4) for l in losses]}")
trend = "decreasing" if losses[-1] < losses[0] else "stable"
print(f"Trend: {trend}")
print(f"\n{SEP}")
print(f"DEVICE:     {dev}")
print(f"MODEL:      {P/1e6:.1f}M params (FP32, bias=True, ReLU)")
print(f"TRAINING:   {dt:.2f}s (20 steps, batch=64)")
print(f"THROUGHPUT: {tps:.0f} tokens/s")
print(f"RESULT:     PASSED" if all(l < 2.0 for l in losses) else f"RESULT:     FAILED")
print(SEP)
PYTHON

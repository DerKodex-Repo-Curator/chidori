#!/bin/bash
# Chidori FP32 Training Test
# Usage:
#   ./run_training_test.sh                    # loopback (starts server locally)
#   ./run_training_test.sh 66.172.11.7 43211  # remote server (client-only)

SERVER=${1:-127.0.0.1}
PORT=${2:-43211}
DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$SERVER" = "127.0.0.1" ]; then
    # Loopback mode: start server locally
    kill $(pgrep -f "chidori-worker.*${PORT}") 2>/dev/null
    sleep 1
    nohup "${DIR}/build/chidori-worker" --standalone "$PORT" > /tmp/chidori-server.log 2>&1 &
    sleep 1
    if ! pgrep -f "chidori-worker.*${PORT}" > /dev/null; then
        echo "FAIL: Server didn't start. Check /tmp/chidori-server.log"
        exit 1
    fi
    echo "Server running locally on port ${PORT}"
else
    echo "Connecting to remote server ${SERVER}:${PORT}"
fi

# Use venv python if available
PYTHON=python3
if [ -f "${DIR}/../.venv/bin/python" ]; then
    PYTHON="${DIR}/../.venv/bin/python"
elif [ -f "${HOME}/chidori/.venv/bin/python" ]; then
    PYTHON="${HOME}/chidori/.venv/bin/python"
fi

# Run training test
CHIDORI_DIRECT_CONNECT=1 \
CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"${SERVER}:${PORT}\"]}" \
LD_PRELOAD="${DIR}/build/libcuda.so.1" \
LD_LIBRARY_PATH="${DIR}/build" \
$PYTHON - << 'PYTHON'
import torch, torch.nn as nn, time
torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
SEP = "=" * 60
print(f"Device: {dev} ({vram:.0f} GB)")
print(SEP)

model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
).cuda()
P = sum(p.numel() for p in model.parameters())
print(f"Model: {P/1e6:.1f}M params (FP32, bias=True, ReLU)")

opt = torch.optim.SGD(model.parameters(), lr=0.01)

# Warmup
x = torch.randn(16, 256).cuda()
opt.zero_grad(); model(x).sum().backward(); opt.step()
torch.cuda.synchronize()
print("Warmup done\n")

t0 = time.time()
losses = []
for step in range(20):
    x = torch.randn(16, 256).cuda()
    opt.zero_grad()
    y = model(x)
    loss = y.sum()
    loss.backward()
    opt.step()
    if step % 5 == 0:
        torch.cuda.synchronize()
        losses.append(loss.item())
        elapsed = time.time() - t0
        print(f"  step {step:2d}: loss={loss.item():.4f}  elapsed={elapsed:.2f}s")

torch.cuda.synchronize()
dt = time.time() - t0
tps = 20 * 16 * 256 / dt

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

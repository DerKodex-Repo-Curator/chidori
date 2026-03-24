#!/usr/bin/env python3
"""Chidori — GPU Stress Test
Hammers the A100 with sustained large matmuls.
Watch nvidia-smi on the server — GPU util should hit 90%+.
"""
import torch
import time
import sys
import math

torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram:.1f} GB)")
print("=" * 60)

# Disable cublasLt batched path — use standard GEMM for max throughput
torch.backends.cuda.preferred_blas_library("cublas")

results = []

# ============================================================
# Phase 1: FP32 large matmul burst (should pin GPU at ~100%)
# ============================================================
N = 8192
ITERS = 500
print(f"\n[FP32] Allocating {N}x{N} matrices...")
a = torch.randn(N, N, device='cuda')
b = torch.randn(N, N, device='cuda')

# Warmup
for _ in range(5):
    c = torch.mm(a, b)
torch.cuda.synchronize()

print(f"[FP32] Running {ITERS} matmuls — GPU should be at 100% NOW")
t0 = time.time()
for i in range(ITERS):
    c = torch.mm(a, b)
torch.cuda.synchronize()
dt = time.time() - t0

flops = 2 * N * N * N * ITERS
tflops = flops / dt / 1e12
print(f"[FP32] {dt:.1f}s — {tflops:.2f} TFLOPS — {ITERS/dt:.0f} matmuls/s")
results.append(("FP32 8192x8192", tflops, dt))
del a, b, c
torch.cuda.empty_cache()

# ============================================================
# Phase 2: FP16 large matmul burst (should hit tensor cores)
# ============================================================
print(f"\n[FP16] Allocating {N}x{N} FP16 matrices...")
a = torch.randn(N, N, device='cuda', dtype=torch.float16)
b = torch.randn(N, N, device='cuda', dtype=torch.float16)

for _ in range(5):
    c = torch.mm(a, b)
torch.cuda.synchronize()

ITERS_FP16 = 1000
print(f"[FP16] Running {ITERS_FP16} matmuls — tensor cores should light up")
t0 = time.time()
for i in range(ITERS_FP16):
    c = torch.mm(a, b)
torch.cuda.synchronize()
dt = time.time() - t0

flops = 2 * N * N * N * ITERS_FP16
tflops = flops / dt / 1e12
print(f"[FP16] {dt:.1f}s — {tflops:.2f} TFLOPS — {ITERS_FP16/dt:.0f} matmuls/s")
results.append(("FP16 8192x8192", tflops, dt))
del a, b, c
torch.cuda.empty_cache()

# ============================================================
# Phase 3: Training stress — large model, big batch
# ============================================================
import torch.nn as nn

HIDDEN = 2048
LAYERS = 4
BATCH = 64
SEQ = 128
STEPS = 30

print(f"\n[TRAIN] Building {LAYERS}-layer model (hidden={HIDDEN})...")
model = nn.Sequential(*[
    nn.Sequential(
        nn.Linear(HIDDEN, HIDDEN * 4, bias=False),
        nn.GELU(),
        nn.Linear(HIDDEN * 4, HIDDEN, bias=False),
    ) for _ in range(LAYERS)
]).cuda()

params = sum(p.numel() for p in model.parameters())
print(f"[TRAIN] {params/1e6:.0f}M params, batch={BATCH}x{SEQ}")

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Warmup
x = torch.randn(BATCH * SEQ, HIDDEN, device='cuda')
t = torch.randn(BATCH * SEQ, HIDDEN, device='cuda')
optimizer.zero_grad()
loss_fn(model(x), t).backward()
optimizer.step()
torch.cuda.synchronize()

print(f"[TRAIN] Running {STEPS} steps — sustained GPU load")
t0 = time.time()
losses = []
for step in range(STEPS):
    x = torch.randn(BATCH * SEQ, HIDDEN, device='cuda')
    target = torch.randn(BATCH * SEQ, HIDDEN, device='cuda')
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        torch.cuda.synchronize()
        losses.append(loss.item())
        tps = (step + 1) * BATCH * SEQ / (time.time() - t0)
        print(f"  step {step:3d}  loss={loss.item():.4f}  tokens/s={tps:.0f}")

torch.cuda.synchronize()
dt = time.time() - t0
tps = STEPS * BATCH * SEQ / dt
print(f"[TRAIN] {dt:.1f}s — {tps:.0f} tokens/s ({tps/1000:.1f}K tok/s)")
results.append(("Training loop", tps / 1000, dt))

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  GPU Stress Results — {dev}")
print(f"{'='*60}")
for name, metric, duration in results:
    if "matmul" in name.lower() or "fp" in name.lower():
        print(f"  {name:25s}  {metric:7.2f} TFLOPS  ({duration:.1f}s)")
    else:
        print(f"  {name:25s}  {metric:7.1f}K tok/s ({duration:.1f}s)")
print(f"{'='*60}")

# Validate
all_ok = True
for name, metric, _ in results:
    if metric <= 0:
        print(f"[VERIFY] FAIL: {name} — zero throughput")
        all_ok = False
    if not math.isfinite(metric):
        print(f"[VERIFY] FAIL: {name} — non-finite metric")
        all_ok = False

all_finite = all(math.isfinite(l) for l in losses)
if not all_finite:
    print("[VERIFY] FAIL: training produced nan/inf losses")
    all_ok = False

if all_ok:
    print("PASS: test_gpu_stress")
    sys.exit(0)
else:
    print("FAIL: test_gpu_stress")
    sys.exit(1)

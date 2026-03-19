#!/usr/bin/env python3
"""Chidori ⚡ — Proper GPU workflow: LOAD → COMPUTE → READ"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

N = 4096
ITERS = 2000

# ============================================================
# LOAD PHASE — upload everything to GPU, accept the latency
# ============================================================
print(f"\n[LOAD] Uploading {N}x{N} matrices to GPU...")
t0 = time.time()
a = torch.randn(N, N).cuda()
b = torch.randn(N, N).cuda()
c = torch.zeros(N, N).cuda()
torch.cuda.synchronize()
load_dt = time.time() - t0
print(f"[LOAD] Done: {load_dt:.1f}s ({N*N*4*3/1e6:.0f} MB transferred)")

# ============================================================
# COMPUTE PHASE — burst matmuls, GPU should stay busy
# ============================================================
print(f"\n[COMPUTE] {ITERS}x mm({N}x{N}) — ALL data already on GPU")
print("[COMPUTE] Watch nvidia-smi NOW...")

# Warmup
c = torch.mm(a, b)
torch.cuda.synchronize()

t0 = time.time()
for i in range(ITERS):
    c = torch.mm(a, b)
# Single sync at the end — GPU processes entire queue
torch.cuda.synchronize()
compute_dt = time.time() - t0

flops = 2 * N * N * N * ITERS
tflops = flops / compute_dt / 1e12
print(f"[COMPUTE] Done: {compute_dt:.2f}s — {tflops:.2f} TFLOPS")
print(f"[COMPUTE] {ITERS/compute_dt:.0f} matmuls/s")

# ============================================================
# READ PHASE — pull back one result
# ============================================================
print(f"\n[READ] Downloading result...")
t0 = time.time()
result = c.cpu()
read_dt = time.time() - t0
print(f"[READ] Done: {read_dt:.1f}s")
print(f"[READ] Sum: {result.sum().item():.2f}")

import sys

# Verify data is not garbage
s = result.sum().item()
passed = True
if s == 0.0:
    print("[VERIFY] FAIL: sum is exactly 0 — GPU may not have computed anything")
    passed = False
elif abs(s) > 1e15:
    print(f"[VERIFY] FAIL: sum is suspiciously large ({s}) — possible garbage data")
    passed = False
else:
    print(f"[VERIFY] OK: sum={s:.2f} looks reasonable for {N}x{N} matmul")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"LOAD:    {load_dt:.1f}s (network transfer)")
print(f"COMPUTE: {compute_dt:.2f}s ({tflops:.2f} TFLOPS, {ITERS} matmuls)")
print(f"READ:    {read_dt:.1f}s (network transfer)")
print(f"TOTAL:   {load_dt + compute_dt + read_dt:.1f}s")
print(f"{'='*60}")

if passed:
    print("PASS: test_train")
    sys.exit(0)
else:
    print("FAIL: test_train")
    sys.exit(1)

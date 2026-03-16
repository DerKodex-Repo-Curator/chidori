#!/usr/bin/env python3
"""Chidori ⚡ — SPIKE THE GPU"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# Allocate big matrices
N = 4096
print(f"Allocating {N}x{N} matrices ({N*N*4/1e6:.0f} MB each)...")
a = torch.randn(N, N).cuda()
b = torch.randn(N, N).cuda()

# Warm up
c = torch.mm(a, b)
torch.cuda.synchronize()
print("Warmup done")

# Burst: 100 matmuls back to back
print(f"\n100x mm({N}x{N}) — watch nvidia-smi...")
t0 = time.time()
for i in range(100):
    c = torch.mm(a, b)
torch.cuda.synchronize()
dt = time.time() - t0

flops = 2 * N * N * N * 100
tflops = flops / dt / 1e12
print(f"Time: {dt:.2f}s — {tflops:.2f} TFLOPS")
print(f"Sum: {c.cpu().sum().item():.2f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

#!/usr/bin/env python3
"""Chidori ⚡ — Heavy matmul training on remote GPU"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# Test matmul at different scales
for N in [64, 256, 1024, 2048]:
    a = torch.randn(N, N).cuda()
    b = torch.randn(N, N).cuda()

    # Warmup
    c = torch.mm(a, b)

    t0 = time.time()
    for _ in range(10):
        c = torch.mm(a, b)
    dt = time.time() - t0

    s = c.cpu().sum().item()
    flops = 2 * N * N * N * 10
    gflops = flops / dt / 1e9 if dt > 0 else 0
    print(f"mm({N:4d}x{N:4d}): {dt:.3f}s  {gflops:.1f} GFLOPS  sum={s:.2f}")

# Training loop with matmul
print("\n--- Training with matmul ---")
DIM = 512
BATCH = 256
EPOCHS = 50
lr = 0.001

W = (torch.randn(DIM, DIM) * 0.01).cuda()
b = torch.zeros(DIM).cuda()

total_t0 = time.time()
for epoch in range(EPOCHS):
    x = torch.randn(BATCH, DIM).cuda()
    target = torch.mm(x, torch.eye(DIM).cuda() * 0.5)  # target: scale by 0.5

    # Forward
    y = torch.mm(x, W) + b

    # Loss
    diff = y - target
    loss = (diff * diff).sum() * (1.0 / (BATCH * DIM))

    # Gradient
    grad_W = torch.mm(x.t(), diff) * (2.0 / (BATCH * DIM))
    grad_b = diff.sum(dim=0) * (2.0 / (BATCH * DIM))

    # Update
    W = W - grad_W * lr
    b = b - grad_b * lr

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        l = loss.cpu().item()
        print(f"  Epoch {epoch:3d}: loss={l:.6f}")

total_dt = time.time() - total_t0
print(f"\nTraining: {total_dt:.1f}s ({EPOCHS/total_dt:.1f} epochs/s)")

# Verify W converged to ~0.5 * I
w_diag = torch.diag(W.cpu()).mean().item()
print(f"W diagonal mean: {w_diag:.4f} (target: 0.5)")

print("\n" + "=" * 60)
print("DONE — REMOTE GPU MATMUL TRAINING COMPLETE")
print("=" * 60)

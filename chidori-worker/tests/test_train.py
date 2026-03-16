#!/usr/bin/env python3
"""Chidori ⚡ — FAT training loop — make the H100 sweat"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

BATCH_SIZE = 4096
DIM = 4096
EPOCHS = 50
lr = 0.00001

# Big tensors on GPU
W = (torch.randn(DIM, DIM) * 0.001).cuda()  # 4096x4096 = 64MB
b = torch.zeros(DIM).cuda()

mem_params = (W.nelement() + b.nelement()) * 4 / 1e6
print(f"Weight matrix: {DIM}x{DIM} = {W.nelement():,} params ({mem_params:.0f} MB)")
print(f"Batch={BATCH_SIZE}, Epochs={EPOCHS}")
print()

total_t0 = time.time()
total_kernels = 0

for epoch in range(EPOCHS):
    t0 = time.time()

    # Generate big batch on CPU, transfer to GPU (64MB)
    x = torch.randn(BATCH_SIZE, DIM).cuda()
    target = x * 0.5  # simple target: scale by 0.5

    # Forward: y = x * W_diagonal (elementwise, treating W as diagonal)
    # Use the diagonal of W as scaling factors
    w_diag = W.sum(dim=0) * (1.0 / DIM)  # approximate diagonal
    y = x * w_diag

    # Loss
    diff = y - target
    loss_per_elem = diff * diff
    loss = loss_per_elem.sum() * (1.0 / (BATCH_SIZE * DIM))

    # Gradient
    grad = diff * (2.0 / (BATCH_SIZE * DIM))
    grad_w = (x * grad).sum(dim=0)

    # Clip gradient
    grad_norm = (grad_w * grad_w).sum()
    # Manual sqrt approximation: just scale down if too large
    grad_w = grad_w * (lr / (grad_norm * 0.001 + 1.0))

    # Update diagonal
    w_update = grad_w.unsqueeze(0).expand(DIM, DIM) * (1.0 / DIM)
    W = W - w_update

    total_kernels += 12  # approximate kernels per epoch
    dt = time.time() - t0

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        l = loss.cpu().item()
        print(f"  Epoch {epoch:3d}: loss={l:.6f}  ({dt:.2f}s/epoch, ~{12/dt:.0f} kernels/s)")

total_dt = time.time() - total_t0

print()
print(f"Training: {total_dt:.1f}s total ({EPOCHS/total_dt:.1f} epochs/s)")
print(f"~{total_kernels} kernel launches, {total_kernels/total_dt:.0f} kernels/s")
print(f"Data moved: ~{BATCH_SIZE * DIM * 4 * EPOCHS / 1e9:.1f} GB to GPU")
print()
print("=" * 60)
print("FAT TRAINING ON REMOTE H100 COMPLETE")
print("=" * 60)

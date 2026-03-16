#!/usr/bin/env python3
"""Chidori ⚡ — Training loop on remote H100"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# Simple linear model: y = Wx + b
# Train to learn y = 3x + 7
INPUT_DIM = 64
BATCH_SIZE = 256
EPOCHS = 100

# Model weights on GPU
W = torch.randn(1, INPUT_DIM, device='cuda') * 0.01
b = torch.zeros(1, device='cuda')
lr = 0.001

print(f"Training: y = 3*mean(x) + 7")
print(f"Batch={BATCH_SIZE}, Input={INPUT_DIM}, Epochs={EPOCHS}")
print(f"W shape={W.shape}, device={W.device}")
print()

total_t0 = time.time()

for epoch in range(EPOCHS):
    # Generate batch
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device='cuda')
    y_true = 3.0 * x.mean(dim=1, keepdim=True) + 7.0

    # Forward: y_pred = x @ W.T + b
    y_pred = x @ W.t() + b

    # Loss: MSE
    diff = y_pred - y_true
    loss = (diff * diff).mean()

    # Backward (manual gradient)
    # dL/dW = 2/N * (y_pred - y_true).T @ x
    # dL/db = 2/N * sum(y_pred - y_true)
    grad_W = (2.0 / BATCH_SIZE) * diff.t() @ x
    grad_b = (2.0 / BATCH_SIZE) * diff.sum()

    # Update
    W = W - lr * grad_W
    b = b - lr * grad_b

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        l = loss.cpu().item()
        w_sum = W.cpu().sum().item()
        b_val = b.cpu().item()
        print(f"  Epoch {epoch:3d}: loss={l:.4f}  W_sum={w_sum:.4f}  b={b_val:.4f}")

total_dt = time.time() - total_t0

print()
print(f"Training complete: {total_dt:.1f}s ({EPOCHS/total_dt:.0f} epochs/s)")
print(f"Final: W_sum={W.cpu().sum().item():.4f}  b={b.cpu().item():.4f}")
print(f"Expected: W_sum≈3.0 (spread across {INPUT_DIM} dims = {3.0/INPUT_DIM:.4f} each)  b≈7.0")
print()
print("=" * 60)
print("TRAINING ON REMOTE H100 COMPLETE")
print("=" * 60)

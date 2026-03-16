#!/usr/bin/env python3
"""Chidori ⚡ — Training loop on remote H100"""
import torch
import time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

BATCH_SIZE = 256
INPUT_DIM = 32
EPOCHS = 200
lr = 0.01

# Create weights on CPU, move to GPU
W = (torch.randn(1, INPUT_DIM) * 0.01).cuda()
b = torch.zeros(1).cuda()

print(f"Training: y = 3*mean(x) + 7")
print(f"Batch={BATCH_SIZE}, Dim={INPUT_DIM}, Epochs={EPOCHS}, lr={lr}")
print()

total_t0 = time.time()

for epoch in range(EPOCHS):
    # Generate data on CPU, move to GPU
    x = torch.randn(BATCH_SIZE, INPUT_DIM).cuda()

    # Target: y = 3 * mean(x) + 7
    x_mean = x.sum(dim=1) * (1.0 / INPUT_DIM)  # mean using sum + multiply
    y_true = x_mean * 3.0 + 7.0

    # Forward: y_pred = sum(x * W) + b (linear layer using elementwise ops)
    xw = x * W  # broadcast multiply
    y_pred = xw.sum(dim=1) + b

    # Loss: MSE using only add/subtract/multiply
    diff = y_pred - y_true
    loss = (diff * diff).sum() * (1.0 / BATCH_SIZE)

    # Gradient (manual)
    # dL/dy_pred = 2 * diff / BATCH_SIZE
    grad_out = diff * (2.0 / BATCH_SIZE)

    # dL/dW = grad_out.T @ x (using elementwise + sum)
    # For each weight dim: dW[j] = sum(grad_out * x[:, j])
    grad_W = (x * grad_out.unsqueeze(1)).sum(dim=0).unsqueeze(0)
    grad_b = grad_out.sum()

    # SGD update
    W = W - grad_W * lr
    b = b - grad_b * lr

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        l = loss.cpu().item()
        w_sum = W.sum().cpu().item()
        b_val = b.cpu().item()
        print(f"  Epoch {epoch:3d}: loss={l:.4f}  W_sum={w_sum:.4f}  b={b_val:.4f}")

total_dt = time.time() - total_t0

print()
print(f"Training: {total_dt:.1f}s ({EPOCHS/total_dt:.0f} epochs/s)")
print(f"Final: W_sum={W.sum().cpu().item():.4f}  b={b.cpu().item():.4f}")
print(f"Target: W_sum≈{3.0/INPUT_DIM * INPUT_DIM:.4f} (3.0)  b≈7.0")
print()

# Verify
test_x = torch.ones(1, INPUT_DIM).cuda()
pred = (test_x * W).sum() + b
print(f"Predict f([1,1,...,1]): {pred.cpu().item():.2f} (expected: {3.0 + 7.0:.2f})")

print()
print("=" * 60)
print("TRAINING ON REMOTE H100 COMPLETE")
print("=" * 60)

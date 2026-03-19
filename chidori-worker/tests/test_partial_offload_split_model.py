#!/usr/bin/env python3
"""Partial GPU offloading — split model training across local + remote GPU.

Layer 0 on local GPU (device 0), Layer 1 on remote GPU (device 1).
Forward pass moves activations between devices.
Backward pass + optimizer step works across both.
"""
import sys
import math
import torch
import torch.nn as nn

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2 devices (local + remote)"

class SplitModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.layer0 = nn.Linear(hidden, hidden, bias=False).to("cuda:0")
        self.layer1 = nn.Linear(hidden, hidden, bias=False).to("cuda:1")

    def forward(self, x):
        h = torch.relu(self.layer0(x))   # compute on local GPU
        h = h.to("cuda:1")               # transfer to remote GPU
        return self.layer1(h)             # compute on remote GPU

model = SplitModel(512)
print(f"layer0 on {next(model.layer0.parameters()).device}")
print(f"layer1 on {next(model.layer1.parameters()).device}")

opt = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []
for step in range(10):
    x = torch.randn(32, 512, device="cuda:0")
    target = torch.randn(32, 512, device="cuda:1")

    opt.zero_grad()
    out = model(x)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    opt.step()

    l = loss.item()
    losses.append(l)
    if step % 3 == 0:
        print(f"  step {step}: loss={l:.4f}")

print(f"Losses: {[round(l,4) for l in losses]}")
assert all(math.isfinite(l) for l in losses), "NaN/Inf in losses"
print(f"Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")

print("PASS: test_partial_offload_split_model")
sys.exit(0)

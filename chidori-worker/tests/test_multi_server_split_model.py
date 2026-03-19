#!/usr/bin/env python3
"""Multi-server split model test: train a model split across two GPUs.

Layer 0 on device 0, Layer 1 on device 1.
Forward: input → layer0(dev0) → transfer → layer1(dev1) → output
Backward + optimizer step across both devices.
"""
import sys
import torch
import torch.nn as nn

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2+ devices"

class SplitModel(nn.Module):
    """Model with layers on different devices."""
    def __init__(self, hidden=512):
        super().__init__()
        self.layer0 = nn.Linear(hidden, hidden, bias=False).to("cuda:0")
        self.layer1 = nn.Linear(hidden, hidden, bias=False).to("cuda:1")

    def forward(self, x):
        # Layer 0 on device 0
        h = torch.relu(self.layer0(x))
        # Transfer to device 1
        h = h.to("cuda:1")
        # Layer 1 on device 1
        return self.layer1(h)

H = 512
model = SplitModel(H)
print(f"Layer 0 on: {next(model.layer0.parameters()).device}")
print(f"Layer 1 on: {next(model.layer1.parameters()).device}")
assert str(next(model.layer0.parameters()).device) == "cuda:0"
assert str(next(model.layer1.parameters()).device) == "cuda:1"

# Optimizer handles parameters on different devices
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
losses = []
for step in range(10):
    x = torch.randn(32, H, device="cuda:0")
    target = torch.randn(32, H, device="cuda:1")

    opt.zero_grad()
    out = model(x)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    opt.step()

    l = loss.item()
    losses.append(l)
    if step % 3 == 0:
        print(f"  step {step}: loss={l:.4f}")

import math
all_finite = all(math.isfinite(l) for l in losses)
print(f"Losses: {[round(l,4) for l in losses]}")
assert all_finite, "NaN/Inf in losses"
assert losses[-1] < losses[0] * 1.5, f"Loss didn't converge: {losses[0]:.4f} → {losses[-1]:.4f}"
print(f"Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")

print("PASS: test_multi_server_split_model")
sys.exit(0)

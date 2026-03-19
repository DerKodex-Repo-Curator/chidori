#!/usr/bin/env python3
"""Partial GPU offloading — cross-device data transfer.

Move tensors between local GPU (device 0) and remote GPU (device 1).
Data must survive the trip through host memory staging.
"""
import sys
import torch

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2 devices (local + remote)"

# Create on local, transfer to remote
x0 = torch.randn(64, 256, device="cuda:0")
sum0 = x0.sum().item()
print(f"Original on cuda:0: sum={sum0:.4f}")

x1 = x0.to("cuda:1")
sum1 = x1.sum().item()
print(f"Transferred to cuda:1: sum={sum1:.4f}")
assert abs(sum0 - sum1) < 0.01, f"Data corrupted local→remote: {sum0} vs {sum1}"
print("  local → remote OK")

# Create on remote, transfer to local
y1 = torch.randn(32, 128, device="cuda:1")
sumy1 = y1.sum().item()
print(f"Original on cuda:1: sum={sumy1:.4f}")

y0 = y1.to("cuda:0")
sumy0 = y0.sum().item()
print(f"Transferred to cuda:0: sum={sumy0:.4f}")
assert abs(sumy1 - sumy0) < 0.01, f"Data corrupted remote→local: {sumy1} vs {sumy0}"
print("  remote → local OK")

# Large transfer (4MB)
big = torch.randn(1024, 1024, device="cuda:0")
big_sum = big.sum().item()
big1 = big.to("cuda:1")
assert abs(big_sum - big1.sum().item()) < 0.1, "Large transfer corrupted"
print(f"  4MB transfer OK")

print("PASS: test_partial_offload_transfer")
sys.exit(0)

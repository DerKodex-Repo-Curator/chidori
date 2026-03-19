#!/usr/bin/env python3
"""Multi-server transfer test: move data between devices on different servers.

Tests .to("cuda:1"), .to("cuda:0"), and verifies data integrity.
"""
import sys
import torch

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2+ devices"

# Create data on device 0
torch.cuda.set_device(0)
x0 = torch.randn(64, 256, device="cuda:0")
sum0 = x0.sum().item()
print(f"Original on cuda:0: shape={x0.shape}, sum={sum0:.4f}")

# Transfer device 0 → device 1
x1 = x0.to("cuda:1")
sum1 = x1.sum().item()
print(f"Transferred to cuda:1: shape={x1.shape}, sum={sum1:.4f}")
assert abs(sum0 - sum1) < 0.01, f"Data corrupted: {sum0} vs {sum1}"
print("  cuda:0 → cuda:1 OK, data preserved")

# Create data on device 1
torch.cuda.set_device(1)
y1 = torch.randn(32, 128, device="cuda:1")
sumy = y1.sum().item()
print(f"Original on cuda:1: shape={y1.shape}, sum={sumy:.4f}")

# Transfer device 1 → device 0
y0 = y1.to("cuda:0")
sumy0 = y0.sum().item()
print(f"Transferred to cuda:0: shape={y0.shape}, sum={sumy0:.4f}")
assert abs(sumy - sumy0) < 0.01, f"Data corrupted: {sumy} vs {sumy0}"
print("  cuda:1 → cuda:0 OK, data preserved")

# Larger transfer
torch.cuda.set_device(0)
big = torch.randn(1024, 1024, device="cuda:0")  # 4MB
big_sum = big.sum().item()
big1 = big.to("cuda:1")
big1_sum = big1.sum().item()
assert abs(big_sum - big1_sum) < 0.1, f"Large transfer corrupted: {big_sum} vs {big1_sum}"
print(f"  Large transfer (4MB) OK: sum diff = {abs(big_sum - big1_sum):.6f}")

print("PASS: test_multi_server_transfer")
sys.exit(0)

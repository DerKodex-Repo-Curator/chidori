#!/usr/bin/env python3
"""Partial GPU offloading — independent compute on each device.

Device 0 (local GPU) and device 1 (remote GPU) can each run matmul independently.
"""
import sys
import torch

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2 devices (local + remote)"

# Compute on local GPU
torch.cuda.set_device(0)
a0 = torch.randn(512, 512, device="cuda:0")
b0 = torch.randn(512, 512, device="cuda:0")
c0 = torch.mm(a0, b0)
torch.cuda.synchronize()
v0 = c0[0, 0].item()
print(f"cuda:0 (local)  matmul: c[0,0]={v0:.4f}")
assert v0 != 0.0, "Local matmul returned zero — real CUDA not working"

# Compute on remote GPU
torch.cuda.set_device(1)
a1 = torch.randn(512, 512, device="cuda:1")
b1 = torch.randn(512, 512, device="cuda:1")
c1 = torch.mm(a1, b1)
torch.cuda.synchronize()
v1 = c1[0, 0].item()
print(f"cuda:1 (remote) matmul: c[0,0]={v1:.4f}")
assert v1 != 0.0, "Remote matmul returned zero — Chidori RPC not working"

# Both should produce different results (different random data)
assert abs(v0 - v1) > 0.001, "Both matmuls returned identical values — suspicious"

print("PASS: test_partial_offload_compute")
sys.exit(0)

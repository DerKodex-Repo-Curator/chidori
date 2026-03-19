#!/usr/bin/env python3
"""Multi-server compute test: matmul on each device independently.

Requires 2 Chidori servers.
"""
import sys
import torch

torch._C._cuda_init()
assert torch.cuda.device_count() >= 2, "Need 2+ devices"

# Matmul on device 0
torch.cuda.set_device(0)
a0 = torch.randn(512, 512, device="cuda:0")
b0 = torch.randn(512, 512, device="cuda:0")
c0 = torch.mm(a0, b0)
torch.cuda.synchronize()
val0 = c0[0, 0].item()
print(f"cuda:0 matmul: c[0,0]={val0:.4f}")
assert val0 != 0.0, "Device 0 matmul returned zero"

# Matmul on device 1
torch.cuda.set_device(1)
a1 = torch.randn(512, 512, device="cuda:1")
b1 = torch.randn(512, 512, device="cuda:1")
c1 = torch.mm(a1, b1)
torch.cuda.synchronize()
val1 = c1[0, 0].item()
print(f"cuda:1 matmul: c[0,0]={val1:.4f}")
assert val1 != 0.0, "Device 1 matmul returned zero"

# Larger benchmark on each
import time
for dev in [0, 1]:
    torch.cuda.set_device(dev)
    x = torch.randn(2048, 2048, device=f"cuda:{dev}")
    y = torch.randn(2048, 2048, device=f"cuda:{dev}")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = 2 * 2048**3 * 100 / dt / 1e12
    print(f"cuda:{dev} benchmark: {tflops:.2f} TFLOPS")

print("PASS: test_multi_server_compute")
sys.exit(0)

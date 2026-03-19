#!/usr/bin/env python3
"""Chidori — Partial GPU offloading test.

Tests that device 0 (local GPU) and device 1 (remote GPU via Chidori)
can both execute CUDA operations. Requires:
  CHIDORI_LOCAL_DEVICES=0
  CHIDORI_REAL_CUDA_PATH=/usr/lib/libcuda.so.1  (or wherever real driver lives)
  CHIDORI_DIRECT_CONNECT=1
  CHIDORI_CFG_OVERRIDE='{"id":"test","servers":["REMOTE:43211"]}'
"""
import torch
import time

torch._C._cuda_init()

num_devices = torch.cuda.device_count()
print(f"Devices: {num_devices}")
for i in range(num_devices):
    print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}")

assert num_devices >= 2, f"Expected at least 2 devices (local + remote), got {num_devices}"

# Test 1: Local GPU (device 0)
print("\n=== Test 1: Local GPU (device 0) ===")
torch.cuda.set_device(0)
a = torch.randn(256, 256, device="cuda:0")
b = torch.randn(256, 256, device="cuda:0")
c = torch.mm(a, b)
torch.cuda.synchronize()
print(f"  matmul on cuda:0 OK: c[0,0]={c[0,0].item():.4f}")
assert c[0, 0].item() != 0.0, "Local matmul returned zeros!"

# Test 2: Remote GPU (device 1)
print("\n=== Test 2: Remote GPU (device 1) ===")
torch.cuda.set_device(1)
d = torch.randn(256, 256, device="cuda:1")
e = torch.randn(256, 256, device="cuda:1")
f = torch.mm(d, e)
torch.cuda.synchronize()
print(f"  matmul on cuda:1 OK: f[0,0]={f[0,0].item():.4f}")

# Test 3: Cross-device data movement
print("\n=== Test 3: Cross-device transfer ===")
x = torch.randn(16, 256, device="cuda:0")
print(f"  x on cuda:0, sum={x.sum().item():.4f}")
y = x.to("cuda:1")
print(f"  y on cuda:1, sum={y.sum().item():.4f}")
# Verify data is the same
assert abs(x.sum().item() - y.sum().item()) < 0.01, "Cross-device transfer corrupted data!"
print("  Data preserved across devices!")

# Test 4: Compute on both devices in sequence
print("\n=== Test 4: Sequential compute across devices ===")
w0 = torch.randn(256, 256, device="cuda:0")
w1 = torch.randn(256, 256, device="cuda:1")
inp = torch.randn(16, 256, device="cuda:0")

# Layer 1 on local
h = torch.mm(inp, w0)
torch.cuda.synchronize()
print(f"  Layer 1 (local): {h.shape}")

# Move to remote
h = h.to("cuda:1")

# Layer 2 on remote
out = torch.mm(h, w1)
torch.cuda.synchronize()
print(f"  Layer 2 (remote): {out.shape}")
print(f"  Output sum={out.sum().item():.2f}")

print("\n" + "=" * 60)
print("PARTIAL OFFLOAD TEST: ALL PASSED")
print("=" * 60)

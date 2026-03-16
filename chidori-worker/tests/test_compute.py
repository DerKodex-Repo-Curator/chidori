#!/usr/bin/env python3
"""Chidori ⚡ — Sustained H100 stress test using operations that work"""
import torch, time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# Warm up — first kernel launch has overhead
x = torch.tensor([1.0]).cuda()
_ = x * 2
_.cpu()

# 1. Scale 10M floats
print("\n[1] Scale 10M floats × 3.14")
big = torch.randn(10_000_000).cuda()
t0 = time.time()
r = big * 3.14
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    {dt:.3f}s — {10/dt:.0f} M elements/s")
v = r.cpu()
print(f"    sum={v.sum().item():.2f}")

# 2. Scale 100M floats
print("\n[2] Scale 100M floats × 2.71")
big = torch.randn(100_000_000).cuda()
t0 = time.time()
r = big * 2.71
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    {dt:.3f}s — {100/dt:.0f} M elements/s")
v = r.cpu()
print(f"    sum={v.sum().item():.2f}")

# 3. Repeated multiply — 500 kernel launches
print("\n[3] 500 consecutive multiplies on 1M tensor")
x = torch.randn(1_000_000).cuda()
t0 = time.time()
for i in range(500):
    x = x * 1.0001
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    {dt:.3f}s — {500/dt:.0f} kernels/s ({dt/500*1000:.1f}ms per kernel)")
v = x.cpu()
print(f"    sum={v.sum().item():.2f}")

# 4. Large tensor — 500M floats (2GB)
print("\n[4] Allocate + scale 500M floats (2GB)")
try:
    big = torch.randn(500_000_000).cuda()
    t0 = time.time()
    r = big * 1.5
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"    {dt:.3f}s — {500/dt:.0f} M elements/s — {2000/dt:.0f} MB/s")
    del big, r
except Exception as e:
    print(f"    FAILED: {e}")

# 5. Sustained burst — 1000 kernel launches
print("\n[5] 1000 kernel launches on 100K tensor")
x = torch.randn(100_000).cuda()
t0 = time.time()
for i in range(1000):
    x = x * 1.00001
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    {dt:.3f}s — {1000/dt:.0f} kernels/s")
v = x.cpu()
print(f"    sum={v.sum().item():.2f}")

print("\n" + "=" * 60)
print(f"ALL TESTS COMPLETE — H100 COMPUTED OVER TCP/IP")
print("=" * 60)

#!/usr/bin/env python3
"""Chidori ⚡ — Make the H100 SWEAT"""
import torch, time

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# 1. Verify basic compute works
print("\n[1] Basic: x * 2")
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
y = x * 2
print(f"    Result: {y.cpu()}")

# 2. Bigger element-wise
print("\n[2] Element-wise: 1M floats * 3.14")
big = torch.randn(1000000).cuda()
t0 = time.time()
result = big * 3.14
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    Time: {dt:.3f}s")
r = result.cpu()
print(f"    sum={r.sum().item():.2f} mean={r.mean().item():.4f}")

# 3. Chain of operations
print("\n[3] Chain: sin(x^2 + 1) * exp(-x)")
x = torch.randn(1000000).cuda()
t0 = time.time()
y = torch.sin(x * x + 1) * torch.exp(-x)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    Time: {dt:.3f}s")
print(f"    sum={y.cpu().sum().item():.2f}")

# 4. Large tensor fill + reduce
print("\n[4] Reduction: sum of 10M random floats")
big = torch.randn(10000000).cuda()
t0 = time.time()
s = big.sum()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    Time: {dt:.3f}s")
print(f"    sum={s.cpu().item():.2f}")

# 5. Multiple kernels back to back
print("\n[5] 100 consecutive kernel launches")
x = torch.randn(100000).cuda()
t0 = time.time()
for i in range(100):
    x = x * 1.001 + 0.001
torch.cuda.synchronize()
dt = time.time() - t0
print(f"    Time: {dt:.3f}s ({dt/100*1000:.1f}ms per kernel)")
print(f"    final sum={x.cpu().sum().item():.2f}")

print("\n" + "=" * 60)
print("ALL COMPUTE TESTS PASSED ON REMOTE H100!")
print("=" * 60)

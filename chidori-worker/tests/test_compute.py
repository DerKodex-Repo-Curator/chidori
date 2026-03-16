#!/usr/bin/env python3
"""Test actual GPU compute through Chidori"""
import torch, time

torch._C._cuda_init()
print("Device:", torch.cuda.get_device_name(0))

# Test 1: transfer and read back
print("\n--- Test 1: Transfer ---")
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
print("x on GPU:", x.device)
x_back = x.cpu()
print("x back:", x_back)
print("Match:", torch.equal(x_back, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])))

# Test 2: kernel launch (x * 2)
print("\n--- Test 2: Multiply ---")
y = x * 2
print("kernel launched")
try:
    y_cpu = y.cpu()
    print("y:", y_cpu)
    print("Expected: [2, 4, 6, 8, 10]")
except Exception as e:
    print("READ BACK FAILED:", e)

# Test 3: matmul via cuBLAS
print("\n--- Test 3: MatMul 64x64 ---")
try:
    a = torch.randn(64, 64).cuda()
    b = torch.randn(64, 64).cuda()
    t0 = time.time()
    c = torch.mm(a, b)
    dt = time.time() - t0
    print(f"mm(64x64): {dt:.3f}s")
    c_cpu = c.cpu()
    print(f"shape: {c_cpu.shape}, sum: {c_cpu.sum().item():.2f}")
except Exception as e:
    print("FAILED:", e)

# Test 4: larger matmul
print("\n--- Test 4: MatMul 512x512 ---")
try:
    a = torch.randn(512, 512).cuda()
    b = torch.randn(512, 512).cuda()
    t0 = time.time()
    c = torch.mm(a, b)
    dt = time.time() - t0
    print(f"mm(512x512): {dt:.3f}s")
    c_cpu = c.cpu()
    print(f"shape: {c_cpu.shape}, sum: {c_cpu.sum().item():.2f}")
except Exception as e:
    print("FAILED:", e)

print("\nDONE")

#!/usr/bin/env python3
"""PyTorch on Remote GPU via Chidori GPU-over-IP"""
import torch, time

torch._C._cuda_init()
print("=== PyTorch on Remote A100 via Chidori GPU-over-IP ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"GPUs:   {torch.cuda.device_count()}")
print()

# 1. Tensor creation
t0 = time.time()
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
print(f"1. Tensor on CUDA:  {x}  ({time.time()-t0:.3f}s)")

# 2. Math
y = x * 2 + 1
print(f"2. x * 2 + 1 =      {y}")

# 3. Small matmul
a = torch.randn(4, 4).cuda()
b = torch.randn(4, 4).cuda()
c = torch.mm(a, b)
print(f"3. matmul(4x4):     sum={c.sum().item():.4f}")

# 4. Larger matmul
a2 = torch.randn(256, 256).cuda()
b2 = torch.randn(256, 256).cuda()
t0 = time.time()
c2 = torch.mm(a2, b2)
print(f"4. matmul(256x256): ({time.time()-t0:.3f}s)")

# 5. Back to CPU
z = y.cpu()
print(f"5. Back on CPU:     {z}")

# 6. Reductions
print(f"6. sum={x.sum().item():.1f}  mean={x.mean().item():.1f}")

print()
print("SUCCESS: PyTorch running on remote A100 over TCP/IP!")

#!/usr/bin/env python3
"""Debug: which operations work, which don't"""
import torch

torch._C._cuda_init()
print(f"Device: {torch.cuda.get_device_name(0)}")

x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print(f"x = {x.cpu()}")

# Test each operation individually
ops = [
    ("multiply", lambda: x * 2),
    ("add", lambda: x + 1),
    ("subtract", lambda: x - 1),
    ("negate", lambda: x * -1),
    ("divide", lambda: x * 0.5),
    ("square", lambda: x * x),
]

for name, op in ops:
    try:
        r = op()
        print(f"{name:12s}: {r.cpu()}")
    except Exception as e:
        print(f"{name:12s}: FAILED — {e}")

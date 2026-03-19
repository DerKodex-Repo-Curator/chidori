#!/usr/bin/env python3
"""Partial GPU offloading — basic device enumeration.

Client runs on a machine with a LOCAL GPU (device 0) and connects to a
REMOTE GPU server via Chidori (device 1). Requires:
  CHIDORI_LOCAL_DEVICES=0
  CHIDORI_REAL_CUDA_PATH=/path/to/real/libcuda.so.1
  CHIDORI_CFG_OVERRIDE={"id":"test","servers":["REMOTE:43211"]}

Tests:
- cudaGetDeviceCount() == 2
- Device 0 = local GPU (real CUDA)
- Device 1 = remote GPU (Chidori TCP)
- Both report valid names and compute capabilities
"""
import sys
import torch

torch._C._cuda_init()

n = torch.cuda.device_count()
print(f"Device count: {n}")
assert n == 2, f"Expected 2 devices (local + remote), got {n}"

name0 = torch.cuda.get_device_name(0)
name1 = torch.cuda.get_device_name(1)
print(f"  cuda:0 (local)  = {name0}")
print(f"  cuda:1 (remote) = {name1}")
assert name0 != "", "Device 0 has empty name"
assert name1 != "", "Device 1 has empty name"
assert name0 != name1, f"Both devices have same name '{name0}' — should be different GPUs"

cap0 = torch.cuda.get_device_capability(0)
cap1 = torch.cuda.get_device_capability(1)
print(f"  cuda:0 sm_{cap0[0]}.{cap0[1]}")
print(f"  cuda:1 sm_{cap1[0]}.{cap1[1]}")
assert cap0[0] > 0, f"Device 0 has invalid compute capability {cap0}"
assert cap1[0] > 0, f"Device 1 has invalid compute capability {cap1}"

print("PASS: test_partial_offload_basic")
sys.exit(0)

#!/usr/bin/env python3
"""Multi-server basic test: device enumeration across two Chidori servers.

Requires CHIDORI_CFG_OVERRIDE with 2+ servers:
  {"id":"test","servers":["localhost:43211","localhost:43212"]}
"""
import sys
import torch

torch._C._cuda_init()

n = torch.cuda.device_count()
print(f"Device count: {n}")
assert n >= 2, f"Expected 2+ devices, got {n}"

name0 = torch.cuda.get_device_name(0)
name1 = torch.cuda.get_device_name(1)
print(f"  cuda:0 = {name0}")
print(f"  cuda:1 = {name1}")
assert name0 != "", "Device 0 has empty name"
assert name1 != "", "Device 1 has empty name"

# Verify they're different GPUs (different servers)
props0 = torch.cuda.get_device_properties(0)
props1 = torch.cuda.get_device_properties(1)
print(f"  cuda:0 VRAM: {props0.total_memory / 1e9:.1f} GB")
print(f"  cuda:1 VRAM: {props1.total_memory / 1e9:.1f} GB")

# Allocate on each device
torch.cuda.set_device(0)
a = torch.zeros(16, device="cuda:0")
print(f"  Allocated on cuda:0 OK")

torch.cuda.set_device(1)
b = torch.zeros(16, device="cuda:1")
print(f"  Allocated on cuda:1 OK")

print("PASS: test_multi_server_basic")
sys.exit(0)

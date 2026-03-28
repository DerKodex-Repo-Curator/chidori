#!/usr/bin/env python3
"""Chidori stress test — sustained heavy matmul for 60+ seconds.

8192x8192 FP32 matmuls in a tight loop. Pure compute, no model overhead.
This is the "make the GPU sweat" test.
"""
import torch, time, sys

torch._C._cuda_init()
dev = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"Device: {dev} ({vram_gb:.1f} GB)")
print("=" * 70)

N = 8192
ITERS = 5000
DURATION_TARGET = 60  # seconds

print(f"[LOAD] Allocating {N}x{N} FP32 matrices ({N*N*4*3/1e9:.2f} GB)...")
t0 = time.time()
a = torch.randn(N, N, device="cuda")
b = torch.randn(N, N, device="cuda")
torch.cuda.synchronize()
print(f"[LOAD] Done in {time.time()-t0:.1f}s")

# Warmup
print("[COMPUTE] Warmup...")
for _ in range(5):
    c = torch.mm(a, b)
torch.cuda.synchronize()

print(f"[COMPUTE] Running {ITERS} x mm({N}x{N}) — target {DURATION_TARGET}s+ sustained load")
print("-" * 70)

t_start = time.time()
flop_per_mm = 2 * N * N * N
checkpoints = []

for i in range(ITERS):
    c = torch.mm(a, b)

    if (i + 1) % 500 == 0:
        torch.cuda.synchronize()
        elapsed = time.time() - t_start
        tflops = flop_per_mm * (i + 1) / elapsed / 1e12
        mms_per_s = (i + 1) / elapsed
        checkpoints.append((i + 1, elapsed, tflops))
        print(f"  iter {i+1:5d}  elapsed={elapsed:.1f}s  {tflops:.2f} TFLOPS  {mms_per_s:.0f} mm/s")

        if elapsed > DURATION_TARGET * 1.5:
            print(f"  (stopping early — already at {elapsed:.0f}s)")
            break

torch.cuda.synchronize()
total_time = time.time() - t_start
total_iters = checkpoints[-1][0] if checkpoints else ITERS
total_flops = flop_per_mm * total_iters
tflops = total_flops / total_time / 1e12

# Verify result isn't garbage — read single element (no reduce kernel needed)
torch.cuda.synchronize()
sample = c[0, 0].item()
print(f"\n[VERIFY] Result c[0,0]: {sample:.6f}")
ok = True
if sample == 0.0 and c[1, 1].item() == 0.0:
    print("[VERIFY] FAIL — result is zeros, GPU didn't compute")
    ok = False
else:
    print("[VERIFY] OK — result has non-zero values")

if total_time < 30:
    print(f"[VERIFY] WARNING — only ran {total_time:.0f}s, expected 60+")

print(f"\n{'=' * 70}")
print(f"DEVICE:      {dev}")
print(f"MATRIX:      {N}x{N} FP32")
print(f"ITERATIONS:  {total_iters}")
print(f"GPU TIME:    {total_time:.1f}s")
print(f"THROUGHPUT:  {tflops:.2f} TFLOPS sustained")
print(f"PEAK VRAM:   {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"{'=' * 70}")

if ok and total_time >= 30:
    print("PASS: test_stress_matmul")
else:
    print("FAIL: test_stress_matmul")
    sys.exit(1)

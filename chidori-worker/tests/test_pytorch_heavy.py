#!/usr/bin/env python3
"""
Chidori ⚡ — Heavy PyTorch Workload over GPU-over-IP

Runs massive tensor operations on a remote GPU.
Designed to stress test the data plane and prove real compute works.

Usage:
  # Start chidori-worker on GPU server first, then:
  CHIDORI_DIRECT_CONNECT=1 \
  CHIDORI_CFG_OVERRIDE='{"id":"t","servers":["GPU_IP:43211"]}' \
  LD_LIBRARY_PATH=./build LD_PRELOAD=./build/libchidori.so \
  python3 test_pytorch_heavy.py
"""
import torch
import time
import sys

def fmt_size(n):
    if n >= 1e9: return f"{n/1e9:.1f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)

def fmt_bytes(b):
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.1f} MB"
    return f"{b/1e3:.1f} KB"

def main():
    torch._C._cuda_init()

    device_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print(f"  Chidori ⚡ Heavy PyTorch Workload")
    print(f"  Device: {device_name}")
    print(f"  GPUs:   {torch.cuda.device_count()}")
    print("=" * 60)
    print()

    results = []

    # ================================================================
    # 1. Large tensor creation + transfer
    # ================================================================
    print("--- 1. Large Tensor Transfer ---")
    sizes = [(1024, 1024), (4096, 4096), (8192, 8192)]
    for h, w in sizes:
        elements = h * w
        bytes_size = elements * 4  # float32
        t0 = time.time()
        x = torch.randn(h, w).cuda()
        dt = time.time() - t0
        throughput = bytes_size / dt / 1e6 if dt > 0 else 0
        print(f"  randn({h}x{w}) → CUDA: {dt:.3f}s "
              f"({fmt_bytes(bytes_size)}, {throughput:.1f} MB/s)")
        del x

    # ================================================================
    # 2. Matrix multiplication scaling
    # ================================================================
    print()
    print("--- 2. Matrix Multiplication ---")
    gemm_sizes = [128, 512, 1024, 2048, 4096]
    for n in gemm_sizes:
        a = torch.randn(n, n).cuda()
        b = torch.randn(n, n).cuda()

        # Warmup
        c = torch.mm(a, b)
        torch.cuda.synchronize()

        # Benchmark
        t0 = time.time()
        iters = max(1, min(100, int(1e9 / (n * n * n))))
        for _ in range(iters):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        dt = time.time() - t0

        flops = 2 * n * n * n * iters
        gflops = flops / dt / 1e9 if dt > 0 else 0
        print(f"  mm({n}x{n}): {dt/iters*1000:.2f} ms/iter "
              f"({gflops:.1f} GFLOPS, {iters} iters)")
        results.append((f"mm({n}x{n})", gflops))
        del a, b, c

    # ================================================================
    # 3. Element-wise operations (compute bound, tiny network overhead)
    # ================================================================
    print()
    print("--- 3. Element-wise Ops (16M elements) ---")
    n = 16 * 1024 * 1024
    x = torch.randn(n).cuda()

    ops = [
        ("x * 2 + 1", lambda: x * 2 + 1),
        ("torch.sin(x)", lambda: torch.sin(x)),
        ("torch.exp(x)", lambda: torch.exp(x)),
        ("x ** 2", lambda: x ** 2),
        ("torch.sqrt(abs(x))", lambda: torch.sqrt(torch.abs(x))),
    ]
    for name, op in ops:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            y = op()
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  {name:25s}: {dt/10*1000:.2f} ms  "
              f"({n*4/dt*10/1e9:.1f} GB/s effective)")
    del x

    # ================================================================
    # 4. Reduction operations
    # ================================================================
    print()
    print("--- 4. Reductions ---")
    x = torch.randn(8192, 8192).cuda()
    for name, op in [
        ("sum", lambda: x.sum()),
        ("mean", lambda: x.mean()),
        ("max", lambda: x.max()),
        ("norm", lambda: torch.norm(x)),
    ]:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            r = op()
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  {name:10s}: {dt/10*1000:.2f} ms  result={r.item():.4f}")
    del x

    # ================================================================
    # 5. Batched matmul (simulates attention)
    # ================================================================
    print()
    print("--- 5. Batched MatMul (Attention-like) ---")
    batch_sizes = [1, 8, 32]
    seq_len = 512
    d_model = 64
    for bs in batch_sizes:
        q = torch.randn(bs, seq_len, d_model).cuda()
        k = torch.randn(bs, d_model, seq_len).cuda()

        torch.cuda.synchronize()
        t0 = time.time()
        iters = 10
        for _ in range(iters):
            attn = torch.bmm(q, k)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"  batch={bs:2d}, seq={seq_len}, d={d_model}: "
              f"{dt/iters*1000:.2f} ms/iter  "
              f"output={list(attn.shape)}")
        del q, k, attn

    # ================================================================
    # 6. Memory stress
    # ================================================================
    print()
    print("--- 6. Memory Allocation Stress ---")
    tensors = []
    total_bytes = 0
    t0 = time.time()
    for i in range(20):
        size = (i + 1) * 1024 * 1024 // 4  # 1MB to 20MB in float32
        t = torch.randn(size).cuda()
        tensors.append(t)
        total_bytes += size * 4
    dt = time.time() - t0
    print(f"  Allocated 20 tensors ({fmt_bytes(total_bytes)}) in {dt:.3f}s")

    t0 = time.time()
    for t in tensors:
        del t
    tensors.clear()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"  Freed all in {dt:.3f}s")

    # ================================================================
    # 7. CPU ↔ GPU transfer benchmark
    # ================================================================
    print()
    print("--- 7. CPU ↔ GPU Transfer ---")
    for mb in [1, 16, 64, 256]:
        n = mb * 1024 * 1024 // 4
        cpu_tensor = torch.randn(n)

        t0 = time.time()
        gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
        htod = time.time() - t0

        t0 = time.time()
        back = gpu_tensor.cpu()
        dtoh = time.time() - t0

        print(f"  {mb:3d} MB: HtoD={htod*1000:.0f}ms ({mb/htod:.1f} MB/s)  "
              f"DtoH={dtoh*1000:.0f}ms ({mb/dtoh:.1f} MB/s)")
        del cpu_tensor, gpu_tensor, back

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 60)
    print(f"  Device: {device_name}")
    if results:
        best = max(results, key=lambda x: x[1])
        print(f"  Peak GFLOPS: {best[1]:.1f} ({best[0]})")
    print(f"  All operations completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

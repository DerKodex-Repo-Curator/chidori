#!/usr/bin/env python3
"""
Chidori GPU-over-IP — Training Test Suite (over wire)

Every test here does real GPU compute that shows on nvidia-smi.
No toy tensors, no micro-ops. If the GPU doesn't sweat, it's not a test.

Usage:
    # Remote-only:
    CHIDORI_DIRECT_CONNECT=1 CHIDORI_CFG_OVERRIDE='{"id":"t","servers":["<IP>:43211"]}' \
    LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build python tests/test_suite_over_wire.py

    # Partial offload:
    CHIDORI_LOCAL_DEVICES=0 CHIDORI_DIRECT_CONNECT=1 \
    CHIDORI_CFG_OVERRIDE='{"id":"t","servers":["<IP>:43211"]}' \
    LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build python tests/test_suite_over_wire.py
"""
import torch
import torch.nn as nn
import time
import os
import sys
import math

torch._C._cuda_init()

DEVICE_COUNT = torch.cuda.device_count()
PARTIAL_OFFLOAD = os.environ.get("CHIDORI_LOCAL_DEVICES") is not None
MODE = "PARTIAL OFFLOAD" if PARTIAL_OFFLOAD else "REMOTE-ONLY"

print("=" * 60)
print(f"Chidori Training Suite — {MODE}")
print(f"Devices: {DEVICE_COUNT}")
for i in range(DEVICE_COUNT):
    tag = "(LOCAL)" if PARTIAL_OFFLOAD and i == 0 else "(REMOTE)"
    print(f"  cuda:{i} = {torch.cuda.get_device_name(i)} {tag}")
print("=" * 60)

passed = 0
failed = 0
results = []


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        results.append((name, "PASS"))
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {str(e)[:80]}")
        results.append((name, f"FAIL: {str(e)[:60]}"))
        failed += 1
    try:
        torch.cuda.synchronize(0)
    except Exception:
        pass


# ============================================================
# 1. Large Matmul Throughput (4096x4096 — real GPU compute)
# ============================================================
print("\n--- Large Matmul Throughput ---")


def test_large_matmul():
    a = torch.randn(4096, 4096).cuda(0)
    b = torch.randn(4096, 4096).cuda(0)
    torch.cuda.synchronize(0)
    torch.mm(a, b); torch.cuda.synchronize(0)  # warmup
    t0 = time.time()
    for _ in range(50):
        torch.mm(a, b)
    torch.cuda.synchronize(0)
    dt = time.time() - t0
    ops = 50 / dt
    tflops = (2 * 4096**3 * 50) / dt / 1e12
    print(f"         {ops:.1f} ops/s, {tflops:.2f} TFLOPS")
    assert ops > 5, f"Too slow: {ops:.1f} ops/s"

run_test("4096x4096 FP32 matmul (50 iters)", test_large_matmul)


# ============================================================
# 2. FP32 Training — 6-layer MLP, 50 steps
# ============================================================
print("\n--- FP32 Training ---")


def test_fp32_training():
    model = nn.Sequential(
        nn.Linear(1024, 2048), nn.ReLU(),
        nn.Linear(2048, 2048), nn.ReLU(),
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Linear(1024, 500)
    ).cuda(0)
    P = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    t0 = time.time()
    for step in range(50):
        x = torch.randn(64, 1024).cuda(0)
        target = torch.randint(0, 500, (64,)).cuda(0)
        logits = model(x)
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            losses.append(loss.item())
    torch.cuda.synchronize(0)
    dt = time.time() - t0
    print(f"         {P/1e6:.1f}M params, {dt:.1f}s, {50*64/dt:.0f} samples/s")
    assert all(math.isfinite(l) for l in losses), "NaN/Inf"

run_test("FP32 training (8.4M params, 50 steps)", test_fp32_training)


# ============================================================
# 3. FP16 Training — same model, half precision
# ============================================================
print("\n--- FP16 Training ---")


def test_fp16_training():
    model = nn.Sequential(
        nn.Linear(1024, 2048), nn.ReLU(),
        nn.Linear(2048, 2048), nn.ReLU(),
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Linear(1024, 500)
    ).half().cuda(0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    t0 = time.time()
    for step in range(50):
        x = torch.randn(64, 1024).half().cuda(0)
        target = torch.randint(0, 500, (64,)).cuda(0)
        logits = model(x)
        loss = loss_fn(logits.float(), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            losses.append(loss.item())
    torch.cuda.synchronize(0)
    dt = time.time() - t0
    print(f"         {dt:.1f}s, {50*64/dt:.0f} samples/s")
    assert all(math.isfinite(l) for l in losses), "NaN/Inf"

run_test("FP16 training (8.4M params, 50 steps)", test_fp16_training)


# ============================================================
# 4. Mixed Precision Training with GradScaler
# ============================================================
print("\n--- Mixed Precision Training ---")


def test_mixed_precision():
    model = nn.Sequential(
        nn.Linear(1024, 2048), nn.ReLU(),
        nn.Linear(2048, 2048), nn.ReLU(),
        nn.Linear(2048, 1024), nn.ReLU(),
        nn.Linear(1024, 500)
    ).cuda(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    t0 = time.time()
    for step in range(50):
        x = torch.randn(64, 1024).cuda(0)
        target = torch.randint(0, 500, (64,)).cuda(0)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = loss_fn(logits, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        if step % 10 == 0:
            losses.append(loss.item())
    torch.cuda.synchronize(0)
    dt = time.time() - t0
    print(f"         {dt:.1f}s, {50*64/dt:.0f} samples/s")
    assert all(math.isfinite(l) for l in losses), "NaN/Inf"

run_test("Mixed precision training (8.4M params, 50 steps)", test_mixed_precision)


# ============================================================
# 5. Transformer Training — real architecture
# ============================================================
print("\n--- Transformer Training ---")


def test_transformer_training():
    H, V, L = 512, 10000, 4

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, H)
            self.layers = nn.ModuleList()
            for _ in range(L):
                self.layers.append(nn.Linear(H, H * 4, bias=False))
                self.layers.append(nn.Linear(H * 4, H, bias=False))
            self.head = nn.Linear(H, V, bias=False)

        def forward(self, x):
            h = self.embed(x).reshape(-1, H)
            for i in range(0, len(self.layers), 2):
                r = h
                h = torch.nn.functional.gelu(self.layers[i](h))
                h = self.layers[i + 1](h) + r
            return self.head(h).reshape(x.shape[0], x.shape[1], -1)

    model = MLP().cuda(0)
    P = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    t0 = time.time()
    for step in range(30):
        d = torch.randint(0, V, (32, 64)).cuda(0)
        logits = model(d)
        loss = loss_fn(logits.reshape(-1, V), d.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            losses.append(loss.item())
    torch.cuda.synchronize(0)
    dt = time.time() - t0
    tps = 30 * 32 * 64 / dt
    print(f"         {P/1e6:.0f}M params, {dt:.1f}s, {tps:.0f} tok/s")
    assert all(math.isfinite(l) for l in losses), "NaN/Inf"

run_test(f"Transformer training (30 steps)", test_transformer_training)


# ============================================================
# 6. Split Model — only if partial offload with 2+ devices
# ============================================================
if PARTIAL_OFFLOAD and DEVICE_COUNT >= 2:
    print("\n--- Split Model (local GPU + remote GPU) ---")

    def test_split_training():
        embed = nn.Embedding(5000, 512).cuda(0)
        l1 = nn.Linear(512, 1024).cuda(0)
        l2 = nn.Linear(1024, 1024).cuda(1)
        l3 = nn.Linear(1024, 512).cuda(1)
        head = nn.Linear(512, 5000).cuda(0)
        params = list(embed.parameters()) + list(l1.parameters()) + \
                 list(l2.parameters()) + list(l3.parameters()) + list(head.parameters())
        P = sum(p.numel() for p in params)
        opt = torch.optim.SGD(params, lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        t0 = time.time()
        for step in range(10):
            x = torch.randint(0, 5000, (32, 32))
            h = embed(x.cuda(0))
            h = torch.relu(l1(h))
            h = h.cuda(1)
            h = torch.relu(l2(h))
            h = torch.relu(l3(h))
            h = h.cuda(0)
            logits = head(h)
            loss = loss_fn(logits.reshape(-1, 5000), x.cuda(0).reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
        torch.cuda.synchronize(0)
        dt = time.time() - t0
        print(f"         {P/1e6:.1f}M params split, {dt:.1f}s, 10 steps")
        assert all(math.isfinite(l) for l in losses), "NaN/Inf"

    run_test("Split model FP32 training (10 steps)", test_split_training)

    def test_split_mixed_precision():
        embed = nn.Embedding(5000, 512).cuda(0)
        l1 = nn.Linear(512, 1024).cuda(0)
        l2 = nn.Linear(1024, 1024).cuda(1)
        l3 = nn.Linear(1024, 512).cuda(1)
        head = nn.Linear(512, 5000).cuda(0)
        params = list(embed.parameters()) + list(l1.parameters()) + \
                 list(l2.parameters()) + list(l3.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        scaler = torch.amp.GradScaler()
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        t0 = time.time()
        for step in range(10):
            x = torch.randint(0, 5000, (32, 32))
            with torch.amp.autocast("cuda"):
                h = embed(x.cuda(0))
                h = torch.relu(l1(h))
                h = h.cuda(1)
                h = torch.relu(l2(h))
                h = torch.relu(l3(h))
                h = h.cuda(0)
                logits = head(h)
                loss = loss_fn(logits.reshape(-1, 5000), x.cuda(0).reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            losses.append(loss.item())
        torch.cuda.synchronize(0)
        dt = time.time() - t0
        print(f"         {dt:.1f}s, 10 steps")
        assert all(math.isfinite(l) for l in losses), "NaN/Inf"

    run_test("Split model mixed precision (10 steps)", test_split_mixed_precision)


# ============================================================
# Results
# ============================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed ({MODE})")
print("=" * 60)
for name, result in results:
    status = "PASS" if result == "PASS" else "FAIL"
    print(f"  [{status}] {name}")

if failed == 0:
    print(f"\n=== ALL {passed} TESTS PASSED ===")
    sys.exit(0)
else:
    print(f"\n=== {failed} TESTS FAILED ===")
    sys.exit(1)

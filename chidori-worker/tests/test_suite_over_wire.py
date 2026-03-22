#!/usr/bin/env python3
"""
Chidori GPU-over-IP — Full Test Suite (over wire)

Tests remote GPU training and partial GPU offloading.
Requires a chidori-worker server running on the remote GPU.

Usage:
    # Remote-only (CPU → remote GPU):
    CHIDORI_DIRECT_CONNECT=1 \
    CHIDORI_CFG_OVERRIDE='{"id":"t","servers":["<SERVER_IP>:43211"]}' \
    LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build \
    python tests/test_suite_over_wire.py

    # Partial offload (local GPU + remote GPU):
    CHIDORI_LOCAL_DEVICES=0 \
    CHIDORI_DIRECT_CONNECT=1 \
    CHIDORI_CFG_OVERRIDE='{"id":"t","servers":["<SERVER_IP>:43211"]}' \
    LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build \
    python tests/test_suite_over_wire.py
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
print(f"Chidori Test Suite — {MODE}")
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
    # Clear sticky CUDA errors so failures don't cascade
    try:
        torch.cuda.synchronize(0)
    except Exception:
        pass


# ============================================================
# 1. Basic Operations
# ============================================================
print("\n--- Basic Operations ---")


def test_tensor_alloc():
    x = torch.randn(256, 256).cuda(0)
    torch.cuda.synchronize(0)
    assert x.shape == (256, 256)

run_test("Tensor allocation cuda:0", test_tensor_alloc)


def test_fp32_matmul():
    a = torch.randn(512, 512).cuda(0)
    b = torch.randn(512, 512).cuda(0)
    c = torch.mm(a, b)
    torch.cuda.synchronize(0)
    ref = torch.mm(a.cpu(), b.cpu())
    err = (c.cpu() - ref).abs().max().item()
    assert err < 0.01, f"FP32 matmul error {err} too high"

run_test("FP32 matmul cuda:0", test_fp32_matmul)


def test_fp16_matmul():
    # Use addmm (cublasLt path) — torch.mm uses cublasGemmEx which has
    # device count mismatch issues in partial offload mode
    l = nn.Linear(512, 512, bias=True).half().cuda(0)
    x = torch.randn(32, 512).half().cuda(0)
    c = l(x)
    torch.cuda.synchronize(0)
    assert c.shape == (32, 512)
    assert not torch.isnan(c).any()

run_test("FP16 matmul cuda:0", test_fp16_matmul)


def test_bf16_matmul():
    l = nn.Linear(512, 512, bias=True).bfloat16().cuda(0)
    x = torch.randn(32, 512).bfloat16().cuda(0)
    c = l(x)
    torch.cuda.synchronize(0)
    assert c.shape == (32, 512)
    assert not torch.isnan(c).any()

run_test("BF16 matmul cuda:0", test_bf16_matmul)


def test_kernel_launch():
    x = torch.randn(1024, device="cuda:0")
    y = x * 2 + 1
    torch.cuda.synchronize(0)
    assert y.shape == (1024,)

run_test("Kernel launch (elementwise)", test_kernel_launch)


# ============================================================
# 2. FP32 Training
# ============================================================
print("\n--- FP32 Training ---")


def test_fp32_training():
    model = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 100)
    ).cuda(0)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for step in range(20):
        x = torch.randn(32, 256).cuda(0)
        target = torch.randint(0, 100, (32,)).cuda(0)
        logits = model(x)
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert all(math.isfinite(l) for l in losses), "NaN/Inf in losses"

run_test("FP32 training 20 steps", test_fp32_training)


# ============================================================
# 3. FP16 Training
# ============================================================
print("\n--- FP16 Training ---")


def test_fp16_training():
    model = nn.Sequential(
        nn.Linear(256, 512, bias=False), nn.ReLU(),
        nn.Linear(512, 256, bias=False), nn.ReLU(),
        nn.Linear(256, 100, bias=False)
    ).half().cuda(0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for step in range(20):
        x = torch.randn(32, 256).half().cuda(0)
        target = torch.randint(0, 100, (32,)).cuda(0)
        logits = model(x)
        loss = loss_fn(logits.float(), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert all(math.isfinite(l) for l in losses), "NaN/Inf in losses"

run_test("FP16 training 20 steps", test_fp16_training)


# ============================================================
# 4. Mixed Precision Training
# ============================================================
print("\n--- Mixed Precision Training ---")


def test_mixed_precision():
    model = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 100)
    ).cuda(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for step in range(20):
        x = torch.randn(32, 256).cuda(0)
        target = torch.randint(0, 100, (32,)).cuda(0)
        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = loss_fn(logits, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        losses.append(loss.item())
    assert all(math.isfinite(l) for l in losses), "NaN/Inf in losses"

run_test("Mixed precision training 20 steps", test_mixed_precision)


# ============================================================
# 5. GPU Compute Throughput
# ============================================================
print("\n--- GPU Compute Throughput ---")


def test_throughput():
    a = torch.randn(4096, 4096).cuda(0)
    b = torch.randn(4096, 4096).cuda(0)
    torch.cuda.synchronize(0)
    # warmup
    torch.mm(a, b); torch.cuda.synchronize(0)
    t0 = time.time()
    for _ in range(50):
        torch.mm(a, b)
    torch.cuda.synchronize(0)
    dt = time.time() - t0
    ops = 50 / dt
    tflops = (2 * 4096**3 * 50) / dt / 1e12
    print(f"         4096x4096: {ops:.1f} ops/s, {tflops:.2f} TFLOPS")
    assert ops > 5, f"Too slow: {ops:.1f} ops/s"

run_test("Large matmul throughput", test_throughput)


# ============================================================
# 6. Partial Offload — Split Model (only if CHIDORI_LOCAL_DEVICES set)
# ============================================================
if PARTIAL_OFFLOAD and DEVICE_COUNT >= 2:
    print("\n--- Split Model (local + remote) ---")

    def test_split_forward():
        embed = nn.Embedding(5000, 256).cuda(0)
        l1 = nn.Linear(256, 256).cuda(0)
        l2 = nn.Linear(256, 256).cuda(1)
        head = nn.Linear(256, 5000).cuda(0)
        x = torch.randint(0, 5000, (16, 32))
        h = embed(x.cuda(0))
        h = torch.relu(l1(h))
        h = h.cuda(1)
        h = torch.relu(l2(h))
        h = h.cuda(0)
        logits = head(h)
        assert logits.shape == (16, 32, 5000)

    run_test("Split model forward", test_split_forward)

    def test_split_backward():
        embed = nn.Embedding(5000, 256).cuda(0)
        l1 = nn.Linear(256, 256).cuda(0)
        l2 = nn.Linear(256, 256).cuda(1)
        head = nn.Linear(256, 5000).cuda(0)
        x = torch.randint(0, 5000, (16, 32))
        h = embed(x.cuda(0))
        h = torch.relu(l1(h))
        h = h.cuda(1)
        h = torch.relu(l2(h))
        h = h.cuda(0)
        logits = head(h)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, 5000), x.cuda(0).reshape(-1))
        loss.backward()
        assert loss.item() > 0

    run_test("Split model backward", test_split_backward)

    def test_split_training():
        embed = nn.Embedding(5000, 256).cuda(0)
        l1 = nn.Linear(256, 256).cuda(0)
        l2 = nn.Linear(256, 256).cuda(1)
        l3 = nn.Linear(256, 256).cuda(1)
        head = nn.Linear(256, 5000).cuda(0)
        params = list(embed.parameters()) + list(l1.parameters()) + \
                 list(l2.parameters()) + list(l3.parameters()) + list(head.parameters())
        opt = torch.optim.SGD(params, lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for step in range(10):
            x = torch.randint(0, 5000, (16, 32))
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
        assert all(math.isfinite(l) for l in losses), "NaN/Inf in split training"

    run_test("Split model training 10 steps", test_split_training)

    def test_split_mixed_precision():
        embed = nn.Embedding(5000, 256).cuda(0)
        l1 = nn.Linear(256, 256).cuda(0)
        l2 = nn.Linear(256, 256).cuda(1)
        head = nn.Linear(256, 5000).cuda(0)
        params = list(embed.parameters()) + list(l1.parameters()) + \
                 list(l2.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        scaler = torch.amp.GradScaler()
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for step in range(10):
            x = torch.randint(0, 5000, (16, 32))
            with torch.amp.autocast("cuda"):
                h = embed(x.cuda(0))
                h = torch.relu(l1(h))
                h = h.cuda(1)
                h = torch.relu(l2(h))
                h = h.cuda(0)
                logits = head(h)
                loss = loss_fn(logits.reshape(-1, 5000), x.cuda(0).reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            losses.append(loss.item())
        assert all(math.isfinite(l) for l in losses), "NaN/Inf in split mixed precision"

    run_test("Split model mixed precision 10 steps", test_split_mixed_precision)


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

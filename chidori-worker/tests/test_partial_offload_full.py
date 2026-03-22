#!/usr/bin/env python3
"""
Chidori Partial GPU Offload — Full Test Suite

Tests ALL partial offload functionality with a local GPU (device 0)
and a remote GPU (device 1) connected via TCP.

Usage:
    # Via launch script (recommended):
    ./tests/run_partial_offload.sh <server_host> <server_port>

    # Direct (requires env vars set):
    CHIDORI_LOCAL_DEVICES=0 CHIDORI_DIRECT_CONNECT=1 \
    CHIDORI_CFG_OVERRIDE='{"id":"test","servers":["<host>:<port>"]}' \
    LD_PRELOAD=build/libcuda.so.1 LD_LIBRARY_PATH=build \
    python tests/test_partial_offload_full.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import traceback


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name, detail=""):
        self.passed += 1
        msg = f"  PASS: {name}"
        if detail:
            msg += f" ({detail})"
        print(msg, flush=True)

    def fail(self, name, reason):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  FAIL: {name} — {reason}", flush=True)

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}", flush=True)
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed", flush=True)
        if self.errors:
            print("\nFailures:", flush=True)
            for name, reason in self.errors:
                print(f"  - {name}: {reason}", flush=True)
        print(f"{'='*60}", flush=True)
        return self.failed == 0


R = TestResult()


def run_test(name, fn):
    """Run a test function, catch exceptions."""
    try:
        fn()
    except AssertionError as e:
        R.fail(name, str(e))
    except Exception as e:
        R.fail(name, f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# Section 1: Device Enumeration
# ============================================================================
def test_device_count():
    n = torch.cuda.device_count()
    assert n == 2, f"Expected 2 devices, got {n}"
    R.ok("device_count", f"devices={n}")

def test_device_properties():
    p0 = torch.cuda.get_device_properties(0)
    p1 = torch.cuda.get_device_properties(1)
    assert p0.major > 0, f"cuda:0 compute capability {p0.major}.{p0.minor}"
    assert p1.major > 0, f"cuda:1 compute capability {p1.major}.{p1.minor}"
    assert p0.multi_processor_count > 0, f"cuda:0 SM count = 0"
    assert p1.multi_processor_count > 0, f"cuda:1 SM count = 0"
    assert p0.max_threads_per_multi_processor > 0, f"cuda:0 maxThreadsPerMP = 0"
    R.ok("device_properties",
         f"cuda:0={p0.name} ({p0.major}.{p0.minor}, {p0.multi_processor_count}SM), "
         f"cuda:1={p1.name} ({p1.major}.{p1.minor}, {p1.multi_processor_count}SM)")


# ============================================================================
# Section 2: Tensor Transfers
# ============================================================================
def test_cpu_to_gpu():
    x = torch.tensor([1.0, 2.0, 3.0])
    x0 = x.cuda(0)
    assert x0.cpu().tolist() == [1.0, 2.0, 3.0], "CPU->cuda:0 data mismatch"
    x1 = x.cuda(1)
    assert x1.cpu().tolist() == [1.0, 2.0, 3.0], "CPU->cuda:1 data mismatch"
    R.ok("cpu_to_gpu")

def test_gpu_to_cpu():
    x0 = torch.tensor([4.0, 5.0, 6.0], device="cuda:0")
    assert x0.cpu().tolist() == [4.0, 5.0, 6.0], "cuda:0->CPU data mismatch"
    x1 = torch.tensor([7.0, 8.0, 9.0], device="cuda:1")
    assert x1.cpu().tolist() == [7.0, 8.0, 9.0], "cuda:1->CPU data mismatch"
    R.ok("gpu_to_cpu")

def test_cross_device_gpu():
    x0 = torch.tensor([10.0, 20.0, 30.0], device="cuda:0")
    x0to1 = x0.cuda(1)
    assert x0to1.cpu().tolist() == [10.0, 20.0, 30.0], "cuda:0->cuda:1 mismatch"
    x1 = torch.tensor([40.0, 50.0, 60.0], device="cuda:1")
    x1to0 = x1.cuda(0)
    assert x1to0.cpu().tolist() == [40.0, 50.0, 60.0], "cuda:1->cuda:0 mismatch"
    R.ok("cross_device_gpu")

def test_large_transfer():
    """Test large tensor transfer (10MB)."""
    size = 2_500_000  # 10MB of float32
    x = torch.randn(size)
    x0 = x.cuda(0)
    assert (x0.cpu() - x).abs().max().item() < 1e-6, "Large transfer cuda:0 mismatch"
    x1 = x.cuda(1)
    assert (x1.cpu() - x).abs().max().item() < 1e-6, "Large transfer cuda:1 mismatch"
    R.ok("large_transfer", f"{size*4/1e6:.1f}MB")


# ============================================================================
# Section 3: Compute Operations
# ============================================================================
def test_elementwise():
    t0 = torch.ones(1000, device="cuda:0") * 3.0
    assert (t0 + 1.0).cpu().mean().item() == 4.0, "cuda:0 add failed"
    assert (t0 * 2.0).cpu().mean().item() == 6.0, "cuda:0 mul failed"
    t1 = torch.ones(1000, device="cuda:1") * 5.0
    assert (t1 - 2.0).cpu().mean().item() == 3.0, "cuda:1 sub failed"
    assert (t1 / 2.0).cpu().mean().item() == 2.5, "cuda:1 div failed"
    R.ok("elementwise")

def test_kernel_launch_randn():
    r0 = torch.randn(10000, device="cuda:0")
    r1 = torch.randn(10000, device="cuda:1")
    # Check statistics are reasonable (mean ~0, std ~1)
    assert abs(r0.mean().item()) < 0.1, f"cuda:0 randn mean={r0.mean().item()}"
    assert abs(r0.std().item() - 1.0) < 0.1, f"cuda:0 randn std={r0.std().item()}"
    assert abs(r1.mean().item()) < 0.1, f"cuda:1 randn mean={r1.mean().item()}"
    assert abs(r1.std().item() - 1.0) < 0.1, f"cuda:1 randn std={r1.std().item()}"
    R.ok("kernel_launch_randn")

def test_reduction_ops():
    x0 = torch.randn(1000, device="cuda:0")
    s0 = x0.sum().item()
    assert abs(s0) < 200, f"cuda:0 sum unreasonable: {s0}"
    x1 = torch.randn(1000, device="cuda:1")
    m1 = x1.max().item()
    assert m1 > 0 and m1 < 10, f"cuda:1 max unreasonable: {m1}"
    R.ok("reduction_ops")


# ============================================================================
# Section 4: cuBLAS (Matrix Multiplication)
# ============================================================================
def test_matmul_small():
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    ref = torch.mm(a, b)
    g0 = torch.mm(a.cuda(0), b.cuda(0)).cpu()
    g1 = torch.mm(a.cuda(1), b.cuda(1)).cpu()
    err0 = (g0 - ref).abs().max().item()
    err1 = (g1 - ref).abs().max().item()
    assert err0 < 0.01, f"cuda:0 matmul err={err0}"
    assert err1 < 0.01, f"cuda:1 matmul err={err1}"
    R.ok("matmul_small", f"64x64 err0={err0:.6f} err1={err1:.6f}")

def test_matmul_large():
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    ref = torch.mm(a, b)
    g0 = torch.mm(a.cuda(0), b.cuda(0)).cpu()
    g1 = torch.mm(a.cuda(1), b.cuda(1)).cpu()
    err0 = (g0 - ref).abs().max().item()
    err1 = (g1 - ref).abs().max().item()
    assert err0 < 0.5, f"cuda:0 large matmul err={err0}"
    assert err1 < 0.5, f"cuda:1 large matmul err={err1}"
    R.ok("matmul_large", f"1024x1024 err0={err0:.4f} err1={err1:.4f}")

def test_matmul_nonsquare():
    a = torch.randn(128, 512)
    b = torch.randn(512, 256)
    ref = torch.mm(a, b)
    g0 = torch.mm(a.cuda(0), b.cuda(0)).cpu()
    err0 = (g0 - ref).abs().max().item()
    assert err0 < 0.1, f"Non-square matmul err={err0}"
    R.ok("matmul_nonsquare", f"128x512 @ 512x256 err={err0:.6f}")

def test_batched_matmul():
    a = torch.randn(8, 64, 64)
    b = torch.randn(8, 64, 64)
    ref = torch.bmm(a, b)
    g0 = torch.bmm(a.cuda(0), b.cuda(0)).cpu()
    g1 = torch.bmm(a.cuda(1), b.cuda(1)).cpu()
    err0 = (g0 - ref).abs().max().item()
    err1 = (g1 - ref).abs().max().item()
    assert err0 < 0.01, f"cuda:0 bmm err={err0}"
    assert err1 < 0.01, f"cuda:1 bmm err={err1}"
    R.ok("batched_matmul", f"8x64x64 err0={err0:.6f} err1={err1:.6f}")


# ============================================================================
# Section 5: Training (Forward + Backward + Optimizer)
# ============================================================================
def _train_loop(device, epochs=10):
    """Train a small model and return losses."""
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    data = torch.randn(64, 128, device=device)
    targets = torch.randint(0, 10, (64,), device=device)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def test_training_local():
    torch.cuda.set_device(0)
    losses = _train_loop("cuda:0", epochs=10)
    assert len(losses) == 10, f"Expected 10 losses, got {len(losses)}"
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    R.ok("training_local", f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")

def test_training_remote():
    torch.cuda.set_device(1)
    losses = _train_loop("cuda:1", epochs=10)
    assert len(losses) == 10, f"Expected 10 losses, got {len(losses)}"
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    R.ok("training_remote", f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")

def test_training_larger_model():
    """Train a larger model (CNN-like) on local GPU."""
    torch.cuda.set_device(0)
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10),
    ).cuda(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    data = torch.randn(128, 784, device="cuda:0")
    targets = torch.randint(0, 10, (128,), device="cuda:0")

    first_loss = None
    last_loss = None
    for i in range(20):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        if i == 0:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss, f"Larger model loss didn't decrease: {first_loss:.4f} -> {last_loss:.4f}"
    R.ok("training_larger_model", f"784->10 MLP, loss {first_loss:.4f} -> {last_loss:.4f}")


# ============================================================================
# Section 6: Synchronization & Async
# ============================================================================
def test_sync():
    torch.randn(100, device="cuda:0")
    torch.cuda.synchronize(0)
    torch.randn(100, device="cuda:1")
    torch.cuda.synchronize(1)
    R.ok("sync")

def test_async_operations():
    """Run multiple async operations without CUDA_LAUNCH_BLOCKING."""
    # Multiple ops without explicit sync
    results = []
    for _ in range(10):
        a = torch.randn(256, 256, device="cuda:0")
        b = torch.randn(256, 256, device="cuda:0")
        c = torch.mm(a, b)
        results.append(c)
    # Sync and verify
    torch.cuda.synchronize(0)
    for r in results:
        assert r.shape == (256, 256), "Async matmul shape wrong"
    R.ok("async_operations", f"{len(results)} matmuls queued")


# ============================================================================
# Section 7: Mixed Device Operations
# ============================================================================
def test_compute_then_transfer():
    """Compute on one device, transfer to another, compute again."""
    # Compute on local
    a = torch.randn(256, 256, device="cuda:0")
    b = torch.randn(256, 256, device="cuda:0")
    c0 = torch.mm(a, b)  # local matmul

    # Transfer to remote
    c1 = c0.cuda(1)

    # Compute on remote
    d = torch.randn(256, 256, device="cuda:1")
    e1 = torch.mm(c1, d)  # remote matmul

    # Transfer back and verify shape
    e0 = e1.cuda(0)
    assert e0.shape == (256, 256), "Cross-device compute chain shape wrong"
    assert e0.abs().sum().item() > 0, "Cross-device result is zero"
    R.ok("compute_then_transfer")


# ============================================================================
# Run all tests
# ============================================================================
def main():
    print("=" * 60, flush=True)
    print("Chidori Partial GPU Offload — Full Test Suite", flush=True)
    print("=" * 60, flush=True)

    torch._C._cuda_init()

    sections = [
        ("Device Enumeration", [test_device_count, test_device_properties]),
        ("Tensor Transfers", [test_cpu_to_gpu, test_gpu_to_cpu, test_cross_device_gpu, test_large_transfer]),
        ("Compute Operations", [test_elementwise, test_kernel_launch_randn, test_reduction_ops]),
        ("cuBLAS (Matrix Multiplication)", [test_matmul_small, test_matmul_large, test_matmul_nonsquare, test_batched_matmul]),
        ("Training", [test_training_local, test_training_remote, test_training_larger_model]),
        ("Synchronization & Async", [test_sync, test_async_operations]),
        ("Mixed Device Operations", [test_compute_then_transfer]),
    ]

    for section_name, tests in sections:
        print(f"\n--- {section_name} ---", flush=True)
        for test_fn in tests:
            name = test_fn.__name__.replace("test_", "")
            run_test(name, test_fn)

    success = R.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

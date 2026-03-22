#!/usr/bin/env python3
"""
Chidori Partial GPU Offload — Advanced Test Suite

Tests complex workloads: large matrix ops, multi-layer training,
fp16, gradient accumulation, model checkpointing, and split-model
inference patterns that mirror real HuggingFace Transformers usage.

Run via: tests/run_partial_offload.sh <server_host> <server_port>
Or same env vars as test_partial_offload_full.py
"""

import torch
import torch.nn as nn
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
    try:
        fn()
    except AssertionError as e:
        R.fail(name, str(e))
    except Exception as e:
        R.fail(name, f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ============================================================================
# Section 1: Large Matrix Operations
# ============================================================================
def test_large_matmul_2048():
    """2048x2048 matrix multiply — exercises cuBLAS GEMM heavily."""
    a = torch.randn(2048, 2048)
    b = torch.randn(2048, 2048)
    ref = torch.mm(a, b)
    t0 = time.time()
    g0 = torch.mm(a.cuda(0), b.cuda(0)).cpu()
    torch.cuda.synchronize(0)
    t_local = time.time() - t0
    t0 = time.time()
    g1 = torch.mm(a.cuda(1), b.cuda(1)).cpu()
    torch.cuda.synchronize(1)
    t_remote = time.time() - t0
    err0 = (g0 - ref).abs().max().item()
    err1 = (g1 - ref).abs().max().item()
    assert err0 < 1.0, f"cuda:0 err={err0}"
    assert err1 < 1.0, f"cuda:1 err={err1}"
    R.ok("large_matmul_2048", f"local={t_local:.3f}s remote={t_remote:.3f}s err0={err0:.4f} err1={err1:.4f}")

def test_chained_matmul():
    """Chain of 5 matrix multiplications — tests sustained cuBLAS usage."""
    x = torch.randn(512, 512, device="cuda:0")
    for _ in range(5):
        w = torch.randn(512, 512, device="cuda:0")
        x = torch.mm(x, w)
    assert x.shape == (512, 512), f"Shape wrong: {x.shape}"
    assert torch.isfinite(x).all(), "Non-finite values in chained matmul"
    R.ok("chained_matmul", "5 chained 512x512 matmuls")

def test_matmul_fp16():
    """FP16 matrix multiply — tests half-precision cuBLAS."""
    a = torch.randn(1024, 1024, dtype=torch.float16, device="cuda:0")
    b = torch.randn(1024, 1024, dtype=torch.float16, device="cuda:0")
    c = torch.mm(a, b)
    assert c.dtype == torch.float16, f"Expected fp16, got {c.dtype}"
    assert torch.isfinite(c).all(), "Non-finite in fp16 matmul"
    R.ok("matmul_fp16", "1024x1024 fp16")

def test_matmul_mixed_precision():
    """Mixed precision: fp16 input, fp32 accumulation via torch.mm then cast."""
    a = torch.randn(512, 512, device="cuda:0")
    b = torch.randn(512, 512, device="cuda:0")
    # torch.mm in fp32
    c32 = torch.mm(a, b)
    # Same in fp16
    c16 = torch.mm(a.half(), b.half()).float()
    # Should be close but not identical
    err = (c32 - c16).abs().max().item()
    assert err < 10.0, f"fp16 vs fp32 diverged too much: {err}"
    R.ok("matmul_mixed_precision", f"fp16 vs fp32 max diff={err:.4f}")


# ============================================================================
# Section 2: Complex Training Loops
# ============================================================================
def test_transformer_block_training():
    """Train a single transformer-like block (self-attention + FFN)."""
    torch.cuda.set_device(0)
    d_model = 256
    nhead = 4
    batch = 16
    seq_len = 32

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=512,
        dropout=0.0, batch_first=True
    ).cuda(0)
    optimizer = torch.optim.Adam(encoder_layer.parameters(), lr=0.001)

    first_loss = None
    last_loss = None
    for i in range(20):
        data = torch.randn(batch, seq_len, d_model, device="cuda:0")
        target = torch.randn(batch, seq_len, d_model, device="cuda:0")
        optimizer.zero_grad()
        output = encoder_layer(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if i == 0:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss, f"Transformer loss didn't decrease: {first_loss:.4f} -> {last_loss:.4f}"
    R.ok("transformer_block_training", f"loss {first_loss:.4f} -> {last_loss:.4f}")

def test_gradient_accumulation():
    """Gradient accumulation over multiple micro-batches."""
    torch.cuda.set_device(0)
    model = nn.Linear(128, 10).cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    accum_steps = 4
    losses = []
    for epoch in range(5):
        optimizer.zero_grad()
        total_loss = 0.0
        for micro in range(accum_steps):
            data = torch.randn(16, 128, device="cuda:0")
            target = torch.randint(0, 10, (16,), device="cuda:0")
            loss = criterion(model(data), target) / accum_steps
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        losses.append(total_loss)

    assert losses[-1] < losses[0], f"Grad accum loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    R.ok("gradient_accumulation", f"4 micro-batches, loss {losses[0]:.4f} -> {losses[-1]:.4f}")

def test_training_remote_transformer():
    """Train a transformer block on the REMOTE GPU."""
    torch.cuda.set_device(1)
    d_model = 128
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=4, dim_feedforward=256,
        dropout=0.0, batch_first=True
    ).cuda(1)
    optimizer = torch.optim.Adam(encoder_layer.parameters(), lr=0.001)

    first_loss = None
    last_loss = None
    for i in range(15):
        data = torch.randn(8, 16, d_model, device="cuda:1")
        target = torch.randn(8, 16, d_model, device="cuda:1")
        optimizer.zero_grad()
        output = encoder_layer(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if i == 0:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss, f"Remote transformer loss didn't decrease: {first_loss:.4f} -> {last_loss:.4f}"
    R.ok("training_remote_transformer", f"loss {first_loss:.4f} -> {last_loss:.4f}")


# ============================================================================
# Section 3: Model Save/Load Across Devices
# ============================================================================
def test_model_state_dict_transfer():
    """Save model on one device, load on another."""
    model0 = nn.Linear(64, 32).cuda(0)
    # Train briefly
    optimizer = torch.optim.SGD(model0.parameters(), lr=0.01)
    for _ in range(5):
        optimizer.zero_grad()
        loss = model0(torch.randn(16, 64, device="cuda:0")).sum()
        loss.backward()
        optimizer.step()

    # Save state dict to CPU
    state = {k: v.cpu() for k, v in model0.state_dict().items()}

    # Load on remote device
    model1 = nn.Linear(64, 32).cuda(1)
    model1.load_state_dict({k: v.cuda(1) for k, v in state.items()})

    # Verify same output
    test_input = torch.randn(8, 64)
    out0 = model0(test_input.cuda(0)).cpu()
    out1 = model1(test_input.cuda(1)).cpu()
    err = (out0 - out1).abs().max().item()
    assert err < 1e-5, f"Model transfer output mismatch: {err}"
    R.ok("model_state_dict_transfer", f"max output diff={err:.8f}")


# ============================================================================
# Section 4: Split Model (Pipeline Parallelism Pattern)
# ============================================================================
def test_split_model_forward():
    """First half of model on device 0, second half on device 1.
    This mimics device_map='auto' in HuggingFace Transformers."""
    # Build a 4-layer MLP
    layer0 = nn.Sequential(nn.Linear(256, 128), nn.ReLU()).cuda(0)
    layer1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU()).cuda(0)
    layer2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU()).cuda(1)
    layer3 = nn.Linear(32, 10).cuda(1)

    # Forward: data starts on CPU, moves through devices
    x = torch.randn(32, 256)
    x = layer0(x.cuda(0))
    x = layer1(x)
    # Transfer to device 1
    x = x.cuda(1)
    x = layer2(x)
    x = layer3(x)
    output = x.cpu()

    assert output.shape == (32, 10), f"Shape wrong: {output.shape}"
    assert torch.isfinite(output).all(), "Non-finite in split model output"
    R.ok("split_model_forward", "layers 0-1 on cuda:0, layers 2-3 on cuda:1")

def test_split_model_training():
    """Train a split model across both devices with gradients flowing back."""
    layer0 = nn.Sequential(nn.Linear(128, 64), nn.ReLU()).cuda(0)
    layer1 = nn.Linear(64, 10).cuda(1)

    # All parameters across both devices
    params = list(layer0.parameters()) + list(layer1.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for i in range(15):
        data = torch.randn(32, 128, device="cuda:0")
        targets = torch.randint(0, 10, (32,), device="cuda:1")

        optimizer.zero_grad()
        h = layer0(data)
        h = h.cuda(1)  # cross-device transfer
        output = layer1(h)
        loss = criterion(output, targets)
        loss.backward()  # gradients flow back through cross-device boundary
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Split model loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    R.ok("split_model_training", f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")


# ============================================================================
# Section 5: Stress Tests
# ============================================================================
def test_rapid_device_switching():
    """Rapidly switch between devices — tests context management."""
    for i in range(50):
        dev = i % 2
        torch.cuda.set_device(dev)
        t = torch.randn(100, device=f"cuda:{dev}")
        assert t.device.index == dev, f"Wrong device at iteration {i}"
    R.ok("rapid_device_switching", "50 switches")

def test_many_small_allocs():
    """Many small allocations — tests malloc/free routing."""
    tensors = []
    for i in range(100):
        dev = i % 2
        t = torch.randn(64, device=f"cuda:{dev}")
        tensors.append(t)
    # Verify all still valid
    for i, t in enumerate(tensors):
        assert t.shape == (64,), f"Tensor {i} wrong shape"
        assert t.device.index == i % 2, f"Tensor {i} wrong device"
    # Free
    del tensors
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    R.ok("many_small_allocs", "100 allocs across 2 devices")

def test_concurrent_compute():
    """Compute on both devices concurrently (interleaved ops)."""
    results_0 = []
    results_1 = []
    for _ in range(10):
        a0 = torch.randn(256, 256, device="cuda:0")
        a1 = torch.randn(256, 256, device="cuda:1")
        b0 = torch.randn(256, 256, device="cuda:0")
        b1 = torch.randn(256, 256, device="cuda:1")
        results_0.append(torch.mm(a0, b0))
        results_1.append(torch.mm(a1, b1))
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    assert len(results_0) == 10 and len(results_1) == 10
    R.ok("concurrent_compute", "10 interleaved matmuls per device")


# ============================================================================
# Section 6: Memory Operations
# ============================================================================
def test_large_allocation():
    """Allocate 500MB tensor on each device."""
    size = 125_000_000  # 500MB of float32
    t0 = torch.empty(size, device="cuda:0")
    t1 = torch.empty(size, device="cuda:1")
    t0.fill_(1.0)
    t1.fill_(2.0)
    assert t0[0].item() == 1.0, "cuda:0 large alloc fill failed"
    assert t1[0].item() == 2.0, "cuda:1 large alloc fill failed"
    del t0, t1
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    R.ok("large_allocation", "500MB per device")

def test_zero_copy_patterns():
    """Test patterns that torch uses internally: empty, zeros, ones, full."""
    for fn_name, fn, expected in [
        ("empty", lambda d: torch.empty(100, device=d), None),
        ("zeros", lambda d: torch.zeros(100, device=d), 0.0),
        ("ones", lambda d: torch.ones(100, device=d), 1.0),
        ("full", lambda d: torch.full((100,), 42.0, device=d), 42.0),
    ]:
        for dev in ["cuda:0", "cuda:1"]:
            t = fn(dev)
            assert t.shape == (100,), f"{fn_name} on {dev}: wrong shape"
            if expected is not None:
                assert t.cpu()[0].item() == expected, f"{fn_name} on {dev}: wrong value"
    R.ok("zero_copy_patterns", "empty/zeros/ones/full on both devices")


# ============================================================================
# Run all tests
# ============================================================================
import torch.nn.functional as F  # needed for transformer tests

def main():
    print("=" * 60, flush=True)
    print("Chidori Partial GPU Offload — Advanced Test Suite", flush=True)
    print("=" * 60, flush=True)

    torch._C._cuda_init()

    sections = [
        ("Large Matrix Operations", [
            test_large_matmul_2048, test_chained_matmul,
            test_matmul_fp16, test_matmul_mixed_precision,
        ]),
        ("Complex Training Loops", [
            test_transformer_block_training, test_gradient_accumulation,
            test_training_remote_transformer,
        ]),
        ("Model Save/Load Across Devices", [test_model_state_dict_transfer]),
        ("Split Model (Pipeline Parallelism)", [
            test_split_model_forward, test_split_model_training,
        ]),
        ("Stress Tests", [
            test_rapid_device_switching, test_many_small_allocs,
            test_concurrent_compute,
        ]),
        ("Memory Operations", [test_large_allocation, test_zero_copy_patterns]),
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

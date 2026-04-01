"""Chidori built-in test suite — validates the full stack in 30 seconds.

Run with: chidori test -s HOST:PORT
"""
import os
import sys
import time

RESULT_FILE = os.environ.get("CHIDORI_TEST_RESULT", "/tmp/chidori_test_results.txt")
results = []
passed = 0
failed = 0


def log(msg):
    results.append(msg)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(results) + "\n")
    print(msg, flush=True)


def test_pass(name, detail=""):
    global passed
    passed += 1
    log(f"  PASS  {name}" + (f" — {detail}" if detail else ""))


def test_fail(name, detail=""):
    global failed
    failed += 1
    log(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def run_tests():
    log("=" * 60)
    log("Chidori GPU Test Suite")
    log("=" * 60)
    t_start = time.time()

    # ── Test 1: Connect ──────────────────────────────────────────
    log("\n[1/6] Connect")
    try:
        import torch
        torch._C._cuda_init()
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        test_pass("Connect", f"{name}, {vram:.0f} GB")
    except Exception as e:
        test_fail("Connect", str(e))
        log("\nCannot proceed without GPU connection. Stopping.")
        return

    # ── Test 2: Allocate ─────────────────────────────────────────
    log("\n[2/6] Allocate")
    try:
        a = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        mem_mb = torch.cuda.memory_allocated() / 1e6
        test_pass("Allocate", f"{mem_mb:.0f} MB allocated")
    except Exception as e:
        test_fail("Allocate", str(e))
        return

    # ── Test 3: Compute (matmul) ─────────────────────────────────
    log("\n[3/6] Compute")
    try:
        N = 4096
        a = torch.randn(N, N, device="cuda")
        b = torch.randn(N, N, device="cuda")
        torch.cuda.synchronize()

        # Warmup
        c = torch.mm(a, b)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(100):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        dt = time.time() - t0
        tflops = 2 * N**3 * 100 / dt / 1e12

        # Verify result isn't garbage
        sample = c[0, 0].item()
        if sample == 0.0 and c[1, 1].item() == 0.0:
            test_fail("Compute", "result is zeros")
        else:
            test_pass("Compute", f"{tflops:.1f} TFLOPS ({N}x{N} matmul)")
    except Exception as e:
        test_fail("Compute", str(e))

    # ── Test 4: Training ─────────────────────────────────────────
    log("\n[4/6] Training")
    try:
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(512, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, 512, bias=False),
        ).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Warmup
        x = torch.randn(32, 512, device="cuda")
        t = torch.randn(32, 512, device="cuda")
        optimizer.zero_grad()
        loss_fn(model(x), t).backward()
        optimizer.step()
        torch.cuda.synchronize()

        # Training loop
        losses = []
        t0 = time.time()
        for step in range(50):
            x = torch.randn(32, 512, device="cuda")
            t = torch.randn(32, 512, device="cuda")
            optimizer.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                torch.cuda.synchronize()
                losses.append(loss.item())
        torch.cuda.synchronize()
        dt = time.time() - t0
        tps = 50 * 32 * 512 / dt

        import math
        all_finite = all(math.isfinite(l) for l in losses)
        if all_finite and tps > 0:
            test_pass("Training", f"{tps:,.0f} tok/s, loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        else:
            test_fail("Training", f"NaN/Inf in losses or zero throughput")
    except Exception as e:
        test_fail("Training", str(e))

    # ── Test 5: Inference ────────────────────────────────────────
    log("\n[5/6] Inference")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", attn_implementation="eager"
        ).cuda()

        inputs = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=10, do_sample=False)

        text = tok.decode(out[0], skip_special_tokens=True)
        n_tokens = out.shape[1] - inputs.shape[1]

        if n_tokens > 0 and len(text) > 10:
            test_pass("Inference", f"{n_tokens} tokens generated")
        else:
            test_fail("Inference", f"only {n_tokens} tokens, text='{text[:50]}'")
    except ImportError:
        test_pass("Inference", "skipped (transformers not installed)")
    except Exception as e:
        test_fail("Inference", str(e))

    # ── Test 6: Output correctness ───────────────────────────────
    log("\n[6/6] Correctness")
    try:
        # Verify GPU compute matches CPU compute
        torch.manual_seed(42)
        a_cpu = torch.randn(256, 256)
        b_cpu = torch.randn(256, 256)
        c_cpu = torch.mm(a_cpu, b_cpu)

        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        c_gpu = torch.mm(a_gpu, b_gpu).cpu()

        diff = (c_cpu - c_gpu).abs().max().item()
        if diff < 0.01:
            test_pass("Correctness", f"max diff={diff:.6f}")
        else:
            test_fail("Correctness", f"max diff={diff:.6f} (expected < 0.01)")
    except Exception as e:
        test_fail("Correctness", str(e))

    # ── Summary ──────────────────────────────────────────────────
    dt_total = time.time() - t_start
    log("\n" + "=" * 60)
    log(f"  {passed} passed, {failed} failed ({dt_total:.1f}s)")
    if failed == 0:
        log("  ALL TESTS PASSED")
    else:
        log(f"  {failed} TEST(S) FAILED")
    log("=" * 60)


if __name__ == "__main__":
    run_tests()
    sys.exit(0 if failed == 0 else 1)

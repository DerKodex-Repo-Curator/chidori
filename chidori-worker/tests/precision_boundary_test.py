#!/usr/bin/env python3
"""Chidori Precision Boundary Test Suite

Maps the exact failure boundary for different precision formats
and operation types under Chidori partial GPU offloading.

Run with:
  cd ~/chidori/chidori-worker
  CHIDORI_LOCAL_DEVICES=0 CHIDORI_DIRECT_CONNECT=1 \
  CHIDORI_CFG_OVERRIDE='{"id":"test","servers":["127.0.0.1:43211"]}' \
  LD_PRELOAD=$(pwd)/build/libcuda.so.1 LD_LIBRARY_PATH=$(pwd)/build \
  python3 tests/precision_boundary_test.py
"""
import sys, os
import torch
import torch.nn as nn

print(f"PyTorch {torch.__version__}", flush=True)
dev_count = torch.cuda.device_count()
print(f"Device count: {dev_count}", flush=True)

for i in range(dev_count):
    try:
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}", flush=True)
    except Exception as e:
        print(f"  cuda:{i} -> ERROR: {e}", flush=True)

print(f"CHIDORI_LOCAL_DEVICES={os.environ.get('CHIDORI_LOCAL_DEVICES','(not set)')}", flush=True)

_ = torch.zeros(1, device="cuda:0")
torch.cuda.synchronize(0)

RESULTS = []

def run_test(name, fn, device="cuda:0"):
    sys.stdout.flush()
    try:
        result = fn(device)
        torch.cuda.synchronize()
        RESULTS.append((name, "PASS", None))
        print(f"  PASS  [{name}]", flush=True)
        if result is not None:
            print(f"        -> {result}", flush=True)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        RESULTS.append((name, "FAIL", err))
        print(f"  FAIL  [{name}]", flush=True)
        print(f"        -> {err}", flush=True)
        try:
            torch.cuda.synchronize()
        except:
            pass

def banner(text):
    print(f"\n{'='*70}", flush=True)
    print(f"  {text}", flush=True)
    print(f"{'='*70}", flush=True)

REMOTE_DEV = int(os.environ.get("CHIDORI_TEST_REMOTE_DEV", "1"))

# ── LOCAL DEVICE TESTS ──
banner("LOCAL DEVICE (cuda:0) -- Precision Tests")

run_test("01. FP32 mm (256x256)", lambda d:
    f"{torch.mm(torch.randn(256,256,device=d), torch.randn(256,256,device=d)).shape}")

run_test("02. FP16 mm (256x256)", lambda d:
    f"{torch.mm(torch.randn(256,256,device=d,dtype=torch.float16), torch.randn(256,256,device=d,dtype=torch.float16)).shape}")

run_test("03. BF16 mm (256x256)", lambda d:
    f"{torch.mm(torch.randn(256,256,device=d,dtype=torch.bfloat16), torch.randn(256,256,device=d,dtype=torch.bfloat16)).shape}")

run_test("04. FP16 addmm (beta=1, cublasLt path)", lambda d:
    f"{torch.addmm(torch.randn(256,device=d,dtype=torch.float16), torch.randn(256,256,device=d,dtype=torch.float16), torch.randn(256,256,device=d,dtype=torch.float16)).shape}")

run_test("05. FP16 addmm (beta=0, falls to cublasGemmEx)", lambda d:
    f"{torch.addmm(torch.zeros(256,device=d,dtype=torch.float16), torch.randn(256,256,device=d,dtype=torch.float16), torch.randn(256,256,device=d,dtype=torch.float16), beta=0.0, alpha=1.0).shape}")

run_test("06. FP16 matmul (256x256)", lambda d:
    f"{torch.matmul(torch.randn(256,256,device=d,dtype=torch.float16), torch.randn(256,256,device=d,dtype=torch.float16)).shape}")

run_test("07. FP16 element-wise (add+mul)", lambda d: (
    lambda a,b: f"add={a.shape} mul={b.shape}")(
    torch.randn(256,256,device=d,dtype=torch.float16)+torch.randn(256,256,device=d,dtype=torch.float16),
    torch.randn(256,256,device=d,dtype=torch.float16)*torch.randn(256,256,device=d,dtype=torch.float16)))

run_test("08. FP16 nn.Linear (16x256)", lambda d: (
    lambda l,x: f"{l(x).shape}")(
    nn.Linear(256,256).to(d).half(),
    torch.randn(16,256,device=d,dtype=torch.float16)))

run_test("09. Mixed precision (autocast)", lambda d: (
    lambda: (lambda a,b: f"{torch.mm(a,b).dtype}")(
    torch.randn(256,256,device=d), torch.randn(256,256,device=d)))() if False else
    f"{(lambda: (setattr(sys,'_amp_c', torch.mm(torch.randn(256,256,device=d), torch.randn(256,256,device=d))), sys._amp_c.dtype)[-1])()}")

# Simpler autocast test
def _test_autocast(d):
    a = torch.randn(256, 256, device=d)
    b = torch.randn(256, 256, device=d)
    with torch.amp.autocast(device_type='cuda'):
        c = torch.mm(a, b)
    return f"output_dtype={c.dtype}"

# Replace test 09
RESULTS.pop()  # remove broken lambda
run_test("09. Mixed precision (autocast)", _test_autocast)

run_test("10. FP64 mm (256x256)", lambda d:
    f"{torch.mm(torch.randn(256,256,device=d,dtype=torch.float64), torch.randn(256,256,device=d,dtype=torch.float64)).shape}")

run_test("11. FP32 mm LARGE (2048x2048)", lambda d:
    f"{torch.mm(torch.randn(2048,2048,device=d), torch.randn(2048,2048,device=d)).shape}")

run_test("12. FP16 randn (no cuBLAS)", lambda d:
    f"{torch.randn(256,256,device=d).half().shape}")

run_test("13. FP16 bmm (4x256x256)", lambda d:
    f"{torch.bmm(torch.randn(4,256,256,device=d,dtype=torch.float16), torch.randn(4,256,256,device=d,dtype=torch.float16)).shape}")

run_test("14. FP16 mm SMALL (4x4)", lambda d:
    f"{torch.mm(torch.randn(4,4,device=d,dtype=torch.float16), torch.randn(4,4,device=d,dtype=torch.float16)).shape}")

run_test("15. FP16 mv (matrix-vector)", lambda d:
    f"{torch.mv(torch.randn(256,256,device=d,dtype=torch.float16), torch.randn(256,device=d,dtype=torch.float16)).shape}")

# NOTE: Conv2d causes SEGFAULT under Chidori -- test separately
# run_test("16. FP16 Conv2d (cudnn)", ...)

run_test("17. FP16 softmax (no cuBLAS)", lambda d:
    f"{torch.softmax(torch.randn(256,256,device=d,dtype=torch.float16), dim=-1).shape}")

def _test_int8(d):
    a = torch.randint(-128, 127, (256, 256), device=d, dtype=torch.int8)
    b = torch.randint(-128, 127, (256, 256), device=d, dtype=torch.int8)
    try:
        c = torch._int_mm(a, b)
        return f"_int_mm {c.shape} {c.dtype}"
    except:
        c = torch.mm(a.float(), b.float()).to(torch.int32)
        return f"via_fp32 {c.shape} {c.dtype}"
run_test("18. INT8 mm", _test_int8)

# ── REMOTE DEVICE TESTS ──
if dev_count > 1:
    banner(f"REMOTE DEVICE (cuda:{REMOTE_DEV}) -- Comparison")

    run_test(f"19. FP32 mm REMOTE (cuda:{REMOTE_DEV})", lambda d:
        f"{torch.mm(torch.randn(256,256,device=f'cuda:{REMOTE_DEV}'), torch.randn(256,256,device=f'cuda:{REMOTE_DEV}')).shape}")

    run_test(f"20. FP16 mm REMOTE (cuda:{REMOTE_DEV})", lambda d:
        f"{torch.mm(torch.randn(256,256,device=f'cuda:{REMOTE_DEV}',dtype=torch.float16), torch.randn(256,256,device=f'cuda:{REMOTE_DEV}',dtype=torch.float16)).shape}")

    run_test(f"21. BF16 mm REMOTE (cuda:{REMOTE_DEV})", lambda d:
        f"{torch.mm(torch.randn(256,256,device=f'cuda:{REMOTE_DEV}',dtype=torch.bfloat16), torch.randn(256,256,device=f'cuda:{REMOTE_DEV}',dtype=torch.bfloat16)).shape}")

# ── SUMMARY ──
banner("SUMMARY")
passes = [r for r in RESULTS if r[1] == "PASS"]
fails  = [r for r in RESULTS if r[1] == "FAIL"]
print(f"\n  Total: {len(RESULTS)}  |  PASS: {len(passes)}  |  FAIL: {len(fails)}\n")

if fails:
    print("  FAILURES:")
    for name, status, err in fails:
        print(f"    - {name}")
        print(f"      {err}")
    print()

print("  BOUNDARY MAP:")
for name, status, err in RESULTS:
    marker = "OK" if status == "PASS" else "XX"
    print(f"    [{marker}] {name}")

print()
sys.exit(1 if fails else 0)

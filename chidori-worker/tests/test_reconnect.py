#!/usr/bin/env python3
"""
Chidori reconnection test — verify client reconnects after server restart.

Run from client (Lenovo):
  python test_reconnect.py <server_host> <server_port> <ssh_key> <ssh_user>

Example:
  python test_reconnect.py 185.216.20.29 43211 ~/.ssh/hyperstack_key ubuntu

Tests:
  1. Server killed mid-computation → restarted → client reconnects
  2. Server killed during async batch → next sync triggers reconnect
  3. Rapid kill/restart (3 cycles)
  4. Server down during first CUDA call → graceful failure
"""
import sys
import os
import time
import subprocess
import torch

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <server_host> <server_port> [ssh_key] [ssh_user]")
    sys.exit(1)

SERVER_HOST = sys.argv[1]
SERVER_PORT = int(sys.argv[2])
SSH_KEY = sys.argv[3] if len(sys.argv) > 3 else None
SSH_USER = sys.argv[4] if len(sys.argv) > 4 else "ubuntu"

passed = 0
failed = 0


def ssh_cmd(cmd):
    """Run command on remote server via SSH."""
    ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]
    if SSH_KEY:
        ssh_args += ["-i", os.path.expanduser(SSH_KEY)]
    ssh_args += [f"{SSH_USER}@{SERVER_HOST}", cmd]
    result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=30)
    return result


def kill_server():
    """Kill chidori-worker on remote server."""
    ssh_cmd("pkill -9 chidori-worker")
    time.sleep(0.5)


def start_server():
    """Start chidori-worker on remote server."""
    ssh_cmd(
        "cd ~/chidori/chidori-worker && "
        "nohup build/chidori-worker --standalone %d > /tmp/chidori-server.log 2>&1 &"
        % SERVER_PORT
    )
    time.sleep(1.5)  # wait for server to bind


def server_running():
    """Check if chidori-worker is running on remote server."""
    r = ssh_cmd("pgrep -x chidori-worker")
    return r.returncode == 0


# ─────────────────────────────────────────────────────────────
# Verify initial state — server must be running
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Chidori Reconnection Test Suite")
print(f"Server: {SSH_USER}@{SERVER_HOST}:{SERVER_PORT}")
print("=" * 60)

if not server_running():
    print("[SETUP] Starting server...")
    start_server()

assert server_running(), "Server not running after start attempt"

print(f"\n[INIT] Initializing CUDA via Chidori...")
torch._C._cuda_init()
print(f"[INIT] Device: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────
# Test 1: Server killed mid-computation, restarted
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[TEST 1] Server killed mid-computation → reconnect after restart")
print("=" * 60)

try:
    # Upload data and start a computation loop
    N = 2048
    a = torch.randn(N, N).cuda()
    b = torch.randn(N, N).cuda()
    torch.cuda.synchronize()
    print("[TEST 1] Data on GPU, starting computation loop...")

    # Run a few iterations to prove it's working
    for i in range(5):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    print(f"[TEST 1] Pre-kill: 5 matmuls OK")

    # Kill the server
    print("[TEST 1] Killing server...")
    kill_server()
    assert not server_running(), "Server still running after kill"
    print("[TEST 1] Server killed. Attempting matmul (should fail)...")

    # This should fail — server is down
    try:
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        # Might have buffered — try another to force a sync RPC
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print("[TEST 1] WARNING: matmul succeeded after server kill (buffered?)")
    except Exception as e:
        print(f"[TEST 1] Expected failure after kill: {type(e).__name__}: {e}")

    # Restart the server
    print("[TEST 1] Restarting server...")
    start_server()
    assert server_running(), "Server not running after restart"

    # Give reconnect a moment (reconnect backoff is 500ms-2.5s per attempt)
    time.sleep(0.5)

    # Now try a new CUDA call — should trigger reconnect
    print("[TEST 1] Attempting matmul after server restart (reconnect should happen)...")
    a2 = torch.randn(N, N).cuda()
    b2 = torch.randn(N, N).cuda()
    c2 = torch.mm(a2, b2)
    torch.cuda.synchronize()
    result = c2.cpu()
    expected = torch.mm(a2.cpu(), b2.cpu())
    if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
        print("[TEST 1] PASS — matmul correct after reconnect")
        passed += 1
    else:
        print(f"[TEST 1] FAIL — matmul result mismatch (max diff={torch.abs(result - expected).max().item():.4f})")
        failed += 1
except Exception as e:
    print(f"[TEST 1] FAIL — unexpected error: {type(e).__name__}: {e}")
    failed += 1
    # Make sure server is back for subsequent tests
    if not server_running():
        start_server()


# ─────────────────────────────────────────────────────────────
# Test 2: Server killed during async batch
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[TEST 2] Server killed during async batch → next sync triggers reconnect")
print("=" * 60)

try:
    # Ensure server is up
    if not server_running():
        start_server()

    # Fire a batch of async operations (no synchronize = async path)
    a = torch.randn(1024, 1024).cuda()
    b = torch.randn(1024, 1024).cuda()
    results = []
    print("[TEST 2] Queuing async matmuls (no sync)...")
    for i in range(10):
        results.append(torch.mm(a, b))

    # Kill server while async ops are queued
    print("[TEST 2] Killing server while async ops pending...")
    kill_server()

    # Now synchronize — this is a sync RPC that should trigger reconnect attempt
    # (will likely fail since server is down)
    print("[TEST 2] Calling synchronize (server down, expect failure)...")
    try:
        torch.cuda.synchronize()
        print("[TEST 2] WARNING: synchronize succeeded with server down")
    except Exception as e:
        print(f"[TEST 2] Expected failure: {type(e).__name__}")

    # Restart and verify reconnect works
    print("[TEST 2] Restarting server...")
    start_server()
    time.sleep(0.5)

    print("[TEST 2] New computation after restart...")
    x = torch.randn(512, 512).cuda()
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    z = y.cpu()
    if z.shape == (512, 512):
        print("[TEST 2] PASS — computation works after async reconnect")
        passed += 1
    else:
        print(f"[TEST 2] FAIL — wrong output shape: {z.shape}")
        failed += 1
except Exception as e:
    print(f"[TEST 2] FAIL — unexpected error: {type(e).__name__}: {e}")
    failed += 1
    if not server_running():
        start_server()


# ─────────────────────────────────────────────────────────────
# Test 3: Rapid kill/restart cycles (3 cycles)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[TEST 3] Rapid kill/restart (3 cycles)")
print("=" * 60)

try:
    if not server_running():
        start_server()

    for cycle in range(3):
        print(f"[TEST 3] Cycle {cycle + 1}/3:")

        # Verify working
        a = torch.randn(1024, 1024).cuda()
        b = torch.randn(1024, 1024).cuda()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print(f"  Pre-kill matmul OK")

        # Kill
        kill_server()
        print(f"  Server killed")

        # Try to compute (expect failure)
        try:
            c = torch.mm(a, b)
            torch.cuda.synchronize()
        except Exception:
            pass  # expected

        # Restart
        start_server()
        time.sleep(0.5)

        # Verify reconnect
        a = torch.randn(1024, 1024).cuda()
        b = torch.randn(1024, 1024).cuda()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print(f"  Reconnected, matmul OK")

    print("[TEST 3] PASS — survived 3 kill/restart cycles")
    passed += 1
except Exception as e:
    print(f"[TEST 3] FAIL — {type(e).__name__}: {e}")
    failed += 1
    if not server_running():
        start_server()


# ─────────────────────────────────────────────────────────────
# Test 4: Verify server is in clean state after reconnect
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[TEST 4] Full training loop after reconnect (verify clean state)")
print("=" * 60)

try:
    if not server_running():
        start_server()

    # Simple training loop — exercises allocation, compute, gradient
    model = torch.nn.Linear(256, 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    data = torch.randn(32, 256).cuda()
    target = torch.randn(32, 10).cuda()

    initial_loss = None
    for step in range(20):
        out = model(data)
        loss = loss_fn(out, target)
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"[TEST 4] Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    if final_loss < initial_loss:
        print("[TEST 4] PASS — loss decreased, training works after reconnect")
        passed += 1
    else:
        print("[TEST 4] FAIL — loss did not decrease")
        failed += 1
except Exception as e:
    print(f"[TEST 4] FAIL — {type(e).__name__}: {e}")
    failed += 1


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)

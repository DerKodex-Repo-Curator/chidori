#!/bin/bash
# Cross-Architecture Test Suite
# Runs each test individually, restarting server between tests.
# Logs ALL output to a timestamped file.
#
# Usage: ./tests/run_crossarch.sh <ssh_port> <ssh_host> <local_port>
# Example: ./tests/run_crossarch.sh 35900 ssh2.vast.ai 43212

SSH_PORT=${1:-35900}
SSH_HOST=${2:-ssh2.vast.ai}
LOCAL_PORT=${3:-43212}
DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGFILE="$DIR/.claude/crossarch_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$DIR/.claude"

PY="$DIR/../.venv/bin/python"
ENV="CHIDORI_DIRECT_CONNECT=1 CHIDORI_CFG_OVERRIDE={\"id\":\"test\",\"servers\":[\"localhost:$LOCAL_PORT\"]} LD_PRELOAD=$DIR/build/libcuda.so.1 LD_LIBRARY_PATH=$DIR/build"

PASSED=0
FAILED=0
TOTAL=0

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOGFILE"; }

restart_server() {
    ssh -o ConnectTimeout=5 -p "$SSH_PORT" "root@$SSH_HOST" '/root/chidori/restart_server.sh' 2>/dev/null | grep -q SERVER_READY
    if [ $? -ne 0 ]; then
        log "FATAL: Server restart failed"
        return 1
    fi
    sleep 1
    return 0
}

run_test() {
    local name="$1"
    local code="$2"
    local timeout_s="${3:-300}"
    TOTAL=$((TOTAL + 1))

    log "--- TEST: $name ---"
    restart_server || { FAILED=$((FAILED+1)); log "RESULT: $name FAIL (server restart)"; return; }

    local output
    output=$(timeout "$timeout_s" env \
        CHIDORI_DIRECT_CONNECT=1 \
        CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"localhost:$LOCAL_PORT\"]}" \
        LD_PRELOAD="$DIR/build/libcuda.so.1" \
        LD_LIBRARY_PATH="$DIR/build" \
        $PY -c "$code" 2>&1)
    local rc=$?

    echo "$output" >> "$LOGFILE"
    # Extract TEST_ lines for summary
    local results=$(echo "$output" | grep "^TEST_")
    local pass_line=$(echo "$output" | grep "TEST_PASS")

    if [ $rc -eq 0 ] && echo "$pass_line" | grep -q "True"; then
        PASSED=$((PASSED+1))
        log "RESULT: $name PASS"
    else
        FAILED=$((FAILED+1))
        log "RESULT: $name FAIL (exit=$rc)"
    fi
    echo "$results" | tee -a "$LOGFILE"
    log ""
}

# Header
log "=== Cross-Architecture Test Suite ==="
log "Server: $SSH_HOST:$SSH_PORT → localhost:$LOCAL_PORT"
log "Client: $(hostname) → $DIR/build"
log "Python: $PY"
log "Log: $LOGFILE"
log ""

# Get server GPU info
restart_server
SERVER_GPU=$(timeout 120 env \
    CHIDORI_DIRECT_CONNECT=1 \
    CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"localhost:$LOCAL_PORT\"]}" \
    LD_PRELOAD="$DIR/build/libcuda.so.1" \
    LD_LIBRARY_PATH="$DIR/build" \
    $PY -c "import torch; torch._C._cuda_init(); print(torch.cuda.get_device_name(0))" 2>/dev/null | tail -1)
log "Server GPU: $SERVER_GPU"
log ""

# TEST 1: Init + device name
run_test "init_device" "
import torch, time
t0 = time.time()
torch._C._cuda_init()
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
print(f'TEST_GPU: {name}')
print(f'TEST_VRAM: {vram} GB')
print(f'TEST_TIME: {time.time()-t0:.1f}s')
print(f'TEST_PASS: {len(name) > 0}')
" 120

# TEST 2: Tensor create + readback
run_test "tensor_readback" "
import torch, time
t0 = time.time()
torch._C._cuda_init()
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cuda()
z = x.cpu().tolist()
print(f'TEST_VALUES: {z}')
print(f'TEST_SUM: {sum(z)}')
print(f'TEST_TIME: {time.time()-t0:.1f}s')
print(f'TEST_PASS: {z == [1.0, 2.0, 3.0, 4.0, 5.0]}')
" 120

# TEST 3: Kernel launch (element-wise)
run_test "kernel_elementwise" "
import torch, time
t0 = time.time()
torch._C._cuda_init()
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
y = x * 2 + 1
torch.cuda.synchronize()
result = y.cpu().tolist()
print(f'TEST_RESULT: {result}')
print(f'TEST_EXPECTED: [3.0, 5.0, 7.0]')
print(f'TEST_TIME: {time.time()-t0:.1f}s')
print(f'TEST_PASS: {result == [3.0, 5.0, 7.0]}')
" 300

# TEST 4: Matmul (cuBLAS)
run_test "matmul_cublas" "
import torch, time
t0 = time.time()
torch._C._cuda_init()
a = torch.randn(256, 256).cuda()
b = torch.randn(256, 256).cuda()
c = torch.mm(a, b)
torch.cuda.synchronize()
s = c.sum().item()
print(f'TEST_SUM: {s:.2f}')
print(f'TEST_NONZERO: {abs(s) > 0.01}')
print(f'TEST_TIME: {time.time()-t0:.1f}s')
print(f'TEST_PASS: {abs(s) > 0.01}')
" 300

# TEST 5: Large matmul
run_test "matmul_large" "
import torch, time
t0 = time.time()
torch._C._cuda_init()
a = torch.randn(1024, 1024).cuda()
b = torch.randn(1024, 1024).cuda()
c = torch.mm(a, b)
torch.cuda.synchronize()
s = c.sum().item()
print(f'TEST_SUM: {s:.2f}')
print(f'TEST_TIME: {time.time()-t0:.1f}s')
print(f'TEST_PASS: {abs(s) > 0.01}')
" 300

# TEST 6: Training step (forward + backward + optimizer)
run_test "training_step" "
import torch, torch.nn as nn, time, math
t0 = time.time()
torch._C._cuda_init()
model = nn.Sequential(nn.Linear(64, 128, bias=False), nn.ReLU(), nn.Linear(128, 10, bias=False)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

losses = []
for step in range(5):
    x = torch.randn(8, 64).cuda()
    t = torch.randint(0, 10, (8,)).cuda()
    opt.zero_grad()
    out = model(x)
    loss = loss_fn(out, t)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    print(f'TEST_STEP{step}: loss={loss.item():.4f}')

all_finite = all(math.isfinite(l) for l in losses)
print(f'TEST_LOSSES: {[round(l,4) for l in losses]}')
print(f'TEST_FINITE: {all_finite}')
print(f'TEST_TIME: {time.time()-t0:.1f}s')
print(f'TEST_PASS: {all_finite}')
" 600

# Summary
log "=== RESULTS ==="
log "$PASSED/$TOTAL passed, $FAILED failed"
log "Log saved to: $LOGFILE"
if [ $FAILED -gt 0 ]; then
    log "SOME TESTS FAILED"
    exit 1
fi
log "ALL TESTS PASSED"
exit 0

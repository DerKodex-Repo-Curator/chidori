#!/bin/bash
#
# Chidori GPU-over-IP — Remote GPU Integration Test
#
# Two-shell approach (simplest, most reliable):
#
# SHELL 1 — GPU Server (ssh into the GPU machine):
#   scp -r gpu-worker/ ubuntu@66.85.148.90:/tmp/chidori-test
#   ssh ubuntu@66.85.148.90
#   cd /tmp/chidori-test/gpu-worker/tests
#   ./start-server.sh 43211
#
# SHELL 2 — Client (ssh into the GPU-less machine):
#   scp -r gpu-worker/ root@66.85.148.120:/tmp/chidori-test
#   ssh root@66.85.148.120
#   cd /tmp/chidori-test/gpu-worker/tests
#   ./run-test.sh 66.85.148.90 43211
#
# Or run everything from your laptop with deploy + test:

set -euo pipefail

# ============================================================
# CONFIGURATION — copy to request.sh and edit
# ============================================================
GPU_SERVER_USER="YOUR_GPU_SERVER_USER"
GPU_SERVER_IP="YOUR_GPU_SERVER_IP"
GPU_SERVER_PORT=43211

CLIENT_USER="YOUR_CLIENT_USER"
CLIENT_IP="YOUR_CLIENT_IP"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_SSH="${GPU_SERVER_USER}@${GPU_SERVER_IP}"
CLIENT_SSH="${CLIENT_USER}@${CLIENT_IP}"
REMOTE_DIR="/tmp/chidori-test"

log()  { echo -e "\033[0;36m[chidori]\033[0m $*"; }
ok()   { echo -e "\033[0;32m[chidori]\033[0m $*"; }
fail() { echo -e "\033[0;31m[chidori]\033[0m $*"; exit 1; }

# --- Preflight ---
log "Checking SSH access..."
ssh -o ConnectTimeout=5 "$GPU_SSH" "echo ok" > /dev/null 2>&1 || fail "Cannot SSH to $GPU_SSH"
ssh -o ConnectTimeout=5 "$CLIENT_SSH" "echo ok" > /dev/null 2>&1 || fail "Cannot SSH to $CLIENT_SSH"
ok "SSH access confirmed"

# --- Deploy to GPU server ---
log "Deploying to GPU server..."
ssh "$GPU_SSH" "test -f $REMOTE_DIR/build/chidori-worker" 2>/dev/null || {
    ssh "$GPU_SSH" "mkdir -p $REMOTE_DIR"
    scp -q -r "$GPU_WORKER_DIR/include" "$GPU_WORKER_DIR/src" "$GPU_WORKER_DIR/tests" \
              "$GPU_WORKER_DIR/Makefile" "$GPU_SSH:$REMOTE_DIR/"
    ssh "$GPU_SSH" "cd $REMOTE_DIR && make server" || fail "Server build failed"
}
ok "GPU server ready"

# --- Deploy to client ---
log "Deploying to client..."
ssh "$CLIENT_SSH" "test -f $REMOTE_DIR/build/test_remote_gpu" 2>/dev/null || {
    ssh "$CLIENT_SSH" "mkdir -p $REMOTE_DIR"
    scp -q -r "$GPU_WORKER_DIR/include" "$GPU_WORKER_DIR/src" "$GPU_WORKER_DIR/tests" \
              "$GPU_WORKER_DIR/Makefile" "$CLIENT_SSH:$REMOTE_DIR/"
    ssh "$CLIENT_SSH" "cd $REMOTE_DIR && make client && gcc -Wall -O2 -I include \
        -o build/test_remote_gpu tests/test_remote_gpu.c -ldl -lm" || fail "Client build failed"
}
ok "Client ready"

# --- Start worker via screen ---
log "Starting gpu-worker on $GPU_SERVER_IP:$GPU_SERVER_PORT..."
ssh "$GPU_SSH" "pkill -f 'chidori-worker.*standalone' 2>/dev/null; true"
sleep 1
ssh "$GPU_SSH" "screen -dmS juice $REMOTE_DIR/build/chidori-worker --standalone $GPU_SERVER_PORT"

log "Waiting for CUDA init..."
for i in $(seq 1 15); do
    if ssh "$GPU_SSH" "pgrep -f 'chidori-worker.*standalone'" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done
ssh "$GPU_SSH" "pgrep -f 'chidori-worker.*standalone'" > /dev/null 2>&1 || fail "gpu-worker not running"
ok "gpu-worker running"

# --- Run test ---
echo ""
log "Running test: $CLIENT_IP → $GPU_SERVER_IP:$GPU_SERVER_PORT"
echo ""

ssh "$CLIENT_SSH" "cd $REMOTE_DIR/build && \
    CHIDORI_DIRECT_CONNECT=1 \
    CHIDORI_CFG_OVERRIDE='{\"id\":\"test\",\"servers\":[\"${GPU_SERVER_IP}:${GPU_SERVER_PORT}\"]}' \
    LD_PRELOAD=./libchidori.so \
    ./test_remote_gpu"
TEST_EXIT=$?

# --- Cleanup ---
ssh "$GPU_SSH" "pkill -f 'chidori-worker.*standalone' 2>/dev/null; true"

echo ""
[ "$TEST_EXIT" -eq 0 ] && ok "ALL TESTS PASSED!" || fail "TESTS FAILED (exit $TEST_EXIT)"
exit $TEST_EXIT

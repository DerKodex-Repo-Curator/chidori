#!/usr/bin/env bash
#
# Juice GPU-over-IP — Remote GPU Integration Test
#
# Tests CUDA-over-IP by deploying to two machines and running a workload
# from a GPU-less client against a remote GPU server.
#
# Usage:
#   cp request_example.sh request.sh
#   # Edit request.sh with your server/client IPs
#   ./request.sh
#
# Prerequisites:
#   - SSH access to both machines (key-based, no password prompt)
#   - GPU server: NVIDIA GPU + CUDA toolkit + GCC
#   - Client: GCC (no GPU needed)
#

set -euo pipefail

# ============================================================
# CONFIGURATION — copy this file to request.sh and edit these
# ============================================================
GPU_SERVER_USER="ubuntu"
GPU_SERVER_IP="YOUR_GPU_SERVER_IP"
GPU_SERVER_PORT=43211          # standalone gpu-worker port
GPU_PCI_BUS=""                 # leave empty for auto-detect (device 0)

CLIENT_USER="root"
CLIENT_IP="YOUR_CLIENT_IP"

# ============================================================
# Don't edit below unless you know what you're doing
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPU_WORKER_DIR="$PROJECT_DIR/gpu-worker"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[juice]${NC} $*"; }
ok()   { echo -e "${GREEN}[juice]${NC} $*"; }
warn() { echo -e "${YELLOW}[juice]${NC} $*"; }
fail() { echo -e "${RED}[juice]${NC} $*"; exit 1; }

GPU_SSH="${GPU_SERVER_USER}@${GPU_SERVER_IP}"
CLIENT_SSH="${CLIENT_USER}@${CLIENT_IP}"
REMOTE_DIR="/tmp/juice-test"

# -----------------------------------------------------------
# Preflight checks
# -----------------------------------------------------------
log "Checking SSH access..."
ssh -o ConnectTimeout=5 "$GPU_SSH" "echo ok" > /dev/null 2>&1 || fail "Cannot SSH to GPU server ($GPU_SSH)"
ssh -o ConnectTimeout=5 "$CLIENT_SSH" "echo ok" > /dev/null 2>&1 || fail "Cannot SSH to client ($CLIENT_SSH)"
ok "SSH access confirmed"

log "Checking GPU server..."
GPU_INFO=$(ssh "$GPU_SSH" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1" 2>/dev/null)
[ -z "$GPU_INFO" ] && fail "No GPU found on $GPU_SERVER_IP"
ok "GPU server: $GPU_INFO"

log "Checking client has no GPU (as expected)..."
if ssh "$CLIENT_SSH" "nvidia-smi" > /dev/null 2>&1; then
    warn "Client has a GPU — test still valid but defeats the purpose"
else
    ok "Client has no GPU — perfect"
fi

# -----------------------------------------------------------
# Deploy gpu-worker to GPU server
# -----------------------------------------------------------
log "Deploying gpu-worker to GPU server..."
ssh "$GPU_SSH" "mkdir -p $REMOTE_DIR/build"

NEED_BUILD=0
if ! ssh "$GPU_SSH" "test -f $REMOTE_DIR/build/juice-gpu-worker" 2>/dev/null; then
    NEED_BUILD=1
fi

if [ "$NEED_BUILD" -eq 1 ]; then
    log "Copying source to GPU server..."
    scp -q -r "$GPU_WORKER_DIR/include" "$GPU_WORKER_DIR/src" "$GPU_WORKER_DIR/Makefile" "$GPU_SSH:$REMOTE_DIR/"

    log "Building gpu-worker on GPU server..."
    ssh "$GPU_SSH" "cd $REMOTE_DIR && make server 2>&1" || fail "Build failed on GPU server"
    ok "gpu-worker built"
else
    ok "gpu-worker already deployed"
fi

# -----------------------------------------------------------
# Deploy client library + test to client machine
# -----------------------------------------------------------
log "Deploying client to $CLIENT_IP..."
ssh "$CLIENT_SSH" "mkdir -p $REMOTE_DIR/build"

NEED_CLIENT=0
if ! ssh "$CLIENT_SSH" "test -f $REMOTE_DIR/build/libjuice-cuda.so" 2>/dev/null; then
    NEED_CLIENT=1
fi

if [ "$NEED_CLIENT" -eq 1 ]; then
    log "Copying source to client..."
    scp -q -r "$GPU_WORKER_DIR/include" "$GPU_WORKER_DIR/src" "$GPU_WORKER_DIR/tests" \
              "$GPU_WORKER_DIR/Makefile" "$CLIENT_SSH:$REMOTE_DIR/"

    log "Building client library..."
    ssh "$CLIENT_SSH" "cd $REMOTE_DIR && make client 2>&1" || fail "Client lib build failed"

    log "Building test programs..."
    ssh "$CLIENT_SSH" "cd $REMOTE_DIR && gcc -Wall -O2 -I include \
        -o build/test_remote_gpu tests/test_remote_gpu.c -ldl -lm 2>&1" || fail "Test build failed"

    ssh "$CLIENT_SSH" "cp $REMOTE_DIR/build/libjuice-cuda.so $REMOTE_DIR/build/"
    ok "Client built"
else
    ok "Client already deployed"
fi

# -----------------------------------------------------------
# Start gpu-worker on GPU server
# -----------------------------------------------------------
log "Starting gpu-worker on GPU server (port $GPU_SERVER_PORT)..."
ssh "$GPU_SSH" "pkill -f 'juice-gpu-worker.*standalone' 2>/dev/null || true"
sleep 1

PCIBUS_ARG=""
[ -n "$GPU_PCI_BUS" ] && PCIBUS_ARG="--pcibus $GPU_PCI_BUS"

ssh "$GPU_SSH" "cd $REMOTE_DIR/build && nohup ./juice-gpu-worker --standalone $GPU_SERVER_PORT $PCIBUS_ARG \
    > /tmp/juice-worker.log 2>&1 &"

# Wait for CUDA init (can take 3-8 seconds on first run)
log "Waiting for CUDA initialization..."
for i in $(seq 1 15); do
    if ssh "$GPU_SSH" "grep -q 'Listening' /tmp/juice-worker.log 2>/dev/null"; then
        break
    fi
    sleep 1
done

if ! ssh "$GPU_SSH" "grep -q 'Listening' /tmp/juice-worker.log 2>/dev/null"; then
    ssh "$GPU_SSH" "cat /tmp/juice-worker.log" 2>/dev/null
    fail "gpu-worker failed to start"
fi

ok "gpu-worker listening on port $GPU_SERVER_PORT"
ssh "$GPU_SSH" "cat /tmp/juice-worker.log" 2>/dev/null

# -----------------------------------------------------------
# Run the test from the client
# -----------------------------------------------------------
echo ""
log "============================================"
log "  Running remote GPU test from CLIENT"
log "  Client:  $CLIENT_IP (no GPU)"
log "  Server:  $GPU_SERVER_IP (GPU)"
log "============================================"
echo ""

ssh "$CLIENT_SSH" "cd $REMOTE_DIR/build && \
    JUICE_DIRECT_CONNECT=1 \
    JUICE_CFG_OVERRIDE='{\"id\":\"test\",\"servers\":[\"${GPU_SERVER_IP}:${GPU_SERVER_PORT}\"]}' \
    LD_PRELOAD=./libjuice-cuda.so \
    ./test_remote_gpu 2>&1"

TEST_EXIT=$?

# -----------------------------------------------------------
# Show server-side logs
# -----------------------------------------------------------
echo ""
log "=== GPU Server Logs ==="
ssh "$GPU_SSH" "cat /tmp/juice-worker.log" 2>/dev/null

# -----------------------------------------------------------
# Cleanup gpu-worker process (leave binaries for next run)
# -----------------------------------------------------------
log "Stopping gpu-worker..."
ssh "$GPU_SSH" "pkill -f 'juice-gpu-worker.*standalone' 2>/dev/null || true"

echo ""
if [ "$TEST_EXIT" -eq 0 ]; then
    ok "============================================"
    ok "  ALL TESTS PASSED!"
    ok "  Remote GPU access working over TCP/IP"
    ok "============================================"
else
    fail "============================================"
    fail "  SOME TESTS FAILED (exit $TEST_EXIT)"
    fail "============================================"
fi

exit $TEST_EXIT

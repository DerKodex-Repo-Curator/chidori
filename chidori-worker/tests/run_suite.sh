#!/bin/bash
# ============================================================
# Chidori Test Suite
#
# One command to test everything over the wire.
#
# Usage:
#   ./tests/run_suite.sh <server_ip> [port] [server_ssh]
#
# Examples:
#   ./tests/run_suite.sh 185.216.20.105
#   ./tests/run_suite.sh 185.216.20.105 43211
#   ./tests/run_suite.sh 185.216.20.105 43211 ubuntu@185.216.20.105
#
# If server_ssh is provided, the script will restart the server
# between tests via SSH. Otherwise it assumes the server is
# already running and stays running.
#
# Run this script ON the client machine (e.g., Lenovo).
# The server machine must have chidori-worker built and ready.
# ============================================================

set -o pipefail

# ── Args ────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Chidori Test Suite"
    echo ""
    echo "Usage: ./tests/run_suite.sh <server_ip> [port] [server_ssh]"
    echo ""
    echo "  server_ip    IP of the GPU server (required)"
    echo "  port         Server port (default: 43211)"
    echo "  server_ssh   SSH target for server restart, e.g. ubuntu@1.2.3.4"
    echo "               If omitted, server must already be running."
    echo ""
    echo "Examples:"
    echo "  ./tests/run_suite.sh 185.216.20.105"
    echo "  ./tests/run_suite.sh 185.216.20.105 43211 ubuntu@185.216.20.105"
    exit 1
fi

SERVER_IP="$1"
PORT="${2:-43211}"
SERVER_SSH="${3:-}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="/tmp/chidori-tests"
mkdir -p "$LOGDIR"

# ── Find Python ─────────────────────────────────────────────
PY=""
for p in "$DIR/../.venv/bin/python" "$HOME/chidori/.venv/bin/python" "$(which python3 2>/dev/null)"; do
    [ -x "$p" ] && PY="$p" && break
done
if [ -z "$PY" ]; then
    echo "FATAL: No Python found"
    exit 1
fi

# ── Check build ─────────────────────────────────────────────
if [ ! -f "$DIR/build/libcuda.so.1" ]; then
    echo "FATAL: No client build. Run:"
    echo "  cd chidori-worker && make all && cd build && ln -sf libchidori.so libcuda.so.1"
    exit 1
fi

# ── Server management ───────────────────────────────────────
restart_server() {
    if [ -n "$SERVER_SSH" ]; then
        ssh -o ConnectTimeout=5 "$SERVER_SSH" \
            "pkill chidori-worker 2>/dev/null; sleep 1; \
             cd ~/chidori/chidori-worker && \
             nohup build/chidori-worker --standalone $PORT > /tmp/server.log 2>&1 & \
             sleep 2; \
             ps aux | grep 'chidori-worker --standalone' | grep -v grep > /dev/null && echo OK || echo FAIL" \
            2>/dev/null
    fi
}

# ── Test runner ─────────────────────────────────────────────
PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0
RESULTS=()

run_test() {
    local name="$1"
    local cmd="$2"
    local timeout_s="${3:-120}"
    local logfile="$LOGDIR/$(echo "$name" | tr ' ' '_').log"
    TOTAL=$((TOTAL + 1))

    # Restart server if we have SSH access
    if [ -n "$SERVER_SSH" ]; then
        local srv_status
        srv_status=$(restart_server)
        if echo "$srv_status" | grep -q "FAIL"; then
            printf "  %-40s SKIP  (server restart failed)\n" "$name"
            SKIPPED=$((SKIPPED + 1))
            RESULTS+=("SKIP  $name (server restart failed)")
            return
        fi
    fi

    printf "  %-40s " "$name"
    local t0=$(date +%s)

    timeout "$timeout_s" env \
        CHIDORI_DIRECT_CONNECT=1 \
        CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"$SERVER_IP:$PORT\"]}" \
        LD_PRELOAD="$DIR/build/libcuda.so.1" \
        LD_LIBRARY_PATH="$DIR/build" \
        bash -c "$cmd" > "$logfile" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))

    # Extract throughput/key metric from log
    local metric=""
    metric=$(grep -oP '\d+\.?\d*K? tok/s' "$logfile" | tail -1)
    [ -z "$metric" ] && metric=$(grep -oP '\d+\.?\d* TFLOPS' "$logfile" | tail -1)
    [ -z "$metric" ] && metric=$(grep -oP '\d+\.?\d* tok/s' "$logfile" | tail -1)

    if [ $rc -eq 0 ] && grep -q "^PASS:" "$logfile"; then
        PASSED=$((PASSED + 1))
        if [ -n "$metric" ]; then
            printf "PASS  %3ds  (%s)\n" "$dt" "$metric"
        else
            printf "PASS  %3ds\n" "$dt"
        fi
        RESULTS+=("PASS  $name")
    elif [ $rc -eq 124 ]; then
        FAILED=$((FAILED + 1))
        printf "FAIL  %3ds  (timeout)\n" "$dt"
        RESULTS+=("FAIL  $name (timeout after ${timeout_s}s)")
    else
        FAILED=$((FAILED + 1))
        local reason
        reason=$(grep -E "^FAIL:|Error|Traceback|error:" "$logfile" | head -1)
        [ -z "$reason" ] && reason="exit $rc"
        printf "FAIL  %3ds  (%s)\n" "$dt" "$reason"
        RESULTS+=("FAIL  $name ($reason)")
        # Show last few relevant lines
        grep -v "fat_binary:\|^\[chidori\]" "$logfile" | tail -3 | sed 's/^/         /'
    fi
}

# ── Header ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Chidori Test Suite"
echo "============================================================"
echo "  Server:     $SERVER_IP:$PORT"
[ -n "$SERVER_SSH" ] && echo "  Server SSH: $SERVER_SSH (auto-restart between tests)"
echo "  Client:     $(hostname)"
echo "  Python:     $PY"
echo "  Build:      $DIR/build"
echo "  Logs:       $LOGDIR/"
echo "============================================================"
echo ""

# ── Connectivity check ──────────────────────────────────────
printf "  %-40s " "connectivity check"
if timeout 5 bash -c "echo > /dev/tcp/$SERVER_IP/$PORT" 2>/dev/null; then
    printf "OK\n"
else
    if [ -n "$SERVER_SSH" ]; then
        restart_server > /dev/null
        sleep 2
        if timeout 5 bash -c "echo > /dev/tcp/$SERVER_IP/$PORT" 2>/dev/null; then
            printf "OK (started server)\n"
        else
            printf "FAIL (cannot connect to $SERVER_IP:$PORT)\n"
            exit 1
        fi
    else
        printf "FAIL (cannot connect to $SERVER_IP:$PORT)\n"
        echo ""
        echo "Server is not reachable. Either:"
        echo "  1. Start it manually: ssh <server> 'cd ~/chidori/chidori-worker && build/chidori-worker --standalone $PORT'"
        echo "  2. Pass server_ssh arg: ./tests/run_suite.sh $SERVER_IP $PORT user@server"
        exit 1
    fi
fi
echo ""

# ── Tests ───────────────────────────────────────────────────
echo "--- Core Tests (single remote GPU) ---"
echo ""

run_test \
    "FP32 MLP training (MSE loss)" \
    "$DIR/tests/run_training_test.sh $SERVER_IP $PORT" \
    60

run_test \
    "FP32 matmul throughput (4096x4096)" \
    "$PY $DIR/tests/test_train.py" \
    180

run_test \
    "FP32 transformer training (23M)" \
    "$PY $DIR/tests/test_training.py" \
    180

run_test \
    "FP16 transformer training (23M)" \
    "$PY $DIR/tests/test_training_fp16.py" \
    180

run_test \
    "Basic PyTorch ops (tensor, matmul)" \
    "$PY $DIR/tests/test_pytorch.py" \
    60

run_test \
    "TinyLlama 1.1B inference" \
    "$PY $DIR/tests/test_inference.py" \
    600

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Results: $PASSED passed, $FAILED failed, $SKIPPED skipped (of $TOTAL)"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "  Failures:"
    for r in "${RESULTS[@]}"; do
        echo "$r" | grep -q "^FAIL" && echo "    $r"
    done
    echo ""
    echo "  Logs: $LOGDIR/"
    echo ""
    echo "TESTS FAILED"
    exit 1
fi

echo ""
echo "ALL TESTS PASSED"
exit 0

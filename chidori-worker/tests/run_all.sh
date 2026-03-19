#!/bin/bash
# Chidori Test Suite Runner
#
# Usage:
#   ./tests/run_all.sh                     # loopback (starts local server)
#   ./tests/run_all.sh <server_ip> <port>  # remote server (client-only)
#
# Runs all tests, prints per-test PASS/FAIL, exits 0 if all pass.

set -o pipefail

SERVER=${1:-127.0.0.1}
PORT=${2:-43211}
DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_PID=""
# If server is remote (not loopback) or port != 43211, don't manage the server
MANAGE_SERVER=1
if [ "$SERVER" != "127.0.0.1" ] || [ "$PORT" != "43211" ]; then
    MANAGE_SERVER=0
fi

# Find Python with PyTorch
PY=python3
for p in "$DIR/../.venv/bin/python" "$HOME/chidori/.venv/bin/python" "$(which python3)"; do
    [ -x "$p" ] && PY="$p" && break
done

# Loopback mode: start server locally
start_server() {
    if [ "$MANAGE_SERVER" = "1" ]; then
        kill $(pgrep -f "chidori-worker.*--standalone.*$PORT") 2>/dev/null
        sleep 1
        "$DIR/build/chidori-worker" --standalone "$PORT" > /dev/null 2>&1 &
        SERVER_PID=$!
        sleep 1
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "FATAL: Server failed to start"
            exit 1
        fi
    fi
}

stop_server() {
    if [ "$MANAGE_SERVER" = "1" ] && [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        SERVER_PID=""
    fi
}

# Check prerequisites
if [ ! -f "$DIR/build/libcuda.so.1" ]; then
    echo "Build first: cd chidori-worker && make all && cd build && ln -sf libchidori.so libcuda.so.1"
    exit 1
fi

# Test counters
PASSED=0
FAILED=0
TOTAL=0
FAILURES=""

run_test() {
    local name="$1"
    local cmd="$2"
    local timeout_s="${3:-120}"
    TOTAL=$((TOTAL + 1))

    # Restart server for each GPU test (loopback only)
    stop_server
    start_server

    printf "  %-35s " "$name"
    local start_time=$(date +%s)
    local output
    output=$(timeout "$timeout_s" env \
        CHIDORI_DIRECT_CONNECT=1 \
        CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"$SERVER:$PORT\"]}" \
        LD_PRELOAD="$DIR/build/libcuda.so.1" \
        LD_LIBRARY_PATH="$DIR/build" \
        bash -c "$cmd" 2>&1)
    local rc=$?
    local end_time=$(date +%s)
    local dt=$((end_time - start_time))

    if [ $rc -eq 0 ]; then
        PASSED=$((PASSED + 1))
        printf "PASS  (%ds)\n" "$dt"
    else
        FAILED=$((FAILED + 1))
        printf "FAIL  (%ds, exit=%d)\n" "$dt" "$rc"
        echo "$output" | grep -v "fat_binary:" | tail -3 | sed 's/^/    /'
        FAILURES="$FAILURES  - $name (exit $rc)\n"
    fi
}

run_test_no_server() {
    local name="$1"
    local cmd="$2"
    local timeout_s="${3:-10}"
    TOTAL=$((TOTAL + 1))

    printf "  %-35s " "$name"
    local start_time=$(date +%s)
    local output
    output=$(timeout "$timeout_s" bash -c "$cmd" 2>&1)
    local rc=$?
    local end_time=$(date +%s)
    local dt=$((end_time - start_time))

    if [ $rc -eq 0 ]; then
        PASSED=$((PASSED + 1))
        printf "PASS  (%ds)\n" "$dt"
    else
        FAILED=$((FAILED + 1))
        printf "FAIL  (%ds, exit=%d)\n" "$dt" "$rc"
        echo "$output" | tail -3 | sed 's/^/    /'
        FAILURES="$FAILURES  - $name (exit $rc)\n"
    fi
}

# Header
echo ""
echo "=== Chidori Test Suite ==="
echo "Server:  $SERVER:$PORT"
echo "Python:  $PY"
echo "Build:   $DIR/build"
echo ""

# Protocol test (no GPU, no server)
if [ -f "$DIR/build/test_protocol" ]; then
    run_test_no_server "protocol (wire format)" "$DIR/build/test_protocol" 10
fi

# GPU tests
run_test "test_pytorch.py (basic ops)" "$PY $DIR/tests/test_pytorch.py" 30
run_test "test_train.py (matmul 4096)" "$PY $DIR/tests/test_train.py" 120
run_test "test_training_fp16.py (FP16)" "$PY $DIR/tests/test_training_fp16.py" 60
run_test "test_training.py (FP32 train)" "$PY $DIR/tests/test_training.py" 120
run_test "test_inference.py (TinyLlama)" "$PY $DIR/tests/test_inference.py" 300

# Cleanup
stop_server

# Summary
echo ""
echo "=== Results ==="
echo "$PASSED/$TOTAL passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failures:"
    printf "$FAILURES"
    echo ""
    echo "SOME TESTS FAILED"
    exit 1
fi
echo "ALL TESTS PASSED"
exit 0

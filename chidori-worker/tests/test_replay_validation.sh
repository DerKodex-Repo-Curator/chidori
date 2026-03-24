#!/bin/bash
# ============================================================
# Chidori Replay Correctness Validation
#
# Runs each replay test TWICE (baseline vs replay) and diffs
# the outputs. Identical output = PASS, any difference = FAIL.
#
# Usage:
#   ./tests/test_replay_validation.sh <server_ip> [port] [server_ssh]
#
# Examples:
#   ./tests/test_replay_validation.sh 185.216.20.105
#   ./tests/test_replay_validation.sh 185.216.20.105 43211
#   ./tests/test_replay_validation.sh 185.216.20.105 43211 ubuntu@185.216.20.105
#
# If server_ssh is provided, the script will restart the server
# between runs via SSH. Otherwise it assumes the server is
# already running and stays running.
#
# Run this script ON the client machine (e.g., Lenovo).
# The server machine must have chidori-worker built and ready.
# ============================================================

set -o pipefail

# ── Args ────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Chidori Replay Validation"
    echo ""
    echo "Usage: ./tests/test_replay_validation.sh <server_ip> [port] [server_ssh]"
    echo ""
    echo "  server_ip     IP of the GPU server (required)"
    echo "  port          Server port (default: 43211)"
    echo "  server_ssh    SSH target for server restart, e.g. ubuntu@1.2.3.4"
    echo ""
    echo "Examples:"
    echo "  ./tests/test_replay_validation.sh 185.216.20.105"
    echo "  ./tests/test_replay_validation.sh 185.216.20.105 43211 ubuntu@185.216.20.105"
    exit 1
fi

SERVER_IP="$1"
PORT="${2:-43211}"
SERVER_SSH="${3:-}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="/tmp/chidori-replay"
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

# ── Common env for test runs ────────────────────────────────
run_with_env() {
    local extra_env="$1"
    shift
    timeout 300 env \
        CHIDORI_DIRECT_CONNECT=1 \
        CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"$SERVER_IP:$PORT\"]}" \
        LD_PRELOAD="$DIR/build/libcuda.so.1" \
        LD_LIBRARY_PATH="$DIR/build" \
        $extra_env \
        "$@"
}

# ── Extract comparable lines from test output ────────────────
# Strips timing, device info, and non-deterministic lines.
# Keeps: LOSSES, WEIGHT HASH, GRADIENT HASH, OUTPUT HASH, step losses.
extract_comparable() {
    local infile="$1"
    grep -E '^\s+step\s+[0-9]+:|^WEIGHT HASH:|^GRADIENT HASH:|^OUTPUT HASH:|^LOSSES:' "$infile"
}

# ── A/B comparison for a single test ────────────────────────
PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0
RESULTS=()

run_ab_test() {
    local name="$1"
    local test_script="$2"
    TOTAL=$((TOTAL + 1))

    # Check that the test script exists
    if [ ! -f "$test_script" ]; then
        printf "  %-45s SKIP  (file not found)\n" "$name"
        SKIPPED=$((SKIPPED + 1))
        RESULTS+=("SKIP  $name ($test_script not found)")
        return
    fi

    local slug
    slug=$(basename "$test_script" .py)
    local baseline="$LOGDIR/${slug}_baseline.log"
    local replay="$LOGDIR/${slug}_replay.log"
    local baseline_cmp="$LOGDIR/${slug}_baseline.cmp"
    local replay_cmp="$LOGDIR/${slug}_replay.cmp"
    local difffile="$LOGDIR/${slug}.diff"

    printf "  %-45s " "$name"
    local t0=$(date +%s)

    # ── Phase A: Baseline (no CHIDORI_REPLAY) ────────────────
    if [ -n "$SERVER_SSH" ]; then
        local srv_status
        srv_status=$(restart_server)
        if echo "$srv_status" | grep -q "FAIL"; then
            printf "FAIL  (server restart failed for baseline)\n"
            FAILED=$((FAILED + 1))
            RESULTS+=("FAIL  $name (server restart failed for baseline)")
            return
        fi
    fi

    run_with_env "" "$PY" "$test_script" > "$baseline" 2>&1
    local rc_a=$?

    if [ $rc_a -ne 0 ]; then
        local dt=$(( $(date +%s) - t0 ))
        local reason
        reason=$(grep -E "^FAIL:|Error|Traceback|error:" "$baseline" | head -1)
        [ -z "$reason" ] && reason="exit $rc_a"
        printf "FAIL  %3ds  (baseline: %s)\n" "$dt" "$reason"
        FAILED=$((FAILED + 1))
        RESULTS+=("FAIL  $name (baseline failed: $reason)")
        grep -v "fat_binary:\|^\[chidori\]" "$baseline" | tail -3 | sed 's/^/         /'
        return
    fi

    # ── Phase B: Replay (CHIDORI_REPLAY=1) ───────────────────
    if [ -n "$SERVER_SSH" ]; then
        local srv_status
        srv_status=$(restart_server)
        if echo "$srv_status" | grep -q "FAIL"; then
            local dt=$(( $(date +%s) - t0 ))
            printf "FAIL  %3ds  (server restart failed for replay)\n" "$dt"
            FAILED=$((FAILED + 1))
            RESULTS+=("FAIL  $name (server restart failed for replay)")
            return
        fi
    fi

    run_with_env "CHIDORI_REPLAY=1" "$PY" "$test_script" > "$replay" 2>&1
    local rc_b=$?

    if [ $rc_b -ne 0 ]; then
        local dt=$(( $(date +%s) - t0 ))
        local reason
        reason=$(grep -E "^FAIL:|Error|Traceback|error:" "$replay" | head -1)
        [ -z "$reason" ] && reason="exit $rc_b"
        printf "FAIL  %3ds  (replay: %s)\n" "$dt" "$reason"
        FAILED=$((FAILED + 1))
        RESULTS+=("FAIL  $name (replay failed: $reason)")
        grep -v "fat_binary:\|^\[chidori\]" "$replay" | tail -3 | sed 's/^/         /'
        return
    fi

    # ── Phase C: Compare ─────────────────────────────────────
    extract_comparable "$baseline" > "$baseline_cmp"
    extract_comparable "$replay"   > "$replay_cmp"

    local dt=$(( $(date +%s) - t0 ))

    if diff -u "$baseline_cmp" "$replay_cmp" > "$difffile" 2>&1; then
        printf "PASS  %3ds  (baseline == replay)\n" "$dt"
        PASSED=$((PASSED + 1))
        RESULTS+=("PASS  $name")
    else
        printf "FAIL  %3ds  (output mismatch)\n" "$dt"
        FAILED=$((FAILED + 1))
        RESULTS+=("FAIL  $name (output mismatch)")
        echo ""
        echo "         --- diff (baseline vs replay) ---"
        head -30 "$difffile" | sed 's/^/         /'
        if [ "$(wc -l < "$difffile")" -gt 30 ]; then
            echo "         ... ($(wc -l < "$difffile") total diff lines, see $difffile)"
        fi
        echo ""
    fi
}

# ── Header ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Chidori Replay Validation"
echo "============================================================"
echo "  Server:     $SERVER_IP:$PORT"
[ -n "$SERVER_SSH" ] && echo "  Server SSH: $SERVER_SSH (auto-restart between runs)"
echo "  Client:     $(hostname)"
echo "  Python:     $PY"
echo "  Build:      $DIR/build"
echo "  Logs:       $LOGDIR/"
echo "============================================================"
echo ""
echo "  Each test runs TWICE: baseline (no replay) then replay"
echo "  (CHIDORI_REPLAY=1). Output must be bitwise identical."
echo ""

# ── Connectivity check ──────────────────────────────────────
printf "  %-45s " "connectivity check"
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
        echo "  2. Pass server_ssh arg: ./tests/test_replay_validation.sh $SERVER_IP $PORT user@server"
        exit 1
    fi
fi
echo ""

# ── Tests ───────────────────────────────────────────────────
echo "--- Replay A/B Comparisons ---"
echo ""

run_ab_test \
    "replay correctness (losses + weight hash)" \
    "$DIR/tests/test_replay_correctness.py"

run_ab_test \
    "replay gradients (gradient hash)" \
    "$DIR/tests/test_replay_gradients.py"

run_ab_test \
    "replay outputs (output hash)" \
    "$DIR/tests/test_replay_outputs.py"

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
    echo "  Full logs in: $LOGDIR/"
    echo "  Each test has: *_baseline.log, *_replay.log, *.diff"
    echo ""
    echo "REPLAY VALIDATION FAILED"
    exit 1
fi

if [ $SKIPPED -eq $TOTAL ]; then
    echo ""
    echo "  All tests skipped — no test scripts found."
    echo "  Expected: test_replay_correctness.py, test_replay_gradients.py, test_replay_outputs.py"
    echo ""
    echo "REPLAY VALIDATION SKIPPED"
    exit 1
fi

echo ""
echo "REPLAY VALIDATION PASSED"
exit 0

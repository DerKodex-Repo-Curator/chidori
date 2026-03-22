#!/bin/bash
# Chidori Partial GPU Offload — Test Runner
#
# Usage:
#   ./tests/run_partial_offload.sh <server_host> <server_port> [ssh_key]
#
# Examples:
#   # Via SSH tunnel (tunnel must be set up separately):
#   ./tests/run_partial_offload.sh 127.0.0.1 43211
#
#   # With SSH key for setting up tunnel to Vast.ai:
#   ./tests/run_partial_offload.sh ssh7.vast.ai 14434 ~/.ssh/vast_key
#
# Prerequisites:
#   - chidori-worker built (make all)
#   - chidori-worker running on the remote server on port 43211
#   - Python venv with PyTorch at ~/chidori/.venv

set -euo pipefail

SERVER_HOST="${1:?Usage: $0 <server_host> <server_port> [ssh_key]}"
SERVER_PORT="${2:?Usage: $0 <server_host> <server_port> [ssh_key]}"
SSH_KEY="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
VENV_PYTHON="${VENV_PYTHON:-$HOME/chidori/.venv/bin/python}"

# Verify build exists
if [ ! -f "$BUILD_DIR/libcuda.so.1" ]; then
    echo "ERROR: build/libcuda.so.1 not found. Run 'make all' first."
    exit 1
fi

# If SSH key provided and host is not localhost, set up tunnel
CONNECT_HOST="$SERVER_HOST"
CONNECT_PORT="$SERVER_PORT"
TUNNEL_PID=""

if [ -n "$SSH_KEY" ] && [ "$SERVER_HOST" != "127.0.0.1" ] && [ "$SERVER_HOST" != "localhost" ]; then
    echo "[setup] Setting up SSH tunnel to $SERVER_HOST:$SERVER_PORT..."
    # Kill any existing tunnel
    pkill -f "ssh.*43211.*$SERVER_HOST" 2>/dev/null || true
    sleep 1
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -f -N \
        -L 43211:localhost:43211 -p "$SERVER_PORT" "root@$SERVER_HOST"
    TUNNEL_PID=$(pgrep -f "ssh.*43211.*$SERVER_HOST" | tail -1)
    echo "[setup] Tunnel PID=$TUNNEL_PID"
    CONNECT_HOST="127.0.0.1"
    CONNECT_PORT="43211"
    sleep 2
fi

echo "[setup] Server: $CONNECT_HOST:$CONNECT_PORT"
echo "[setup] Build: $BUILD_DIR"
echo "[setup] Python: $VENV_PYTHON"
echo ""

# Run the full test suite
CHIDORI_LOCAL_DEVICES=0 \
CHIDORI_DIRECT_CONNECT=1 \
CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"$CONNECT_HOST:$CONNECT_PORT\"]}" \
LD_PRELOAD="$BUILD_DIR/libcuda.so.1" \
LD_LIBRARY_PATH="$BUILD_DIR" \
"$VENV_PYTHON" "$SCRIPT_DIR/test_partial_offload_full.py"
EXIT_CODE=$?

# Also run remote-only regression (no CHIDORI_LOCAL_DEVICES)
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "--- Remote-Only Regression ---"
    CHIDORI_DIRECT_CONNECT=1 \
    CHIDORI_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"$CONNECT_HOST:$CONNECT_PORT\"]}" \
    LD_PRELOAD="$BUILD_DIR/libcuda.so.1" \
    LD_LIBRARY_PATH="$BUILD_DIR" \
    "$VENV_PYTHON" "$SCRIPT_DIR/test_inference.py"
    REGRESSION_CODE=$?
    if [ $REGRESSION_CODE -ne 0 ]; then
        echo "FAIL: Remote-only regression test failed"
        EXIT_CODE=1
    else
        echo "PASS: Remote-only regression"
    fi
fi

# Cleanup tunnel
if [ -n "$TUNNEL_PID" ]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
fi

exit $EXIT_CODE

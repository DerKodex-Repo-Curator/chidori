#!/bin/bash
# Run this on the CLIENT (no GPU needed)
# Usage: ./run-test.sh <gpu-server-ip> [port]
#
# Example:
#   ./run-test.sh 66.85.148.90
#   ./run-test.sh 66.85.148.90 43211

set -e

SERVER_IP="${1:?Usage: ./run-test.sh <gpu-server-ip> [port]}"
PORT="${2:-43211}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"

# Always rebuild to ensure correct architecture
echo "Building client library..."
make -C "$SCRIPT_DIR/.." client

echo "Building test program..."
gcc -Wall -O2 -I "$SCRIPT_DIR/../include" \
    -o "$BUILD_DIR/test_remote_gpu" \
    "$SCRIPT_DIR/test_remote_gpu.c" -ldl -lm

echo "Running test against $SERVER_IP:$PORT..."
echo ""

cd "$BUILD_DIR"
JUICE_DIRECT_CONNECT=1 \
JUICE_CFG_OVERRIDE="{\"id\":\"test\",\"servers\":[\"${SERVER_IP}:${PORT}\"]}" \
LD_PRELOAD=./libjuice-cuda.so \
./test_remote_gpu

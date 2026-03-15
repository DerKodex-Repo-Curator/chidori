#!/bin/bash
# Run this on the GPU SERVER
# Usage: ./start-server.sh [port] [pcibus]
#
# Example:
#   ./start-server.sh 43211
#   ./start-server.sh 43211 00000000:0B:00.0

set -e

PORT="${1:-43211}"
PCIBUS_ARG=""
[ -n "${2:-}" ] && PCIBUS_ARG="--pcibus $2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"

# Build if needed
if [ ! -f "$BUILD_DIR/juice-gpu-worker" ]; then
    echo "Building gpu-worker..."
    make -C "$SCRIPT_DIR/.." server
fi

# Kill any existing worker
pkill -f "juice-gpu-worker.*standalone" 2>/dev/null || true
sleep 1

echo "Starting juice-gpu-worker on port $PORT..."
cd "$BUILD_DIR"
./juice-gpu-worker --standalone "$PORT" $PCIBUS_ARG

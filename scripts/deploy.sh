#!/usr/bin/env bash
# Deploy this project to an RDK S100P board and build it there.
#
#   BOARD=root@192.168.3.191 ./scripts/deploy.sh
#
# Copies cpp/ web/ assets/ scripts/ to /userdata/yolo26_dual_demo on the board,
# then runs the CMake build on-board (the BPU toolchain/libs live on the board).
set -euo pipefail

BOARD="${BOARD:-root@192.168.3.191}"
DEST="${DEST:-/userdata/yolo26_dual_demo}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SSH_OPTS="-o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o BatchMode=yes"

echo "[deploy] target: $BOARD:$DEST"
ssh $SSH_OPTS "$BOARD" "mkdir -p $DEST/models"

# Single tar over one ssh connection (avoids per-file scp round trips).
tar -C "$HERE" -czf - cpp web assets scripts \
  | ssh $SSH_OPTS "$BOARD" "tar -xzf - -C $DEST"

echo "[deploy] building on board..."
ssh $SSH_OPTS "$BOARD" "cd $DEST/cpp && rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"

echo "[deploy] done. Run with:"
echo "  ssh $BOARD 'cd $DEST && bash scripts/run.sh'"

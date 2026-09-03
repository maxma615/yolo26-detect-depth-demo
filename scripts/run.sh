#!/bin/sh
# On-board launcher for the C++ YOLO26 detect+depth dual-model demo.
# Usage:
#   bash scripts/run.sh                          # live camera + web UI on :8080
#   bash scripts/run.sh --max-seconds 60         # auto stop after 60 s
#   bash scripts/run.sh --image assets/bus.jpg   # single image mode
#   bash scripts/run.sh --save output/rec.mp4    # also record the composite
#   bash scripts/run.sh --source 2 --cam-fps 60  # pick camera / fps
#   bash scripts/run.sh --hdmi                   # HDMI direct-out
#   bash scripts/run.sh --dep-variant l          # depth l (lite) instead of x
#   bash scripts/run.sh --score 0.4              # detection threshold
cd "$(dirname "$0")/.." || exit 1   # project root (parent of scripts/)
BIN=cpp/build/yolo26_dual
if [ ! -x "$BIN" ]; then
  echo "[run.sh] building $BIN ..."
  (cd cpp && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j4) || exit 1
fi
exec "$BIN" "$@"

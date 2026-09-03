#!/bin/sh
# Launch the C++ demo fully detached from the calling ssh session.
# Extra args are forwarded to yolo26_dual. Log goes to output/live.log.
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # project root (parent of scripts/)
mkdir -p "$HERE/output"
cd "$HERE" || exit 1
BIN=cpp/build/yolo26_dual
[ -x "$BIN" ] || { echo "missing $BIN; run bash scripts/run.sh first"; exit 1; }
setsid nohup "$BIN" "$@" > output/live.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$!"

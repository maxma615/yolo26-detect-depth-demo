#!/usr/bin/env bash
# Download the YOLO26 .hbm models into ./models.
#
# The models are NOT committed to git (they are ~200MB). Host them yourself
# (GitHub Releases, OSS, S3, an internal HTTP server, ...) and point this
# script at the base URL:
#
#   MODEL_URL=https://github.com/<you>/yolo26-detect-depth-demo/releases/download/v1.0 \
#       ./scripts/download_models.sh
#
# Or copy them from an existing board / local dir:
#
#   MODEL_SRC=/userdata/yolo26_dual_demo/models ./scripts/download_models.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HERE/models"
mkdir -p "$DEST"

# The two models the demo loads by default (detect + depth).
REQUIRED=(
  "yolo26x_detect_nashm_640x640_nv12.hbm"
  "yolo26x_depth_lite_nashm_768x768.hbm"
)
# Optional extras (other sizes / variants).
OPTIONAL=(
  "yolo26m_detect_nashm_640x640_nv12.hbm"
  "yolo26n_detect_nashm_640x640_nv12.hbm"
  "yolo26n_depth_nashm_768x768_nv12.hbm"
  "yolo26l_depth_lite_nashm_768x768.hbm"
)

if [[ -n "${MODEL_SRC:-}" ]]; then
  echo "[models] copying from $MODEL_SRC"
  for m in "${REQUIRED[@]}" "${OPTIONAL[@]}"; do
    [[ -f "$MODEL_SRC/$m" ]] && cp -v "$MODEL_SRC/$m" "$DEST/" || true
  done
elif [[ -n "${MODEL_URL:-}" ]]; then
  echo "[models] downloading from $MODEL_URL"
  for m in "${REQUIRED[@]}"; do
    curl -fL --retry 3 -o "$DEST/$m" "$MODEL_URL/$m"
  done
  for m in "${OPTIONAL[@]}"; do
    curl -fL --retry 3 -o "$DEST/$m" "$MODEL_URL/$m" || echo "[models] optional $m not fetched"
  done
else
  echo "ERROR: set MODEL_URL=<base http url> or MODEL_SRC=<local dir> first." >&2
  echo "  e.g. MODEL_SRC=/userdata/yolo26_dual_demo/models $0" >&2
  exit 1
fi

echo "[models] done:"
ls -la "$DEST"

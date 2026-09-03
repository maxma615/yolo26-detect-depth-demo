#!/bin/sh
# Install the HDMI demo as a boot service on the S100P.
# - builds the C++ binary if needed
# - installs yolo26-dual.service (autostart at boot, auto-restart)
# - disables the GNOME desktop (gdm) so the demo owns the DRM/HDMI output
# Revert any time with: bash uninstall_boot.sh  (restores graphical boot)
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # project root (parent of scripts/)
BIN="$HERE/cpp/build/yolo26_dual"

# 1) build if missing
if [ ! -x "$BIN" ]; then
  echo "[install] building ..."
  (cd "$HERE/cpp" && mkdir -p build && cd build && \
   cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j4) || exit 1
fi

# 2) stop the demo if it is already running
systemctl stop yolo26-dual 2>/dev/null || true

# 3) install the unit
cp "$HERE/scripts/yolo26-dual.service" /etc/systemd/system/yolo26-dual.service
systemctl daemon-reload

# 4) take over the display: stop gdm and boot to multi-user (no desktop).
#    The demo service then becomes DRM master for HDMI.
systemctl stop gdm3 2>/dev/null || true
systemctl disable gdm3 2>/dev/null || true
systemctl set-default multi-user.target

# 5) enable + start now
systemctl enable yolo26-dual
systemctl start yolo26-dual

echo "[install] done."
echo "  status    : systemctl status yolo26-dual"
echo "  journal   : journalctl -u yolo26-dual -f"
echo "  web panel : http://<board-ip>:8080/"
echo "  boot mode : multi-user (desktop disabled; revert with uninstall_boot.sh)"

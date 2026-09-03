#!/bin/sh
# Remove the boot service and restore the desktop graphical boot.
set -e
systemctl stop yolo26-dual 2>/dev/null || true
systemctl disable yolo26-dual 2>/dev/null || true
rm -f /etc/systemd/system/yolo26-dual.service
systemctl daemon-reload
systemctl enable gdm3 2>/dev/null || true
systemctl set-default graphical.target
echo "[uninstall] boot service removed; graphical desktop restored."
echo "  reboot to get the desktop back: reboot"

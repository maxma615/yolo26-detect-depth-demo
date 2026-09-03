# HDMI direct-out (DRM/KMS)

`--hdmi` renders the composite straight to a display via **DRM/KMS** — no X11, no
Wayland, no browser. `cpp/src/kms_display.h` opens `/dev/dri/card0`, becomes DRM
master, modesets the first connected connector (prefers HDMI-A-1), and presents
frames with double-buffered, vsync-aligned `drmModePageFlip`.

## Run

```bash
bash scripts/run.sh --hdmi --grid-cols 6 --grid-rows 4
```

## Autostart at boot (kiosk)

```bash
bash scripts/install_boot.sh    # installs yolo26-dual.service, disables gdm,
                                # sets default target to multi-user
bash scripts/uninstall_boot.sh  # restores graphical desktop
```

`install_boot.sh` disables the GNOME desktop so the demo can own the DRM master;
the systemd unit restarts the demo on failure and retries camera open (boot-race
safety).

## Notes

- The HDMI path adds a light-theme metrics panel + QR column drawn with OpenCV
  (resolution-adaptive), independent of the web UI.
- Hot-plug: if no monitor is attached at start, the KMS thread polls and picks up
  the display within ~2s; unplugging mid-run triggers re-init.
- If another compositor (gdm/GNOME) holds the DRM master, `--hdmi` cannot acquire
  it — stop the desktop first (`install_boot.sh` does this).

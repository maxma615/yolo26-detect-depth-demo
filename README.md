# YOLO26 Detect + Depth Dual-Model Realtime Demo (RDK S100P)

**[English](README.md)** | **[中文文档](README_cn.md)**

A realtime **dual-model** demo for the **D-Robotics RDK S100P** (Nash) edge board:
**YOLO26 object detection** and **YOLO26 monocular depth estimation** run
*concurrently* on the single-core **BPU**, fused into one composite view and
served over **HTTP/MJPEG** (browser) and/or **HDMI** (DRM/KMS direct-out).

```
+----------------------------+     +-----------------------------+
|  top : camera + detections |     |  left  : metrics panel      |
|        + person tracks     |     |  center: detect + depth     |
+----------------------------+     |  right : QR / brand         |
|  bottom: depth (turbo)     |     +-----------------------------+
|          + distance grid   |
+----------------------------+
```

- **Detect**: YOLO26 (n/m/x) `640x640 NV12`, anchor-free, person tracking via a
  native C++ **ByteTrack** implementation.
- **Depth**: YOLO26-Depth (n/l/x) `768x768`, colorized (turbo) with an optional
  **distance grid** (per-intersection relative distance, or metric with
  `--depth-meters`).
- **Inference**: `hobot_dnn` C API (`hbDNN`/`hbUCP`, `libdnn.so` + `libhbucp.so`) —
  the same native stack TROS uses. Two worker threads submit async UCP tasks so
  both models are resident on the BPU scheduler at once; CPU pre/post-processing
  of one model overlaps the other's BPU execution.
- **Display**: MJPEG + JSON stats over HTTP (`:8080`), and optional **HDMI**
  direct-out via DRM/KMS (`--hdmi`) with vsync page-flips.

## Screenshots

**Web dashboard** (browser, `:8080`) — metrics panel + live detect/depth + QR:

![web dashboard](docs/images/web_dashboard.png)

**Composite output** — top: camera + detections + person tracks; bottom: depth
turbo + distance grid:

![composite](docs/images/composite_detect_depth.jpg)

## Benchmarks (S100P, BPU @1.5GHz, 720p@30, steady-state)

| Metric | Value |
|---|---|
| camera / detect / depth frames | 1:1, zero drop |
| end-to-end | ~29.5 fps |
| detect BPU | avg ~9.2 ms (p95 ~9.4) |
| depth BPU  | avg ~15.8 ms (p95 ~16.4) |
| BPU utilization | ~65% |
| display stream | ~50 fps |

At **1080p@30** the numbers are essentially identical (composite is downscaled and
model inputs are fixed), so 1080p is free in terms of BPU cost.

## Requirements

- **Board**: D-Robotics RDK S100P (aarch64), TROS/Hobot image.
- **On-board libs**: OpenCV (4.x), `libdnn.so` + `libhbucp.so` (`/usr/hobot/lib`),
  `libdrm`, Qt5 *not* required (web UI is plain HTML/JS).
- **Camera**: any V4L2 UVC camera (`/dev/videoN`). An **EMEET PIXY** PTZ camera is
  supported including firmware-level AI auto-tracking (see
  [docs/camera-pixy.md](docs/camera-pixy.md)).
- **Models**: YOLO26 `.hbm` (nash-m). Not in git — see *Get models* below.

## Repository layout

```
cpp/
├── CMakeLists.txt            # builds yolo26_dual + tests + CPack .deb
├── src/main.cpp              # pipeline + compositor + HTTP + HDMI(KMS)
├── src/kms_display.h         # DRM/KMS HDMI direct-out (page-flip, vsync)
├── src/tracking/             # native C++ ByteTrack (byte_tracker.*)
├── src/ui/browser_overlay.*  # track boxes / labels / depth focus marker
└── tests/                    # byte_tracker + browser_overlay unit tests
web/index.html                # browser dashboard (live-reloaded by the server)
assets/coco_classes.names     # COCO labels
scripts/
├── run.sh                    # on-board launcher
├── deploy.sh                 # push + build on the board
├── download_models.sh        # fetch .hbm models (not in git)
├── install_boot.sh           # systemd autostart + HDMI takeover
└── uninstall_boot.sh         # restore graphical boot
docs/                         # architecture / web-ui / hdmi / camera / packaging
```

## Quick start

### 1. Get the models

Models are ~200MB and are **not** committed. Host them (GitHub Releases / OSS /
internal HTTP) and fetch, or copy from an existing board:

```bash
MODEL_SRC=/userdata/yolo26_dual_demo/models ./scripts/download_models.sh
# or
MODEL_URL=https://github.com/<you>/<repo>/releases/download/v1.0 ./scripts/download_models.sh
```

### 2. Deploy + build on the board

```bash
BOARD=root@192.168.3.191 ./scripts/deploy.sh
```

(or build manually: `cd cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`)

### 3. Run

```bash
# on the board
bash scripts/run.sh                          # live camera + web UI on :8080
bash scripts/run.sh --source 2 --cam-fps 60  # pick camera index / fps
bash scripts/run.sh --hdmi                   # HDMI direct-out (DRM/KMS)
bash scripts/run.sh --image assets/bus.jpg   # single-image mode
```

Open `http://<board-ip>:8080/` in a browser.

### CLI options (selected)

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `0` | V4L2 camera index (or image/video path) |
| `--cam-w/--cam-h` | `1280/720` | capture resolution |
| `--cam-fps` | `30` | capture frame rate (PIXY supports 60) |
| `--port` | `8080` | HTTP port |
| `--hdmi` | off | enable HDMI/DRM direct-out |
| `--grid-cols/--grid-rows` | `8/6` | depth distance-grid density (`--no-grid` disables) |
| `--depth-meters K` | `0` | show metric depth (m) instead of relative % |
| `--score` / `--nms` | `0.25/0.45` | detection thresholds |
| `--dep-variant` | `x` | depth model size n/l/x |

## Web dashboard

The server reads `web/index.html` from disk **on every request**, so you can edit
the UI and just refresh the browser — no recompile. The dashboard shows:

- **Left**: HBM inference latency (detect/depth), system utilization
  (BPU/CPU/MEMORY), display FPS, active persons, ByteTrack stats.
- **Center**: live composite (detect + tracks on top, depth + distance grid below).
- **Right**: brand + QR codes.

BPU utilization is read from the hardware counter
`/sys/devices/system/bpu/ratio` (real occupancy, not an estimate). See
[docs/web-ui.md](docs/web-ui.md).

## HDMI direct-out

`--hdmi` takes over `/dev/dri/card0` as DRM master and presents the composite with
double-buffered, vsync-aligned page flips. Use `scripts/install_boot.sh` to make it
autostart at boot (disables the GNOME desktop so the demo owns the display);
`scripts/uninstall_boot.sh` restores the desktop. See
[docs/hdmi.md](docs/hdmi.md).

## Camera: EMEET PIXY + AI tracking

The demo auto-detects any numeric `--source` as a V4L2 camera. For the EMEET PIXY
PTZ camera you can additionally enable firmware-level AI person tracking over HID.
See [docs/camera-pixy.md](docs/camera-pixy.md).

## Packaging (.deb)

CPack is configured. On the board:

```bash
cd cpp/build && cpack -G DEB
# -> yolo26-detect-depth-demo_1.0.0_arm64.deb
sudo dpkg -i yolo26-detect-depth-demo_1.0.0_arm64.deb
```

See [docs/packaging.md](docs/packaging.md).

## Testing

```bash
cd cpp/build && ctest --output-on-failure
```

Covers the ByteTrack tracker (ID stability, Hungarian tie-breaking, lifecycle) and
the browser overlay (labels, colors, clamping, focus marker).

## License

MIT — see [LICENSE](LICENSE).

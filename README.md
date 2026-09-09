<div align="center">

# YOLO26 Vision Lab

**Detect. Track. Depth. — on one RDK S100P.**

A native C++ dual-model vision pipeline with a live browser dashboard.

[English](README.md) · [简体中文](README_cn.md) · [Architecture](docs/architecture.md) · [Web UI](docs/web-ui.md)

![C++17](https://img.shields.io/badge/C%2B%2B-17-28313b)
![Platform](https://img.shields.io/badge/RDK-S100P-ff3c00)
![License](https://img.shields.io/badge/license-MIT-536170)

</div>

## See the pipeline

YOLO26 detection and monocular depth estimation share the BPU scheduler;
native C++ ByteTrack associates person detections on the CPU. The browser brings
the two views and live telemetry together without a frontend build step.

![Silver dashboard: telemetry, live detection/depth and community QR codes](docs/images/web_dashboard_silver.png)

*Captured on September 9, 2026, with EMEET PIXY at 1920×1080@30. A live snapshot, not a benchmark report.*

| Detect | Track | Depth |
| :--- | :--- | :--- |
| YOLO26x · 640 × 640 NV12 | ByteTrack · person IDs | YOLO26x Depth Lite · 768 × 768 |
| Object boxes and class labels | Native C++ association and lifecycle | Turbo visualization and relative-depth grid |

> **Depth is relative by default.** The demo is not a calibrated rangefinder.
> `--depth-meters K` applies a scalar conversion; it does not by itself establish
> metric accuracy. ByteTrack is a tracking algorithm, not a third neural model.

## Dashboard

The latest UI uses a silver background, a dot-matrix title, graphite latency
panel, and muted orange utilization bars.

- **Left:** separate Detect/Depth HBM latency and reciprocal FPS; BPU, CPU and
  memory utilization; display FPS; active persons; a framed ByteTrack panel;
  model names and input specifications.
- **Center:** complete detection/tracking view above depth, without encoded
  sidebars or duplicate diagnostic HUD text.
- **Right:** branding and three readable QR codes.
- **Responsive:** desktop-first, 16:9-oriented layout with narrow-screen fallbacks.
  Reduced-motion preferences are respected.

Inter and Bubbledot load from external font providers in the **viewing browser**.
Offline system-font fallbacks are available. Customize the UI in
[`web/index.html`](web/index.html); the server rereads it on each page request.

## Quick start

### 1 · Prepare the board and models

Requires RDK S100P with the Hobot DNN/UCP runtime, OpenCV 4.x, libdrm,
CMake and a C++17 compiler. Use a V4L2 camera; the demonstrated EMEET PIXY
may appear as `/dev/video0` or `/dev/video2`; verify the actual device index.

Run these commands **on the board**:

```bash
git clone https://github.com/maxma615/yolo26-detect-depth-demo.git
cd yolo26-detect-depth-demo
# Replace with a directory containing compatible Nash-m HBM files.
MODEL_SRC=/path/to/existing/models bash scripts/download_models.sh
```

The default pair is **not included in Git**:

```text
models/yolo26x_detect_nashm_640x640_nv12.hbm
models/yolo26x_depth_lite_nashm_768x768.hbm
```

You must supply compatible models. Alternatively, set `MODEL_URL` to your own
model hosting base URL; the repository does not promise a public model download.

### 2 · Build and run

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j4

# Default capture is 1080p@30, specified explicitly here for reproducibility.
# Change --source to the camera's actual video index (0 for this screenshot).
bash scripts/run.sh --source 0 --cam-w 1920 --cam-h 1080 --cam-fps 30 \
  --grid-cols 6 --grid-rows 4
```

Open **http://BOARD_IP:8080/**. Stop the foreground process with **Ctrl+C**.

<details>
<summary>Deploy from a Linux / WSL host instead</summary>

```bash
BOARD=root@BOARD_IP bash scripts/deploy.sh
```

This replaces project files and rebuilds on the board. It does **not** transfer
the local `models/` directory: place the required HBM files in the destination's
`models/` directory separately. The default destination is
`/userdata/yolo26_dual_demo`. Inspect the script before using it on an existing
installation; it rebuilds the build directory and disables SSH host-key checking.

</details>

## Performance: read the numbers correctly

Representative observations with the default model pair on S100P
(BPU at 1.5 GHz) are listed below, **not guaranteed benchmarks**.
Earlier project notes report approximately 29.5 FPS with 1080p@30 input;
the September 8 UI validation used 720p@30. These are different runs, not a
controlled resolution comparison.

| Metric | Approximate reference | Meaning |
| :--- | :--- | :--- |
| Detect HBM latency | 9.2 ms | Mean BPU task latency |
| Depth HBM latency | 15.8 ms | Mean BPU task latency |
| Detection / depth processing | 29–30 FPS at 30 FPS input | Worker processing rate |
| Display updates | 50 FPS | May reuse inference results; not 50 unique inferred frames/s |

The HBM panel's **FPS = 1000 / mean HBM latency (ms)** excludes preprocessing,
postprocessing and shared-resource contention. It is not measured end-to-end
throughput. Display FPS is also separate from capture and model FPS.

The pipeline uses **latest-frame-wins** buffers to prioritize freshness:
frames can be skipped under load. Similar frame rates do not prove zero drops.
A zero-drop claim requires frame-ID accounting over a defined test interval.
This repository does not include a reproducible long-duration benchmark report.

## Camera tracking & HDMI

```bash
# Optional EMEET PIXY firmware tracking; physically moves the camera.
python3 -m pip install hidapi
sudo python3 scripts/pixy_tracking.py on
sudo python3 scripts/pixy_tracking.py off

# Optional DRM/KMS direct output.
bash scripts/run.sh --source 0 --hdmi
```

Camera firmware tracking and ByteTrack are independent. HDMI shows the native
output, **not a browser rendering of the silver dashboard**. Autostart scripts
can disable the desktop to acquire DRM ownership; read the documentation first.

## Explore

| Guide | Contents |
| :--- | :--- |
| [Architecture](docs/architecture.md) | Workers, frame exchange and output |
| [Web UI](docs/web-ui.md) | Stats endpoints, visual customization and states |
| [EMEET PIXY](docs/camera-pixy.md) | Camera selection and firmware tracking |
| [HDMI](docs/hdmi.md) | DRM/KMS output and desktop interaction |
| [Packaging](docs/packaging.md) | Building a Debian package |

```text
cpp/       Native inference, ByteTrack, compositor, HTTP, KMS and tests
web/       Browser UI and branding / QR assets
scripts/   Launch, deployment, model preparation and camera control
docs/      Guides and screenshots
assets/    Class labels and sample assets
```

Run the native tests after building:

```bash
ctest --test-dir cpp/build --output-on-failure
```

## License

Repository code is [MIT licensed](LICENSE). Model weights, datasets, fonts and
other third-party assets retain their respective licenses.

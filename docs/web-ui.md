# Web UI

The C++ server serves `web/index.html` **from disk on every request**
(`LoadIndexHtml()`), so editing the HTML/CSS/JS and refreshing the browser takes
effect immediately — no recompile, no restart.

## Endpoints

| Path | Purpose |
|---|---|
| `/` | dashboard (`web/index.html`) |
| `/stream` | MJPEG live stream |
| `/snapshot.jpg` | single composite frame |
| `/stats.json` | live metrics (latency/fps/tracking/BPU/CPU/memory) |
| `/qr/qrN.png` | QR images from `web/` |
| `/stop` | graceful shutdown |

## Layout

Three-column grid:

- **Left (performance)** — HBM inference latency (DETECT/DEPTH ms), SYSTEM
  UTILIZATION (BPU/CPU/MEMORY bars), DISPLAY FPS / PERSON ACTIVE, ByteTrack
  (Active/Total/Lost/Latency).
- **Center (inference)** — the live composite (detect+tracks top, depth+grid
  bottom), `object-fit:cover` to fill the pane.
- **Right (community)** — brand logo + QR codes.

## Typography

Loads Google Fonts (Space Grotesk for display, Inter for body, JetBrains Mono for
data). On an offline board the fonts fall back gracefully to system faces.

## BPU utilization

Read from the hardware counter `/sys/devices/system/bpu/ratio` (0–100). This is
real occupancy, not an estimate derived from latency.

## Customizing

- Colors/sizes are CSS custom properties in `:root` (`--brand`, `--page`, ...).
- QR images: drop `web/qr1.png..qr3.png`.
- Brand logo: `web/brand.png` (served at `/qr/brand.png`).

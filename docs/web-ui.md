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
- HBM FPS is calculated as `1000 / average HBM latency in milliseconds` for
  each model. Its tooltip explains that this excludes pre/postprocessing and is
  not measured pipeline throughput. Invalid or zero latency displays a dash.
- The model information section reads names and input formats from
  `stats.models`, with missing values shown as a dash.
- **Center (inference)** — the live composite (detect+tracks top, depth+grid
  bottom), `object-fit:contain` to preserve all detection boxes and depth cells.
- **Right (community)** — three equal square QR slots, ordered: D-Robotics
  developer community, Discord, WeChat group. The public-account entry is hidden.

The desktop layout targets 16:9 displays. The center column follows the stacked
camera aspect ratio and available viewport height. The compositor also sizes its
canvas to the scaled camera width, removing encoded sidebars instead of cropping
the image in CSS. Native browser labels replace the live stream's diagnostic HUD;
the relative/metric depth label follows `depth.grid.unit`.

Below 1100px, QR codes move below the
main panels; below 600px, the video and telemetry stack vertically. Small screens
can scroll rather than clipping content.

## Typography

Adapts the AI Runtime visual reference into a silver laboratory interface:
soft silver lighting, subtle dot texture, graphite latency panel, orange bars,
and a dot-matrix YOLO26 title. The live stream remains the main visual; there is
no decorative background video or fabricated marketing telemetry.

Inter loads from Google Fonts; BubbledotICG-FinePos loads from OnlineWebFonts.
The viewing browser needs internet access for these fonts, not the board.
System sans-serif and monospace fallbacks keep the interface usable offline,
but the dot-matrix title requires the external display font. Numeric metrics
use local tabular monospace fonts to remain readable and stable.

Short staggered entrance animations and a subtle LIVE pulse honor
`prefers-reduced-motion`. No scanning animation covers the inference images.

## Connection states

The status begins in a connecting state. Statistics requests run sequentially;
after three failures the panel shows OFFLINE and clears stale values. Missing
values display as a dash, not zero. Stream reconnection remains independent.

## BPU utilization

Read from the hardware counter `/sys/devices/system/bpu/ratio` (0–100). This is
real occupancy, not an estimate derived from latency.

## Customizing

- Colors/fonts are CSS custom properties in `:root` (`--brand`, `--ink`, ...).
- QR images: drop `web/qr1.png..qr3.png`.
- Discord: `web/discord.png`, linking to https://discord.gg/rzqg8TgUgb.
  Regenerate with `python scripts/generate_discord_qr.py` (requires `qrcode[pil]`, `svglib`, and `reportlab`).
  Uses high error correction and the official Discord symbol centered on a Blurple badge.
  The compact two-module image border relies on the white card for additional scan clearspace.
- Brand logo: `web/brand.png` (served at `/qr/brand.png`).

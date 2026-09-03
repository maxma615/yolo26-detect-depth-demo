# Architecture

## Pipeline

```
V4L2 (MJPG) capture  ──>  FrameHub (latest-wins)
                              ├── detect worker ── hbDNNInferV2 ── UCP task (prio 8)
                              │        └─ ByteTrack (person) ── tracks
                              └── depth  worker ── hbDNNInferV2 ── UCP task (prio 4)
                                         └─ colorize + distance grid
        compositor (20ms cadence) ──> CompositeSlot ──> JPEG encoder
                              ├── HTTP /stream (MJPEG) + /snapshot.jpg + /stats.json
                              └── (optional) HDMI / KMS page-flip
```

## Concurrency model

The S100P BPU is a **single core**. We run both models "simultaneously" by
submitting two async UCP tasks (one per worker thread) with distinct priorities;
the DNN scheduler interleaves them while each thread's CPU pre/post-processing
overlaps the other model's BPU execution. This saturates the BPU (~65%) while
keeping a 1:1 camera→inference frame ratio at 30fps.

Input tensors are allocated with `hbUCPMallocCached` and written zero-copy via
`cv::Mat`. Dynamic (`-1`) strides are normalized to 64-byte alignment exactly as
`hbm_runtime` does, otherwise `hbDNNInferV2` rejects the tensor.

NV12 models expose separate Y / UV plane inputs; we locate them **by shape**
(`dim[3]==1` → Y, `dim[3]==2` → UV) because the order differs between model sizes.

## Composite

Top pane = camera frame + detection boxes + person track boxes.
Bottom pane = depth turbo colormap + distance grid + NEAR/FAR scale.
Each pane is letterboxed to a fixed height so the composite keeps a stable,
landscape-ish aspect ratio regardless of capture resolution (no stretching, no
vertical clipping in the web view).

## Stats

`/stats.json` exposes per-model latency (avg/min/max/p95), fps, tracking stats,
and **real BPU utilization** read from `/sys/devices/system/bpu/ratio`
(hardware counter), plus CPU/memory from `/proc`.

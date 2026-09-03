# Camera: EMEET PIXY (USB PTZ) + AI auto-tracking

The demo treats any numeric `--source` as a V4L2 camera (it forces the V4L2
backend + MJPG + a boot-race retry loop). The **EMEET PIXY**
(`0x328f:0x00c0`) is a USB PTZ camera that works out of the box and additionally
supports **firmware-level AI person tracking** over HID.

## Device nodes

On the RDK the PIXY typically enumerates as `/dev/video2` (capture) plus a
secondary metadata node (do not use). Identify it with:

```bash
v4l2-ctl -d /dev/video2 --info | grep 'Card'
# -> Card type : EMEET PIXY
```

## Run the demo on the PIXY at 1080p/60

```bash
bash scripts/run.sh --source 2 --cam-w 1920 --cam-h 1080 --cam-fps 60
```

The PIXY supports MJPG `1920x1080@60` and `1280x720@60`. Because the composite
is downscaled and model inputs are fixed, 1080p/60 costs essentially the same BPU
as 720p/30.

## AI auto-tracking (recommended)

Firmware tracking is much smoother than software PID (the camera tracks at 30Hz
internally). Enable it once; the demo then just consumes the moving frame:

```bash
sudo pip3 install hidapi
sudo python3 scripts/pixy_tracking.py on    # camera follows people
sudo python3 scripts/pixy_tracking.py off   # manual / fixed framing
```

If `hidapi` fails with permission denied, run as root or add a udev rule for
`0x328f:0x00c0`.

## V4L2 PTZ (optional, software control)

For manual pan/tilt/zoom use `v4l2-ctl` absolute controls:

| Control | Range | Unit |
|---|---|---|
| `pan_absolute` | -540000 … 540000 | 1/3600° |
| `tilt_absolute` | -324000 … 324000 | 1/3600° |
| `zoom_absolute` | 100 … 150 | percent |

Prefer HID AI tracking over software PTZ for following people.

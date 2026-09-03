#!/usr/bin/env python3
"""
EMEET PIXY firmware-level AI auto-tracking (HID).

Enables/disables the PIXY's on-camera AI person tracking, which is far smoother
than any software PID loop (the firmware tracks at 30Hz internally).

Usage:
    sudo pip3 install hidapi
    sudo python3 scripts/pixy_tracking.py on     # enable AI tracking
    sudo python3 scripts/pixy_tracking.py off    # disable

Packet: 32-byte HID report, [8]=0x01 on / 0x00 off (vendor 0x328f:0x00c0).
"""
import sys

VENDOR_ID = 0x328F
PRODUCT_ID = 0x00C0
REPORT_LEN = 32


def _tracking_packet(enable: bool) -> bytes:
    val = 0x01 if enable else 0x00
    return (bytes([0x09, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, val])
            + bytes(REPORT_LEN - 9))


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("on", "off"):
        print(__doc__)
        return 2
    enable = sys.argv[1] == "on"
    try:
        import hid
    except ImportError:
        print("ERROR: hidapi not installed. Run: pip3 install hidapi",
              file=sys.stderr)
        return 1
    try:
        dev = hid.device()
        dev.open(VENDOR_ID, PRODUCT_ID)
    except Exception as e:  # permission / not found
        print(f"ERROR: cannot open PIXY HID ({VENDOR_ID:04x}:{PRODUCT_ID:04x}): {e}",
              file=sys.stderr)
        print("Run as root or add a udev rule for 0x328f:0x00c0.", file=sys.stderr)
        return 1
    dev.write(_tracking_packet(enable))
    dev.close()
    print(f"PIXY AI tracking: {'ON' if enable else 'OFF'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

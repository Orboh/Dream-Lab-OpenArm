#!/usr/bin/env python
"""Hardware-reset RealSense cameras (they re-enumerate on USB, ~8 s). Safe: it only restarts the camera, never the robot.

  .venv/bin/python scripts/reset_realsense.py [serial ...]     # no serial = all connected cameras

Use it when a camera opens but delivers no frames (typically the depth stream after a USB hiccup), then restart the console.
Stop anything that holds the cameras first (the console, camera_probe_web.py, lerobot-record).
A camera that is not listed at all is not connected: check its USB cable.
"""

import sys
import time

import pyrealsense2 as rs


def main(serials: list[str]) -> int:
    devs = list(rs.context().query_devices())
    if not devs:
        print("no RealSense camera found: check the USB cables")
        return 1
    for d in devs:
        s, name = d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name)
        if serials and s not in serials:
            continue
        print(f"resetting {name} {s} ...")
        d.hardware_reset()
    time.sleep(8)
    now = [(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name)) for d in rs.context().query_devices()]
    print("connected now:", *[f"{n} {s}" for s, n in now], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

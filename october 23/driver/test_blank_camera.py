"""The blank camera behind Dream's own CameraReader: black frames at the configured size and rate, no device."""

import time
import types

import numpy as np

import openarm_blank_camera as blank


def test_frames_are_black_at_the_configured_size_and_rate():
    cam = blank.BlankCamera(types.SimpleNamespace(width=64, height=48, fps=50))
    start = time.monotonic()
    frames = [cam.read() for _ in range(5)]
    assert time.monotonic() - start >= 0.07  # 5 frames at 50 Hz take about 0.1 s
    assert all(ok and f.shape == (48, 64, 3) and f.dtype == np.uint8 and not f.any() for ok, f in frames)
    cam.release()


def test_dreams_camera_reader_loads_it_from_a_station_entry_and_delivers_black_frames():
    from robots import cameras, config

    cfg = config.CameraConfig(name="wrist_l", width=320, height=176, fps=30, source=f"{blank.__file__}:BlankCamera")
    reader = cameras.CameraReader.from_config(cfg, stale_after_s=0.2)
    reader.start()
    try:
        deadline = time.monotonic() + 3
        while reader.frame is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert reader.frame is not None and not np.asarray(reader.frame.image).any()
    finally:
        reader.stop()

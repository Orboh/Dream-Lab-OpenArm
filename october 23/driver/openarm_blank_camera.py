"""A camera source that shows nothing: every frame is black.

Dream's docs say that for a wrist camera the robot does not use (here the locked left arm, whose real camera looks at the robot's own body), the model should get a black frame, never another picture.
Use it in the station file instead of the RealSense serial:  `{name: wrist_l, source: /path/to/openarm_blank_camera.py:BlankCamera, width: 1280, height: 720, fps: 15, rotate: 0}`.
It opens no device.
"""

import time

import numpy as np


class BlankCamera:
    """Has the `cv2.VideoCapture` shape Dream's camera reader expects: `read() -> (ok, frame_bgr)` and `release()`."""

    def __init__(self, cfg) -> None:
        """@param cfg The station file's camera entry: the frame size and the rate."""
        self._frame = np.zeros((int(cfg.height), int(cfg.width), 3), dtype=np.uint8)
        self._period = 1.0 / max(float(cfg.fps), 1.0)
        self._next = time.monotonic()

    def read(self) -> tuple[bool, np.ndarray]:
        """Waits for the next tick of the configured rate, then returns a black frame."""
        self._next += self._period
        time.sleep(max(0.0, self._next - time.monotonic()))
        return True, self._frame.copy()

    def release(self) -> None:
        """Nothing to release."""

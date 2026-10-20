"""Camera sources: Intel RealSense devices, or MuJoCo renders when running in sim."""

from __future__ import annotations

import io
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image


@dataclass
class CameraDevice:
    serial: str
    name: str
    usb: str = ""
    port: str = ""
    firmware: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CameraSource(ABC):
    device: CameraDevice

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Latest RGB frame (H, W, 3) uint8, or None."""

    def close(self) -> None:
        pass


# ----------------------------------------------------------------- RealSense
def _rs():
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise RuntimeError(
            "pyrealsense2 not installed. x86: pip install pyrealsense2. "
            "Jetson/aarch64: build librealsense with -DBUILD_PYTHON_BINDINGS=ON."
        ) from e
    return rs


def list_realsense() -> list[CameraDevice]:
    rs = _rs()
    out = []
    for dev in rs.context().query_devices():
        def info(key):
            return dev.get_info(key) if dev.supports(key) else ""
        out.append(CameraDevice(
            serial=info(rs.camera_info.serial_number), name=info(rs.camera_info.name),
            usb=info(rs.camera_info.usb_type_descriptor), port=info(rs.camera_info.physical_port),
            firmware=info(rs.camera_info.firmware_version)))
    return out


class RealSenseSource(CameraSource):
    def __init__(self, device: CameraDevice, width: int, height: int, fps: int):
        rs = _rs()
        self.device = device
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(device.serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipe.start(cfg)

    def read(self) -> np.ndarray | None:
        frames = self._pipe.poll_for_frames()
        color = frames.get_color_frame() if frames else None
        if not color:
            frames = self._pipe.wait_for_frames(1000)
            color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy() if color else None

    def close(self) -> None:
        try:
            self._pipe.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------- sim
SIM_DEVICES = [
    # Deliberately named neutrally so discovery has to work out the roles.
    CameraDevice(serial="sim-cam-a", name="MuJoCo camera_wrist_right"),
    CameraDevice(serial="sim-cam-b", name="MuJoCo tablecam"),
]


class SimCameraSource(CameraSource):
    def __init__(self, sim, device: CameraDevice, width: int = 480, height: int = 360):
        self.sim, self.device, self.w, self.h = sim, device, width, height
        cams = {sim.model.camera(i).name for i in range(sim.model.ncam)}
        wanted = device.name.removeprefix("MuJoCo ")
        self._cam: str | int = wanted if wanted in cams else -1  # -1 = free camera

    def read(self) -> np.ndarray | None:
        img = self.sim.render(self.w, self.h, self._cam)
        if img is None:  # no GL: synthetic frame so the UI still works
            img = np.full((self.h, self.w, 3), 40, np.uint8)
        return img


# -------------------------------------------------------------------- helpers
def to_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class CameraManager:
    """Owns open camera sources by role; caches the latest JPEG per role."""

    def __init__(self):
        self.sources: dict[str, CameraSource] = {}
        self._jpeg: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def open(self, sources: dict[str, CameraSource]) -> None:
        self.close()
        self.sources = sources

    def close(self) -> None:
        for s in self.sources.values():
            s.close()
        self.sources = {}
        self._jpeg = {}

    def snapshot(self, role: str, max_age_s: float = 0.05) -> bytes | None:
        with self._lock:
            cached = self._jpeg.get(role)
            if cached and time.monotonic() - cached[0] < max_age_s:
                return cached[1]
            src = self.sources.get(role)
            img = src.read() if src else None
            if img is None:
                return None
            jpg = to_jpeg(img)
            self._jpeg[role] = (time.monotonic(), jpg)
            return jpg

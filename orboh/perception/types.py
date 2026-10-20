"""Shared perception types: an RGB-D frame and a detected object (camera frame, optionally base frame)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RGBD:
    """Colour + metric depth aligned to colour. Camera frame: x right, y down, z forward (metres)."""

    color: np.ndarray  # (H, W, 3) uint8
    depth_m: np.ndarray  # (H, W) float32, 0 = no measurement
    fx: float
    fy: float
    cx: float
    cy: float

    def points(self) -> np.ndarray:
        """(H, W, 3) camera-frame points; (0, 0, 0) where depth is missing."""
        h, w = self.depth_m.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = self.depth_m
        return np.stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z], axis=-1)

    def deproject(self, u: float, v: float, z: float) -> np.ndarray:
        return np.array([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])


@dataclass
class Detection:
    """One object on the table. Lengths in millimetres, positions in metres."""

    pixel: tuple[float, float]  # centre of the top face in the colour image
    cam_xyz: np.ndarray  # object centre (middle of its height) in the camera frame
    height_mm: float  # above the fitted table plane
    size_mm: tuple[float, float]  # footprint (long, short side) in the table plane
    yaw_rad: float  # long-side direction within the table plane, camera-plane basis (see block_finder)
    confidence: float
    matches: list[str] = field(default_factory=list)  # catalog pieces that fit within tolerance
    plane: tuple[np.ndarray, float] | None = None  # (unit normal toward the camera, offset): n.p + d = 0 on the table
    mask: np.ndarray | None = None  # (H, W) bool
    base_xyz: np.ndarray | None = None  # filled once a camera->base calibration is available

    def to_dict(self) -> dict:
        return {"pixel": [round(float(x), 1) for x in self.pixel], "cam_xyz": self.cam_xyz.round(4).tolist(),
                "base_xyz": None if self.base_xyz is None else self.base_xyz.round(4).tolist(),
                "height_mm": round(self.height_mm, 1), "size_mm": [round(s, 1) for s in self.size_mm],
                "yaw_deg": round(float(np.degrees(self.yaw_rad)), 1), "confidence": round(self.confidence, 2),
                "matches": self.matches}

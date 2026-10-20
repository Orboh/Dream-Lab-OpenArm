"""Convert between LeRobot key/value dicts (robot units) and canonical joint vectors."""

from __future__ import annotations

import math

import numpy as np

from orboh.config import LeRobotConfig
from orboh.types import GRIPPER_IDX, JOINT_NAMES, NUM_JOINTS


class JointMapping:
    def __init__(self, cfg: LeRobotConfig):
        self.cfg = cfg
        self.keys: list[str] = []
        for name in JOINT_NAMES:
            side, part = name.split("_", 1)
            if part == "gripper":
                self.keys.append(cfg.gripper_key.format(side=side))
            else:
                self.keys.append(cfg.joint_key.format(side=side, index=part.removeprefix("joint")))
        scale = math.pi / 180 if cfg.joint_units == "degrees" else 1.0
        self._scale = np.full(NUM_JOINTS, scale)
        self._sign = np.array([cfg.joint_signs.get(n, 1.0) for n in JOINT_NAMES])
        self._offset = np.array([cfg.joint_offsets.get(n, 0.0) for n in JOINT_NAMES])

    def to_canonical(self, values: dict) -> np.ndarray:
        missing = [k for k in self.keys if k not in values]
        if missing:
            raise KeyError(f"robot data missing keys {missing}; check robot.lerobot.joint_key/gripper_key")
        raw = np.array([float(values[k]) for k in self.keys])
        q = self._sign * raw * self._scale + self._offset
        span = self.cfg.gripper_open - self.cfg.gripper_closed
        for i in GRIPPER_IDX:
            q[i] = float(np.clip((raw[i] - self.cfg.gripper_closed) / span, 0.0, 1.0))
        return q

    def from_canonical(self, q: np.ndarray) -> dict[str, float]:
        raw = (np.asarray(q, dtype=float) - self._offset) / (self._sign * self._scale)
        span = self.cfg.gripper_open - self.cfg.gripper_closed
        for i in GRIPPER_IDX:
            raw[i] = self.cfg.gripper_closed + float(np.clip(q[i], 0.0, 1.0)) * span
        return {k: float(v) for k, v in zip(self.keys, raw)}

    def vector_to_canonical(self, vec: np.ndarray, names: list[str]) -> np.ndarray:
        """Convert a dataset row (values ordered by `names`) to canonical."""
        return self.to_canonical(dict(zip(names, vec)))

"""Canonical joint layout and trajectory type shared by every stage.

Canonical joint vector (16 values):
  [left_joint1..7, left_gripper, right_joint1..7, right_gripper]
Arm joints are radians in the MuJoCo OpenArm convention; grippers are
normalized 0 (closed) .. 1 (open). Robot/dataset units are converted at the
edges (see robot/mapping.py), so everything in between speaks one language.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIDES = ("left", "right")


def _joint_names() -> tuple[str, ...]:
    names: list[str] = []
    for side in SIDES:
        names += [f"{side}_joint{i}" for i in range(1, 8)] + [f"{side}_gripper"]
    return tuple(names)


JOINT_NAMES = _joint_names()
NUM_JOINTS = len(JOINT_NAMES)
GRIPPER_IDX = (7, 15)
ARM_IDX = tuple(i for i in range(NUM_JOINTS) if i not in GRIPPER_IDX)


def joint_index(name: str) -> int:
    return JOINT_NAMES.index(name)


@dataclass
class Trajectory:
    """Joint-space waypoints sampled at a fixed rate."""

    positions: np.ndarray  # (T, 16) canonical
    hz: float

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != NUM_JOINTS:
            raise ValueError(f"trajectory must be (T, {NUM_JOINTS}), got {self.positions.shape}")
        if self.hz <= 0:
            raise ValueError("hz must be positive")

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def duration(self) -> float:
        return (len(self) - 1) / self.hz if len(self) else 0.0

    def resample(self, hz: float | None = None, speed_scale: float = 1.0) -> "Trajectory":
        """Linearly resample to `hz`; speed_scale < 1 plays the motion slower."""
        hz = hz or self.hz
        if len(self) < 2:
            return Trajectory(self.positions.copy(), hz)
        t_src = np.arange(len(self)) / self.hz / speed_scale
        n = int(round(t_src[-1] * hz)) + 1
        t_dst = np.linspace(0.0, t_src[-1], n)
        out = np.stack([np.interp(t_dst, t_src, self.positions[:, j]) for j in range(NUM_JOINTS)], axis=1)
        return Trajectory(out, hz)

    def to_dict(self) -> dict:
        return {"hz": self.hz, "joint_names": list(JOINT_NAMES), "positions": self.positions.round(5).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(np.asarray(d["positions"]), float(d["hz"]))


@dataclass
class Plan:
    """What the DreamLab stage hands to safety + execution."""

    trajectory: Trajectory
    source: str
    meta: dict = field(default_factory=dict)


def min_jerk(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    """n waypoints from q0 to q1 (inclusive) with zero start/end velocity and acceleration."""
    if n < 2:
        return np.asarray([q1], dtype=np.float64)
    s = np.linspace(0.0, 1.0, n)[:, None]
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    return q0[None, :] + (q1 - q0)[None, :] * blend

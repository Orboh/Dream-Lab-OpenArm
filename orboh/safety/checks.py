"""Static (math-only) safety checks on a trajectory, before it ever touches sim or hardware."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np

from orboh.config import SafetyConfig
from orboh.types import ARM_IDX, GRIPPER_IDX, JOINT_NAMES, Trajectory


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    warning: bool = False  # passed, but worth a human look


@dataclass
class SafetyReport:
    passed: bool
    checks: list[CheckResult]
    fingerprint: str  # hash of the exact command sequence that was checked
    start_q: list[float]  # robot state when checked
    sim: dict = field(default_factory=dict)
    # Large, non-JSON outputs (measured joints, preview frames); kept out of to_dict().
    artifacts: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("artifacts")
        return d


def fingerprint(traj: Trajectory) -> str:
    return hashlib.sha256(traj.positions.round(6).tobytes() + str(traj.hz).encode()).hexdigest()[:16]


def _worst(values: np.ndarray, idx: tuple[int, ...]) -> tuple[float, str, int]:
    sub = np.abs(values[:, idx])
    t, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return float(sub[t, j]), JOINT_NAMES[idx[j]], int(t)


def static_checks(traj: Trajectory, lo: np.ndarray, hi: np.ndarray, cfg: SafetyConfig) -> list[CheckResult]:
    q = traj.positions
    results: list[CheckResult] = []

    finite = bool(np.isfinite(q).all())
    results.append(CheckResult("finite_values", finite, "all values finite" if finite else "NaN/inf in trajectory"))
    if not finite:
        return results

    ok = traj.duration <= cfg.max_plan_duration_s and len(traj) >= 2
    results.append(CheckResult("duration", ok, f"{traj.duration:.1f}s, {len(traj)} steps @ {traj.hz:g}Hz"))

    m = cfg.joint_limit_margin_rad
    lo_m, hi_m = lo.copy(), hi.copy()
    lo_m[list(ARM_IDX)] += m
    hi_m[list(ARM_IDX)] -= m
    below, above = q < lo_m - 1e-9, q > hi_m + 1e-9
    bad = below | above
    if bad.any():
        t, j = np.argwhere(bad)[0]
        detail = f"{JOINT_NAMES[j]}={q[t, j]:.3f} outside [{lo_m[j]:.3f}, {hi_m[j]:.3f}] at t={t / traj.hz:.2f}s"
    else:
        detail = f"within limits (margin {m} rad)"
    results.append(CheckResult("joint_limits", not bad.any(), detail))

    if len(traj) >= 2:
        vel = np.diff(q, axis=0) * traj.hz
        v, name, t = _worst(vel, ARM_IDX)
        results.append(CheckResult(
            "joint_velocity", v <= cfg.max_joint_velocity_rad_s,
            f"peak {v:.2f} rad/s on {name} at t={t / traj.hz:.2f}s (limit {cfg.max_joint_velocity_rad_s})"))
        g, name, t = _worst(vel, GRIPPER_IDX)
        results.append(CheckResult(
            "gripper_velocity", g <= cfg.max_gripper_velocity,
            f"peak {g:.2f}/s on {name} (limit {cfg.max_gripper_velocity})"))
    if len(traj) >= 3:
        acc = np.diff(q, n=2, axis=0) * traj.hz**2
        a, name, t = _worst(acc, ARM_IDX)
        results.append(CheckResult(
            "joint_acceleration", a <= cfg.max_joint_accel_rad_s2,
            f"peak {a:.1f} rad/s² on {name} at t={t / traj.hz:.2f}s (limit {cfg.max_joint_accel_rad_s2})"))
    return results

"""Make a policy chunk feasible: bounded velocity and acceleration, braking before targets, joint limits.

The policy proposes where the joints should be; this decides how fast the arms may get there. The output
satisfies the same limits `safety.static_checks` enforces, so a chunk that passes through here only fails a
check for geometric reasons (collisions), not because the model asked for an impossible move.
"""

from __future__ import annotations

import numpy as np

from orboh.config import SafetyConfig
from orboh.types import ARM_IDX, GRIPPER_IDX


def slew_limit(q0: np.ndarray, v0: np.ndarray, targets: np.ndarray, hz: float, cfg: SafetyConfig,
               lo: np.ndarray, hi: np.ndarray, margin_scale: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Follow `targets` (T, 16) from state (q0, v0) under velocity/acceleration caps.

    Returns (commands (T, 16), final velocity (16,)). Caps are scaled by `margin_scale` so rounding never
    tips a later static check over its limit.
    """
    dt = 1.0 / hz
    arm, grip = list(ARM_IDX), list(GRIPPER_IDX)
    vmax = np.zeros_like(q0)
    vmax[arm] = cfg.max_joint_velocity_rad_s * margin_scale
    vmax[grip] = cfg.max_gripper_velocity * margin_scale
    amax = np.full_like(q0, np.inf)
    amax[arm] = cfg.max_joint_accel_rad_s2 * margin_scale
    lo_m, hi_m = lo.copy(), hi.copy()
    lo_m[arm] += cfg.joint_limit_margin_rad
    hi_m[arm] -= cfg.joint_limit_margin_rad

    q, v = np.asarray(q0, dtype=float).copy(), np.asarray(v0, dtype=float).copy()
    out = np.empty_like(targets, dtype=float)
    for t, target in enumerate(targets):
        goal = np.clip(target, lo_m, hi_m)
        err = goal - q
        # Fastest speed from which the arm can still stop on the goal. Planned at half the acceleration cap:
        # discrete steps plus the acceleration clamp cannot quite deliver the full cap, which overshoots.
        brake = np.sqrt(np.where(np.isfinite(amax), amax, 1e9) * np.abs(err))
        v_des = np.clip(err / dt, -np.minimum(vmax, brake), np.minimum(vmax, brake))
        v = np.clip(v_des, v - amax * dt, v + amax * dt)
        q = np.clip(q + v * dt, lo_m, hi_m)
        out[t] = q
    return out, v

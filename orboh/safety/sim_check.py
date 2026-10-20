"""Roll the exact command sequence through MuJoCo and look for collisions / infeasible motion."""

from __future__ import annotations

from collections import Counter

import numpy as np

from orboh.config import SafetyConfig
from orboh.safety.checks import CheckResult
from orboh.sim.mujoco_sim import MujocoSim
from orboh.types import ARM_IDX, JOINT_NAMES, Trajectory


def simulate(
    traj: Trajectory, sim: MujocoSim, cfg: SafetyConfig, preview_frames: int = 6
) -> tuple[list[CheckResult], dict]:
    sim.reset(traj.positions[0])
    period = 1.0 / traj.hz
    bad_kinds = {"self", "arm_arm", "env"} | (set() if cfg.allow_finger_env_contact else {"finger_env"})
    collisions: Counter = Counter()
    first_collision = None
    finger_env = 0
    max_err, max_err_joint, max_err_t = 0.0, "", 0.0
    measured = np.zeros_like(traj.positions)
    frames: list[np.ndarray] = []
    frame_at = set(np.linspace(0, len(traj) - 1, preview_frames).astype(int)) if preview_frames else set()

    for t, q_cmd in enumerate(traj.positions):
        sim.set_command(q_cmd)
        sim.step(period)
        q = sim.joints()
        measured[t] = q
        err = np.abs(q - q_cmd)[list(ARM_IDX)]
        if err.max() > max_err:
            max_err = float(err.max())
            max_err_joint, max_err_t = JOINT_NAMES[ARM_IDX[int(err.argmax())]], t * period
        for kind, a, b in sim.contacts():
            if kind in bad_kinds:
                collisions[(kind, a, b)] += 1
                if first_collision is None:
                    first_collision = {"t": round(t * period, 2), "kind": kind, "bodies": [a, b]}
            elif kind == "finger_env":
                finger_env += 1
        if t in frame_at:
            img = sim.render()
            if img is not None:
                frames.append(img)

    results = [
        CheckResult(
            "sim_collisions",
            not collisions,
            "no robot self/arm-arm/environment collisions" if not collisions
            else f"{sum(collisions.values())} colliding steps; first at t={first_collision['t']}s: "
                 f"{first_collision['kind']} {first_collision['bodies'][0]} <-> {first_collision['bodies'][1]}",
        ),
        CheckResult(
            "sim_tracking",
            max_err <= cfg.sim_max_tracking_error_rad,
            f"max tracking error {max_err:.3f} rad on {max_err_joint} at t={max_err_t:.2f}s "
            f"(limit {cfg.sim_max_tracking_error_rad})",
        ),
    ]
    if finger_env:
        results.append(CheckResult("finger_environment_contact", True,
                                   f"gripper touches table/fixture on {finger_env} steps", warning=True))
    summary = {
        "scene": sim.scene,
        "collisions": [{"kind": k, "a": a, "b": b, "steps": n} for (k, a, b), n in collisions.most_common(10)],
        "first_collision": first_collision,
        "max_tracking_error_rad": round(max_err, 4),
        "measured": measured,
        "frames": frames,
    }
    return results, summary

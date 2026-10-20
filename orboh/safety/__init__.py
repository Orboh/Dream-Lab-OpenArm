from __future__ import annotations

import numpy as np

from orboh.config import SafetyConfig
from orboh.safety.checks import CheckResult, SafetyReport, fingerprint, static_checks
from orboh.safety.sim_check import simulate
from orboh.sim.mujoco_sim import MujocoSim
from orboh.types import Trajectory


def check_trajectory(traj: Trajectory, sim: MujocoSim, cfg: SafetyConfig, start_q: np.ndarray,
                     preview_frames: int = 6) -> SafetyReport:
    """Static checks first; only a statically sane trajectory is worth simulating."""
    lo, hi = sim.limits()
    checks = static_checks(traj, lo, hi, cfg)
    sim_summary: dict = {}
    artifacts: dict = {}
    if all(c.passed for c in checks):
        sim_checks, sim_summary = simulate(traj, sim, cfg, preview_frames)
        checks += sim_checks
        artifacts = {k: sim_summary.pop(k) for k in ("measured", "frames")}
    else:
        checks.append(CheckResult("sim_rollout", False, "skipped: static checks failed"))
    return SafetyReport(
        passed=all(c.passed for c in checks),
        checks=checks,
        fingerprint=fingerprint(traj),
        start_q=[round(float(v), 5) for v in start_q],
        sim=sim_summary,
        artifacts=artifacts,
    )


__all__ = ["CheckResult", "SafetyReport", "check_trajectory", "fingerprint", "static_checks", "simulate"]

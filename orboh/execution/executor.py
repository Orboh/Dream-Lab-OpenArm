"""Stream a safety-checked trajectory to the robot under a runtime supervisor."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from orboh.config import SafetyConfig
from orboh.robot.base import Robot
from orboh.types import ARM_IDX, JOINT_NAMES, Trajectory, min_jerk


def build_execution_trajectory(current_q: np.ndarray, plan: Trajectory, cfg: SafetyConfig,
                               speed_scale: float = 1.0) -> tuple[Trajectory, int]:
    """Approach (current pose -> plan start, min-jerk) + the plan at speed_scale.

    This is the exact sequence the sim check sees and the executor streams.
    Returns (trajectory, number of approach steps).
    """
    body = plan.resample(speed_scale=speed_scale)
    start = body.positions[0]
    dist = float(np.max(np.abs(start - current_q)[list(ARM_IDX)]))
    # min-jerk peaks at 1.875x average speed; size the move so the peak stays at approach_speed.
    duration = max(0.5, 1.875 * dist / cfg.approach_speed_rad_s)
    n = int(round(duration * body.hz)) + 1
    approach = min_jerk(np.asarray(current_q, dtype=float), start, n)
    full = np.vstack([approach[:-1], body.positions])
    return Trajectory(full, body.hz), n - 1


def supervise(target: np.ndarray, last_cmd: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_step: float) -> np.ndarray:
    """Last line of defence: clip to joint limits and cap the per-step arm move, whatever was asked."""
    cmd = np.clip(target, lo, hi)
    arm = list(ARM_IDX)
    cmd[arm] = last_cmd[arm] + np.clip(cmd[arm] - last_cmd[arm], -max_step, max_step)
    return cmd


@dataclass
class ExecutionResult:
    completed: bool
    steps_sent: int
    total_steps: int
    reason: str
    max_tracking_error_rad: float = 0.0
    events: list[str] = field(default_factory=list)


class Executor:
    def __init__(self, robot: Robot, cfg: SafetyConfig, lo: np.ndarray, hi: np.ndarray,
                 on_progress: Callable[[int, int], None] | None = None):
        self.robot, self.cfg, self.lo, self.hi = robot, cfg, lo, hi
        self.on_progress = on_progress

    def run(self, traj: Trajectory, stop: threading.Event) -> ExecutionResult:
        period = 1.0 / traj.hz
        max_step = self.cfg.max_joint_velocity_rad_s * period * 1.5
        last_cmd = self.robot.read_joints()
        max_err = 0.0
        next_tick = time.monotonic()
        for i, target in enumerate(traj.positions):
            if stop.is_set():
                return self._halt(i, len(traj), "stopped by operator (E-STOP)", max_err)
            t0 = time.monotonic()
            q = self.robot.read_joints()
            err = float(np.max(np.abs(q - last_cmd)[list(ARM_IDX)]))
            max_err = max(max_err, err)
            if err > self.cfg.runtime_max_tracking_error_rad:
                j = ARM_IDX[int(np.argmax(np.abs(q - last_cmd)[list(ARM_IDX)]))]
                return self._halt(i, len(traj), f"tracking error {err:.3f} rad on {JOINT_NAMES[j]} "
                                  f"(limit {self.cfg.runtime_max_tracking_error_rad})", max_err)
            # Supervisor: never exceed limits or the per-step velocity cap, whatever the plan says.
            cmd = supervise(target, last_cmd, self.lo, self.hi, max_step)
            self.robot.send_joints(cmd)
            last_cmd = cmd
            if time.monotonic() - t0 > self.cfg.watchdog_s:
                return self._halt(i + 1, len(traj), f"control loop overran watchdog ({self.cfg.watchdog_s}s)", max_err)
            if self.on_progress and (i % 5 == 0 or i == len(traj) - 1):
                self.on_progress(i + 1, len(traj))
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            else:
                next_tick = time.monotonic()
        return ExecutionResult(True, len(traj), len(traj), "completed", round(max_err, 4))

    def _halt(self, sent: int, total: int, reason: str, max_err: float) -> ExecutionResult:
        """Hold the current measured pose so the arms stop where they are."""
        try:
            self.robot.send_joints(self.robot.read_joints())
        except Exception as e:  # keep the original reason; the hold failure is reported too
            reason += f"; hold command failed: {e}"
        return ExecutionResult(False, sent, total, reason, round(max_err, 4))

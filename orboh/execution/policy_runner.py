"""Closed-loop execution of a Dream-1 policy under the same safety rules as a planned trajectory.

Three threads share one small state:
  * control (caller's thread): one row per tick -> supervisor -> robot; E-STOP, watchdog, tracking error
  * observer: hands camera frames + joints to the client at `obs_hz`
  * fetcher: asks the client when `refill_due`, slew-limits the chunk, rolls it through MuJoCo and only then
    queues it. A chunk that fails the sim check is dropped and the arms hold.

Every row that reaches the robot has been slew-limited (velocity, acceleration, braking, joint limits) and
passes through `supervise`. The sim check covers the chunk from the commanded state a moment before it is
installed; the control thread re-slews from its exact state at install time, which is a few rows of drift at most.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from orboh.config import PolicyConfig, SafetyConfig
from orboh.dreamlab.policy import PolicyClient, PolicyError
from orboh.execution.executor import supervise
from orboh.execution.slew import slew_limit
from orboh.robot.base import Robot
from orboh.safety import check_trajectory
from orboh.sim.mujoco_sim import MujocoSim
from orboh.types import ARM_IDX, JOINT_NAMES, Trajectory

WARMUP_BUDGET_S = 120.0  # how long a cold model may keep answering `pool_warming`


@dataclass
class PolicyResult:
    completed: bool
    reason: str
    steps_sent: int = 0
    chunks_received: int = 0
    chunks_rejected: int = 0
    max_tracking_error_rad: float = 0.0
    duration_s: float = 0.0
    dry_run: bool = False
    events: list[str] = field(default_factory=list)
    timings: list[dict] = field(default_factory=list)  # per get_action: rtt / inference ms


class PolicyRunner:
    def __init__(self, robot: Robot, client: PolicyClient, snapshots: Callable[[], dict[str, bytes | None]],
                 cfg: PolicyConfig, safety: SafetyConfig, lo: np.ndarray, hi: np.ndarray, check_sim: MujocoSim,
                 execute: bool = True, log: Callable[[str, str], None] | None = None,
                 on_progress: Callable[[int, int], None] | None = None):
        self.robot, self.client, self.snapshots = robot, client, snapshots
        self.cfg, self.safety, self.lo, self.hi, self.check_sim = cfg, safety, lo, hi, check_sim
        self.execute = execute
        self._log = log or (lambda msg, level="info": None)
        self.on_progress = on_progress

    # ---------------------------------------------------------------- run
    def run(self, prompt: str, max_duration_s: float | None, stop: threading.Event) -> PolicyResult:
        hz = self.client.action_hz
        period, max_step = 1.0 / hz, self.safety.max_joint_velocity_rad_s / hz * 1.5
        limit = max_duration_s or self.cfg.max_duration_s
        res = PolicyResult(False, "not started", dry_run=not self.execute)

        self._lock = threading.Lock()
        self._done = threading.Event()
        self._fatal: str | None = None
        self._rows: deque[np.ndarray] = deque()
        self._pending: np.ndarray | None = None  # raw chunk that passed the sim check, waiting to install
        self._executed = 0
        self._last_cmd = self.robot.read_joints()
        self._last_vel = np.zeros_like(self._last_cmd)
        self._latest_q = self._last_cmd.copy()
        self._res = res
        self._period_hz = hz

        self.client.open(prompt)
        self.client.observe(self._latest_q, self.snapshots())
        threads = [threading.Thread(target=self._observer, name="policy-observe", daemon=True),
                   threading.Thread(target=self._fetcher, name="policy-fetch", daemon=True)]
        for t in threads:
            t.start()

        t_start = time.monotonic()
        next_tick = t_start
        last_row_t = t_start
        got_first = False
        try:
            while True:
                now = time.monotonic()
                res.duration_s = now - t_start
                if stop.is_set():
                    return self._finish(res, False, "stopped by operator (E-STOP)")
                if self._fatal:
                    return self._finish(res, False, self._fatal)
                if res.duration_s >= limit:
                    return self._finish(res, True, f"completed ({limit:g}s)")

                t0 = time.monotonic()
                q = self.robot.read_joints() if self.execute else self._last_cmd
                if self.execute:
                    err_v = np.abs(q - self._last_cmd)[list(ARM_IDX)]
                    res.max_tracking_error_rad = max(res.max_tracking_error_rad, float(err_v.max()))
                    if err_v.max() > self.safety.runtime_max_tracking_error_rad:
                        j = ARM_IDX[int(np.argmax(err_v))]
                        return self._finish(res, False, f"tracking error {err_v.max():.3f} rad on {JOINT_NAMES[j]} "
                                            f"(limit {self.safety.runtime_max_tracking_error_rad})")
                with self._lock:
                    self._latest_q = q if self.execute else self.robot.read_joints()
                    if self._pending is not None and self.execute:
                        cmds, _ = slew_limit(self._last_cmd, self._last_vel, self._pending, hz, self.safety,
                                             self.lo, self.hi)
                        self._rows, self._pending, self._executed = deque(cmds), None, 0
                    row = self._rows.popleft() if self._rows else None
                    self._executed += row is not None
                if row is not None:
                    last_row_t, got_first = now, True
                    target = row
                else:
                    target = self._last_cmd  # hold
                    waited = now - last_row_t
                    budget = self.cfg.starve_s if got_first else WARMUP_BUDGET_S
                    if self.execute and waited > budget:
                        return self._finish(res, False, f"policy starved: no usable chunk for {waited:.1f}s")
                cmd = supervise(target, self._last_cmd, self.lo, self.hi, max_step) if self.execute else target
                if self.execute:
                    self.robot.send_joints(cmd)
                    res.steps_sent += 1
                with self._lock:
                    self._last_vel = (cmd - self._last_cmd) * hz
                    self._last_cmd = cmd
                if self.execute and time.monotonic() - t0 > self.safety.watchdog_s:
                    return self._finish(res, False, f"control loop overran watchdog ({self.safety.watchdog_s}s)")
                if self.on_progress:
                    self.on_progress(int(res.duration_s * 10), int(limit * 10))
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    stop.wait(delay)
                else:
                    next_tick = time.monotonic()
        finally:
            self._done.set()
            for t in threads:
                t.join(timeout=5)
            self.client.close()

    def _finish(self, res: PolicyResult, ok: bool, reason: str) -> PolicyResult:
        """Hold the measured pose (real motion only) and report."""
        self._done.set()
        if self.execute:
            try:
                self.robot.send_joints(self.robot.read_joints())
            except Exception as e:
                reason += f"; hold command failed: {e}"
        res.completed, res.reason = ok, reason
        return res

    # ------------------------------------------------------------ threads
    def _event(self, msg: str, level: str = "info") -> None:
        self._res.events.append(msg)
        self._log(msg, level)

    def _observer(self) -> None:
        period = 1.0 / self.cfg.obs_hz
        while not self._done.wait(period):
            try:
                with self._lock:
                    q = self._latest_q.copy()
                self.client.observe(q, self.snapshots())
            except Exception as e:
                self._fatal = f"observation failed: {e}"
                return

    def _fetcher(self) -> None:
        hz, period = self._period_hz, 1.0 / self._period_hz
        fails = 0
        warm_since = time.monotonic()
        while not self._done.is_set():
            with self._lock:
                busy = self._pending is not None
                executed, remaining = self._executed, len(self._rows)
            if busy or not self.client.refill_due(executed, remaining):
                self._done.wait(period)
                continue
            try:
                chunk = self.client.get_chunk()
            except PolicyError as e:
                if e.retryable and time.monotonic() - warm_since < WARMUP_BUDGET_S:
                    wait = min(e.retry_after_s or 1.0, 10.0)
                    self._event(f"model not ready ({e}); retrying in {wait:.1f}s", "warn")
                    self._done.wait(wait)
                    continue
                self._fatal = str(e)
                return
            except Exception as e:
                self._fatal = f"get_chunk crashed: {e}"
                return
            warm_since = time.monotonic()
            self._res.chunks_received += 1
            timing = self.client.timing()
            if timing and len(self._res.timings) < 200:
                self._res.timings.append({k: timing.get(k) for k in ("rtt_ms", "serve_infer_ms", "gateway_ms")})
            ok, why = self._check(chunk, hz)
            if ok:
                fails = 0
                if self.execute:
                    with self._lock:
                        self._pending = chunk.positions
                else:
                    self._event(f"dry-run: chunk {self._res.chunks_received} passes ({len(chunk)} rows, not sent)")
            else:
                fails += 1
                self._res.chunks_rejected += 1
                self._event(f"chunk {self._res.chunks_received} rejected: {why}", "error")
                if fails >= self.cfg.max_check_failures:
                    self._fatal = f"{fails} consecutive chunks failed the safety check: {why}"
                    return

    def _check(self, chunk: Trajectory, hz: float) -> tuple[bool, str]:
        """Slew-limit the chunk from the current command and roll it through MuJoCo."""
        if not np.isfinite(chunk.positions).all():
            return False, "non-finite values in chunk"
        with self._lock:
            q0, v0 = self._last_cmd.copy(), self._last_vel.copy()
        cmds, _ = slew_limit(q0, v0, chunk.positions, hz, self.safety, self.lo, self.hi)
        report = check_trajectory(Trajectory(np.vstack([q0, cmds]), hz), self.check_sim, self.safety, q0,
                                  preview_frames=0)
        bad = [f"{c.name}: {c.detail}" for c in report.checks if not c.passed]
        if bad:
            bad.append("start q=" + np.array2string(q0, precision=3, suppress_small=True, max_line_width=200))
        return report.passed, "; ".join(bad)

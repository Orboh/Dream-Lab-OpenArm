"""Smooth, motor-safe motion for the OpenArm driver. Pure numpy: no CAN, no Dream runtime, tested without hardware.

- #MotionLimits: per joint speed and acceleration limits (the "Gentle" table) and the torque budget: how far a commanded target may lead the measured position, so that
  the PD torque `kp x lead` stays at half of what the motor is rated for.
- #SmoothTracker: follows any target (a step, Dream's linear Home ramp, a command stream) with bounded speed and acceleration, a soft start and a soft stop, and keeps settling
  on its own when the targets stop coming.
- #smooth_path: a spline through recorded waypoints with an S-curve speed profile; it only stops at its two ends, never at a waypoint.
"""

import dataclasses
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")
## "Gentle": degrees per second and degrees per second squared, per joint.
GENTLE_VEL_DPS = (40.0, 40.0, 40.0, 40.0, 70.0, 70.0, 70.0, 80.0)
GENTLE_ACC_DPS2 = (120.0, 120.0, 120.0, 120.0, 250.0, 250.0, 250.0, 300.0)
## Rated (peak) torque by motor model, N.m (the fork's `motors/damiao/tables.py`).
RATED_NM = {"openarm_dm8009": 54.0, "openarm_dm4340": 28.0, "openarm_dm4310": 10.0}
## The share of the rated torque the PD term may use.
TORQUE_BUDGET = 0.5
## The gripper's squeeze limit, degrees of lead: explicit, because its budget would allow far more heat than it can take.
GRIPPER_LEAD_DEG = 8.0
## Targets stay this far inside a joint's limits, degrees.
SOFT_LIMIT_MARGIN_DEG = 3.0
## How fast the tracker closes the last distance to its target, 1/s (the speed is this times the distance, within the limits).
APPROACH_RATE = 4.0


@dataclasses.dataclass(frozen=True)
class MotionLimits:
    """Per joint limits in the order of #NAMES."""

    names: tuple[str, ...]
    vel_dps: np.ndarray
    acc_dps2: np.ndarray
    lead_deg: np.ndarray


def build_limits(kp: dict[str, float], models: dict[str, str]) -> MotionLimits:
    """The Gentle table with the torque budget worked out from each joint's gain and motor model.

    @param kp MIT position gain per joint name, N.m per rad.
    @param models Motor model per joint name, a key of #RATED_NM.
    @return The limits.
    """
    lead = []
    for name in NAMES:
        if name == "gripper":
            lead.append(GRIPPER_LEAD_DEG)
        else:
            lead.append(math.degrees(TORQUE_BUDGET * RATED_NM[models[name]] / kp[name]))
    return MotionLimits(NAMES, np.array(GENTLE_VEL_DPS), np.array(GENTLE_ACC_DPS2), np.array(lead))


class SmoothTracker:
    """Follows a target with bounded speed and acceleration; never overshoots it."""

    def __init__(self, limits: MotionLimits, x0: np.ndarray) -> None:
        """@param limits The limits. @param x0 Where it starts, degrees per joint (at rest)."""
        self.limits = limits
        self.x = np.array(x0, dtype=float)
        self.v = np.zeros_like(self.x)

    def reset(self, x0: np.ndarray) -> None:
        """Puts it at `x0`, at rest."""
        self.x, self.v = np.array(x0, dtype=float), np.zeros(len(self.x))

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        """Advances by `dt` seconds toward `target`.

        @return The new position, degrees per joint.
        """
        err = np.asarray(target, dtype=float) - self.x
        acc, vmax = self.limits.acc_dps2, self.limits.vel_dps
        brake = 0.9 * np.sqrt(2.0 * acc * np.abs(err))                      # the fastest speed from which it can still stop on the target
        v_des = np.sign(err) * np.minimum(np.minimum(vmax, brake), APPROACH_RATE * np.abs(err))
        self.v = self.v + np.clip(v_des - self.v, -acc * dt, acc * dt)
        new_x = self.x + self.v * dt
        crossed = (target - new_x) * err < 0.0                               # it would pass the target: stop on it
        self.x = np.where(crossed, target, new_x)
        self.v = np.where(crossed, 0.0, self.v)
        return self.x

    def settled(self, target: np.ndarray, tol_deg: float = 0.05) -> bool:
        """Whether it is on `target` and at rest."""
        return bool(np.all(np.abs(np.asarray(target) - self.x) < tol_deg) and np.all(np.abs(self.v) < 0.5))


@dataclasses.dataclass(frozen=True)
class SmoothPath:
    """Samples of a smooth path: `[n, joints]` degrees, `dt` apart, the first and last exactly the end points."""

    samples: np.ndarray
    duration: float


def smooth_path(points_deg: np.ndarray, limits: MotionLimits, dt: float) -> SmoothPath:
    """A spline through the waypoints with an S-curve speed profile that keeps every joint inside its speed and acceleration limits.

    @param points_deg `[n, 7]` waypoints in degrees, the first is where the arm is.
    @param limits The limits (the first seven joints are used).
    @param dt Seconds between samples.
    @return The path; one sample and zero duration when the end points coincide.
    """
    pts = np.asarray(points_deg, dtype=float)
    keep = [0] + [i for i in range(1, len(pts)) if np.max(np.abs(pts[i] - pts[i - 1])) > 1e-6]
    pts = pts[keep]
    if len(pts) < 2:
        return SmoothPath(pts[:1].copy(), 0.0)
    vel, acc = limits.vel_dps[: pts.shape[1]], limits.acc_dps2[: pts.shape[1]]
    knots = np.concatenate([[0.0], np.cumsum(np.max(np.abs(np.diff(pts, axis=0)) / vel, axis=1))])   # seconds at full speed to each waypoint
    spline = PchipInterpolator(knots, pts, axis=0)
    total = float(knots[-1])
    duration = max(1.5 * total, 0.5)
    while True:
        n = max(2, math.ceil(duration / dt))
        s = np.linspace(0.0, 1.0, n + 1)
        u = 10 * s**3 - 15 * s**4 + 6 * s**5                                  # minimum-jerk time scaling: zero speed only at the ends
        q = spline(u * total)
        step = duration / n
        v = np.diff(q, axis=0) / step
        a = np.diff(v, axis=0) / step
        if (np.all(np.abs(v) <= vel * 0.98) and np.all(np.abs(a) <= acc * 0.98)) or duration > 600.0:
            break
        duration *= 1.05
    q[0], q[-1] = pts[0], pts[-1]
    return SmoothPath(q, duration)


class RequestTally:
    """Counts what Dream asks of one arm, run by run, so the log can show whether an arm that should stay still is being asked to move.

    A run is a stretch of writes with no silence longer than #IDLE_S. Pure: the caller supplies the clock.
    """

    IDLE_S = 3.0        # silence that ends a run
    REPORT_S = 2.0      # between notices while a run keeps asking
    NOTICE_DEG = 5.0    # the smallest net move worth a line

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._origin: np.ndarray | None = None
        self._last_q: np.ndarray | None = None
        self._path = 0.0
        self._net = 0.0
        self._joint = 0
        self._last_t: float | None = None
        self._reported_t: float | None = None

    def update(self, q_deg: np.ndarray, now: float) -> list[tuple[str, float, int, float]]:
        """Takes one requested pose.

        @param q_deg The requested joint angles, degrees.
        @param now Seconds on any steady clock.
        @return Notices due: `("run ended", net, joint index, path)` for the run that just ended (only if it was a real move), and
            `("asks", net, joint index, path)` once the run passes #NOTICE_DEG and then every #REPORT_S. `net` is the largest joint's
            distance from where the run began; `path` is the joint travel summed over joints and writes, degrees.
        """
        out: list[tuple[str, float, int, float]] = []
        if self._last_t is not None and now - self._last_t > self.IDLE_S:
            if self._net >= self.NOTICE_DEG:
                out.append(("run ended", self._net, self._joint, self._path))
            self._reset()
        q = np.asarray(q_deg, dtype=float)
        if self._origin is None or self._last_q is None:
            self._origin, self._last_q = q.copy(), q.copy()
        self._path += float(np.abs(q - self._last_q).sum())
        self._last_q = q.copy()
        diff = np.abs(q - self._origin)
        self._joint, self._net = int(diff.argmax()), float(diff.max())
        self._last_t = now
        if self._net >= self.NOTICE_DEG and (self._reported_t is None or now - self._reported_t >= self.REPORT_S):
            self._reported_t = now
            out.append(("asks", self._net, self._joint, self._path))
        return out

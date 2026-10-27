"""Arm-to-arm clearance: how close the left and right arm come to each other at a pair of poses. Nothing here talks to a motor.

Each arm is three capsules (a segment with a radius) in the robot's body frame: upper arm (joint 2 to joint 4), forearm (joint 4 to joint 6) and hand
(joint 6 to the grasp point), all with Dream's 4 cm link radius, plus its open gripper: points on the hand housing and both fingers from the official meshes
(`openarm_gripper.surface_points`, log-039), each a ball of `openarm_gripper.SAMPLE_RADIUS_M`. The arms' base frames sit at #BASE_OFFSET_M in the body frame (log-013; the same as the workspace editor's BASE_OFFSET).
The joint chain is the one the guard uses (`openarm_zones.ArmKinematics`, logical angles through the joint mapping in use).

The gap between the arms is the smallest distance between any left and any right capsule surface; #problem flags a pair of poses closer than #MIN_GAP_M.

During a task (log-034) both arms run in the station's one process: each arm publishes its measured pose and its latest target here (#publish), and each
arm's `_limited` cuts a new target with #cut against the other arm's fresh poses (#others), so a target that would bring the arms within #MIN_GAP_M stops
at the last pose that keeps it. An arm not heard from within #FRESH_S counts as hanging at its rest pose.
Known limits: the gripper is counted fully open (its widest); the robot's own body and head are not included here (they are zones, log-039).
"""

import dataclasses
import threading
import time

import numpy as np

import openarm_gripper as gripper
import openarm_guard
import openarm_mapping as mapping
import openarm_poses as poses
import openarm_zones as zones

## Each arm's base frame in the robot's body frame, metres (x forward, y left, z up from the base plate).
BASE_OFFSET_M = {"left": np.array([0.0, 0.031, 0.698]), "right": np.array([0.0, -0.031, 0.698])}
## Upper arm and forearm radius: the link radius Dream's IK uses.
LINK_RADIUS_M = zones.LINK_RADIUS_M
## The smallest allowed gap between the two arms' surfaces, metres (the 2 cm clearance Dream keeps from a zone).
MIN_GAP_M = 0.02
## A pose another arm published longer ago than this, seconds, is no longer used (the arm counts as hanging at rest).
FRESH_S = 0.5
## Halvings of the move when a target is cut back to the last pose that keeps #MIN_GAP_M.
CUT_STEPS = 12


@dataclasses.dataclass(frozen=True)
class Capsule:
    """A segment `a`-`b` (body frame, metres) with a radius."""

    name: str
    a: np.ndarray
    b: np.ndarray
    radius: float


_KINEMATICS: dict[str, zones.ArmKinematics] = {}


def _kin(side: str) -> zones.ArmKinematics:
    if side not in _KINEMATICS:
        _KINEMATICS[side] = zones.ArmKinematics(side, logical=True)
    return _KINEMATICS[side]


def capsules(side: str, q_logical_rad) -> list[Capsule]:
    """One arm's capsules in the body frame. @param q_logical_rad Seven joint angles, radians, in our (logical) angles."""
    pts = [p + BASE_OFFSET_M[side] for p in _kin(side).chain(np.asarray(q_logical_rad, dtype=float))]
    return [Capsule("upper_arm", pts[1], pts[3], LINK_RADIUS_M),
            Capsule("forearm", pts[3], pts[5], LINK_RADIUS_M),
            Capsule("hand", pts[5], pts[7], LINK_RADIUS_M)]


_GRIP_LOCAL: dict[tuple[str, float], np.ndarray] = {}   # the gripper's surface points per side and finger opening (mm steps)


def _grip_local(side: str, opening_m: float) -> np.ndarray:
    key = (side, round(float(np.clip(opening_m, 0.0, gripper.FINGER_OPEN_M)), 3))
    if key not in _GRIP_LOCAL:
        _GRIP_LOCAL[key] = gripper.surface_points(side, key[1])
    return _GRIP_LOCAL[key]


def gripper_points(side: str, q_logical_rad, opening_m: float = gripper.FINGER_OPEN_M) -> np.ndarray:
    """The gripper's surface points of one arm in the body frame, `[n, 3]` metres (log-039); fully open unless told its opening (log-040)."""
    tool = _kin(side).tool_pose(mapping.to_model(side, np.asarray(q_logical_rad, dtype=float)))
    return _grip_local(side, opening_m) @ tool[:3, :3].T + tool[:3, 3] + BASE_OFFSET_M[side]


@dataclasses.dataclass(frozen=True)
class ArmShape:
    """One arm at one pose: its link capsules and its gripper points (body frame)."""

    capsules: list[Capsule]
    grip: np.ndarray


def shape(side: str, q_logical_rad, opening_m: float = gripper.FINGER_OPEN_M) -> ArmShape:
    """One arm's capsules and gripper points at a pose (logical radians) and finger opening (metres, fully open by default)."""
    return ArmShape(capsules(side, q_logical_rad), gripper_points(side, q_logical_rad, opening_m))


def _as_pose(item) -> tuple[np.ndarray, float]:
    """A pose of the other arm: `(q, opening)` as #others gives it, or a bare `q` (fully open)."""
    if isinstance(item, tuple):
        return np.asarray(item[0], dtype=float), float(item[1])
    return np.asarray(item, dtype=float), gripper.FINGER_OPEN_M


def _points_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((points - a) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1)


def segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """The shortest distance between the segments p1-q1 and p2-q2 (either may be a single point)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= eps:
            s, t = min(max(-c / a, 0.0), 1.0), 0.0
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = min(max(-c / a, 0.0), 1.0), 0.0
            elif t > 1.0:
                s, t = min(max((b - c) / a, 0.0), 1.0), 1.0
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _shape_gap(left: ArmShape, right: ArmShape) -> tuple[float, str, str]:
    """Links against links, each gripper against the other arm's links, and gripper against gripper."""
    r = gripper.SAMPLE_RADIUS_M
    best = (np.inf, "", "")
    for cl in left.capsules:
        for cr in right.capsules:
            gap = segment_distance(cl.a, cl.b, cr.a, cr.b) - cl.radius - cr.radius
            if gap < best[0]:
                best = (gap, cl.name, cr.name)
    for cr in right.capsules:
        gap = float(_points_to_segment(left.grip, cr.a, cr.b).min()) - cr.radius - r
        if gap < best[0]:
            best = (gap, "gripper", cr.name)
    for cl in left.capsules:
        gap = float(_points_to_segment(right.grip, cl.a, cl.b).min()) - cl.radius - r
        if gap < best[0]:
            best = (gap, cl.name, "gripper")
    diff = left.grip[:, None, :] - right.grip[None, :, :]
    gap = float(np.sqrt(np.einsum("ijk,ijk->ij", diff, diff).min())) - 2.0 * r
    if gap < best[0]:
        best = (gap, "gripper", "gripper")
    return best


def gap_m(q_left, q_right, opening_left_m: float = gripper.FINGER_OPEN_M, opening_right_m: float = gripper.FINGER_OPEN_M) -> tuple[float, str, str]:
    """The gap between the arms' surfaces, metres (negative: they overlap), and the left and right part where it is smallest (grippers fully open unless told)."""
    return _shape_gap(shape("left", q_left, opening_left_m), shape("right", q_right, opening_right_m))


def problem(q_left, q_right, min_gap_m: float = MIN_GAP_M, opening_left_m: float = gripper.FINGER_OPEN_M,
            opening_right_m: float = gripper.FINGER_OPEN_M) -> str | None:
    """A sentence when the arms come closer than `min_gap_m`, else None."""
    gap, left_part, right_part = gap_m(q_left, q_right, opening_left_m, opening_right_m)
    if gap >= min_gap_m:
        return None
    where = f"{-gap * 100:.1f} cm into" if gap < 0 else f"{gap * 100:.1f} cm from"
    return f"left {left_part} is {where} the right {right_part} (the arms must keep {min_gap_m * 100:.0f} cm)"


# -- during a task: the two arms of one station process (log-034) -------------------------------------------------------------------

_lock = threading.Lock()
_latest: dict[str, dict[str, tuple[np.ndarray, float, float]]] = {}   # side -> {"measured" | "target": (logical radians, finger opening m, monotonic time)}


def publish(side: str, kind: str, q_logical_rad, opening_m: float = gripper.FINGER_OPEN_M) -> None:
    """Records an arm's `"measured"` pose or its latest `"target"` (logical radians) and its finger opening for the other arm's check."""
    with _lock:
        _latest.setdefault(side, {})[kind] = (np.array(q_logical_rad, dtype=float), float(opening_m), time.monotonic())


def others(side: str) -> list[tuple[np.ndarray, float]]:
    """The other arm's fresh poses `(logical radians, finger opening)`: measured and target; its rest pose, gripper open, when nothing fresh was heard."""
    other = "right" if side == "left" else "left"
    now = time.monotonic()
    with _lock:
        fresh = [(q.copy(), opening) for q, opening, t in _latest.get(other, {}).values() if now - t <= FRESH_S]
    return fresh or [(np.radians(poses.REST_DEG[other]), gripper.FINGER_OPEN_M)]


def reset() -> None:
    """Forgets every published pose (tests, and a station that restarts its arms)."""
    with _lock:
        _latest.clear()


def _smallest_gap(side: str, q, others: list[ArmShape], opening_m: float = gripper.FINGER_OPEN_M) -> tuple[float, str]:
    """`others`: the other arm's shapes, one per pose (computed once per #cut: the other arm does not move during one check)."""
    mine = shape(side, q, opening_m)
    best, why = np.inf, ""
    for other in others:
        gap, left_part, right_part = _shape_gap(mine, other) if side == "left" else _shape_gap(other, mine)
        if gap < best:
            best, why = gap, f"{side} {left_part if side == 'left' else right_part} {gap * 100:.1f} cm from the other arm's {right_part if side == 'left' else left_part}"
    return best, why


def cut(side: str, prev_logical_rad, q_logical_rad, other_poses: list, min_gap_m: float = MIN_GAP_M,
        opening_m: float = gripper.FINGER_OPEN_M) -> tuple[np.ndarray, str | None]:
    """A target cut back along the straight joint line from `prev` so the arm keeps `min_gap_m` from every pose in `other_poses`, on the whole way there
    (checked every `openarm_guard.SWEEP_STEP_DEG`), not only at the target (log-035).

    A target that keeps the gap comes back unchanged with None. When `prev` itself is already too close, only a target that opens the gap is allowed;
    otherwise the arm stays at `prev`. @return `(pose to send, logical radians; a sentence why it was cut, or None)`.
    """
    prev, q = np.asarray(prev_logical_rad, dtype=float), np.asarray(q_logical_rad, dtype=float)
    other_side = "right" if side == "left" else "left"
    other_poses = [shape(other_side, *_as_pose(pose)) for pose in other_poses]
    gap_q, why = _smallest_gap(side, q, other_poses, opening_m)
    gap_prev, _ = _smallest_gap(side, prev, other_poses, opening_m)
    if gap_prev < min_gap_m:
        return (q, None) if gap_q > gap_prev else (prev, why)
    # the way there (log-035): the first step of the straight joint line that comes too close, then the edge inside that step
    steps = openarm_guard.sweep_steps(prev, q)
    hit = None
    for k in range(1, steps + 1):
        gap_k, why_k = (gap_q, why) if k == steps else _smallest_gap(side, prev + (k / steps) * (q - prev), other_poses, opening_m)
        if gap_k < min_gap_m:
            hit, why = k, (why_k if k == steps else f"{why_k} on the way to the target")
            break
    if hit is None:
        return q, None
    low, high = (hit - 1) / steps, hit / steps   # `low` keeps the gap, `high` does not
    for _ in range(CUT_STEPS):
        mid = (low + high) / 2.0
        if _smallest_gap(side, prev + mid * (q - prev), other_poses, opening_m)[0] >= min_gap_m:
            low = mid
        else:
            high = mid
    return prev + low * (q - prev), why


def way_problem(side: str, prev_logical_rad, q_logical_rad, other_poses: list, min_gap_m: float = MIN_GAP_M,
                opening_m: float = gripper.FINGER_OPEN_M) -> str | None:
    """As `openarm_guard.Guard.way_problem`, against the other arm: the first place the straight line comes within `min_gap_m`, or None (log-040)."""
    prev, q = np.asarray(prev_logical_rad, dtype=float), np.asarray(q_logical_rad, dtype=float)
    other_side = "right" if side == "left" else "left"
    others = [shape(other_side, *_as_pose(pose)) for pose in other_poses]
    steps = openarm_guard.sweep_steps(prev, q)
    leaving = True
    for k in range(steps + 1):
        gap, why = _smallest_gap(side, prev + (k / steps) * (q - prev), others, opening_m)
        if gap < min_gap_m and not leaving:
            return why
        if gap >= min_gap_m:
            leaving = False
    return "the way back never gets clear of the other arm" if leaving else None

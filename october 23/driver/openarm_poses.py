"""The rest pose, the ready pose and the path between them, for each OpenArm follower. Pure data and path maths, no hardware.

The table top is about 30 cm above where the gripper hangs at rest, so an arm that must work on the table first swings back
a little and then rises to a ready pose from which its wrist camera sees the table top. The way there is the recorded
path of the frozen start pose set `kitakyushu-wood-l1-v1` in openarm-orboh (`config/pose_sets/`, frozen 2026-10-20, made for
this table and used for real data collection): its waypoints, in order, and its final pose. Angles are the fork's logical
degrees (zero = arm hanging straight down), which the driver passes to Dream's model unchanged.

The rest pose is the hanging pose read on the arms on 2026-10-23, wrist untwisted, gripper open.
"""

import math

import numpy as np

JOINTS = 7

## Hanging rest pose read through the driver on 2026-10-23.
REST_DEG = {
    "left": [-0.5, -0.8, -4.3, 4.0, 2.1, -0.5, 6.4],
    "right": [0.3, 0.7, -1.4, 2.6, 1.3, -6.5, -3.8],
}

## The recorded path from rest toward the ready pose: swing back, then bend and raise.
WAYPOINTS_DEG = {
    "left": [
        [13.84, -0.86, -1.27, 25.55, 0.74, 1.79, 1.04],
        [13.79, -0.75, -1.27, 25.55, 0.76, 1.79, 4.04],
        [28.48, -3.57, -1.27, 25.68, 4.76, 1.75, -40.92],
        [36.37, -3.64, -1.27, 25.79, 4.76, 4.54, -69.51],
        [42.65, -3.75, -1.27, 33.4, 0.28, 4.59, -85.51],
        [48.05, -3.66, -1.23, 43.39, 0.32, 10.2, -85.38],
        [63.45, -3.62, -1.23, 85.2, 0.78, 10.16, -63.98],
        [63.94, -3.62, -1.23, 94.86, 0.85, 10.42, -63.15],
    ],
    "right": [
        [-6.74, -0.84, 0.12, 22.45, -2.98, -0.92, -9.12],
        [-6.76, -0.84, 0.12, 22.45, -3.0, -0.92, -4.84],
        [-20.31, 0.54, 0.01, 23.19, -2.96, -0.96, 31.88],
        [-47.04, 1.3, 0.85, 65.53, -2.98, -1.14, 41.72],
        [-53.84, 1.3, 3.62, 71.08, -2.98, -1.14, 57.24],
        [-55.15, 1.3, 3.62, 71.08, -2.98, -1.14, 63.34],
        [-58.16, 1.3, 3.62, 85.39, -3.0, -1.14, 63.27],
        [-58.01, 1.23, 3.62, 85.39, 5.68, -1.14, 63.95],
        [-58.01, 1.23, 3.56, 85.39, -0.86, -1.14, 63.95],
    ],
}

## The ready pose: the frozen start pose's final joints (joint 7 as set for the leader's resting wrist).
READY_DEG = {
    "left": [64.42, -3.62, -1.23, 95.91, -2.48, 19.01, 21.6],
    "right": [-59.61, 1.23, 7.53, 87.17, 2.62, -1.23, -15.0],
}


def path_to_ready(side: str) -> list[list[float]]:
    """Rest, then the waypoints in order, then the ready pose.

    @param side `"left"` or `"right"`.
    @return Joint angles in degrees, first point = rest, last = ready.
    """
    return [list(REST_DEG[side]), *map(list, WAYPOINTS_DEG[side]), list(READY_DEG[side])]


def ready_rad(side: str) -> list[float]:
    """The ready pose in radians, for the station file's `home`.

    @param side `"left"` or `"right"`.
    @return Seven angles in radians.
    """
    return [round(math.radians(q), 4) for q in READY_DEG[side]]


def nearest(points: list[list[float]], pose: list[float]) -> tuple[int, float]:
    """The path point closest to `pose`, by the largest joint difference.

    @param points Path points, degrees.
    @param pose The arm's joints, degrees.
    @return `(index, distance)`: the point's index and the largest joint difference to it, degrees.
    """
    distances = [float(np.max(np.abs(np.asarray(p) - np.asarray(pose)))) for p in points]
    index = int(np.argmin(distances))
    return index, distances[index]


def simplify_path(points: list[list[float]], tol_deg: float) -> list[list[float]]:
    """Ramer-Douglas-Peucker in joint space: keeps the ends and the points that matter.

    @param points Joint poses in degrees.
    @param tol_deg Largest accepted distance of a dropped point to the line between kept points, degrees.
    @return A subset of `points` including the first and the last.
    """
    if len(points) < 3:
        return [list(p) for p in points]
    a, b = np.asarray(points[0], dtype=float), np.asarray(points[-1], dtype=float)
    ab = b - a
    denom = float(ab @ ab)
    worst, index = 0.0, 0
    for i in range(1, len(points) - 1):
        p = np.asarray(points[i], dtype=float)
        t = 0.0 if denom == 0.0 else min(max(float((p - a) @ ab) / denom, 0.0), 1.0)
        d = float(np.max(np.abs(p - (a + t * ab))))
        if d > worst:
            worst, index = d, i
    if worst <= tol_deg:
        return [list(points[0]), list(points[-1])]
    left = simplify_path(points[: index + 1], tol_deg)
    right = simplify_path(points[index:], tol_deg)
    return left[:-1] + right


def densify(points: list[list[float]], max_dps: float, dt: float) -> list[list[float]]:
    """Samples a path every `dt` seconds so no joint moves faster than `max_dps` on average, easing in and out of every point.

    @param points Path points, degrees; the arm is assumed to start at the first.
    @param max_dps Peak joint speed target, degrees per second (the easing keeps the peak under 1.5 times it).
    @param dt Seconds between samples.
    @return The samples after the first point, ending exactly on the last point.
    """
    samples: list[list[float]] = []
    for start, end in zip(points, points[1:], strict=False):
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        steps = max(1, math.ceil(1.5 * float(np.max(np.abs(b - a))) / (max_dps * dt)))
        for i in range(1, steps + 1):
            s = i / steps
            samples.append((a + (b - a) * (s * s * (3 - 2 * s))).tolist())
    return samples

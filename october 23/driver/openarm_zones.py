"""Safety zones and forward kinematics for the OpenArm follower arms. Pure Python and numpy: no CAN, no Dream runtime.

A zones file (`~/.dreamlabs/zones.yaml`, written by `workspace/workspace.py apply`) holds, per arm and in that arm's base frame (the
frame of Dream's `openarm2_faas` model: x forward, y left, z up, metres): the painted work area (`reach`, points seen from above),
the danger zones (`keep_out`, boxes and upright capsules) and when and how they were captured. The driver's guard and the workspace tools
read the same file.

The arm's points are the model's own joint frames: elbow (joint 4), wrist pitch (joint 6), wrist (joint 7), the middle of the forearm,
the hand base and the grasp point (tcp). Links count as spheres of #LINK_RADIUS_M (Dream's IK uses 4 cm), plus #CLEARANCE_M of margin.
"""

import dataclasses
import datetime as dt
import importlib.util
import math
import pathlib
from typing import Any

import numpy as np
import yaml

JOINTS = 7
## Dream's collision sphere around every link, and the margin it keeps from a keep-out zone, metres.
LINK_RADIUS_M = 0.04
CLEARANCE_M = 0.02
## Dream's reach disk around each painted point, and the slack the guard allows beyond it, metres.
REACH_RADIUS_M = 0.04
REACH_SLACK_M = 0.02
## A point may be this far outside the allowed region before it counts as outside, metres (the box is stored to 0.1 mm and the arm's own readings are no better).
REGION_TOLERANCE_M = 0.001
## A keep-out zone marked `allow_tcp_touch` (the table) may be touched by the grasp point to this depth, metres.
TOUCH_TOLERANCE_M = 0.005

## The grasp point in the `ee_base_link` frame, from `dream_ik/specs.py` (`openarm2_faas`).
TCP_IN_EE = {"left": (-0.0016, -0.0001, -0.159), "right": (-0.0016, 0.0001, -0.159)}


def _rotation_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    x, y, z = rpy
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rz @ ry @ rx


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    return out


def model_urdf(side: str) -> pathlib.Path:
    """Dream's kinematics-only URDF of the OpenArm v2 arm on `side`, from the installed `dream_ik` package.

    @param side `"left"` or `"right"`.
    @return The file's path.
    @throws FileNotFoundError `dream_ik` is not importable here, or has no such asset.
    """
    spec = importlib.util.find_spec("dream_ik")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("dream_ik is not installed in this Python")
    path = pathlib.Path(next(iter(spec.submodule_search_locations))) / "assets" / f"openarm2_faas_{side}.urdf"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class ArmKinematics:
    """Forward kinematics of one arm of Dream's OpenArm v2 model, in numpy (fast enough for the 250 Hz command loop)."""

    def __init__(self, side: str, urdf: pathlib.Path | None = None) -> None:
        """Reads the arm's seven revolute joints from its URDF.

        @param side `"left"` or `"right"`.
        @param urdf The URDF file; the installed model's by default.
        """
        import xml.etree.ElementTree as ET

        self.side = side
        root = ET.parse(urdf or model_urdf(side)).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        self._origins: list[np.ndarray] = []
        self._axes: list[np.ndarray] = []
        for i in range(1, JOINTS + 1):
            joint = joints[f"openarm_{side}_joint{i}"]
            origin = joint.find("origin")
            frame = np.eye(4)
            frame[:3, :3] = _rotation_rpy(tuple(float(v) for v in origin.get("rpy").split()))
            frame[:3, 3] = [float(v) for v in origin.get("xyz").split()]
            self._origins.append(frame)
            self._axes.append(np.array([float(v) for v in joint.find("axis").get("xyz").split()]))
        self._tcp = np.array(TCP_IN_EE[side])

    def points(self, q_rad: np.ndarray) -> dict[str, np.ndarray]:
        """The arm's named points in its base frame.

        @param q_rad Seven joint angles, radians, in Dream's model.
        @return `elbow`, `wrist_pitch`, `wrist`, `forearm`, `hand` and `tcp`, each `[3]` metres.
        """
        frame = np.eye(4)
        origins: list[np.ndarray] = []
        for i in range(JOINTS):
            frame = frame @ self._origins[i]
            origins.append(frame[:3, 3].copy())
            frame = frame @ _rotation_about(self._axes[i], float(q_rad[i]))
        return {
            "elbow": origins[3],
            "wrist_pitch": origins[5],
            "wrist": origins[6],
            "forearm": (origins[3] + origins[6]) / 2.0,
            "hand": frame[:3, 3].copy(),
            "tcp": (frame @ np.append(self._tcp, 1.0))[:3],
        }

    def tool_pose(self, q_rad: np.ndarray) -> np.ndarray:
        """The grasp point's 4x4 pose in the base frame (for cross-checks against Dream's own FK)."""
        frame = np.eye(4)
        for i in range(JOINTS):
            frame = frame @ self._origins[i] @ _rotation_about(self._axes[i], float(q_rad[i]))
        tool = np.eye(4)
        tool[:3, 3] = self._tcp
        return frame @ tool


# -- geometry ----------------------------------------------------------------------------------------------------------

def distance_to_box(point: np.ndarray, center: np.ndarray, size: np.ndarray) -> float:
    """Signed distance from `point` to an axis-aligned box's surface, metres: positive outside, negative inside (the depth)."""
    offset = np.abs(point - center) - size / 2.0
    outside = float(np.linalg.norm(np.maximum(offset, 0.0)))
    return outside if outside > 0.0 else float(np.max(offset))


def distance_to_capsule(point: np.ndarray, center: np.ndarray, radius: float, height: float) -> float:
    """Distance from `point` to an upright capsule's surface (negative inside), metres: a vertical segment of `height`, centred, plus `radius`."""
    low, high = center[2] - height / 2.0, center[2] + height / 2.0
    nearest = np.array([center[0], center[1], min(max(point[2], low), high)])
    return float(np.linalg.norm(point - nearest)) - radius


def distance_to_reach(xy: np.ndarray, cells: np.ndarray) -> float:
    """Distance from a hand position seen from above to the nearest painted point, metres."""
    return float(np.min(np.linalg.norm(cells - xy, axis=1)))


# -- zones -------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class ArmZones:
    """One arm's zones: the painted work area (None = no check) and the keep-out zones."""

    reach: np.ndarray | None
    keep_out: list[dict[str, Any]]
    region: dict[str, list[float]] | None = None


@dataclasses.dataclass
class Zones:
    """A zones file in force."""

    captured_at: dt.datetime
    source: str
    arms: dict[str, ArmZones]
    raw: dict[str, Any]
    home: dict[str, list[float]] | None = None
    home_path: dict[str, list[list[float]]] | None = None

    def age_hours(self, now: dt.datetime | None = None) -> float:
        """Hours since the zones were captured."""
        now = now or dt.datetime.now(self.captured_at.tzinfo)
        return (now - self.captured_at).total_seconds() / 3600.0

    def violations(self, side: str, points: dict[str, np.ndarray], check_reach: bool = True, check_region: bool = True,
                   region_slack: float = 0.0) -> list[str]:
        """Why an arm pose is not allowed, one text per problem; empty when it is.

        @param side `"left"` or `"right"`.
        @param points The pose's points from #ArmKinematics.points().
        @param check_region Also require every point to be inside the arm's allowed `region` box, when it has one.
        @param region_slack Metres a point may be outside the region before it counts (for judging where an arm measurably is, not a target).
        @param check_reach Also require the grasp point to be over the painted work area (the driver guard does not: Dream clips to it).
        @return The violations: a point inside or too close to a keep-out zone, or the grasp point off the painted area.
        """
        zones = self.arms[side]
        out: list[str] = []
        for name, point in points.items():
            radius = 0.0 if name == "tcp" else LINK_RADIUS_M
            for zone in zones.keep_out:
                center = np.asarray(zone["center"], dtype=float)
                reach = radius + CLEARANCE_M
                if name == "tcp" and zone.get("allow_tcp_touch"):
                    reach = -TOUCH_TOLERANCE_M
                elif name == "tcp":
                    reach = CLEARANCE_M
                if zone["kind"] == "box":
                    distance = distance_to_box(point, center, np.asarray(zone["size"], dtype=float))
                else:
                    distance = distance_to_capsule(point, center, float(zone["radius"]), float(zone["height"]))
                if distance < reach:
                    out.append(f"{side} {name} is {distance * 100:.1f} cm from keep-out zone '{zone.get('name', zone['kind'])}'")
        if check_region and zones.region is not None:
            lo, hi = np.asarray(zones.region["lo"]), np.asarray(zones.region["hi"])
            for name, point in points.items():
                outside = float(np.max(np.maximum(lo - point, point - hi)))
                if outside > REGION_TOLERANCE_M + region_slack:
                    out.append(f"{side} {name} is {outside * 100:.1f} cm outside the allowed region")
        if check_reach and zones.reach is not None:
            miss = distance_to_reach(points["tcp"][:2], zones.reach)
            if miss > REACH_RADIUS_M + REACH_SLACK_M:
                out.append(f"{side} hand is {miss * 100:.1f} cm outside the painted work area")
        return out


def parse_zones(raw: dict[str, Any]) -> Zones:
    """Checks a zones mapping and turns it into #Zones.

    @param raw The loaded YAML.
    @return The zones.
    @throws ValueError The mapping is missing a key or a value is out of shape.
    """
    try:
        captured = dt.datetime.fromisoformat(str(raw["captured_at"]))
        if captured.tzinfo is None:
            captured = captured.astimezone()
        arms: dict[str, ArmZones] = {}
        for side in ("left", "right"):
            section = raw["arms"][side]
            cells = section.get("reach")
            reach = None if not cells else np.asarray(cells, dtype=float).reshape(-1, 2)
            keep_out = []
            for zone in section.get("keep_out") or []:
                kind = zone["kind"]
                if kind not in ("box", "capsule"):
                    raise ValueError(f"keep_out kind {kind!r} must be box or capsule")
                center = [float(v) for v in zone["center"]]
                if len(center) != 3:
                    raise ValueError("keep_out center needs three numbers")
                entry: dict[str, Any] = {"kind": kind, "center": center, "name": zone.get("name", kind),
                                         "allow_tcp_touch": bool(zone.get("allow_tcp_touch", False))}
                if kind == "box":
                    size = [float(v) for v in zone["size"]]
                    if len(size) != 3 or min(size) <= 0:
                        raise ValueError("keep_out box size needs three numbers above 0")
                    entry["size"] = size
                else:
                    entry["radius"], entry["height"] = float(zone["radius"]), float(zone["height"])
                    if entry["radius"] <= 0 or entry["height"] <= 0:
                        raise ValueError("keep_out capsule radius and height must be above 0")
                keep_out.append(entry)
            region = None
            if section.get("region") is not None:
                lo = [float(v) for v in section["region"]["lo"]]
                hi = [float(v) for v in section["region"]["hi"]]
                if len(lo) != 3 or len(hi) != 3 or any(a >= b for a, b in zip(lo, hi, strict=True)):
                    raise ValueError("region needs lo and hi, three numbers each, with lo below hi on every axis")
                region = {"lo": lo, "hi": hi}
            arms[side] = ArmZones(reach, keep_out, region)
        home = None
        if raw.get("home") is not None:
            home = {side: [float(v) for v in raw["home"][side]] for side in ("left", "right")}
            if any(len(h) != JOINTS for h in home.values()):
                raise ValueError(f"home needs {JOINTS} joint angles (radians) per arm")
        home_path = None
        if raw.get("home_path"):
            home_path = {}
            for side, path in raw["home_path"].items():
                points = [[float(v) for v in p] for p in path]
                if side not in ("left", "right") or len(points) < 2 or any(len(p) != JOINTS for p in points):
                    raise ValueError(f"home_path of the {side} arm needs at least two poses of {JOINTS} joint angles (degrees)")
                home_path[side] = points
        return Zones(captured, str(raw.get("source", "?")), arms, raw, home, home_path)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"zones file is missing or has a bad key (region, keep_out, home...): {exc!r}") from exc


def load_zones(path: str | pathlib.Path) -> Zones:
    """Reads and checks a zones file.

    @param path The YAML file.
    @return The zones.
    @throws FileNotFoundError No such file.
    @throws ValueError The file is not valid zones.
    """
    return parse_zones(yaml.safe_load(pathlib.Path(path).read_text()))

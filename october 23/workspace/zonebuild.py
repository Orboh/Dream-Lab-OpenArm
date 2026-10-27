"""Builds the zones file and the station.yaml fragment from captured points. Pure maths and file handling, no hardware.

The zones file is the one source of truth of the daily safety area (schema read by `driver/openarm_zones.py`). Every number is in the
arm's own base frame (Dream's `openarm2_faas` model: x forward, y left, z up), metres.
"""

import dataclasses
import datetime as dt
import pathlib
import shutil
from typing import Any

import numpy as np
import yaml

import openarm_mapping as mapping
import openarm_poses as poses
import openarm_zones as zones

## The painted work area's grid, metres: the same 2 cm Dream's setup wizard uses.
REACH_GRID_M = 0.02
## How thick the table slab is taken to be below its top, metres (covers the desk body next to the arm).
TABLE_THICKNESS_M = 0.30
## A hazard box is the marked points' bounding box grown by this, metres.
HAZARD_MARGIN_M = 0.03
## Two reference points touched again must agree within this, metres.
VERIFY_TOLERANCE_M = 0.015
SIDES = ("left", "right")
## The gentle station settings `apply` writes: a slower start-up ramp and the slow hand speed cap (Dream's own ramp and cap, not the driver's).
GENTLE_STATION = {"ramp_s": 3.0, "ramp_max_rad_s": 0.3, "max_ee_speed_m_s": 0.6}  # 0.6 = Dream's default speed cap (robots/kinematics.py; log-029, user 2026-10-27: no slow-down)
## Written into every bundle so the numbers can be read without this code.
FRAME_NOTE = ("each arm's base frame = the robot body frame shifted by (0, +0.031, 0.698) m for the left arm and (0, -0.031, 0.698) m for the right, "
              "no rotation; x forward, y left, z up; metres")


@dataclasses.dataclass
class ArmCapture:
    """What was captured for one arm."""

    reach: set[tuple[int, int]] = dataclasses.field(default_factory=set)
    table_points: list[tuple[float, float, float]] = dataclasses.field(default_factory=list)
    hazards: dict[str, list[tuple[float, float, float]]] = dataclasses.field(default_factory=dict)
    references: list[tuple[float, float, float]] = dataclasses.field(default_factory=list)


def cell(xy: np.ndarray) -> tuple[int, int]:
    """The grid cell a position seen from above is in."""
    i, j = np.round(np.asarray(xy, dtype=float)[:2] / REACH_GRID_M)
    return int(i), int(j)


def cells_to_points(cells: set[tuple[int, int]]) -> list[list[float]]:
    """Grid cells as sorted `[x, y]` points, metres to the millimetre."""
    return [[round(i * REACH_GRID_M, 3), round(j * REACH_GRID_M, 3)] for i, j in sorted(cells)]


def _bbox(points: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=float)
    return array.min(axis=0), array.max(axis=0)


def table_keep_out(points: list[tuple[float, float, float]], thickness: float = TABLE_THICKNESS_M) -> dict[str, Any]:
    """The table as a keep-out slab: its top at the touched height, its footprint the touched points' bounding box.

    @param points Three or more points touched on the table top, base frame.
    @param thickness Slab thickness below the top, metres.
    @return A keep-out box the grasp point may touch (`allow_tcp_touch`).
    @throws ValueError Fewer than three points.
    """
    if len(points) < 3:
        raise ValueError("touch the table top in at least three points")
    low, high = _bbox(points)
    top = float(np.median([p[2] for p in points]))
    size_x, size_y = max(float(high[0] - low[0]), 0.05), max(float(high[1] - low[1]), 0.05)
    return {
        "name": "table", "kind": "box", "allow_tcp_touch": True,
        "center": [round(float((low[0] + high[0]) / 2), 4), round(float((low[1] + high[1]) / 2), 4), round(top - thickness / 2, 4)],
        "size": [round(size_x, 4), round(size_y, 4), round(thickness, 4)],
    }


def hazard_keep_out(name: str, points: list[tuple[float, float, float]], margin: float = HAZARD_MARGIN_M) -> dict[str, Any]:
    """A danger zone: the marked points' bounding box grown by `margin`.

    @param name The zone's name.
    @param points One or more marked points, base frame.
    @param margin Metres added on every side.
    @return A keep-out box.
    """
    low, high = _bbox(points)
    low, high = low - margin, high + margin
    return {"name": name, "kind": "box", "center": [round(float(v), 4) for v in (low + high) / 2],
            "size": [round(float(v), 4) for v in high - low]}


def build_zones(captures: dict[str, ArmCapture], source: str, operator: str, now: dt.datetime,
                homes: dict[str, list[float]] | None = None) -> dict[str, Any]:
    """The zones file's contents.

    @param captures Per side, what was captured.
    @param source `"follower"` or `"leader"`.
    @param operator Who captured it.
    @param now When (timezone-aware).
    @param homes Per side, the home pose in radians (seven angles): where the arm starts and ends a run; the ready pose when None.
    @return The mapping to write as YAML; `verify.status` starts as `unverified`.
    @throws ValueError An arm has too few table points.
    """
    arms: dict[str, Any] = {}
    refs: dict[str, Any] = {}
    for side in SIDES:
        c = captures[side]
        table = table_keep_out(c.table_points)
        keep_out = [table] + [hazard_keep_out(name, pts) for name, pts in c.hazards.items()]
        arms[side] = {"table_z": round(float(np.median([p[2] for p in c.table_points])), 4),
                      "reach": cells_to_points(c.reach) or None, "keep_out": keep_out}
        refs[side] = [[round(float(v), 4) for v in p] for p in c.references]
    return {"version": 1, "captured_at": now.isoformat(timespec="seconds"), "source": source, "operator": operator,
            "frame": FRAME_NOTE, "home": homes or {side: poses.ready_rad(side) for side in SIDES}, "arms": arms, "references": refs,
            "verify": {"status": "unverified", "worst_m": None, "checked_at": None}}


def home_adjustments(raw: dict[str, Any]) -> dict[str, list[tuple[int, float]]]:
    """Per side, the joints of the bundle's home that the station file moves (degrees) to keep them inside Dream's joint range: `{}` when none moves."""
    out = {}
    for side in SIDES:
        if raw.get("home"):
            moved = [(joint, deg) for joint, deg in mapping.clip_to_model_limits(side, mapping.to_model(side, raw["home"][side]))[1] if abs(deg) >= 0.01]  # below 0.01 deg is the bundle's 4-decimal rounding of a home the editor already fitted
            if moved:
                out[side] = moved
    return out


def _keep_out_of(raw: dict[str, Any], side: str) -> list[dict[str, Any]]:
    """The arm's danger zones in the exact form Dream's station file takes (`kind`, `center`, `size` for a box; `radius`, `height` for a capsule)."""
    keep_out = []
    for zone in raw["arms"][side]["keep_out"]:
        if zone["kind"] == "box":
            keep_out.append({"kind": "box", "center": list(zone["center"]), "size": list(zone["size"])})
        else:
            keep_out.append({"kind": "capsule", "center": list(zone["center"]), "radius": zone["radius"], "height": zone["height"]})
    return keep_out


def station_fragment(raw: dict[str, Any], homes: dict[str, tuple[float, float]], danger_only: bool | None = None) -> dict[str, dict[str, Any]]:
    """The keys to write into each arm's section of station.yaml: `reach` (painted cells plus the home hand) and `keep_out`.

    A **danger-only** bundle (`mode: danger_only`, or `danger_only=True`) writes `keep_out` and `home` and nothing else: no `reach` and no `region`, so Dream's hand can go anywhere except into the
    danger zones (Dream: "None when the arm's section has no `reach`: the hand can go anywhere"). Nothing about the task or the table's usable area is handed to the model.

    Dream's keep_out takes exactly `kind`, `center`, `size` for a box. `region` is not written: the rest and ready poses have the hand below the table top.
    """
    out: dict[str, dict[str, Any]] = {}
    danger = (raw.get("mode") == "danger_only") if danger_only is None else danger_only
    for side in SIDES:
        if danger:
            out[side] = {"keep_out": _keep_out_of(raw, side)}
            if raw.get("home"):
                out[side]["home"] = [float(v) for v in mapping.clip_to_model_limits(side, mapping.to_model(side, raw["home"][side]))[0]]
            continue
        cells = {cell(np.array(p)) for p in (raw["arms"][side].get("reach") or [])}
        cells.add(cell(np.array(homes[side])))
        out[side] = {"reach": cells_to_points(cells), "keep_out": _keep_out_of(raw, side)}
        if raw.get("home"):  # the bundle's home is the arm's logical angles; the station file holds Dream's model angles (the same under the identity mapping)
            out[side]["home"] = [float(v) for v in mapping.clip_to_model_limits(side, mapping.to_model(side, raw["home"][side]))[0]]
        if raw["arms"][side].get("region"):
            out[side]["region"] = {"lo": list(raw["arms"][side]["region"]["lo"]), "hi": list(raw["arms"][side]["region"]["hi"])}
    return out


def pose_violations(parsed: zones.Zones, side: str, named_poses: dict[str, list[float]], kin: zones.ArmKinematics) -> list[str]:
    """Which of the given poses (joint degrees, Dream's model) the zones would refuse.

    @param parsed The zones.
    @param side `"left"` or `"right"`.
    @param named_poses Name to seven joint angles in degrees.
    @param kin That arm's kinematics.
    @return One text per problem, naming the pose; empty when all are allowed.
    """
    out: list[str] = []
    for name, pose in named_poses.items():
        # the allowed region holds the work poses (home, live); rest and the raise/park path hang below the table and are exempt from it
        in_region = name == "home" or name.startswith("live")
        for text in parsed.violations(side, kin.points(np.radians(pose)), check_reach=False, check_region=in_region):
            out.append(f"{name}: {text}")
    return out


def home_poses(side: str, steps: int = 6) -> dict[str, list[float]]:
    """The poses the zones must leave free for each arm: rest, ready and samples along the path between them."""
    path = poses.path_to_ready(side)
    named = {"rest": path[0], "ready": path[-1]}
    for index, point in enumerate(path[1:-1], start=1):
        named[f"path {index}"] = point
    return named


def compare_references(stored: list[tuple[float, float, float]], measured: list[tuple[float, float, float]], tol: float = VERIFY_TOLERANCE_M) -> dict[str, Any]:
    """Compares reference points touched again with the stored ones.

    @param stored The points stored with the zones.
    @param measured The same points touched now, in the same order.
    @param tol Largest accepted distance, metres.
    @return `{"ok", "worst_m", "distances_m"}`.
    @throws ValueError The lists differ in length or are empty.
    """
    if len(stored) != len(measured) or not stored:
        raise ValueError(f"touch the same {len(stored)} reference points as when the zones were captured")
    distances = [float(np.linalg.norm(np.asarray(a) - np.asarray(b))) for a, b in zip(stored, measured, strict=True)]
    return {"ok": max(distances) <= tol, "worst_m": max(distances), "distances_m": distances}


def apply_to_station(station: pathlib.Path, fragment: dict[str, dict[str, Any]], tuning: dict[str, float] | None = None) -> pathlib.Path:
    """Writes the fragment into station.yaml's `follower_l` / `follower_r`, after saving a timestamped backup beside it.

    @param station The station file.
    @param fragment From #station_fragment().
    @param tuning Extra top-level settings to write (for example #GENTLE_STATION); with it `custom.driver` is also pointed at the `Arm` class of the same driver file.
    @return The backup's path.
    """
    text = station.read_text()
    backup = station.with_name(f"{station.name}.{dt.datetime.now():%Y%m%d-%H%M%S}.bak")
    shutil.copy2(station, backup)
    header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    data = yaml.safe_load(text)
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        if "reach" in fragment[side]:
            data[key]["reach"] = fragment[side]["reach"]
        else:
            data[key].pop("reach", None)  # danger-only: no painted area, the hand can go anywhere outside the danger zones
        data[key]["keep_out"] = fragment[side]["keep_out"]
        if "home" in fragment[side]:
            data[key]["home"] = fragment[side]["home"]
        if "region" in fragment[side]:
            data[key]["region"] = fragment[side]["region"]
        else:
            data[key].pop("region", None)
    if tuning:
        data.update(tuning)
        driver = (data.get("custom") or {}).get("driver")
        if isinstance(driver, str) and driver.rsplit(":", 1)[-1] != "Arm":
            data["custom"]["driver"] = driver.rsplit(":", 1)[0] + ":Arm"
    station.write_text(header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
    return backup


def merge_leader(base: dict[str, Any], extra: dict[str, ArmCapture], operator: str, now: dt.datetime) -> dict[str, Any]:
    """Adds what was painted and marked with the leader arms to a follower-captured zones file.

    The table and the reference points stay the follower's (ground truth); the painted cells are added to the work area and each marked
    hazard becomes a new keep-out box (a name already in use gets a numeric suffix).

    @param base The follower's zones mapping.
    @param extra Per side, the leader session's capture (only `reach` and `hazards` are used).
    @param operator Who captured it.
    @param now When.
    @return A new zones mapping.
    """
    import copy

    merged = copy.deepcopy(base)
    for side in SIDES:
        section = merged["arms"][side]
        cells = {cell(np.array(p)) for p in (section.get("reach") or [])} | extra[side].reach
        section["reach"] = cells_to_points(cells) or None
        names = {z["name"] for z in section["keep_out"]}
        for name, points in extra[side].hazards.items():
            unique, n = name, 2
            while unique in names:
                unique, n = f"{name} {n}", n + 1
            names.add(unique)
            section["keep_out"].append(hazard_keep_out(unique, points))
    merged["source"] = "follower+leader"
    merged["operator"] = operator
    merged["captured_at"] = now.isoformat(timespec="seconds")
    return merged


PATH_SAMPLE_S = 0.1   # the planned move is checked every 0.1 s
PARK_MAX_S = 45.0     # a raise or park along the home path must take less than this: `dreamcli robot down` gives up after 60 s


def planned_path(path_deg: list[list[float]]):
    """The driver's own smooth move through `path_deg` (degrees, seven joints, the first is where the arm is): its samples every #PATH_SAMPLE_S and its duration in seconds."""
    import openarm_motion as motion
    import openarm_v2 as drv

    plan = motion.smooth_path(np.asarray(path_deg, dtype=float), drv.LIMITS, drv.SAMPLE_DT_S)
    return plan.samples[:: max(1, int(round(PATH_SAMPLE_S / drv.SAMPLE_DT_S)))], float(plan.duration)


def path_seconds(path_deg: list[list[float]]) -> float:
    """How long the driver's smooth move along `path_deg` takes, seconds."""
    return planned_path(path_deg)[1]


def taught_path_slack_cm(raw: dict[str, Any], side: str, path_deg: list[list[float]]) -> float:
    """The smallest clearance left along the driver's planned move through `path_deg`, in cm beyond what is required: the grasp point must keep 2 cm from a zone, every link sphere 6 cm.

    Negative means a zone is too close somewhere (by that many cm); positive is room to spare. No zones: a large number.
    """
    import copy

    strict = copy.deepcopy(raw)
    kin = zones.ArmKinematics(side, logical=True)
    samples, _ = planned_path(path_deg)
    slack = 1e3
    for zone in strict["arms"][side]["keep_out"]:
        center = np.asarray(zone["center"], dtype=float)
        for q in samples:
            for name, point in kin.points(np.radians(q)).items():
                required = zones.CLEARANCE_M + (0.0 if name == "tcp" else zones.LINK_RADIUS_M)
                if zone["kind"] == "box":
                    distance = zones.distance_to_box(point, center, np.asarray(zone["size"], dtype=float))
                else:
                    distance = zones.distance_to_capsule(point, center, float(zone["radius"]), float(zone["height"]))
                slack = min(slack, (distance - required) * 100)
    return float(slack)


def taught_path_problems(raw: dict[str, Any], side: str, path_deg: list[list[float]] | None = None, limit: int = 3) -> list[str]:
    """Where the move along an arm's home path (the one the driver follows to raise it and to park it) comes too close to a danger zone.

    Stricter than the guard for Dream's targets: the grasp point may NOT touch a zone here (it keeps 2 cm, links 6 cm), because this is the driver's own move, not a grasp.
    Checked on the driver's planned smooth move (its corners rounded), not on the straight segments between the waypoints.

    @param raw The zones bundle. @param path_deg The path to check; the bundle's own `home_path` for `side` by default.
    @return At most `limit` texts naming the place on the path and the zone; empty when the whole move keeps clear.
    """
    path = path_deg if path_deg is not None else (raw.get("home_path") or {}).get(side)
    if not path:
        return []
    import copy

    strict = copy.deepcopy(raw)
    for arm in strict["arms"].values():
        arm["reach"], arm["region"] = None, None
        for zone in arm["keep_out"]:
            zone.pop("allow_tcp_touch", None)
    parsed = zones.parse_zones(strict)
    kin = zones.ArmKinematics(side, logical=True)
    samples, _ = planned_path(path)
    out: list[str] = []
    for index, q in enumerate(samples):
        found = parsed.violations(side, kin.points(np.radians(q)), check_reach=False, check_region=False)
        if found:
            out.append(f"{side} taught path, {index * PATH_SAMPLE_S:.1f} s into the move: {found[0]}")
            if len(out) >= limit:
                break
    return out


def free_pose_problems(raw: dict[str, Any]) -> list[str]:
    """Violations of the rest pose, the ready pose, the bundle's home pose and the path between them against the bundle's zones (empty when all are free)."""
    parsed = zones.parse_zones(raw)
    out: list[str] = []
    for side in SIDES:
        named = home_poses(side)
        if parsed.home:
            named["home"] = list(np.degrees(parsed.home[side]))
        out += [f"{side} {text}" for text in pose_violations(parsed, side, named, zones.ArmKinematics(side, logical=True))]
    return out

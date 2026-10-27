"""The zone editor's state, with no GUI: the table, the danger boxes, the work area and the home pose of each arm.

Everything is edited in the ROBOT BODY frame (the frame of the URDF and its meshes: z up, x forward, y left). Each arm's own base frame, the one the zones file and Dream use,
is that frame shifted by #BASE_OFFSET with no rotation, so `to_bundle` converts every zone for each arm.
"""

import dataclasses
import datetime as dt
from typing import Any

import numpy as np

import openarm_mapping as mapping
import openarm_poses as poses
import openarm_zones as zones
import zonebuild as zb

SIDES = ("left", "right")
## Each arm's base frame in the body frame, metres (checked against Dream's FK with a Kabsch fit: residual 0.0 m).
HOME_ROOM_M = 0.05  # free space around the home pose inside the allowed area: home is only the pose from which the cameras see the table, not a limit
BASE_OFFSET = {"left": np.array([0.0, 0.031, 0.698]), "right": np.array([0.0, -0.031, 0.698])}
## How far the danger box's top may be lowered below the real table surface, cm. 0 (the default) puts the box top at the surface: the safest. Dream keeps the arm's 4 cm safety balls 2 cm off the box, so the arm
## stays 6 cm (ball centre) from it; lowering the box by d cm lets the balls come d cm lower. Past #SAFE_LOWER_CM a ball can dip into the real table, so the editor warns.
LOWER_MAX_CM = 4.0
SAFE_LOWER_CM = 2.0
## A shortened home path may leave this much less clearance than the recorded one (cm) when the recorded one was already short of the required clearance; a path that was clear must stay clear.
SHORTCUT_SLACK_LOSS_CM = 0.5
## A NEGATIVE lowering raises the box top above the real surface: an extra safety margin for the arm over the table (added 2026-10-24 after the arm seemed to drag on the table in the first live run).
RAISE_MAX_CM = 15.0
## The table top is this far above the hand hanging at rest (user, 2026-10-23).
TABLE_ABOVE_HANGING_HAND_M = 0.30


def to_base(side: str, xyz) -> list[float]:
    """A body-frame point in `side`'s base frame."""
    return [float(v) for v in np.asarray(xyz, dtype=float) - BASE_OFFSET[side]]


def to_world(side: str, xyz) -> list[float]:
    """A point in `side`'s base frame in the body frame."""
    return [float(v) for v in np.asarray(xyz, dtype=float) + BASE_OFFSET[side]]


@dataclasses.dataclass
class Table:
    """The table as a slab: top height, footprint and thickness (body frame, metres)."""

    top_z: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    thickness: float = zb.TABLE_THICKNESS_M
    lower_cm: float = 0.0   # the danger box's top sits this far below the real surface (0: at the surface; negative: above it, an extra margin); see #LOWER_MAX_CM and #RAISE_MAX_CM

    def lower_warning(self) -> str | None:
        """Text when the box top is lowered so far that a safety ball (4 cm radius) can dip into the real table, else None."""
        if self.lower_cm > SAFE_LOWER_CM + 1e-9:
            return (f"the danger box top is {self.lower_cm:g} cm below the table surface: the arm's 4 cm safety balls can dip {self.lower_cm - SAFE_LOWER_CM:g} cm into the real table "
                    f"(up to {SAFE_LOWER_CM:g} cm is the limit that still keeps them out)")
        return None

    def keep_out(self) -> dict[str, Any]:
        """The slab as a keep-out box the grasp point may touch (its top is #lower_cm below the surface)."""
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        top = self.top_z - self.lower_cm / 100.0
        return {"name": "table", "kind": "box", "allow_tcp_touch": True,
                "center": [round((x0 + x1) / 2, 4), round((y0 + y1) / 2, 4), round(top - self.thickness / 2, 4)],
                "size": [round(x1 - x0, 4), round(y1 - y0, 4), round(self.thickness, 4)]}


def table_from_measurements(front_cm: float, top_above_base_plate_cm: float, depth_cm: float, width_cm: float, centre_y_cm: float = 0.0, thickness_m: float = zb.TABLE_THICKNESS_M,
                            lower_cm: float = 0.0) -> Table:
    """The table from tape measurements (body frame: x forward from the robot's centre line, y left, z up from the floor-level base plate).

    @param front_cm How far in front of the robot's centre line the near edge is.
    @param top_above_base_plate_cm The table top's height above the base plate the arms stand on (the arm's own base joint is #BASE_OFFSET z = 69.8 cm above it).
    @param depth_cm How far the table runs away from the robot. @param width_cm Its width. @param centre_y_cm Where its centre is sideways (0: on the centre line).
    @param lower_cm How far the danger box's top is lowered below the real surface (0 keeps it at the surface; at most #LOWER_MAX_CM). Negative values raise it: an extra safety margin, at most #RAISE_MAX_CM.
    """
    if depth_cm <= 0 or width_cm <= 0:
        raise ValueError("depth and width must be above 0")
    if not -RAISE_MAX_CM <= lower_cm <= LOWER_MAX_CM:
        raise ValueError(f"lower_cm must be between {-RAISE_MAX_CM:g} (raised) and {LOWER_MAX_CM:g} (lowered)")
    front, half = front_cm / 100.0, width_cm / 200.0
    centre = centre_y_cm / 100.0
    return Table(top_z=top_above_base_plate_cm / 100.0, x_range=(front, front + depth_cm / 100.0), y_range=(centre - half, centre + half), thickness=thickness_m, lower_cm=lower_cm)


@dataclasses.dataclass
class DangerBox:
    """A forbidden axis-aligned box (body frame) for one arm or both."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    side: str = "both"


@dataclasses.dataclass
class ArmState:
    """One arm's home pose (radians) and painted work area (grid cells)."""

    home: list[float]
    reach: set[tuple[int, int]] = dataclasses.field(default_factory=set)
    home_source: str = "default"                           # default | teach | rest | ready | live | bundle (not saved; the page's checklist uses it)
    region: dict[str, list[float]] | None = None            # the allowed box (arm base frame), from the teach sweep
    home_path: list[list[float]] | None = None              # degrees: rest -> home, as the user moved the arm


class EditorState:
    """What the editor holds, and the operations the GUI calls."""

    def __init__(self, table: Table, boxes: list[DangerBox], arms: dict[str, ArmState]) -> None:
        self.table, self.boxes, self.arms = table, boxes, arms
        self.park_tolerance_deg = 0.0  # > 0: the saved home path is simplified to this tolerance (degrees) when that keeps it clear of the zones: a shorter raise and park
        self.last_fit: dict[str, list[tuple[int, float]]] = {}   # what the last to_bundle() moved to fit each home into Dream's range: [(joint, degrees)]
        self.last_shortening: dict[str, dict] = {}   # what the last to_bundle() did to each arm's home path: seconds and points before and after
        self.danger_only = False  # True: the bundle names only the danger zones (no painted work area, no allowed region): nothing about the task is handed to the model
        self.table_set = False   # True once the user placed the table (or it came from a saved bundle)
        self.references: dict[str, list[tuple[float, float, float]]] = {side: [] for side in SIDES}   # table points touched with the follower, for the daily verify

    @classmethod
    def default(cls) -> "EditorState":
        """A table 30 cm above the hanging hand, 35-80 cm in front of the body, 60 cm wide; both arms at the ready pose, no work area."""
        hanging = zones.ArmKinematics("left", logical=True).points(np.zeros(7))["tcp"][2]
        table = Table(top_z=float(hanging + BASE_OFFSET["left"][2] + TABLE_ABOVE_HANGING_HAND_M), x_range=(0.35, 0.80), y_range=(-0.30, 0.30))
        return cls(table, [], {side: ArmState(home=list(poses.ready_rad(side))) for side in SIDES})

    # -- danger boxes ---------------------------------------------------------------------------------------------------
    def _box(self, name: str) -> DangerBox:
        for box in self.boxes:
            if box.name == name:
                return box
        raise KeyError(name)

    def add_box(self, name: str, center, size, side: str = "both") -> None:
        """Adds a danger box (body frame). @throws ValueError The name is taken or is 'table', the size is not positive, or the side is unknown."""
        if name == "table" or any(b.name == name for b in self.boxes):
            raise ValueError(f"the name '{name}' is already used (and 'table' is reserved)")
        if len(size) != 3 or min(size) <= 0:
            raise ValueError("size needs three numbers above 0")
        if side not in (*SIDES, "both"):
            raise ValueError("side must be left, right or both")
        self.boxes.append(DangerBox(name, tuple(float(v) for v in center), tuple(float(v) for v in size), side))

    def move_box(self, name: str, center) -> None:
        self._box(name).center = tuple(float(v) for v in center)

    def resize_box(self, name: str, size) -> None:
        if min(size) <= 0:
            raise ValueError("size needs three numbers above 0")
        self._box(name).size = tuple(float(v) for v in size)

    def remove_box(self, name: str) -> None:
        self.boxes.remove(self._box(name))

    # -- work area and home ---------------------------------------------------------------------------------------------
    def set_work_area(self, side: str, x_range, y_range) -> None:
        """Fills the 2 cm grid over a rectangle seen from above (the arm's own base-frame x, y); None, None clears it."""
        if x_range is None or y_range is None:
            self.arms[side].reach = set()
            return
        xs = np.arange(round(x_range[0] / zb.REACH_GRID_M), round(x_range[1] / zb.REACH_GRID_M) + 1)
        ys = np.arange(round(y_range[0] / zb.REACH_GRID_M), round(y_range[1] / zb.REACH_GRID_M) + 1)
        self.arms[side].reach = {(int(i), int(j)) for i in xs for j in ys}

    def add_reference(self, side: str, xyz) -> None:
        """Records a reference point (arm base frame, metres): a place on the table touched with the follower, to be touched again by `verify`."""
        self.references[side].append(tuple(round(float(v), 4) for v in xyz))

    def use_rest(self, side: str) -> None:
        self.arms[side].home = [float(v) for v in np.radians(poses.REST_DEG[side])]
        self.arms[side].home_source = "rest"

    def use_ready(self, side: str) -> None:
        self.arms[side].home = list(poses.ready_rad(side))
        self.arms[side].home_source = "ready"

    def use_pose(self, side: str, q_rad) -> None:
        """Takes a pose (for example the live arm's) as the home. @throws ValueError Not seven angles."""
        if len(q_rad) != zones.JOINTS:
            raise ValueError(f"home needs {zones.JOINTS} angles")
        self.arms[side].home = [float(v) for v in q_rad]
        self.arms[side].home_source = "live"

    # -- output -------------------------------------------------------------------------------------------------------
    def _arm_keep_out(self, side: str) -> list[dict[str, Any]]:
        out = []
        for zone in [self.table.keep_out(), *({"name": b.name, "kind": "box", "center": list(b.center), "size": list(b.size)}
                                                for b in self.boxes if b.side in (side, "both"))]:
            zone = dict(zone)
            zone["center"] = [round(v, 4) for v in to_base(side, zone["center"])]
            out.append(zone)
        return out

    def fit_home_to_dream(self, side: str) -> list[tuple[int, float]]:
        """Moves a home that is past Dream's joint range just inside it, in our angles, and the end of its path with it (log-031).

        The station file can only hold a home inside Dream's range, and the driver follows a taught path only when it ends within
        `mapping.HOME_MATCH_DEG` of that home: fitting both here keeps them identical. @return `[(joint number, degrees moved)]`, empty when it fits.
        """
        arm = self.arms[side]
        fitted, moved = mapping.fit_logical(side, arm.home)
        if moved:
            arm.home = [float(v) for v in fitted]
            if arm.home_path:
                end, fitted_deg = list(arm.home_path[-1]), np.degrees(fitted)
                for joint, _ in moved:
                    end[joint - 1] = float(fitted_deg[joint - 1])
                arm.home_path = [list(p) for p in arm.home_path[:-1]] + [end]
        return moved

    def to_bundle(self, operator: str, now: dt.datetime) -> dict[str, Any]:
        """The zones bundle in the documented output format (arm base frames); `source` is `editor`. Each home is fitted into Dream's range first (#fit_home_to_dream)."""
        self.last_fit = {side: moved for side in SIDES if (moved := self.fit_home_to_dream(side))}
        arms = {side: {"table_z": round(float(self.table.top_z - BASE_OFFSET[side][2]), 4),
                       "reach": zb.cells_to_points(self.arms[side].reach) or None, "keep_out": self._arm_keep_out(side)} for side in SIDES}
        if self.danger_only:
            for side in SIDES:
                arms[side].pop("reach")
        for side in SIDES:
            if self.arms[side].region and not self.danger_only:
                arms[side]["region"] = {"lo": list(self.arms[side].region["lo"]), "hi": list(self.arms[side].region["hi"])}
        paths = {side: self.arms[side].home_path for side in SIDES if self.arms[side].home_path}
        if self.danger_only:  # the known rest-to-ready path is the home path of an arm whose home is the ready pose
            paths.update({side: poses.path_to_ready(side) for side in SIDES if not paths.get(side) and self.arms[side].home_source == "ready"})
        bundle = {"version": 1, "captured_at": now.isoformat(timespec="seconds"), "source": "editor", "operator": operator, "frame": zb.FRAME_NOTE,
                "home": {side: [round(v, 4) for v in self.arms[side].home] for side in SIDES}, "arms": arms,
                "references": {side: [list(p) for p in self.references[side]] for side in SIDES}, "verify": {"status": "unverified", "worst_m": None, "checked_at": None}}
        if self.table.lower_cm:
            bundle["table_danger_lowered_cm"] = float(self.table.lower_cm)  # the box top is this far below the real surface
        if self.danger_only:
            bundle["mode"] = "danger_only"
        self.last_shortening = {}
        if self.park_tolerance_deg > 0:
            for side, path in list(paths.items()):
                shorter = poses.simplify_path(path, self.park_tolerance_deg)
                if len(shorter) >= len(path):
                    continue
                slack_before, slack_after = zb.taught_path_slack_cm(bundle, side, path), zb.taught_path_slack_cm(bundle, side, shorter)
                worse = slack_after < min(slack_before, 0.0) - SHORTCUT_SLACK_LOSS_CM   # a shortcut may not come closer to a zone than the recorded path did (a clear path must stay clear)
                self.last_shortening[side] = {"shortened": not worse, "points_before": len(path), "points_after": len(shorter), "slack_cm_before": round(slack_before, 1), "slack_cm_after": round(slack_after, 1),
                                              "seconds_before": round(zb.path_seconds(path), 1), "seconds_after": round(zb.path_seconds(shorter), 1)}
                if not worse:
                    paths[side] = shorter
        if paths:
            bundle["home_path"] = {side: [list(map(float, p)) for p in path] for side, path in paths.items()}
        return bundle

    @classmethod
    def from_bundle(cls, raw: dict[str, Any]) -> "EditorState":
        """Rebuilds the editor from a bundle (the left arm's frame gives the table; boxes seen by both arms merge into one)."""
        left = raw["arms"]["left"]
        table_zone = next(z for z in left["keep_out"] if z.get("name") == "table")
        c, s = to_world("left", table_zone["center"]), table_zone["size"]
        lower = float(raw.get("table_danger_lowered_cm", 0.0))
        table = Table(top_z=c[2] + s[2] / 2 + lower / 100.0, x_range=(c[0] - s[0] / 2, c[0] + s[0] / 2), y_range=(c[1] - s[1] / 2, c[1] + s[1] / 2), thickness=s[2], lower_cm=lower)
        boxes: dict[str, DangerBox] = {}
        for side in SIDES:
            for zone in raw["arms"][side]["keep_out"]:
                if zone.get("name") == "table" or zone["kind"] != "box":
                    continue
                center = tuple(round(v, 4) for v in to_world(side, zone["center"]))
                existing = boxes.get(zone["name"])
                if existing and existing.center == center and existing.size == tuple(zone["size"]):
                    existing.side = "both"
                else:
                    boxes[zone["name"]] = DangerBox(zone["name"], center, tuple(zone["size"]), side)
        arms = {side: ArmState(home=list(raw["home"][side]) if raw.get("home") else list(poses.ready_rad(side)),
                               reach={zb.cell(np.array(p)) for p in (raw["arms"][side].get("reach") or [])},
                               region=raw["arms"][side].get("region"),
                               home_path=(raw.get("home_path") or {}).get(side)) for side in SIDES}
        for arm in arms.values():
            arm.home_source = "bundle"
        state = cls(table, list(boxes.values()), arms)
        state.danger_only = raw.get("mode") == "danger_only"
        state.table_set = True
        state.references = {side: [tuple(p) for p in (raw.get("references") or {}).get(side, [])] for side in SIDES}
        return state

    def fit_home_into_region(self) -> list[str]:
        """Widens an arm's allowed region so its home pose has #HOME_ROOM_M of free space on every side.

        The home is only the pose the cameras see the table from: it must not sit against a wall of the allowed area, or the arm
        could hardly move away from it. A side that already has the room is left alone.

        @return The sides whose region was widened (arms without a region are left alone).
        """
        widened = []
        for side in SIDES:
            region = self.arms[side].region
            if not region:
                continue
            pts = np.array(list(zones.ArmKinematics(side, logical=True).points(np.asarray(self.arms[side].home, dtype=float)).values()))
            old_lo, old_hi = np.asarray(region["lo"], dtype=float), np.asarray(region["hi"], dtype=float)
            need_lo, need_hi = pts.min(axis=0) - HOME_ROOM_M, pts.max(axis=0) + HOME_ROOM_M
            lo = np.where(need_lo < old_lo - 1e-4, np.floor(need_lo * 1e4) / 1e4, old_lo)   # only a side that has to move moves (rounded outward to 0.1 mm)
            hi = np.where(need_hi > old_hi + 1e-4, np.ceil(need_hi * 1e4) / 1e4, old_hi)
            if not (np.allclose(lo, old_lo) and np.allclose(hi, old_hi)):
                self.arms[side].region = {"lo": [float(v) for v in lo], "hi": [float(v) for v in hi]}
                widened.append(side)
        return widened

    def reach_over_table(self, side: str) -> tuple[int, int]:
        """How much of an arm's painted work area lies over the table seen from above: `(cells over the table, all cells)`; `(0, 0)` when nothing is painted."""
        cells = self.arms[side].reach
        if not cells:
            return 0, 0
        (x0, x1), (y0, y1) = self.table.x_range, self.table.y_range
        base = BASE_OFFSET[side]
        over = sum(1 for i, j in cells if x0 <= i * zb.REACH_GRID_M + base[0] <= x1 and y0 <= j * zb.REACH_GRID_M + base[1] <= y1)
        return over, len(cells)

    def path_warnings(self) -> list[str]:
        """Where the home path as it would be saved (each arm's rest-to-home move) comes too close to a danger zone; empty when it keeps clear (grasp point 2 cm, links 6 cm)."""
        raw = self.to_bundle("editor", dt.datetime.now(dt.timezone.utc))
        out: list[str] = []
        for side in SIDES:
            out += zb.taught_path_problems(raw, side, limit=1)
        return out

    def region_overlap_cm(self) -> float:
        """How far the two arms' allowed areas run into each other (cm along the body's y axis), 0 when they do not or one is missing."""
        boxes = []
        for side in SIDES:
            region = self.arms[side].region
            if not region:
                return 0.0
            boxes.append((np.asarray(region["lo"], dtype=float) + BASE_OFFSET[side], np.asarray(region["hi"], dtype=float) + BASE_OFFSET[side]))
        depth = np.minimum(boxes[0][1], boxes[1][1]) - np.maximum(boxes[0][0], boxes[1][0])
        return round(float(depth[1]) * 100, 1) if bool(np.all(depth > 0)) else 0.0

    # -- clashes --------------------------------------------------------------------------------------------------------
    def clashes(self, live: dict[str, np.ndarray] | None = None) -> list[str]:
        """Every way the zones block the rest, ready and home poses, the path between them, or a live pose (radians per side)."""
        now = dt.datetime.now(dt.timezone.utc)
        parsed = zones.parse_zones(self.to_bundle("editor", now))
        out: list[str] = []
        for side in SIDES:
            kin = zones.ArmKinematics(side, logical=True)
            named = zb.home_poses(side)
            named["home"] = list(np.degrees(self.arms[side].home))
            out += [f"{side} {t}" for t in zb.pose_violations(parsed, side, named, kin)]
            if live and side in live:
                out += [f"{side} live: {t}" for t in parsed.violations(side, kin.points(np.asarray(live[side], dtype=float)), check_reach=False)]
        return out

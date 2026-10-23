"""The zone editor's state, with no GUI: the table, the danger boxes, the work area and the home pose of each arm.

Everything is edited in the ROBOT BODY frame (the frame of the URDF and its meshes: z up, x forward, y left). Each arm's own base frame, the one the zones file and Dream use,
is that frame shifted by #BASE_OFFSET with no rotation, so `to_bundle` converts every zone for each arm.
"""

import dataclasses
import datetime as dt
from typing import Any

import numpy as np

import openarm_poses as poses
import openarm_zones as zones
import zonebuild as zb

SIDES = ("left", "right")
## Each arm's base frame in the body frame, metres (checked against Dream's FK with a Kabsch fit: residual 0.0 m).
BASE_OFFSET = {"left": np.array([0.0, 0.031, 0.698]), "right": np.array([0.0, -0.031, 0.698])}
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

    def keep_out(self) -> dict[str, Any]:
        """The slab as a keep-out box the grasp point may touch."""
        (x0, x1), (y0, y1) = self.x_range, self.y_range
        return {"name": "table", "kind": "box", "allow_tcp_touch": True,
                "center": [round((x0 + x1) / 2, 4), round((y0 + y1) / 2, 4), round(self.top_z - self.thickness / 2, 4)],
                "size": [round(x1 - x0, 4), round(y1 - y0, 4), round(self.thickness, 4)]}


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
        self.table_set = False   # True once the user placed the table (or it came from a saved bundle)

    @classmethod
    def default(cls) -> "EditorState":
        """A table 30 cm above the hanging hand, 35-80 cm in front of the body, 60 cm wide; both arms at the ready pose, no work area."""
        hanging = zones.ArmKinematics("left").points(np.zeros(7))["tcp"][2]
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

    def to_bundle(self, operator: str, now: dt.datetime) -> dict[str, Any]:
        """The zones bundle in the documented output format (arm base frames); `source` is `editor`."""
        arms = {side: {"table_z": round(float(self.table.top_z - BASE_OFFSET[side][2]), 4),
                       "reach": zb.cells_to_points(self.arms[side].reach) or None, "keep_out": self._arm_keep_out(side)} for side in SIDES}
        for side in SIDES:
            if self.arms[side].region:
                arms[side]["region"] = {"lo": list(self.arms[side].region["lo"]), "hi": list(self.arms[side].region["hi"])}
        paths = {side: self.arms[side].home_path for side in SIDES if self.arms[side].home_path}
        bundle = {"version": 1, "captured_at": now.isoformat(timespec="seconds"), "source": "editor", "operator": operator, "frame": zb.FRAME_NOTE,
                "home": {side: [round(v, 4) for v in self.arms[side].home] for side in SIDES}, "arms": arms,
                "references": {side: [] for side in SIDES}, "verify": {"status": "unverified", "worst_m": None, "checked_at": None}}
        if paths:
            bundle["home_path"] = {side: [list(map(float, p)) for p in path] for side, path in paths.items()}
        return bundle

    @classmethod
    def from_bundle(cls, raw: dict[str, Any]) -> "EditorState":
        """Rebuilds the editor from a bundle (the left arm's frame gives the table; boxes seen by both arms merge into one)."""
        left = raw["arms"]["left"]
        table_zone = next(z for z in left["keep_out"] if z.get("name") == "table")
        c, s = to_world("left", table_zone["center"]), table_zone["size"]
        table = Table(top_z=c[2] + s[2] / 2, x_range=(c[0] - s[0] / 2, c[0] + s[0] / 2), y_range=(c[1] - s[1] / 2, c[1] + s[1] / 2), thickness=s[2])
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
        state.table_set = True
        return state

    # -- clashes --------------------------------------------------------------------------------------------------------
    def clashes(self, live: dict[str, np.ndarray] | None = None) -> list[str]:
        """Every way the zones block the rest, ready and home poses, the path between them, or a live pose (radians per side)."""
        now = dt.datetime.now(dt.timezone.utc)
        parsed = zones.parse_zones(self.to_bundle("editor", now))
        out: list[str] = []
        for side in SIDES:
            kin = zones.ArmKinematics(side)
            named = zb.home_poses(side)
            named["home"] = list(np.degrees(self.arms[side].home))
            out += [f"{side} {t}" for t in zb.pose_violations(parsed, side, named, kin)]
            if live and side in live:
                out += [f"{side} live: {t}" for t in parsed.violations(side, kin.points(np.asarray(live[side], dtype=float)), check_reach=False)]
        return out

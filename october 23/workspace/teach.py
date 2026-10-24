"""The teach session: what is learned while you move the follower with the leader arm. Pure maths, no hardware and no GUI.

- `observe(side, q)` is fed the follower's joints (radians) all the time. While SWEEP is on, every pose adds the arm's points to the swept set.
- The swept set gives the allowed REGION (bounding box of every arm point plus a margin) and the work area (`reach` cells) where the hand has been.
- HOME is the pose you choose (the arm looks at the work space from it). If PATH recording was started at the rest pose, the way you moved the arm from rest to home is kept
  (simplified) as the home path: the driver raises and parks the arm along it, so it never cuts across the table.
"""

import numpy as np

import editor as ed
import openarm_poses as poses
import openarm_zones as zones
import zonebuild as zb

MIN_SAMPLES = 10
## Path recording only starts when the arm is within this of the rest pose, degrees (largest joint difference).
REST_TOLERANCE_DEG = 12.0
## A path sample is kept only when some joint moved this much since the last kept one, degrees.
PATH_STEP_DEG = 1.5
## The recorded path is simplified until no dropped point is farther than this from the straight line between kept points, degrees.
PATH_TOLERANCE_DEG = 4.0


def simplify_path(points: list[list[float]], tol_deg: float = PATH_TOLERANCE_DEG) -> list[list[float]]:
    """Ramer-Douglas-Peucker in joint space (see `openarm_poses.simplify_path`), with the teach tool's default tolerance."""
    return poses.simplify_path(points, tol_deg)


def _fill_hull(xy: np.ndarray) -> set[tuple[int, int]]:
    """The 2 cm grid cells whose centre lies inside the convex hull of `xy` (empty for a degenerate hull)."""
    pts = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in xy})
    if len(pts) < 3:
        return set()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return set()
    grid = zb.REACH_GRID_M
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    cells = set()
    for i in range(int(np.floor(min(xs) / grid)), int(np.ceil(max(xs) / grid)) + 1):
        for j in range(int(np.floor(min(ys) / grid)), int(np.ceil(max(ys) / grid)) + 1):
            c = (i * grid, j * grid)
            if all(cross(hull[k], hull[(k + 1) % len(hull)], c) >= -1e-9 for k in range(len(hull))):
                cells.add((i, j))
    return cells


class TeachSession:
    """The learned home, swept region, work area and path of both arms."""

    def __init__(self, margin_m: float = 0.0) -> None:
        """@param margin_m Metres added around the recorded points when the region is made (0: exactly the points you showed)."""
        self.margin = margin_m
        self._kin = {side: zones.ArmKinematics(side, logical=True) for side in ed.SIDES}
        self._sweeping = {side: False for side in ed.SIDES}
        self._points: dict[str, list[np.ndarray]] = {side: [] for side in ed.SIDES}
        self._tcp: dict[str, list[np.ndarray]] = {side: [] for side in ed.SIDES}
        self._marks: dict[str, list[tuple[str, list[float]]]] = {side: [] for side in ed.SIDES}
        self._path: dict[str, list[list[float]] | None] = {side: None for side in ed.SIDES}
        self._home: dict[str, np.ndarray | None] = {side: None for side in ed.SIDES}
        self._finished_path: dict[str, list[list[float]] | None] = {side: None for side in ed.SIDES}
        self._prefix: dict[str, list[list[float]]] = {side: [] for side in ed.SIDES}   # the known rest -> ready path kept in front of a taught segment
        # automatic tracking: since the arm last stood at rest or ready: the kept prefix and the poses recorded after it (None: it never did)
        self._auto: dict[str, dict | None] = {side: None for side in ed.SIDES}
        self._recorded: dict[str, list[dict]] = {side: [] for side in ed.SIDES}      # single points recorded on purpose: {"pts": {name: xyz}}
        self._touched = {side: False for side in ed.SIDES}                            # something was recorded or cleared: apply_to may overwrite the state

    # -- feeding -----------------------------------------------------------------------------------------------------
    def observe(self, side: str, q_rad: np.ndarray) -> None:
        """One follower pose (radians, seven joints)."""
        q = np.asarray(q_rad, dtype=float)
        if self._sweeping[side]:
            pts = self._kin[side].points(q)
            self._points[side].extend(pts.values())
            self._tcp[side].append(pts["tcp"])
        path = self._path[side]
        deg = np.degrees(q).tolist()
        if path is not None:
            if not path or float(np.max(np.abs(np.asarray(deg) - np.asarray(path[-1])))) >= PATH_STEP_DEG:
                path.append(deg)
        self._track(side, deg)

    def _track(self, side: str, deg: list[float]) -> None:
        """Automatic path tracking: every dwell at the rest or ready pose restarts the record; poses after it are kept (downsampled)."""
        from_rest = float(np.max(np.abs(np.asarray(deg) - np.asarray(poses.REST_DEG[side]))))
        from_ready = float(np.max(np.abs(np.asarray(deg) - np.asarray(poses.READY_DEG[side]))))
        if from_rest <= REST_TOLERANCE_DEG:
            self._auto[side] = {"prefix": [], "buf": []}
        elif from_ready <= REST_TOLERANCE_DEG:
            self._auto[side] = {"prefix": poses.path_to_ready(side), "buf": []}
        elif self._auto[side] is not None:
            buf = self._auto[side]["buf"]
            if not buf or float(np.max(np.abs(np.asarray(deg) - np.asarray(buf[-1])))) >= PATH_STEP_DEG:
                buf.append(deg)

    # -- sweep -------------------------------------------------------------------------------------------------------
    def set_sweep(self, side: str, on: bool) -> None:
        self._sweeping[side] = bool(on)

    def sweeping(self, side: str) -> bool:
        return self._sweeping[side]

    def clear_sweep(self, side: str) -> None:
        self._points[side], self._tcp[side] = [], []
        self._touched[side] = True

    def sweep_count(self, side: str) -> int:
        """How many poses have been swept."""
        return len(self._tcp[side])

    def trail(self, side: str) -> np.ndarray:
        """The swept hand positions, `[n, 3]`, arm base frame."""
        return np.array(self._tcp[side]).reshape(-1, 3)

    # -- recorded points (the safe place is made from them) ---------------------------------------------------------------
    def add_point(self, side: str, q_rad: np.ndarray) -> None:
        """Records the follower's pose now as one extreme point."""
        self._recorded[side].append({"pts": self._kin[side].points(np.asarray(q_rad, dtype=float))})
        self._touched[side] = True

    def undo_point(self, side: str) -> None:
        """Removes the last recorded point (nothing happens if there is none)."""
        if self._recorded[side]:
            self._recorded[side].pop()
            self._touched[side] = True

    def clear_points(self, side: str) -> None:
        self._recorded[side] = []
        self._touched[side] = True

    def point_count(self, side: str) -> int:
        return len(self._recorded[side])

    def point_list(self, side: str) -> list[list[float]]:
        """The recorded hand positions, arm base frame, in the order they were recorded."""
        return [[float(v) for v in p["pts"]["tcp"]] for p in self._recorded[side]]

    def region(self, side: str) -> dict[str, list[float]] | None:
        """The bounding box of every arm point of the recorded poses (two or more) and of a sweep (#MIN_SAMPLES or more), plus the margin; None before that."""
        sources: list[np.ndarray] = []
        if self.sweep_count(side) >= MIN_SAMPLES:
            sources += self._points[side]
        if self.point_count(side) >= 2:
            sources += [p for rec in self._recorded[side] for p in rec["pts"].values()]
        if not sources:
            return None
        pts = np.array(sources)
        return {"lo": [float(np.floor((v) * 1e4) / 1e4) for v in pts.min(axis=0) - self.margin],
                "hi": [float(np.ceil((v) * 1e4) / 1e4) for v in pts.max(axis=0) + self.margin]}

    def reach_cells(self, side: str) -> set[tuple[int, int]]:
        """The 2 cm cells the hand has been over, plus the area inside the polygon through three or more recorded hand positions."""
        cells = {zb.cell(p) for p in self._tcp[side]}
        xy = np.array([p["pts"]["tcp"][:2] for p in self._recorded[side]])
        if len(xy) >= 3:
            cells |= _fill_hull(xy)
        cells |= {zb.cell(p) for p in xy}
        return cells

    def reach_holes(self, side: str, max_gap_m: float = 0.15) -> int:
        """How many work-area cells are farther than `max_gap_m` (metres, seen from above) from every hand position that was actually visited (recorded points, swept trail).

        A polygon through a few far-apart points marks everything inside as reachable; a cell this far from anything the hand visited is space nobody showed to be safe.
        """
        visited = [np.asarray(p[:2], dtype=float) for p in self.point_list(side)] + [np.asarray(p[:2], dtype=float) for p in self._tcp[side]]
        if not visited:
            return 0
        visited_xy = np.array(visited)
        cells = np.array([[i * zb.REACH_GRID_M, j * zb.REACH_GRID_M] for i, j in self.reach_cells(side)])
        if not len(cells):
            return 0
        gaps = np.min(np.linalg.norm(cells[:, None, :] - visited_xy[None, :, :], axis=2), axis=1)
        return int(np.sum(gaps > max_gap_m))

    # -- marks -------------------------------------------------------------------------------------------------------
    def mark(self, side: str, label: str, q_rad: np.ndarray) -> None:
        """Remembers where the hand is now under a label (for example 'far edge')."""
        self._marks[side].append((label, [float(v) for v in self._kin[side].points(np.asarray(q_rad, dtype=float))["tcp"]]))

    def marks(self, side: str) -> list[tuple[str, list[float]]]:
        return list(self._marks[side])

    # -- home and path -----------------------------------------------------------------------------------------------
    def start_path(self, side: str, q_rad: np.ndarray) -> None:
        """Starts recording the way to the home you are about to choose.

        From the rest pose the whole way is recorded. From the old READY pose only the new segment is recorded and the built-in rest-to-ready path is kept in front of it.

        @throws ValueError The arm is at neither the rest pose nor the ready pose.
        """
        deg = np.degrees(np.asarray(q_rad, dtype=float))
        from_rest = float(np.max(np.abs(deg - np.asarray(poses.REST_DEG[side]))))
        from_ready = float(np.max(np.abs(deg - np.asarray(poses.READY_DEG[side]))))
        if from_rest <= REST_TOLERANCE_DEG:
            self._prefix[side] = []
        elif from_ready <= REST_TOLERANCE_DEG:
            self._prefix[side] = poses.path_to_ready(side)
        else:
            raise ValueError(f"the {side} arm is {min(from_rest, from_ready):.0f} deg from the rest or the ready pose: put it at the rest (hanging) or the ready pose before recording the path")
        self._path[side] = []
        self.observe(side, q_rad)

    def _build_path(self, side: str, prefix: list[list[float]], recorded: list[list[float]], home: list[float]) -> list[list[float]]:
        """rest ... home from a kept built-in prefix (not simplified) or from rest, plus the simplified recorded poses."""
        if prefix:
            taught = simplify_path([list(prefix[-1]), *recorded, home])
            path = [list(p) for p in prefix] + taught[1:]
        else:
            path = simplify_path([list(poses.REST_DEG[side]), *recorded, home])
            path[0] = list(poses.REST_DEG[side])
        path[-1] = home
        return path

    def finish_path(self, side: str, home_rad: np.ndarray) -> list[list[float]]:
        """Ends a MANUAL recording at `home_rad`; returns the path in degrees, first = rest, last = home."""
        recorded = self._path[side] or []
        self._path[side] = None
        home = np.degrees(np.asarray(home_rad, dtype=float)).tolist()
        self._finished_path[side] = self._build_path(side, self._prefix[side], recorded, home)
        return self._finished_path[side]

    def has_path(self, side: str) -> bool:
        """Whether a path from rest to the chosen home exists."""
        return self._finished_path[side] is not None

    def set_home(self, side: str, q_rad: np.ndarray) -> list[list[float]] | None:
        """Takes `q_rad` as the home and returns the path from rest, or None when there is none.

        A manual recording (start_path) ends here. Otherwise the automatic record is used: the poses since the arm last stood at the rest or the ready pose.
        """
        q = np.asarray(q_rad, dtype=float)
        self._home[side] = q.copy()
        if self._path[side] is not None:
            return self.finish_path(side, q)
        auto = self._auto[side]
        if auto is None:
            self._finished_path[side] = None
            return None
        self._finished_path[side] = self._build_path(side, auto["prefix"], auto["buf"], np.degrees(q).tolist())
        return self._finished_path[side]

    def home(self, side: str) -> np.ndarray | None:
        return self._home[side]

    def recording_path(self, side: str) -> bool:
        return self._path[side] is not None

    # -- result ------------------------------------------------------------------------------------------------------
    def apply_to(self, state: ed.EditorState) -> None:
        """Writes what was learned into the editor state: home, home path, and (for an arm with sweeps or points) its region and work area."""
        for side in ed.SIDES:
            arm = state.arms[side]
            if self._home[side] is not None:
                arm.home = [float(v) for v in self._home[side]]
            if self._touched[side] or self.sweep_count(side) or self.point_count(side):
                arm.region = self.region(side)
                arm.reach = set(self.reach_cells(side))
            if self._finished_path[side] is not None:
                arm.home_path = self._finished_path[side]

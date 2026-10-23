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


class TeachSession:
    """The learned home, swept region, work area and path of both arms."""

    def __init__(self, margin_m: float = 0.01) -> None:
        """@param margin_m Metres added around the swept points when the region is made."""
        self.margin = margin_m
        self._kin = {side: zones.ArmKinematics(side) for side in ed.SIDES}
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

    def sweep_count(self, side: str) -> int:
        """How many poses have been swept."""
        return len(self._tcp[side])

    def trail(self, side: str) -> np.ndarray:
        """The swept hand positions, `[n, 3]`, arm base frame."""
        return np.array(self._tcp[side]).reshape(-1, 3)

    def region(self, side: str) -> dict[str, list[float]] | None:
        """The bounding box of every swept arm point plus the margin, or None before #MIN_SAMPLES poses."""
        if self.sweep_count(side) < MIN_SAMPLES:
            return None
        pts = np.array(self._points[side])
        return {"lo": [round(float(v), 4) for v in pts.min(axis=0) - self.margin], "hi": [round(float(v), 4) for v in pts.max(axis=0) + self.margin]}

    def reach_cells(self, side: str) -> set[tuple[int, int]]:
        """The 2 cm cells the hand has been over."""
        return {zb.cell(p) for p in self._tcp[side]}

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
        """Writes what was learned into the editor state: home, region, work area and home path (only what exists)."""
        for side in ed.SIDES:
            arm = state.arms[side]
            if self._home[side] is not None:
                arm.home = [float(v) for v in self._home[side]]
            region = self.region(side)
            if region is not None:
                arm.region = region
            cells = self.reach_cells(side)
            if cells:
                arm.reach = set(cells)
            if self._finished_path[side] is not None:
                arm.home_path = self._finished_path[side]

"""The visual zone editor: the real OpenArm model (URDF meshes) in a browser with the table, danger boxes, work area and home poses on top.

`RobotModel` loads `urdf/openarm_bimanual.urdf` with its meshes and poses it from the fork's joint angles (the v10 URDF's own convention). `EditorPanel` holds the GUI
(table, danger boxes with draggable handles, work-area rectangles, home choices, Save and Export) and redraws the zones whenever the state changes. All the logic of the
panel is in methods the buttons call, so it is tested without a browser; the pure state is in `editor.py`.
"""

import datetime as dt
import pathlib
from collections.abc import Callable

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

import cards
import editor as ed
import openarm_poses as poses
import openarm_zones as zones
import outputs
import teach as th
import zonebuild as zb

HERE = pathlib.Path(__file__).resolve().parent
URDF = HERE.parent / "urdf" / "openarm_bimanual.urdf"
MESH_ROOT = pathlib.Path("~/projects/openarm-orboh/openarm_description").expanduser()
BLUE, RED, AMBER, GREEN, GREY = (60, 120, 220), (220, 50, 47), (255, 191, 0), (60, 180, 75), (150, 150, 150)
JOINTS = 7


class RobotModel:
    """The real robot drawn from the URDF, posed by the fork's logical joint angles (radians, seven per arm)."""

    def __init__(self, server: viser.ViserServer, urdf_path: pathlib.Path = URDF, mesh_root: pathlib.Path = MESH_ROOT, detail: str = "visual") -> None:
        """@param detail `"visual"` (full meshes) or `"collision"` (simplified, lighter)."""
        def handler(fname: str) -> str:  # yourdfpy calls it with the keyword `fname`
            return str(mesh_root / fname.split("openarm_description/", 1)[-1])

        urdf = yourdfpy.URDF.load(str(urdf_path), filename_handler=handler, build_scene_graph=True, load_meshes=True,
                                  build_collision_scene_graph=True, load_collision_meshes=True)
        visual = detail == "visual"
        self._model = ViserUrdf(server, urdf, root_node_name="/robot", load_meshes=visual, load_collision_meshes=not visual)
        self._names = list(self._model.get_actuated_joint_names())
        self.cfg = {name: 0.0 for name in self._names}

    def pose(self, side: str, q_rad) -> None:
        """Poses one arm. @param q_rad Seven joint angles, radians (logical, the fork's convention)."""
        for i in range(JOINTS):
            self.cfg[f"openarm_{side}_joint{i + 1}"] = float(q_rad[i])
        self._model.update_cfg(np.array([self.cfg[n] for n in self._names]))


class EditorPanel:
    """The editor's GUI and scene. Buttons and fields call the public methods below."""

    def __init__(self, server: viser.ViserServer, state: ed.EditorState, out_dir: pathlib.Path, operator: str, robot: RobotModel | None = None,
                 read_q: Callable[[str], np.ndarray] | None = None, station_keys: pathlib.Path | None = None,
                 cameras: list[dict[str, str]] | None = None) -> None:
        """@param server The viser server. @param state The editor state. @param out_dir Where Save writes. @param operator Who is editing.
        @param robot The drawn robot (None in tests). @param read_q Live joints per side (radians), or None. @param station_keys Where Export writes.
        @param cameras RealSense name/serial pairs for the switchable feeds in the Home card, or None."""
        self.server, self.state, self.out_dir, self.operator = server, state, pathlib.Path(out_dir), operator
        self.robot, self.read_q, self.cameras = robot, read_q, cameras
        self.cams = None
        self.station_keys = station_keys or self.out_dir / "station_keys.yaml"
        self.selected: str | None = None
        self.pose_view = "live" if read_q is not None else "home"
        self.teach = th.TeachSession()
        self.live_q: dict[str, np.ndarray] = {}
        self.receiving = {side: False for side in ed.SIDES}
        self.kin = {side: zones.ArmKinematics(side) for side in ed.SIDES}
        self.hanging_z = float(self.kin["left"].points(np.zeros(7))["tcp"][2] + ed.BASE_OFFSET["left"][2])
        self.last_saved: tuple[pathlib.Path, pathlib.Path] | None = None
        self._nodes: list = []
        self._busy = False
        self._playing = False
        self._path_nodes: list = []
        self._build_gui()
        self.redraw()

    # -- GUI ------------------------------------------------------------------------------------------------------------
    def _build_gui(self) -> None:
        g, t = self.server.gui, self.state.table
        g.configure_theme(control_width="large", show_logo=False, show_share_button=False, brand_color=(31, 157, 85))
        try:
            g.main_panel.dock_right()
        except Exception:  # older viser: the theme alone is enough
            pass
        self.status = g.add_html(cards.status_strip(self.receiving, None))
        self.out_msg = g.add_markdown("")
        tabs = g.add_tab_group()
        self.tab_labels = ["1 Position", "2 Home", "3 Danger area", "4 Save"]
        # -- 1 Position -------------------------------------------------------------------------------------------------
        with tabs.add_tab(self.tab_labels[0]):
            self.pos_cards = {side: g.add_html(cards.position_card(side, None, None, None, fresh=False)) for side in ed.SIDES}
        # -- 2 Home -----------------------------------------------------------------------------------------------------
        with tabs.add_tab(self.tab_labels[1]):
            g.add_markdown("Move the **leader** until the camera sees the work space, then press the button for that arm. The way from rest is recorded for you.")
            self.home_cards, self.b_home = {}, {}
            for side in ed.SIDES:
                self.home_cards[side] = g.add_html(cards.home_card(side, False, False))
                self.b_home[side] = g.add_button(f"Set {side.upper()} home here", color="green")
                self.b_home[side].on_click(lambda _e, s=side: self.set_home_clicked(s))
            g.add_markdown("**See the move first:** plays the recorded way from rest to home on the 3D robot (nothing moves for real).")
            self.b_preview = {}
            for label, sides in (("LEFT", ("left",)), ("RIGHT", ("right",)), ("both arms", ed.SIDES)):
                b = g.add_button(f"Preview {label}: rest to home", color="gray")
                b.on_click(lambda _e, ss=sides: self.play_path(list(ss)))
                self.b_preview[label] = b
            if self.cameras is not None:
                import viewer

                self.cams = viewer.CameraPanel(self.server, self.cameras)
        # -- 3 Danger area ----------------------------------------------------------------------------------------------
        with tabs.add_tab(self.tab_labels[2]):
            with g.add_folder("Allowed area: show the farthest safe points"):
                g.add_markdown("Move the leader to a farthest point the arm may safely reach (down to just above the table), press **Record this point**. Do this for every corner and edge. The safe place is made from your points (green).")
                self.record_side = g.add_button_group("Arm", ("left", "right", "both"))
                self.record_side.value = "both"
                self.b_point = g.add_button("Record this point", color="green")
                self.b_undo = g.add_button("Undo last point", color="gray")
                self.b_clear = g.add_button("Clear all points", color="gray")
                self.record_info = g.add_markdown("")
                self.b_point.on_click(lambda _e: self.record_point_clicked())
                self.b_undo.on_click(lambda _e: self.undo_point_clicked())
                self.b_clear.on_click(lambda _e: self.clear_clicked())
            with g.add_folder("Table (cm)"):
                self.f_height_cm = g.add_number("top is this far above the hanging hand", (t.top_z - self.hanging_z) * 100, step=1.0)
                self.f_front_cm = g.add_number("front edge, forward from the body", t.x_range[0] * 100, step=1.0)
                self.f_depth_cm = g.add_number("depth (away from you)", (t.x_range[1] - t.x_range[0]) * 100, min=5.0, step=1.0)
                self.f_width_cm = g.add_number("width (left to right)", (t.y_range[1] - t.y_range[0]) * 100, min=5.0, step=1.0)
                for f in (self.f_height_cm, self.f_front_cm, self.f_depth_cm, self.f_width_cm):
                    f.on_update(lambda _e: self.table_cards_changed())
            with g.add_folder("Danger boxes"):
                self.b_add = g.add_button("Add a danger box", color="red")
                self.f_select = g.add_dropdown("box", ("(none)",), initial_value="(none)")
                self.f_size_cm = g.add_vector3("size (cm)", (10.0, 10.0, 10.0), step=1.0)
                self.f_side = g.add_dropdown("applies to", ("both", "left", "right"), initial_value="both")
                self.b_delete = g.add_button("Delete this box", color="gray")
                g.add_markdown("Drag the arrows in the 3D view to move the box.")
                self.b_add.on_click(lambda _e: self.add_box_clicked())
                self.b_delete.on_click(lambda _e: self.delete_clicked())
                self.f_select.on_update(lambda _e: self.select_box(self.f_select.value))
                for f in (self.f_size_cm, self.f_side):
                    f.on_update(lambda _e: self._box_fields_changed())
        # -- 4 Save -----------------------------------------------------------------------------------------------------
        with tabs.add_tab(self.tab_labels[3]):
            self.check_html = g.add_html("")
            self.b_save = g.add_button("Save bundle + layout picture", color="green")
            self.b_export = g.add_button("Export station keys", color="gray")
            self.b_save.on_click(lambda _e: self.save_clicked())
            self.b_export.on_click(lambda _e: self.export_clicked())
            g.add_markdown("Reference points: touch a spot on the table with the follower, press the button. Do it twice per arm; `verify` checks them every day.")
            for side in ed.SIDES:
                b_ref = g.add_button(f"Record reference point: {side.upper()} hand here", color="gray")
                b_ref.on_click(lambda _e, s=side: self.record_reference_clicked(s))
        # -- Advanced (rarely needed) -----------------------------------------------------------------------------------
        with g.add_folder("Advanced", expand_by_default=False):
            self.f_operator = g.add_text("operator", self.operator)
            self.b_record = g.add_button("Start recording continuously (sweep)", color="gray")
            self.b_record.on_click(lambda _e: self.record_toggle_clicked())
            self.f_margin = g.add_number("margin around the sweep (cm)", self.teach.margin * 100, min=0.0, step=0.5)
            self.f_margin.on_update(lambda _e: setattr(self.teach, "margin", float(self.f_margin.value) / 100.0))
            self.f_pose = g.add_dropdown("show robot at", ("live", "home", "rest", "ready"), initial_value=self.pose_view)
            self.f_pose.on_update(lambda _e: self.set_pose_view(self.f_pose.value))
            self.f_name = g.add_text("new danger box name", "box 1")
            self.f_center = g.add_vector3("selected box centre (m, body frame)", (0.0, 0.0, 0.0), step=0.01)
            self.f_center.on_update(lambda _e: self._box_fields_changed())
            self.f_mark = g.add_text("mark label", "far edge")
            self.teach_info = g.add_markdown("")
            for side in ed.SIDES:
                b_mark = g.add_button(f"Mark {side} hand here", color="gray")
                b_mark.on_click(lambda _e, s=side: self.mark_clicked(s))
                b_path = g.add_button(f"{side}: record path manually (arm at rest)", color="gray")
                b_path.on_click(lambda _e, s=side: self.start_path_clicked(s))
                rest = g.add_button(f"{side} home = rest", color="gray")
                ready = g.add_button(f"{side} home = ready", color="gray")
                rest.on_click(lambda _e, s=side: self.use_home(s, "rest"))
                ready.on_click(lambda _e, s=side: self.use_home(s, "ready"))
            self.f_area: dict[str, tuple] = {}
            for side in ed.SIDES:
                x = g.add_vector2(f"{side}: work area x from / to (m)", (0.40, 0.70), step=0.02)
                y = g.add_vector2(f"{side}: work area y from / to (m)", (-0.20, 0.20), step=0.02)
                fill, clear = g.add_button(f"{side}: fill the rectangle", color="gray"), g.add_button(f"{side}: clear work area", color="gray")
                fill.on_click(lambda _e, s=side: self.fill_area(s))
                clear.on_click(lambda _e, s=side: self.clear_area(s))
                self.f_area[side] = (x, y)
            b_apply = g.add_button("Apply the sweep now", color="gray")
            b_apply.on_click(lambda _e: self.apply_sweep_clicked())
        self.gizmo_table = self.server.scene.add_transform_controls("/gizmo_table", scale=0.12, disable_rotations=True, visible=True)
        self.gizmo_table.on_update(lambda _e: self._table_gizmo_moved())
        self.gizmo_box = self.server.scene.add_transform_controls("/gizmo_box", scale=0.12, disable_rotations=True, visible=False)
        self.gizmo_box.on_update(lambda _e: self._box_gizmo_moved())

    def _path_for(self, side: str) -> list[list[float]] | None:
        """The path from rest to the arm's home to preview: the recorded one, or the built-in rest-to-ready path when home is the ready pose."""
        arm = self.state.arms[side]
        if arm.home_path:
            return arm.home_path
        if arm.home_source == "ready":
            return poses.path_to_ready(side)
        return None

    def play_path(self, sides: list[str], blocking: bool = False, realtime: bool = True, speed_dps: float = 25.0, fps: float = 25.0) -> None:
        """Plays the way from rest to home on the 3D robot at the speed the real arm would use (nothing real moves).

        @param sides The arms to play, together (their frames are interleaved).
        @param blocking Wait until it has finished (tests); otherwise it runs on its own thread.
        @param realtime Sleep between frames (False in tests).
        """
        if self.robot is None:
            return
        paths = {s: self._path_for(s) for s in sides}
        missing = [s for s, p in paths.items() if p is None]
        if missing:
            self.out_msg.content = f"No path recorded for the {' and '.join(missing)} arm: move the leader from the rest or ready pose and press Set home again."
            return
        if self._playing:
            self.out_msg.content = "A preview is already playing."
            return
        frames = {}
        for s, path in paths.items():
            samples = [list(path[0])] + poses.densify(path, speed_dps, 1.0 / fps)
            frames[s] = [np.radians(p) for p in samples]
        count = max(len(f) for f in frames.values())
        self._show_hand_paths({s: frames[s] for s in frames})

        def run() -> None:
            import time

            self._playing, previous = True, self.pose_view
            self.pose_view = "preview"
            try:
                for i in range(count):
                    for s in sides:
                        self.robot.pose(s, frames[s][min(i, len(frames[s]) - 1)])
                    if realtime:
                        time.sleep(1.0 / fps)
            finally:
                self.pose_view, self._playing = previous, False
                self.out_msg.content = f"Played the way from rest to home for the {' and '.join(sides)} arm ({count / fps:.0f} s at {speed_dps:g} deg/s)."

        if blocking:
            run()
        else:
            import threading

            threading.Thread(target=run, daemon=True).start()

    def _show_hand_paths(self, frames: dict[str, list[np.ndarray]]) -> None:
        """Draws the hand's way (the orange line) for the previewed arm(s) in the 3D view."""
        for node in self._path_nodes:
            try:
                node.remove()
            except Exception:
                pass
        self._path_nodes = []
        for side, qs in frames.items():
            pts = np.array([self.kin[side].points(q)["tcp"] + ed.BASE_OFFSET[side] for q in qs[:: max(1, len(qs) // 80)]])
            if len(pts) >= 2:
                self._path_nodes.append(self.server.scene.add_line_segments(f"/zones/hand_path_{side}", np.stack([pts[:-1], pts[1:]], axis=1), (255, 120, 0), thickness=0.008))

    def _apply_teach(self) -> list[str]:
        """Writes what the teach session learned into the state, then widens each allowed box to hold its arm's home pose; returns the arms widened."""
        self.teach.apply_to(self.state)
        return self.state.fit_home_into_region()

    def _guard(self, fn: Callable[[], None]) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            fn()
        finally:
            self._busy = False

    # -- table ---------------------------------------------------------------------------------------------------------
    def set_table(self, top_z: float, x_range, y_range, thickness: float) -> None:
        """Sets the table numbers (metres, body frame), marks the table as placed, and redraws."""
        t = self.state.table
        t.top_z, t.x_range, t.y_range, t.thickness = float(top_z), tuple(map(float, x_range)), tuple(map(float, y_range)), float(thickness)
        self.state.table_set = True
        self.redraw()

    def table_cards_changed(self) -> None:
        """The table card's four numbers (cm) changed: height above the hanging hand, front edge, depth, width (kept centred where it is)."""
        def apply() -> None:
            t = self.state.table
            front, depth, width = self.f_front_cm.value / 100.0, max(self.f_depth_cm.value, 5.0) / 100.0, max(self.f_width_cm.value, 5.0) / 100.0
            cy = (t.y_range[0] + t.y_range[1]) / 2
            self.set_table(self.hanging_z + self.f_height_cm.value / 100.0, (front, front + depth), (cy - width / 2, cy + width / 2), t.thickness)
        self._guard(apply)

    def _table_gizmo_moved(self) -> None:
        def move() -> None:
            t = self.state.table
            cx, cy, cz = (float(v) for v in self.gizmo_table.position)
            half_x, half_y = (t.x_range[1] - t.x_range[0]) / 2, (t.y_range[1] - t.y_range[0]) / 2
            self.set_table(cz, (cx - half_x, cx + half_x), (cy - half_y, cy + half_y), t.thickness)
        self._guard(move)

    # -- danger boxes ---------------------------------------------------------------------------------------------------
    def add_box_clicked(self) -> None:
        """Adds a 10 cm box above the table centre with the typed name (a numbered name if empty or taken)."""
        name = (self.f_name.value or "").strip() or f"box {len(self.state.boxes) + 1}"
        used = {b.name for b in self.state.boxes} | {"table"}
        base, n = name, 2
        while name in used:
            name, n = f"{base} {n}", n + 1
        t = self.state.table
        self.state.add_box(name, ((t.x_range[0] + t.x_range[1]) / 2, 0.0, t.top_z + 0.15), (0.1, 0.1, 0.1))
        self.select_box(name)

    def select_box(self, name: str | None) -> None:
        """Chooses the box the fields and the handle edit."""
        self.selected = None if name in (None, "(none)") else name
        self.redraw()

    def _box_fields_changed(self) -> None:
        def apply() -> None:
            if self.selected is None:
                return
            box = next(b for b in self.state.boxes if b.name == self.selected)
            size = tuple(float(v) / 100.0 for v in self.f_size_cm.value)
            if min(size) > 0:
                box.size = size
            box.center = tuple(float(v) for v in self.f_center.value)
            box.side = self.f_side.value
            self.redraw()
        self._guard(apply)

    def _box_gizmo_moved(self) -> None:
        def move() -> None:
            if self.selected is not None:
                self.state.move_box(self.selected, tuple(float(v) for v in self.gizmo_box.position))
                self.redraw()
        self._guard(move)

    def delete_clicked(self) -> None:
        """Deletes the selected box."""
        if self.selected is not None:
            self.state.remove_box(self.selected)
            self.selected = None
            self.redraw()

    # -- work area and home -----------------------------------------------------------------------------------------------
    def fill_area(self, side: str) -> None:
        """Fills the work-area rectangle typed for `side` (its base frame)."""
        x, y = self.f_area[side]
        self.state.set_work_area(side, tuple(x.value), tuple(y.value))
        self.redraw()

    def clear_area(self, side: str) -> None:
        self.state.set_work_area(side, None, None)
        self.redraw()

    def use_home(self, side: str, which: str) -> None:
        """Sets an arm's home to rest, ready or the live pose; a message says when there is no live pose."""
        if which == "rest":
            self.state.use_rest(side)
        elif which == "ready":
            self.state.use_ready(side)
        elif self.read_q is None:
            self.out_msg.content = "No live arm connected: start with --observe (listening while your teleop runs) or --live follower (limp arm)."
            return
        else:
            self.state.use_pose(side, self.read_q(side))
        self.redraw()

    def set_pose_view(self, which: str) -> None:
        self.pose_view = which
        self.show_pose()

    def refresh_strip(self, clashes: list[str]) -> None:
        """The thin strip at the top: a light per arm and the first clash."""
        self.status.content = cards.status_strip(self.receiving, clashes[0] if clashes else None)

    def update_status(self, live_q: dict[str, np.ndarray] | None = None) -> None:
        """Refreshes the strip (including the live arms) without redrawing the scene."""
        self.refresh_strip(self.state.clashes(live=live_q))

    def update_position_cards(self, live: dict[str, np.ndarray], fresh: dict[str, bool]) -> None:
        """Card 1: each arm's live joints (degrees) and hand position (cm), or 'waiting' for an arm that is silent."""
        self.receiving = {side: bool(fresh.get(side)) for side in ed.SIDES}
        for side in ed.SIDES:
            q = live.get(side)
            if self.receiving[side] and q is not None:
                hand = self.kin[side].points(np.asarray(q, dtype=float))["tcp"] * 100.0
                self.pos_cards[side].content = cards.position_card(side, np.degrees(q).tolist(), None, tuple(hand), fresh=True)
            else:
                self.pos_cards[side].content = cards.position_card(side, None, None, None, fresh=False)
        self.refresh_strip(self.state.clashes(live={s: np.asarray(live[s], dtype=float) for s in live}))

    def refresh_home_cards(self) -> None:
        """Card 2: the state of each arm's home (set, path recorded)."""
        for side in ed.SIDES:
            arm = self.state.arms[side]
            has_path = bool(arm.home_path) or arm.home_source in ("ready", "rest")
            self.home_cards[side].content = cards.home_card(side, arm.home_source != "default", has_path, np.degrees(arm.home).tolist())

    def refresh_checklist(self) -> None:
        """Card 4: the checklist of what is done and what is missing."""
        self.check_html.content = cards.checklist(cards.build_checklist(self.state, self.teach))

    def show_pose(self, live_q: dict[str, np.ndarray] | None = None) -> None:
        """Poses the drawn robot per the chosen view (home, rest, ready, or the live arms)."""
        if self.robot is None or self.pose_view == "preview":   # a preview owns the robot while it plays
            return
        for side in ed.SIDES:
            if self.pose_view == "rest":
                q = np.radians(poses.REST_DEG[side])
            elif self.pose_view == "ready":
                q = np.radians(poses.READY_DEG[side])
            elif self.pose_view == "live" and live_q and side in live_q:
                q = live_q[side]
            else:
                q = self.state.arms[side].home
            self.robot.pose(side, q)

    # -- teach ----------------------------------------------------------------------------------------------------------
    _NO_LIVE = "No live follower: start with --observe (it listens while your teleop runs)."

    def teach_tick(self, live: dict[str, np.ndarray]) -> None:
        """Feeds the follower's latest joints (radians per side) to the teach session; while recording, the allowed region grows live."""
        self.live_q = live
        for side, q in live.items():
            self.teach.observe(side, q)
        sweeping = [s for s in ed.SIDES if self.teach.sweeping(s)]
        if sweeping:
            self._apply_teach()
        self.refresh_record_info()

    def sweep_changed(self, side: str, on: bool) -> None:
        self.teach.set_sweep(side, on)

    def clear_sweep_clicked(self, side: str) -> None:
        self.teach.clear_sweep(side)
        self.redraw()

    def _live(self, side: str) -> np.ndarray | None:
        if self.read_q is None:
            self.out_msg.content = self._NO_LIVE
            return None
        return np.asarray(self.read_q(side), dtype=float)

    def start_path_clicked(self, side: str) -> None:
        """Starts recording the way from the rest pose to the home you are about to choose."""
        q = self._live(side)
        if q is None:
            return
        try:
            self.teach.start_path(side, q)
            self.out_msg.content = f"Recording the {side} arm's path from rest. Move it to the home pose, then press Set HOME."
        except ValueError as exc:
            self.out_msg.content = str(exc)

    def set_home_from_follower(self, side: str) -> None:
        """Takes the follower's pose now as that arm's home with the path recorded automatically since it last stood at rest or ready."""
        q = self._live(side)
        if q is None:
            return
        path = self.teach.set_home(side, q)
        self._apply_teach()
        arm = self.state.arms[side]
        arm.home_source = "teach"
        if path is None:
            arm.home_path = None
        self.out_msg.content = f"{side} home set from the follower" + (" with the recorded path from rest." if arm.home_path else ", but there is NO path from rest.")
        self.redraw()

    def set_home_clicked(self, side: str) -> None:
        """Card 2 button: Set <SIDE> home here."""
        self.set_home_from_follower(side)

    def record_toggle_clicked(self) -> None:
        """Card 3 button: starts or stops recording the extremes for the chosen arm(s)."""
        sides = ed.SIDES if self.record_side.value == "both" else (self.record_side.value,)
        starting = not any(self.teach.sweeping(s) for s in sides)
        if starting and self.read_q is None:
            self.out_msg.content = self._NO_LIVE
            return
        for side in ed.SIDES:
            self.teach.set_sweep(side, starting and side in sides)
        self.b_record.label = "Stop recording extremes" if starting else "Start recording extremes"
        self.b_record.color = "red" if starting else "green"
        self.out_msg.content = ("Recording: move the leader to the farthest safe points, then press Stop." if starting else "Stopped. The green box is the allowed area.")
        if not starting:
            self._apply_teach()
            self.redraw()

    def refresh_record_info(self) -> None:
        """The Allowed area card's text: how many points each arm has and what is still missing."""
        lines = []
        for side in ed.SIDES:
            n, swept = self.teach.point_count(side), self.teach.sweep_count(side)
            text = f"**{side}**: {n} point{'s' if n != 1 else ''}" + (f", {swept} swept poses" if swept else "")
            if n and n < 3:
                text += " (record at least 3 for the work area; 2 are enough for the box)"
            elif n == 0 and not swept:
                text += " (none yet)"
            lines.append(text)
        self.record_info.content = "  \n".join(lines)

    def _record_sides(self) -> tuple[str, ...]:
        return ed.SIDES if self.record_side.value == "both" else (self.record_side.value,)

    def record_point_clicked(self) -> None:
        """Card 3 button: records the follower's pose now as one extreme point for the chosen arm(s); the safe place follows from the points."""
        if self.read_q is None:
            self.out_msg.content = self._NO_LIVE
            return
        for side in self._record_sides():
            self.teach.add_point(side, np.asarray(self.read_q(side), dtype=float))
        self._apply_teach()
        self.redraw()

    def undo_point_clicked(self) -> None:
        """Card 3 button: removes the last recorded point of the chosen arm(s)."""
        for side in self._record_sides():
            self.teach.undo_point(side)
        self._apply_teach()
        self.redraw()

    def clear_clicked(self) -> None:
        """Card 3 button: forgets every recorded point and sweep of the chosen arm(s) and removes their safe place."""
        for side in self._record_sides():
            self.teach.clear_points(side)
            self.teach.clear_sweep(side)
        self._apply_teach()
        self.redraw()

    def record_reference_clicked(self, side: str) -> None:
        """Save card button: records the live hand position as a reference point (a place touched on the table)."""
        q = self._live(side)
        if q is None:
            return
        self.state.add_reference(side, self.kin[side].points(q)["tcp"])
        self.out_msg.content = f"{side} reference point {len(self.state.references[side])} recorded."
        self.refresh_checklist()

    def mark_clicked(self, side: str) -> None:
        """Advanced: remembers the hand's position now under the typed label."""
        q = self._live(side)
        if q is not None:
            self.teach.mark(side, self.f_mark.value or "mark", q)
            self.redraw()

    def apply_sweep_clicked(self) -> None:
        """Advanced: turns the recorded points and sweeps into the safe place of each arm now."""
        self._apply_teach()
        done = [s for s in ed.SIDES if self.state.arms[s].region]
        self.out_msg.content = ("Applied to the safe place of: " + ", ".join(done) + ".") if done else \
            f"Nothing to apply yet: record at least 2 points (or sweep {th.MIN_SAMPLES} poses)."
        self.redraw()

    # -- outputs --------------------------------------------------------------------------------------------------------
    def save_clicked(self) -> None:
        """Writes the bundle and the layout picture; the message names both files and any clash."""
        widened = self._apply_teach()
        self.operator = self.f_operator.value or self.operator
        problems = self.state.clashes()
        self.last_saved = outputs.save_bundle(self.state, self.operator, self.out_dir)
        warning = f"\n\n**Saved with {len(problems)} clash(es): fix them before apply.**" if problems else ""
        note = f"\n\nThe allowed box of the {' and '.join(widened)} arm was widened to hold its home pose." if widened else ""
        self.out_msg.content = f"Saved `{self.last_saved[0].name}` and `{self.last_saved[1].name}` in `{self.out_dir}`.{note}{warning}"

    def export_clicked(self) -> pathlib.Path:
        """Writes the station keys file (home, reach, keep_out per arm)."""
        raw = self.state.to_bundle(self.operator, dt.datetime.now().astimezone())
        path = outputs.export_station_keys(raw, self.station_keys)
        self.out_msg.content = f"Wrote the station keys to `{path}`."
        return path

    # -- scene ----------------------------------------------------------------------------------------------------------
    def redraw(self, live_q: dict[str, np.ndarray] | None = None) -> None:
        """Redraws the zones, the work area and the status from the state."""
        for node in self._nodes:
            try:
                node.remove()
            except Exception:
                pass
        self._nodes = []
        scene, margin = self.server.scene, zones.LINK_RADIUS_M + zones.CLEARANCE_M
        clashes = self.state.clashes(live=live_q)
        clashing = " ".join(clashes)

        def box(name: str, center, size, color, label: str) -> None:
            bad = label in clashing
            self._nodes.append(scene.add_box(f"/zones/{name}", RED if bad and color != BLUE else color, tuple(size), opacity=0.3, position=tuple(center)))
            self._nodes.append(scene.add_box(f"/zones/{name}_margin", AMBER, tuple(np.asarray(size) + 2 * margin), wireframe=True, position=tuple(center)))

        t = self.state.table
        tz = t.keep_out()
        box("table", tz["center"], tz["size"], BLUE, "table")
        for b in self.state.boxes:
            box(b.name, b.center, b.size, RED, b.name)
        for side in ed.SIDES:
            cells = np.array(sorted(self.state.arms[side].reach), dtype=float) * zb.REACH_GRID_M if self.state.arms[side].reach else None
            if cells is not None:
                base = ed.BASE_OFFSET[side]
                pts = np.column_stack([cells[:, 0] + base[0], cells[:, 1] + base[1], np.full(len(cells), t.top_z + 0.004)])
                self._nodes.append(scene.add_point_cloud(f"/zones/work_area_{side}", pts, GREEN, point_size=0.012))
        for side in ed.SIDES:
            base = ed.BASE_OFFSET[side]
            region = self.state.arms[side].region
            if region:
                lo, hi = np.asarray(region["lo"]), np.asarray(region["hi"])
                self._nodes.append(scene.add_box(f"/zones/region_{side}", GREEN, tuple(hi - lo), wireframe=True, position=tuple((lo + hi) / 2 + base)))
            trail = self.teach.trail(side)
            if len(trail):
                self._nodes.append(scene.add_point_cloud(f"/zones/trail_{side}", trail[-3000:] + base, (30, 140, 255), point_size=0.006))
            for i, xyz in enumerate(self.teach.point_list(side)):
                self._nodes.append(scene.add_icosphere(f"/zones/point_{side}_{i}", 0.014, (255, 120, 0), position=tuple(np.asarray(xyz) + base)))
                self._nodes.append(scene.add_label(f"/zones/point_{side}_{i}_label", str(i + 1), position=tuple(np.asarray(xyz) + base + [0, 0, 0.025])))
            for i, (label, xyz) in enumerate(self.teach.marks(side)):
                self._nodes.append(scene.add_icosphere(f"/zones/mark_{side}_{i}", 0.012, (255, 0, 160), position=tuple(np.asarray(xyz) + base)))
                self._nodes.append(scene.add_label(f"/zones/mark_{side}_{i}_label", f"{side}: {label}", position=tuple(np.asarray(xyz) + base + [0, 0, 0.03])))
        self.gizmo_table.position = ((t.x_range[0] + t.x_range[1]) / 2, (t.y_range[0] + t.y_range[1]) / 2, t.top_z)
        names = ["(none)"] + [b.name for b in self.state.boxes]
        self.f_select.options = tuple(names)
        if self.selected not in names:
            self.selected = None
        self.f_select.value = self.selected or "(none)"
        self.gizmo_box.visible = self.selected is not None
        if self.selected is not None:
            sel = next(b for b in self.state.boxes if b.name == self.selected)
            self.gizmo_box.position = sel.center
            self._busy = True
            try:
                self.f_center.value, self.f_size_cm.value, self.f_side.value = sel.center, tuple(v * 100.0 for v in sel.size), sel.side
            finally:
                self._busy = False
        self.show_pose(live_q)
        self._busy = True
        try:
            self.f_height_cm.value = (t.top_z - self.hanging_z) * 100.0
            self.f_front_cm.value = t.x_range[0] * 100.0
            self.f_depth_cm.value = (t.x_range[1] - t.x_range[0]) * 100.0
            self.f_width_cm.value = (t.y_range[1] - t.y_range[0]) * 100.0
        finally:
            self._busy = False
        self.refresh_record_info()
        self.refresh_strip(clashes)
        self.refresh_home_cards()
        self.refresh_checklist()


def serve_designer(state: ed.EditorState, out_dir: pathlib.Path, operator: str, port: int = 8080, detail: str = "visual",
                   read_q: Callable[[str], np.ndarray] | None = None, cameras: list[dict[str, str]] | None = None,
                   observer_status: Callable[[], str] | None = None) -> None:
    """Starts the editor in a browser (http://localhost:<port>) and runs until Ctrl-C; with `read_q` the robot follows the limp arms when 'live' is chosen.

    @param state The editor state to start from.
    @param out_dir Where Save and Export write.
    @param operator Who is editing.
    @param port The browser port.
    @param detail `"visual"` or `"collision"` meshes.
    @param read_q Live joints per side in radians, or None.
    @param cameras RealSense name/serial pairs for the switchable camera feeds, or None.
    @param observer_status Text about whether teleop frames are arriving (for --observe), or None.
    """
    import time

    server = viser.ViserServer(port=port)
    robot = RobotModel(server, detail=detail)
    panel = EditorPanel(server, state, out_dir, operator, robot=robot, read_q=read_q, cameras=cameras)
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)
    print(f"editor: open http://localhost:{port}   (Ctrl-C to stop)")
    last_draw = 0.0
    try:
        while True:
            if read_q is not None:
                live, fresh = {}, {}
                for side in ed.SIDES:
                    try:
                        live[side], fresh[side] = np.asarray(read_q(side), dtype=float), True
                    except Exception:  # no frames from this arm yet, or its teleop stopped: the card says "waiting"
                        fresh[side] = False
                panel.update_position_cards(live, fresh)
                if live:
                    panel.teach_tick(live)
                    if panel.pose_view == "live":
                        panel.show_pose(live)
                if observer_status is not None and not all(fresh.values()):
                    panel.teach_info.content = observer_status()
                if time.monotonic() - last_draw > 1.0 and any(panel.teach.sweeping(s) for s in ed.SIDES):
                    panel.redraw(live)
                    last_draw = time.monotonic()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if panel.cams is not None:
            panel.cams.close()
        server.stop()

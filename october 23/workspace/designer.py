"""The visual zone editor: the real OpenArm model (URDF meshes) in a browser with the table, danger boxes, work area and home poses on top.

`RobotModel` loads `urdf/openarm_bimanual.urdf` with its meshes and poses it from the fork's joint angles (the v10 URDF's own convention). `EditorPanel` holds the GUI
(table, danger boxes with draggable handles, work-area rectangles, home choices, Save and Export) and redraws the zones whenever the state changes. All the logic of the
panel is in methods the buttons call, so it is tested without a browser; the pure state is in `editor.py`.
"""

import datetime as dt
import html
import os
import pathlib
import time
from collections.abc import Callable

import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

import cards
import collect
import editor as ed
import handpose
import i18n
import openarm_poses as poses
import openarm_zones as zones
import outputs
import startpose
import teach as th
import zonebuild as zb

HERE = pathlib.Path(__file__).resolve().parent
URDF = HERE.parent / "urdf" / "openarm_bimanual.urdf"
MESH_ROOT = pathlib.Path("~/projects/openarm-orboh/openarm_description").expanduser()
STATION = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
BRANCH_EVERY_S = 5.0  # how often the top of the page re-reads the branches of openarm-orboh and lerobot (V8)
MARK = "✅ "  # in front of the chosen button of a button group: viser draws all of them alike (2026-10-28)
BLUE, RED, AMBER, GREEN, GREY = (60, 120, 220), (220, 50, 47), (255, 191, 0), (60, 180, 75), (150, 150, 150)
JOINTS = 7


def unmarked(text: str) -> str:
    """A button group's option text without the chosen-button mark."""
    return text.removeprefix(MARK)


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
                 cameras: list[dict[str, str]] | None = None, collector: collect.CollectSession | None = None,
                 collect_env: pathlib.Path = collect.DEFAULT_ENV, branches: Callable[[], list[dict]] = collect.branches) -> None:
        """@param collector Runs episode collection for the Collect tab (None: a new one). @param collect_env The collection settings file.
        @param server The viser server. @param state The editor state. @param out_dir Where Save writes. @param operator Who is editing.
        @param robot The drawn robot (None in tests). @param read_q Live joints per side (radians), or None. @param station_keys Where Export writes.
        @param cameras RealSense name/serial pairs for the switchable feeds in the Home card, or None.
        @param branches Reads the branches shown at the top (tests replace it)."""
        self.server, self.state, self.out_dir, self.operator = server, state, pathlib.Path(out_dir), operator
        self.robot, self.read_q, self.cameras = robot, read_q, cameras
        self.lang = i18n.Lang("en")
        self._retext: list[Callable[[], None]] = []
        self._marked: list = []  # button groups whose chosen button carries MARK
        self._marking = False
        self.collector = collector or collect.CollectSession()
        self.branches = branches
        has_default = (startpose.pose_sets_dir() / collect.DEFAULT_POSE_SET).is_dir()
        self.collect_opts = collect.Options(env_file=collect_env, pose_set=collect.DEFAULT_POSE_SET if has_default else "")
        self._last_live: dict[str, np.ndarray] = {}
        self._last_fresh: dict[str, bool] = {}
        self._collect_seen: tuple = ()
        self.cams = None
        self.station_keys = station_keys or self.out_dir / "station_keys.yaml"
        self.selected: str | None = None
        self.pose_view = "live" if read_q is not None else "home"
        self.teach = th.TeachSession()
        self.live_q: dict[str, np.ndarray] = {}
        self.receiving = {side: False for side in ed.SIDES}
        self.kin = {side: zones.ArmKinematics(side, logical=True) for side in ed.SIDES}
        self.hanging_z = float(self.kin["left"].points(np.zeros(7))["tcp"][2] + ed.BASE_OFFSET["left"][2])
        self.last_saved: tuple[pathlib.Path, pathlib.Path] | None = None
        self._nodes: list = []
        self._busy = False
        self._playing = False
        self._path_nodes: list = []
        self._build_gui()
        self.redraw()

    # -- GUI ------------------------------------------------------------------------------------------------------------
    def _tx(self, handle, attr: str, key: str, **fields) -> Callable[[], None]:
        """Gives a widget its text in the current language now, and again whenever the language changes. A field given as a function is called each time.
        @return The function that sets the text again (to refresh it when a field's value changed)."""
        def apply() -> None:
            text = self.lang.t(key, **{k: (v() if callable(v) else v) for k, v in fields.items()})
            if isinstance(handle, viser.GuiTabHandle):  # viser has no public tab label setter: change the label and rebuild the tab row
                handle._label = text
                handle._parent._rebuild_tab_props()
            else:
                setattr(handle, attr, text)
        self._retext.append(apply)
        apply()
        return apply

    def _choice(self, handle, keys: tuple[str, ...]) -> str:
        """The language-neutral key (`left`, `right`, `both`) of the option chosen in a dropdown or button group whose texts change with the language."""
        options = [unmarked(o) for o in handle.options]
        value = unmarked(handle.value)
        return keys[options.index(value)] if value in options else keys[0]

    def _show_chosen(self, handle) -> None:
        """Marks the chosen button of a button group (viser draws them all alike): the chosen one gets `MARK`, the others lose it."""
        plain = [unmarked(o) for o in handle.options]
        value = unmarked(handle.value)
        index = plain.index(value) if value in plain else 0
        self._marking = True  # viser calls the group's handlers when its value changes, also from code: this is not a click
        try:
            handle.options = tuple((MARK if i == index else "") + o for i, o in enumerate(plain))
            handle.value = handle.options[index]
        finally:
            self._marking = False

    def _mark_on_click(self, handle) -> None:
        """Keeps a button group's chosen button marked, from now and after every click."""
        self._marked.append(handle)
        handle.on_click(lambda _e: None if self._marking else self._show_chosen(handle))
        self._show_chosen(handle)

    def _set_choice(self, handle, keys: tuple[str, ...], key: str) -> None:
        handle.value = list(handle.options)[keys.index(key)]

    def _arm_texts(self) -> tuple[str, ...]:
        return (self.lang.t("side.left"), self.lang.t("side.right"), self.lang.t("arm.both"))

    def _build_gui(self) -> None:
        g, t, tx = self.server.gui, self.state.table, self._tx
        g.configure_theme(control_width="large", show_logo=False, show_share_button=False, brand_color=(31, 157, 85))
        try:
            g.main_panel.dock_right()
        except Exception:  # older viser: the theme alone is enough
            pass
        g.add_html(cards.PAGE_CSS + cards.KEYS_HTML)
        try:
            g.main_panel.set_width(560)  # clamped to the window by viser, so narrow screens still fit
        except Exception:
            pass
        self.b_lang = g.add_button_group("Language / 言語", i18n.LANGS)
        self.b_lang.value = i18n.LANGS[0]
        self.b_lang.on_click(lambda _e: None if self._marking else self.set_language(unmarked(self.b_lang.value)))
        self._mark_on_click(self.b_lang)
        self.status = g.add_html(cards.status_strip(self.receiving, None, self.lang))
        self.branch_strip = g.add_html("")  # V8: the branches of openarm-orboh and lerobot, yellow when not the default
        self._branches_at = 0.0
        self._retext.append(lambda: self.branch_tick(force=True))
        self.branch_tick(force=True)
        self.out_msg = g.add_markdown("")
        if self.cameras is not None:
            import viewer

            self.cam_tiles = viewer.add_camera_strip(self.server, self.cameras)  # outside the tabs, so it shows on every tab
            self.b_show_cams = g.add_checkbox("", True)
            tx(self.b_show_cams, "label", "cam.show")
            self.b_show_cams.on_update(lambda _e: self.show_cameras(self.b_show_cams.value))
        tabs = self.tabs = g.add_tab_group()
        tab_keys = ("tab.position", "tab.home", "tab.danger", "tab.save", "tab.collect")
        self.tab_labels = [self.lang.t(k) for k in tab_keys]
        # -- 1 Position -------------------------------------------------------------------------------------------------
        tab = tabs.add_tab(self.tab_labels[0])
        tx(tab, "label", tab_keys[0])
        with tab:
            self.pos_cards = {side: g.add_html(cards.position_card(side, None, None, None, fresh=False, lang=self.lang)) for side in ed.SIDES}
        # -- 2 Home -----------------------------------------------------------------------------------------------------
        tab = tabs.add_tab(self.tab_labels[1])
        tx(tab, "label", tab_keys[1])
        with tab:
            self._build_move_folder(g)
            tx(g.add_markdown(""), "content", "home.help")
            self.home_cards, self.b_home = {}, {}
            for side in ed.SIDES:
                self.home_cards[side] = g.add_html(cards.home_card(side, False, False, lang=self.lang))
                self.b_home[side] = g.add_button("", color="green")
                tx(self.b_home[side], "label", "home.set", side=side)
                self.b_home[side].on_click(lambda _e, s=side: self.set_home_clicked(s))
            tx(g.add_markdown(""), "content", "home.previewhelp")
            self.b_preview = {}
            for label, sides in (("LEFT", ("left",)), ("RIGHT", ("right",)), ("both", ed.SIDES)):
                b = g.add_button("", color="gray")
                tx(b, "label", "home.preview", which=lambda label=label: self.lang.t(f"which.{label}"))
                b.on_click(lambda _e, ss=sides: self.play_path(list(ss)))
                self.b_preview[label] = b
            self._build_start_pose_folder(g)
            if self.cameras is not None:
                import viewer

                self.cams = viewer.CameraPanel(self.server, self.cameras, tiles=self.cam_tiles)
                tx(self.cams.folder, "label", "cam.folder")
                for cam_name, box in self.cams.boxes.items():
                    tx(box, "label", "cam.on", name=cam_name)
                if self.cams.none_note is not None:
                    tx(self.cams.none_note, "content", "cam.none")
        # -- 3 Danger area ----------------------------------------------------------------------------------------------
        tab = tabs.add_tab(self.tab_labels[2])
        tx(tab, "label", tab_keys[2])
        with tab:
            folder = g.add_folder("Allowed area")
            tx(folder, "label", "area.folder")
            with folder:
                tx(g.add_markdown(""), "content", "area.help")
                self.record_side = g.add_button_group("Arm", self._arm_texts())
                tx(self.record_side, "label", "area.arm")
                self._retext.append(lambda: self._relabel_choice(self.record_side))
                self._set_choice(self.record_side, ed.SIDES + ("both",), "both")
                self._mark_on_click(self.record_side)
                self.b_point = g.add_button("", color="green")
                self.b_undo = g.add_button("", color="gray")
                self.b_clear = g.add_button("", color="gray")
                tx(self.b_point, "label", "area.record")
                tx(self.b_undo, "label", "area.undo")
                tx(self.b_clear, "label", "area.clear")
                self.record_info = g.add_markdown("")
                self.b_point.on_click(lambda _e: self.record_point_clicked())
                self.b_undo.on_click(lambda _e: self.undo_point_clicked())
                self.b_clear.on_click(lambda _e: self.clear_clicked())
            folder = g.add_folder("Table (cm)")
            tx(folder, "label", "table.folder")
            with folder:
                self.f_height_cm = g.add_number("", (t.top_z - self.hanging_z) * 100, step=1.0)
                self.f_front_cm = g.add_number("", t.x_range[0] * 100, step=1.0)
                self.f_depth_cm = g.add_number("", (t.x_range[1] - t.x_range[0]) * 100, min=5.0, step=1.0)
                self.f_width_cm = g.add_number("", (t.y_range[1] - t.y_range[0]) * 100, min=5.0, step=1.0)
                self.f_lower_cm = g.add_number("", t.lower_cm, min=-ed.RAISE_MAX_CM, max=ed.LOWER_MAX_CM, step=0.5)
                tx(self.f_lower_cm, "label", "table.lower")
                tx(g.add_markdown(""), "content", "table.lower.help")
                self.f_lower_cm.on_update(lambda _e: self.lower_changed())
                for handle, key in ((self.f_height_cm, "table.height"), (self.f_front_cm, "table.front"), (self.f_depth_cm, "table.depth"), (self.f_width_cm, "table.width")):
                    tx(handle, "label", key)
                for f in (self.f_height_cm, self.f_front_cm, self.f_depth_cm, self.f_width_cm):
                    f.on_update(lambda _e: self.table_cards_changed())
            folder = g.add_folder("Danger boxes")
            tx(folder, "label", "box.folder")
            with folder:
                self.b_add = g.add_button("", color="red")
                tx(self.b_add, "label", "box.add")
                self.f_select = g.add_dropdown("", ("(none)",), initial_value="(none)")
                tx(self.f_select, "label", "box.select")
                self.f_size_cm = g.add_vector3("", (10.0, 10.0, 10.0), step=1.0)
                tx(self.f_size_cm, "label", "box.size")
                self.f_side = g.add_dropdown("", ("both",) + ed.SIDES, initial_value="both")
                self.f_side.options = (self.lang.t("arm.both"),) + tuple(self.lang.t(f"side.{x}") for x in ed.SIDES)
                self.f_side.value = self.f_side.options[0]
                tx(self.f_side, "label", "box.applies")
                self._retext.append(lambda: self._relabel_choice(self.f_side, ("both",) + ed.SIDES))
                self.b_delete = g.add_button("", color="gray")
                tx(self.b_delete, "label", "box.delete")
                tx(g.add_markdown(""), "content", "box.drag")
                self.b_add.on_click(lambda _e: self.add_box_clicked())
                self.b_delete.on_click(lambda _e: self.delete_clicked())
                self.f_select.on_update(lambda _e: self.select_box(self.f_select.value))
                for f in (self.f_size_cm, self.f_side):
                    f.on_update(lambda _e: self._box_fields_changed())
        # -- 4 Save -----------------------------------------------------------------------------------------------------
        tab = tabs.add_tab(self.tab_labels[3])
        tx(tab, "label", tab_keys[3])
        with tab:
            self.check_html = g.add_html("")
            self.b_save = g.add_button("", color="green")
            self.b_export = g.add_button("", color="gray")
            tx(self.b_save, "label", "save.bundle")
            tx(self.b_export, "label", "save.export")
            self.b_save.on_click(lambda _e: self.save_clicked())
            self.b_export.on_click(lambda _e: self.export_clicked())
            tx(g.add_markdown(""), "content", "save.refhelp")
            for side in ed.SIDES:
                b_ref = g.add_button("", color="gray")
                tx(b_ref, "label", "save.ref", side=side)
                b_ref.on_click(lambda _e, s=side: self.record_reference_clicked(s))
        # -- 5 Collect --------------------------------------------------------------------------------------------------
        tab = tabs.add_tab(self.tab_labels[4])
        tx(tab, "label", tab_keys[4])
        with tab:
            self._build_collect_tab(g)
        # -- Advanced (rarely needed; kept in English) ---------------------------------------------------------------------
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
        self.gizmo_table = self.server.scene.add_transform_controls("/gizmo_table", scale=0.2, disable_rotations=True, visible=True)
        self.gizmo_table.on_update(lambda _e: self._table_gizmo_moved())
        self.gizmo_box = self.server.scene.add_transform_controls("/gizmo_box", scale=0.2, disable_rotations=True, visible=False)
        self.gizmo_box.on_update(lambda _e: self._box_gizmo_moved())

    def _relabel_choice(self, handle, keys: tuple[str, ...] | None = None) -> None:
        """Puts the arm names of the new language on a choice, keeping what was chosen."""
        keys = keys or ed.SIDES + ("both",)
        old = [unmarked(o) for o in handle.options]
        index = old.index(unmarked(handle.value)) if unmarked(handle.value) in old else 0
        if keys[0] == "both":
            texts = (self.lang.t("arm.both"),) + tuple(self.lang.t(f"side.{x}") for x in ed.SIDES)
        else:
            texts = self._arm_texts()
        handle.options = texts
        handle.value = texts[index]
        if handle in self._marked:
            self._show_chosen(handle)

    # -- start pose set (Home tab) --------------------------------------------------------------------------------
    def _base_set(self) -> str:
        """The start-pose set the collection settings file uses (else the repo's CURRENT): what a new set is copied from."""
        name = collect.env_value(self.collect_opts.env_file, "POSE_SET")
        if name:
            return name
        try:
            return (startpose.pose_sets_dir() / "CURRENT").read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _build_start_pose_folder(self, g) -> None:
        tx = self._tx
        folder = g.add_folder("Start pose for collection")
        tx(folder, "label", "start.folder")
        with folder:
            tx(g.add_markdown(""), "content", "start.help")
            self.pitch_info = g.add_markdown("")
            base = self._base_set()
            self.f_setbase = g.add_text("", base)
            tx(self.f_setbase, "label", "start.base")
            self.f_setname = g.add_text("", startpose.next_name(base) if base else "")
            tx(self.f_setname, "label", "start.name")
            self.b_setsave = g.add_button("", color="green")
            tx(self.b_setsave, "label", "start.save")
            self.b_setsave.on_click(lambda _e: self.save_start_set_clicked())

    def _pitch_text(self, live: dict[str, np.ndarray]) -> str:
        """`left: -69°  right: -75°` for the Home tab, from the arms' live joints (empty when an arm is silent)."""
        parts = [self.lang.t("start.pitch", side=side, deg=handpose.approach_elevation_deg(side, live[side])) for side in ed.SIDES if side in live]
        return "  \n".join(parts)

    def save_start_set_clicked(self) -> None:
        """Home tab: writes the arms' pose now as a new start-pose set and makes the Collect tab use it."""
        if not all(self.receiving.values()) or not all(side in self._last_live for side in ed.SIDES):
            self.out_msg.content = self.lang.t("start.err.nolive")
            return
        name = self.f_setname.value.strip()
        joints = {side: np.degrees(np.asarray(self._last_live[side], dtype=float)).tolist() for side in ed.SIDES}
        try:
            startpose.save(name, self.f_setbase.value.strip(), joints)
        except startpose.StartPoseError as exc:
            key, _, detail = str(exc).partition("|")
            self.out_msg.content = self.lang.t(key, what=detail)
            return
        self.out_msg.content = self.lang.t("start.saved", name=name, pitch=self._pitch_text(self._last_live).replace("  \n", "  "))
        self.f_poseset.value = name  # the Collect tab now starts from it
        self.f_setname.value = startpose.next_name(name)

    # -- 5 Collect ------------------------------------------------------------------------------------------------------
    def _build_collect_tab(self, g) -> None:
        """The Collect card: start and stop `collect_bimanual.sh`, type its control keys, and show what it is doing."""
        tx, o = self._tx, self.collect_opts
        tx(g.add_markdown(""), "content", "col.help")
        self.keys_hint = g.add_markdown("")
        tx(self.keys_hint, "content", "col.keys")
        self.f_collector = g.add_text("", os.environ.get("OPERATOR", ""))  # empty until typed (or OPERATOR=<name> at launch): Start waits for it
        tx(self.f_collector, "label", "col.operator")
        self.f_collector.on_update(lambda _e: self._collect_buttons())
        self.f_envfile = g.add_dropdown("", ("-",))  # the task: which settings file collect_bimanual.sh reads (V2); options from collect_env_refresh
        tx(self.f_envfile, "label", "col.env")
        self.env_info = g.add_markdown("")
        self._retext.append(self._env_info_text)
        self._env_paths: dict[str, pathlib.Path] = {}
        self.f_mode = g.add_button_group("Mode", ("dry", "gate", "session"))
        tx(self.f_mode, "label", "col.mode")
        self._retext.append(self._relabel_mode)
        self.f_mode.options = tuple(self.lang.t(f"col.mode.{m}") for m in collect.MODES)
        self.f_mode.value = self.f_mode.options[collect.MODES.index(o.mode)]
        self._mark_on_click(self.f_mode)
        self.f_mode.on_click(lambda _e: self._keep_mode())
        self.f_tag = g.add_text("", o.tag)
        tx(self.f_tag, "label", "col.tag")
        self.tag_warn = g.add_html("")  # V6: a full-width or other bad character is refused as it is typed
        self._retext.append(self._tag_check)
        self.f_tag.on_update(lambda _e: self._tag_check())
        # A text field, not a number: viser's number input never sends an emptied field, so clearing "12" left "1" behind (2026-10-26)
        self.f_episodes = g.add_text("", str(o.episodes) if o.episodes else "")
        tx(self.f_episodes, "label", "col.episodes")
        self.f_time = g.add_number("", o.episode_time_s, min=0, max=3600, step=10)
        self.f_poseset = g.add_text("", o.pose_set)  # not f_pose: that is the Advanced "show robot at" dropdown
        self._env_labels = (  # their "empty = settings file" hints follow the chosen file
            tx(self.f_time, "label", "col.time", default=lambda: collect.env_value(self.collect_opts.env_file, "EPISODE_TIME_S") or "?"),
            tx(self.f_poseset, "label", "col.pose", default=lambda: collect.env_value(self.collect_opts.env_file, "POSE_SET") or "?"))
        self.pose_warn = g.add_html("")  # V3: yellow when the start pose set differs from the task file's POSE_SET
        self._retext.append(self._pose_warning)
        self.f_poseset.on_update(lambda _e: self._pose_warning())
        self.f_envfile.on_update(lambda _e: self.collect_env_changed())
        self.collect_env_refresh()
        folder = g.add_folder("Options", expand_by_default=False)
        tx(folder, "label", "col.opts")
        with folder:
            self.c_sounds, self.c_bilateral, self.c_saveend = (g.add_checkbox("", v) for v in (o.play_sounds, o.bilateral, o.save_on_episode_end))
            for handle, key in ((self.c_sounds, "col.sounds"), (self.c_bilateral, "col.bilateral"), (self.c_saveend, "col.saveend")):
                tx(handle, "label", key)
            self.c_return = g.add_checkbox("", o.return_on_episode_end)
            tx(self.c_return, "label", "col.return")
            self.c_hold = g.add_checkbox("", o.start_pose_hold)
            tx(self.c_hold, "label", "col.hold")
            self.f_countdown = g.add_number("", o.start_countdown_s, min=0, max=10, step=1)
            tx(self.f_countdown, "label", "col.countdown")
        self.collect_card = g.add_html(cards.collect_card(self.collector.snapshot(), self.lang))
        self.review_box = g.add_folder("")  # V4: shown after a finished gate until the reviewer's OK is ticked
        tx(self.review_box, "label", "col.review.folder")
        with self.review_box:
            self.review_info = g.add_markdown("")
            self.f_reviewer = g.add_text("", "")
            tx(self.f_reviewer, "label", "col.reviewer")
            self.c_reviewed = g.add_checkbox("", False)
            tx(self.c_reviewed, "label", "col.reviewed")
            self.c_reviewed.on_update(lambda _e: self.review_ticked())
        self._retext.append(self._review_text)
        self.b_cstart = g.add_button("", color="green")
        tx(self.b_cstart, "label", "col.start")
        self.resume_info = g.add_markdown("")
        self._retext.append(self._resume_text)
        self.b_cresume = g.add_button("", color="green")
        tx(self.b_cresume, "label", "col.resume")
        self.b_crefresh = g.add_button("", color="gray")  # V7: re-reads the set and its dataset now (the page also does it by itself when they change)
        tx(self.b_crefresh, "label", "col.refresh")
        self.b_crefresh.on_click(lambda _e: self.collect_refresh_clicked())
        self.c_replace = g.add_checkbox("", False)  # V6: shown while a stopped set can be resumed; a new gate/session forgets it only when ticked
        tx(self.c_replace, "label", "col.replace")
        self.b_cnext = g.add_button("", color="green")
        tx(self.b_cnext, "label", "col.next")
        self.b_crerecord = g.add_button("", color="gray")
        tx(self.b_crerecord, "label", "col.rerecord")
        self.b_calign = g.add_button("", color="gray")
        tx(self.b_calign, "label", "col.align")
        self.b_cstop = g.add_button("", color="orange")
        tx(self.b_cstop, "label", "col.stop")
        self.b_ckill = g.add_button("", color="red")
        tx(self.b_ckill, "label", "col.kill")
        self.b_cstart.on_click(lambda _e: self.collect_start_clicked())
        self.b_cresume.on_click(lambda _e: self.collect_resume_clicked())
        self.b_cnext.on_click(lambda _e: self.collect_key("next"))
        self.b_crerecord.on_click(lambda _e: self.collect_key("rerecord"))
        self.b_calign.on_click(lambda _e: self.collect_key("align"))
        self.b_cstop.on_click(lambda _e: self.collect_key("stop"))
        self.b_ckill.on_click(lambda _e: self.collect_kill_clicked())
        self._collect_buttons()

    def collect_env_refresh(self) -> None:
        """Lists the settings files of openarm-orboh's config folder (plus the one given at launch, if it lives elsewhere) and selects the current one."""
        config = pathlib.Path(self.collector.repo) / "config"
        paths = collect.env_files(self.collector.repo)
        current = pathlib.Path(self.collect_opts.env_file)
        if current not in paths:
            paths.append(current)
        self._env_paths = {(f.name if f.parent == config else str(f)): f for f in paths}
        self.f_envfile.options = tuple(self._env_paths)
        self.f_envfile.value = next(name for name, f in self._env_paths.items() if f == current)
        self.collect_env_changed()

    def collect_env_changed(self) -> None:
        """A task file was chosen: the next run reads it, and the page shows its TASK, SCENE, POSE_SET and EPISODE_TIME_S."""
        path = self._env_paths.get(self.f_envfile.value)
        if path is None:
            return
        self.collect_opts.env_file = path
        self._env_info_text()
        for apply in self._env_labels:
            apply()
        self._pose_warning()

    def _pose_warning(self) -> None:
        """Warns (yellow) when the start pose set typed on the page is not the one the task file names: on 10/26 sets A and B/C
        used different start poses (wrist J7 97° left, 73° right apart) and nobody noticed until afterwards (V3, 2026-10-28)."""
        page = self.f_poseset.value.strip()
        file = collect.env_value(self.collect_opts.env_file, "POSE_SET")
        differs = bool(page and file and page != file)
        self.pose_warn.content = cards.warning(self.lang.t("col.pose.differs", page=html.escape(page), file=html.escape(file))) if differs else ""

    def _env_info_text(self) -> None:
        v = collect.env_summary(self.collect_opts.env_file)
        self.env_info.content = self.lang.t("col.env.info", task=v["TASK"] or "?", scene=v["SCENE"] or "?", pose=v["POSE_SET"] or "?", time=v["EPISODE_TIME_S"] or "?")

    def _relabel_mode(self) -> None:
        old = [unmarked(o) for o in self.f_mode.options]
        index = old.index(unmarked(self.f_mode.value)) if unmarked(self.f_mode.value) in old else 0
        self.f_mode.options = tuple(self.lang.t(f"col.mode.{m}") for m in collect.MODES)
        self.f_mode.value = self.f_mode.options[index]
        self._show_chosen(self.f_mode)

    def _collect_options(self) -> collect.Options:
        """The chosen options, read from the card."""
        o = self.collect_opts
        o.mode = self._choice(self.f_mode, collect.MODES) if unmarked(self.f_mode.value) in [unmarked(x) for x in self.f_mode.options] else o.mode
        count = self.f_episodes.value.strip()
        if count and not count.isdigit():
            raise collect.CollectError(f"col.err.episodes|{count}")
        o.operator, o.tag, o.episodes = self.f_collector.value.strip(), self.f_tag.value.strip(), int(count or 0)
        o.env_file = self._env_paths.get(self.f_envfile.value, o.env_file)
        o.episode_time_s, o.pose_set = int(self.f_time.value), self.f_poseset.value.strip()
        o.play_sounds, o.bilateral, o.save_on_episode_end = bool(self.c_sounds.value), bool(self.c_bilateral.value), bool(self.c_saveend.value)
        o.start_pose_hold, o.start_countdown_s = bool(self.c_hold.value), int(self.f_countdown.value)
        o.return_on_episode_end = bool(self.c_return.value)
        o.replace_set = bool(self.c_replace.value)
        return o

    def _error_text(self, exc: collect.CollectError) -> str:
        key, _, detail = str(exc).partition("|")
        return self.lang.t(key, what=detail, path=detail)

    def collect_start_clicked(self, opts: collect.Options | None = None) -> None:
        """Collect tab, Start: runs the script. Releases this page's cameras first (the script needs them) and says so.

        @param opts What to run (None: the options on the card)."""
        note = ""
        if self.cams is not None and any(w.running for w in self.cams.workers.values()):
            note = " " + self.lang.t("col.camsoff")
        try:
            opts = opts or self._collect_options()
        except collect.CollectError as exc:
            self.out_msg.content = self._error_text(exc)
            return
        if opts.pose_set and not (startpose.pose_sets_dir() / opts.pose_set).is_dir():  # say so here, before the cameras are released and the script runs
            self.out_msg.content = self.lang.t("start.err.base", what=opts.pose_set)
            return
        try:
            if self.cams is not None:
                self.cams.close()
            self.collector.start(opts)
            self.c_replace.value = False  # one confirmation per new set
            self.out_msg.content = self.lang.t("col.state.starting") + " " + self.lang.t("col.count", mode=opts.mode, n=self.collector.target) + note
        except collect.CollectError as exc:
            self.out_msg.content = self._error_text(exc)
        self.collect_tick(force=True)

    def collect_resume_clicked(self) -> None:
        """Collect tab, Resume: records only the episodes the stopped set still needs, into the same dataset, with the card's other options."""
        try:
            opts = self.collector.resume_options(self._collect_options())
        except collect.CollectError as exc:
            self.out_msg.content = self._error_text(exc)
            return
        self.collect_start_clicked(opts)

    def _resume_text(self) -> None:
        """The line above Resume: which set stopped short and how many it still needs (empty when none did)."""
        todo = self.collector.pending()
        if todo is None:
            self.resume_info.content = ""
            return
        name = todo["mode"] + (f" {todo['tag']}" if todo.get("tag") else "")
        self.resume_info.content = self.lang.t("col.resume.info", name=name, have=todo["have"], total=todo["target"], left=todo["left"])

    def collect_key(self, name: str) -> None:
        """Collect tab: types one control key (next, rerecord, align, stop) into the run."""
        try:
            self.collector.send(name)
        except collect.CollectError as exc:
            self.out_msg.content = self._error_text(exc)
        self.collect_tick(force=True)

    def collect_kill_clicked(self) -> None:
        """Collect tab, emergency stop: Ctrl-C for the whole run."""
        try:
            self.collector.kill()
        except collect.CollectError as exc:
            self.out_msg.content = self._error_text(exc)
        self.collect_tick(force=True)

    def review_ticked(self) -> None:
        """The gate review box was ticked: records the reviewer's OK, so a session may start; refused (and unticked) without a reviewer name."""
        if not self.c_reviewed.value:
            return
        who = self.f_reviewer.value.strip()
        try:
            self.collector.mark_reviewed(who)
            self.out_msg.content = self.lang.t("col.review.done", who=who)
        except collect.CollectError as exc:
            self.out_msg.content = self._error_text(exc)
        self.c_reviewed.value = False
        self.collect_tick(force=True)

    def _review_text(self) -> None:
        """Shows the review box only while a finished gate waits for its OK, with how many episodes it holds and where."""
        gate = self.collector.review_pending()
        self.review_box.visible = gate is not None
        self.review_info.content = "" if gate is None else self.lang.t(
            "col.review.info", have=gate["have"], target=gate["target"], repo=gate.get("repo_id") or "?")

    def _tag_check(self) -> None:
        """The session tag as it is typed: a full-width `s１` or any other bad character gets a yellow box under the field at once, and a session will not start (V6)."""
        problem = collect.tag_problem(self.f_tag.value)
        self.tag_warn.content = "" if problem is None else cards.warning(self._error_text(collect.CollectError(problem)))
        if problem is not None:
            self.out_msg.content = self._error_text(collect.CollectError(problem))

    def _keep_mode(self) -> None:
        """A click on Mode while a run is going is undone: the mark stays on the mode that is running."""
        if self._marking or not self.collector.active:
            return
        self._marking = True  # setting the value calls this handler again
        try:
            self._set_choice(self.f_mode, collect.MODES, self.collector._mode)
        finally:
            self._marking = False
        self._show_chosen(self.f_mode)

    def _collect_fields(self) -> tuple:
        """The Collect tab's settings: frozen while a run is going, since one session keeps one operator and one set of values (instructions §1, 2026-10-28).
        Mode is not here: viser cannot disable a button group, so `_keep_mode` undoes a click on it instead."""
        return (self.f_collector, self.f_envfile, self.f_tag, self.f_episodes, self.f_time, self.f_poseset,
                self.c_sounds, self.c_bilateral, self.c_saveend, self.c_return, self.c_hold, self.f_countdown)

    def _collect_buttons(self) -> None:
        """Shows only what makes sense now: Start while idle, the key buttons while running; no discard when every episode is saved the instant it ends."""
        running = self.collector.active
        for b in (self.b_cnext, self.b_calign, self.b_cstop, self.b_ckill):
            b.disabled = not running
        self.b_crerecord.disabled = not running or (bool(self.c_saveend.value) and not bool(self.c_return.value))  # with the automatic return, ← discards and returns
        for field in self._collect_fields():
            field.disabled = running
        nobody = not self.f_collector.value.strip()  # every run is recorded under a name, so nothing starts without one
        self.b_cstart.disabled = running or nobody
        self.b_cresume.disabled = running or nobody or self.collector.pending() is None
        self.c_replace.visible = not running and self.collector.pending() is not None
        self._resume_text()
        self._review_text()

    def branch_tick(self, force: bool = False) -> None:
        """Re-reads the branches of openarm-orboh and lerobot every few seconds, so a switch made in a terminal shows without a restart (V8)."""
        if not force and time.monotonic() - self._branches_at < BRANCH_EVERY_S:
            return
        self._branches_at = time.monotonic()
        self.branch_strip.content = cards.branch_strip(self.branches(), self.lang)

    def collect_refresh_clicked(self) -> None:
        """Collect tab, Refresh: reads the set and its dataset from disk again and says what it found (V7)."""
        self.collect_tick(force=True)
        todo = self.collector.pending()
        self.out_msg.content = self.lang.t("col.refreshed") + ("" if todo is None else " " + self.resume_info.content)

    def collect_tick(self, force: bool = False) -> None:
        """Refreshes the Collect card and its buttons when the run, or the set and its dataset on disk, changed (called a few times a second)."""
        snap = self.collector.snapshot()
        seen = (snap["phase"], snap["episode"], snap["started"], snap["rc"], tuple(snap["lines"][-3:]), self.lang.code, bool(self.c_saveend.value), bool(self.c_return.value), snap["saved"], snap["done"],
                self.collector.disk_stamp())  # V7: 10/26 the tab kept "0 left" and a disabled Resume after the dataset changed outside the page
        if seen == self._collect_seen and not force:
            return
        self._collect_seen = seen
        self.branch_tick(force=True)  # a run that just started or ended: show the branches it used
        self.collect_card.content = cards.collect_card(snap, self.lang)
        self._collect_buttons()

    def _build_move_folder(self, g) -> None:
        """Go to home / Park for the real arms, in the Home tab only: the Collect tab must not offer it, since collection drives the same arms (2026-10-28)."""
        import homemove

        self.mover = homemove.HomeMover(STATION)
        folder = g.add_folder("")
        self._tx(folder, "label", "move.folder")
        with folder:
            self._tx(g.add_markdown(""), "content", "move.help")
            self.b_go_home = g.add_button("", color="orange")
            self._tx(self.b_go_home, "label", "move.go")
            self.b_go_home.on_click(lambda _e: self.go_home_clicked())
            self.b_park = g.add_button("", color="red")
            self._tx(self.b_park, "label", "move.park")
            self.b_park.on_click(lambda _e: self.mover.park())
            self.move_info = g.add_markdown("")
        self._move_shown = ""

    def go_home_clicked(self) -> None:
        """Go to home: raises both arms (the lock decides which may move) and holds them until Park."""
        if not self.mover.go(list(ed.SIDES)):
            self.out_msg.content = self.lang.t("move.busy")

    def move_tick(self) -> None:
        """Shows the move's state and report lines when they change (called from the page loop)."""
        text = f"**{self.mover.state}**" + ("\n\n```\n" + self.mover.text() + "\n```" if self.mover.messages else "")
        if text != self._move_shown:
            self.move_info.content = self._move_shown = text

    def show_cameras(self, on: bool) -> None:
        """Shows or hides the camera strip at the top. Hiding keeps the cameras open (the Home tab's checkboxes release them); the 3D view gets its full height back."""
        for tile in self.cam_tiles.values():
            tile.visible = on

    def set_language(self, name: str) -> None:
        """The language buttons: every label, card and message switches. The page is shared, so everyone watching sees the same language."""
        self.lang.set(name)
        for apply in self._retext:
            apply()
        self.tab_labels = [self.lang.t(k) for k in ("tab.position", "tab.home", "tab.danger", "tab.save", "tab.collect")]
        self.out_msg.content = ""
        self.refresh_strip(self.state.clashes())
        self.update_position_cards(self._last_live, self._last_fresh)
        self.refresh_home_cards()
        self.refresh_checklist()
        self.refresh_record_info()
        self.collect_tick(force=True)

    def _names(self, sides) -> str:
        """`left and right` / `左と右`, in the current language."""
        return (" and " if self.lang.code == "en" else "と").join(self.lang.t(f"side.{s}") for s in sides)

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
            self.out_msg.content = self.lang.t("msg.nopath", sides=self._names(missing))
            return
        if self._playing:
            self.out_msg.content = self.lang.t("msg.playing")
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
                self.out_msg.content = self.lang.t("msg.played", sides=self._names(sides), sec=count / fps, dps=speed_dps)

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

    def lower_changed(self) -> None:
        """Danger area tab: how far the table's danger box top is lowered below the real surface (0 keeps it at the surface); warns past the safe limit."""
        self.state.table.lower_cm = min(max(float(self.f_lower_cm.value), -ed.RAISE_MAX_CM), ed.LOWER_MAX_CM)
        warning = self.state.table.lower_warning()
        self.out_msg.content = self.lang.t("msg.lowerwarn", cm=self.state.table.lower_cm, dip=self.state.table.lower_cm - ed.SAFE_LOWER_CM, safe=ed.SAFE_LOWER_CM) if warning else ""
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
            box.side = self._choice(self.f_side, ("both",) + ed.SIDES)
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
            self.out_msg.content = self.lang.t("msg.nolive")
            return
        else:
            self.state.use_pose(side, self.read_q(side))
        self.redraw()

    def set_pose_view(self, which: str) -> None:
        self.pose_view = which
        self.show_pose()

    def refresh_strip(self, clashes: list[str]) -> None:
        """The thin strip at the top: a light per arm and the first clash."""
        self.status.content = cards.status_strip(self.receiving, clashes[0] if clashes else None, self.lang)

    def update_status(self, live_q: dict[str, np.ndarray] | None = None) -> None:
        """Refreshes the strip (including the live arms) without redrawing the scene."""
        self.refresh_strip(self.state.clashes(live=live_q))

    def update_position_cards(self, live: dict[str, np.ndarray], fresh: dict[str, bool]) -> None:
        """Card 1: each arm's live joints (degrees) and hand position (cm), or 'waiting' for an arm that is silent."""
        self._last_live, self._last_fresh = dict(live), dict(fresh)
        self.receiving = {side: bool(fresh.get(side)) for side in ed.SIDES}
        for side in ed.SIDES:
            q = live.get(side)
            if self.receiving[side] and q is not None:
                hand = self.kin[side].points(np.asarray(q, dtype=float))["tcp"] * 100.0
                pitch = handpose.approach_elevation_deg(side, q)
                self.pos_cards[side].content = cards.position_card(side, np.degrees(q).tolist(), None, tuple(hand), fresh=True, lang=self.lang, pitch_deg=pitch)
            else:
                self.pos_cards[side].content = cards.position_card(side, None, None, None, fresh=False, lang=self.lang)
        self.pitch_info.content = self._pitch_text({s: np.asarray(live[s], dtype=float) for s in live if self.receiving.get(s)})
        self.refresh_strip(self.state.clashes(live={s: np.asarray(live[s], dtype=float) for s in live}))

    def refresh_home_cards(self) -> None:
        """Card 2: the state of each arm's home (set, path recorded)."""
        for side in ed.SIDES:
            arm = self.state.arms[side]
            has_path = bool(arm.home_path) or arm.home_source in ("ready", "rest")
            self.home_cards[side].content = cards.home_card(side, arm.home_source != "default", has_path, np.degrees(arm.home).tolist(), self.lang)

    def refresh_checklist(self) -> None:
        """Card 4: the checklist of what is done and what is missing."""
        self.check_html.content = cards.checklist(cards.build_checklist(self.state, self.teach, self.lang), self.lang)

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
    @property
    def _NO_LIVE(self) -> str:
        return self.lang.t("msg.nolive")

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
            self.out_msg.content = self.lang.t("msg.pathrec", side=side)
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
        moved = self.state.fit_home_to_dream(side)  # the home and its path's end, together, inside Dream's range (log-031)
        fit = "".join(self.lang.t("msg.homefit", joint=joint, deg=deg) for joint, deg in moved)
        self.out_msg.content = self.lang.t("msg.homeset", side=side) + self.lang.t("msg.homepath" if arm.home_path else "msg.homenopath") + fit
        self.redraw()

    def set_home_clicked(self, side: str) -> None:
        """Card 2 button: Set <SIDE> home here."""
        self.set_home_from_follower(side)

    def record_toggle_clicked(self) -> None:
        """Card 3 button: starts or stops recording the extremes for the chosen arm(s)."""
        sides = self._record_sides()
        starting = not any(self.teach.sweeping(s) for s in sides)
        if starting and self.read_q is None:
            self.out_msg.content = self._NO_LIVE
            return
        for side in ed.SIDES:
            self.teach.set_sweep(side, starting and side in sides)
        self.b_record.label = "Stop recording extremes" if starting else "Start recording extremes"
        self.b_record.color = "red" if starting else "green"
        self.out_msg.content = self.lang.t("msg.sweep.on" if starting else "msg.sweep.off")
        if not starting:
            self._apply_teach()
            self.redraw()

    def refresh_record_info(self) -> None:
        """The Allowed area card's text: how many points each arm has and what is still missing."""
        lines = []
        for side in ed.SIDES:
            n, swept = self.teach.point_count(side), self.teach.sweep_count(side)
            text = self.lang.t("record.points", side=side, n=n)
            if self.lang.code == "en" and n != 1:
                text = text.replace("point(s)", "points")
            elif self.lang.code == "en":
                text = text.replace("point(s)", "point")
            if swept:
                text += self.lang.t("record.swept", n=swept)
            if n and n < 3:
                text += self.lang.t("record.few")
            elif n == 0 and not swept:
                text += self.lang.t("record.none")
            lines.append(text)
        self.record_info.content = "  \n".join(lines)

    def _record_sides(self) -> tuple[str, ...]:
        chosen = self._choice(self.record_side, ed.SIDES + ("both",))
        return ed.SIDES if chosen == "both" else (chosen,)

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
        self.out_msg.content = self.lang.t("msg.refrec", side=side, n=len(self.state.references[side]))
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
        self.out_msg.content = self.lang.t("msg.applied", sides=self._names(done)) if done else self.lang.t("msg.nothing", n=th.MIN_SAMPLES)
        self.redraw()

    # -- outputs --------------------------------------------------------------------------------------------------------
    def save_clicked(self) -> None:
        """Writes the bundle and the layout picture; the message names both files and any clash."""
        widened = self._apply_teach()
        self.operator = self.f_operator.value or self.operator
        problems = self.state.clashes()
        self.last_saved = outputs.save_bundle(self.state, self.operator, self.out_dir)
        warning = "\n\n" + self.lang.t("msg.clashes", n=len(problems)) if problems else ""
        note = "\n\n" + self.lang.t("msg.widened", sides=self._names(widened)) if widened else ""
        self.out_msg.content = self.lang.t("msg.saved", bundle=self.last_saved[0].name, layout=self.last_saved[1].name, dir=self.out_dir) + note + warning

    def export_clicked(self) -> pathlib.Path:
        """Writes the station keys file (home, reach, keep_out per arm)."""
        raw = self.state.to_bundle(self.operator, dt.datetime.now().astimezone())
        path = outputs.export_station_keys(raw, self.station_keys)
        self.out_msg.content = self.lang.t("msg.keys", path=path)
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
                self.f_center.value, self.f_size_cm.value = sel.center, tuple(v * 100.0 for v in sel.size)
                self._set_choice(self.f_side, ("both",) + ed.SIDES, sel.side)
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
                   observer_status: Callable[[], str] | None = None, collect_env: pathlib.Path = collect.DEFAULT_ENV) -> None:
    """Starts the editor in a browser (http://localhost:<port>) and runs until Ctrl-C; with `read_q` the robot follows the limp arms when 'live' is chosen.

    @param state The editor state to start from.
    @param out_dir Where Save and Export write.
    @param operator Who is editing.
    @param port The browser port.
    @param detail `"visual"` or `"collision"` meshes.
    @param read_q Live joints per side in radians, or None.
    @param cameras RealSense name/serial pairs for the switchable camera feeds, or None.
    @param collect_env The collection settings file the Collect tab uses.
    @param observer_status Text about whether teleop frames are arriving (for --observe), or None.
    """
    import time

    server = viser.ViserServer(port=port)
    robot = RobotModel(server, detail=detail)
    panel = EditorPanel(server, state, out_dir, operator, robot=robot, read_q=read_q, cameras=cameras, collect_env=collect_env)
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)
    print(f"editor: open http://localhost:{port}   (Ctrl-C to stop)")
    last_draw = 0.0
    try:
        while True:
            panel.collect_tick()
            panel.branch_tick()
            panel.move_tick()
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
        panel.mover.close()  # an arm held at the home parks and switches off before the page goes
        if panel.cams is not None:
            panel.cams.close()
        if panel.collector.active:  # closing the page server must not leave a run holding the arms unseen
            print("collection is still running: it keeps going; stop it with the Stop button next time or Ctrl-C its process group")
        server.stop()

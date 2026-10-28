"""Tests for the daily safety-area tools: building zones from captured points, the capture session, build/apply/verify. No hardware."""

import datetime as dt
import json

import numpy as np
import pytest
import yaml

import capture as cap
import collect
import startpose
import openarm_poses as poses
import openarm_zones as zones
import zonebuild as zb

NOW = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


# -- zonebuild: pure maths ---------------------------------------------------------------------------------------------

def test_cells_round_to_the_two_cm_grid_and_dedupe():
    cells = {zb.cell(np.array(p)) for p in ([0.101, 0.2], [0.099, 0.201], [0.3, -0.05])}
    assert len(cells) == 2
    assert zb.cells_to_points(cells) == [[0.1, 0.2], [0.3, -0.04]] or zb.cells_to_points(cells)[0] == [0.1, 0.2]


def test_table_box_from_touch_points():
    pts = [(0.10, -0.20, -0.30), (0.50, -0.20, -0.31), (0.50, 0.20, -0.29), (0.10, 0.20, -0.30)]
    box = zb.table_keep_out(pts, thickness=0.30)
    assert box["kind"] == "box" and box["allow_tcp_touch"] is True
    assert box["center"] == pytest.approx([0.30, 0.0, -0.30 - 0.15])
    assert box["size"] == pytest.approx([0.40, 0.40, 0.30])


def test_table_needs_three_points():
    with pytest.raises(ValueError, match="three"):
        zb.table_keep_out([(0, 0, 0), (1, 0, 0)])


def test_hazard_box_adds_a_margin_around_the_marked_points():
    box = zb.hazard_keep_out("torso", [(0.0, 0.0, 0.0), (0.1, 0.0, 0.2)], margin=0.03)
    assert box["name"] == "torso"
    assert box["center"] == pytest.approx([0.05, 0.0, 0.1])
    assert box["size"] == pytest.approx([0.16, 0.06, 0.26])
    one = zb.hazard_keep_out("pole", [(1.0, 1.0, 1.0)], margin=0.05)
    assert one["size"] == pytest.approx([0.1, 0.1, 0.1])


def capture_fixture():
    left = zb.ArmCapture()
    left.reach = {zb.cell(np.array([0.3, 0.0])), zb.cell(np.array([0.32, 0.0]))}
    left.table_points = [(0.1, -0.2, -0.3), (0.5, -0.2, -0.3), (0.5, 0.2, -0.3)]
    left.hazards = {"torso": [(-0.1, 0.0, -0.1), (-0.05, 0.0, 0.0)]}
    left.references = [(0.3, 0.0, -0.3), (0.4, 0.1, -0.3)]
    right = zb.ArmCapture()
    right.table_points = list(left.table_points)
    right.references = list(left.references)
    return {"left": left, "right": right}


def test_build_zones_has_the_schema_the_driver_reads():
    raw = zb.build_zones(capture_fixture(), source="follower", operator="kota", now=NOW)
    parsed = zones.parse_zones(raw)
    assert parsed.source == "follower" and parsed.age_hours(NOW) == 0
    names = [z["name"] for z in parsed.arms["left"].keep_out]
    assert names == ["table", "torso"]
    assert raw["arms"]["left"]["table_z"] == pytest.approx(-0.3)
    assert raw["verify"]["status"] == "unverified"


def test_station_fragment_uses_only_keys_dream_accepts_and_adds_the_home_hand():
    raw = zb.build_zones(capture_fixture(), source="follower", operator="kota", now=NOW)
    frag = zb.station_fragment(raw, homes={"left": (0.05, 0.12), "right": (0.05, -0.12)})
    left = frag["left"]
    assert set(left) == {"reach", "keep_out", "home"}
    assert [0.06, 0.12] in left["reach"] or any(abs(p[0] - 0.05) < 0.02 and abs(p[1] - 0.12) < 0.02 for p in left["reach"])
    for zone in left["keep_out"]:
        assert set(zone) == {"kind", "center", "size"}  # Dream's keep_out takes exactly these for a box
    assert frag["right"]["reach"]  # the home cell alone is still a reach area


def test_pose_violations_flags_a_pose_inside_a_zone():
    kin = zones.ArmKinematics("left")
    tcp = kin.points(np.radians(poses.READY_DEG["left"]))["tcp"].tolist()
    raw = zb.build_zones(capture_fixture(), source="follower", operator="k", now=NOW)
    raw["arms"]["left"]["keep_out"].append({"name": "in_the_way", "kind": "box", "center": tcp, "size": [0.2, 0.2, 0.2]})
    found = zb.pose_violations(zones.parse_zones(raw), "left", {"ready": poses.READY_DEG["left"]}, kin)
    assert found and "ready" in found[0] and "in_the_way" in found[0]


def test_compare_references():
    stored = [(0.3, 0.0, -0.3), (0.4, 0.1, -0.3)]
    ok = zb.compare_references(stored, [(0.305, 0.0, -0.3), (0.4, 0.105, -0.302)], tol=0.015)
    assert ok["ok"] and ok["worst_m"] < 0.01
    bad = zb.compare_references(stored, [(0.33, 0.0, -0.3), (0.4, 0.1, -0.3)], tol=0.015)
    assert not bad["ok"] and bad["worst_m"] == pytest.approx(0.03)
    with pytest.raises(ValueError):
        zb.compare_references(stored, [(0.3, 0.0, -0.3)], tol=0.015)


def test_apply_writes_the_station_file_with_a_backup_and_keeps_other_keys(tmp_path):
    station = tmp_path / "station.yaml"
    station.write_text("# header\nbody: custom\nfollower_l: {can_channel: can_fol_l, home: [0.0]}\nfollower_r: {can_channel: can_fol_r, home: [0.0]}\nmax_ee_speed_m_s: 0.15\n")
    frag = {"left": {"reach": [[0.1, 0.1]], "keep_out": [{"kind": "box", "center": [0, 0, 0], "size": [1, 1, 1]}]},
            "right": {"reach": [[0.1, -0.1]], "keep_out": []}}
    backup = zb.apply_to_station(station, frag)
    assert backup.exists() and "# header" in backup.read_text()
    new = yaml.safe_load(station.read_text())
    assert new["follower_l"]["reach"] == [[0.1, 0.1]] and new["follower_l"]["can_channel"] == "can_fol_l"
    assert new["follower_r"]["keep_out"] == [] and new["max_ee_speed_m_s"] == 0.15
    assert station.read_text().startswith("# header")


# -- capture session ---------------------------------------------------------------------------------------------------

class Tcp:
    def __init__(self):
        self.xyz = {"left": np.array([0.3, 0.0, -0.3]), "right": np.array([0.3, 0.0, -0.3])}

    def __call__(self, side):
        return self.xyz[side].copy()


def test_session_paints_only_while_painting_is_on():
    tcp = Tcp(); s = cap.CaptureSession("left", tcp)
    s.tick()
    assert not s.captures["left"].reach
    s.command("paint on"); s.tick()
    tcp.xyz["left"] = np.array([0.5, 0.1, -0.3]); s.tick()
    s.command("paint off"); tcp.xyz["left"] = np.array([0.9, 0.9, -0.3]); s.tick()
    assert len(s.captures["left"].reach) == 2


def test_session_records_table_hazard_and_reference_points_for_the_chosen_arm():
    tcp = Tcp(); s = cap.CaptureSession("left", tcp)
    s.command("table"); s.command("hazard torso"); s.command("ref")
    s.command("arm right"); s.command("table")
    assert len(s.captures["left"].table_points) == 1 and len(s.captures["right"].table_points) == 1
    assert s.captures["left"].hazards == {"torso": [(0.3, 0.0, -0.3)]}
    assert len(s.captures["left"].references) == 1


def test_session_undo_removes_the_last_recorded_point():
    tcp = Tcp(); s = cap.CaptureSession("left", tcp)
    s.command("table"); s.command("ref"); s.command("undo")
    assert len(s.captures["left"].table_points) == 1 and not s.captures["left"].references


def test_leader_sessions_cannot_record_table_or_reference_points():
    tcp = Tcp(); s = cap.CaptureSession("left", tcp, ground_truth=False)
    assert "follower" in s.command("table")[1].lower()
    assert "follower" in s.command("ref")[1].lower()
    assert not s.captures["left"].table_points and not s.captures["left"].references
    s.command("hazard torso")  # painting and hazards are fine from a leader
    assert s.captures["left"].hazards


def test_session_commands_return_save_quit_and_unknown():
    s = cap.CaptureSession("left", Tcp())
    assert s.command("save")[0] == "save" and s.command("quit")[0] == "quit"
    assert "unknown" in s.command("dance")[1].lower()


# -- leader merge ------------------------------------------------------------------------------------------------------

def test_merge_leader_keeps_the_follower_table_and_adds_painted_cells_and_hazards():
    base = zb.build_zones(capture_fixture(), source="follower", operator="k", now=NOW)
    base["verify"] = {"status": "ok", "worst_m": 0.004, "checked_at": NOW.isoformat()}
    extra = {"left": zb.ArmCapture(), "right": zb.ArmCapture()}
    extra["left"].reach = {zb.cell(np.array([0.6, 0.1]))}
    extra["left"].hazards = {"camera mount": [(0.2, 0.3, 0.0)]}
    merged = zb.merge_leader(base, extra, operator="k", now=NOW + dt.timedelta(minutes=5))
    left = merged["arms"]["left"]
    assert [0.6, 0.1] in left["reach"] and len(left["reach"]) == len(base["arms"]["left"]["reach"]) + 1
    assert [z["name"] for z in left["keep_out"]] == ["table", "torso", "camera mount"]
    assert merged["source"] == "follower+leader"
    assert merged["verify"]["status"] == "ok"  # the table and references are the follower's, unchanged
    assert merged["references"] == base["references"]
    zones.parse_zones(merged)  # still a valid file


# -- CLI flow (no hardware): build, apply, status -----------------------------------------------------------------------

import shutil  # noqa: E402

import workspace as ws  # noqa: E402


def far_table_captures():
    """A table whose slab is well in front of the hanging arms, so rest, ready and the path are clear of it."""
    captures = capture_fixture()
    for side in ("left", "right"):
        captures[side].table_points = [(0.35, -0.25, -0.30), (0.75, -0.25, -0.30), (0.75, 0.25, -0.30), (0.35, 0.25, -0.30)]
        captures[side].hazards = {}
        captures[side].reach = {zb.cell(np.array([0.5, 0.0]))}
    return captures


@pytest.fixture
def station_env(tmp_path, monkeypatch):
    real = ws.STATION
    station = tmp_path / "station.yaml"
    shutil.copy2(real, station)
    zdir = tmp_path / "zones"
    zdir.mkdir()
    raw = zb.build_zones(far_table_captures(), "follower", "test", dt.datetime.now(dt.timezone.utc))
    path = zdir / "zones-20261023-1200.yaml"
    ws.save_raw(path, raw)
    monkeypatch.setattr(ws, "ZONES_DIR", zdir)
    monkeypatch.setenv("OPENARM_ZONES", str(tmp_path / "in_force.yaml"))
    monkeypatch.setenv("OPENARM_MAPPING_FILE", str(tmp_path / "model_mapping"))  # apply records the mapping here, never in the real ~/.dreamlabs
    return station, path


def test_build_passes_when_the_zones_leave_rest_ready_and_path_free(station_env, capsys):
    station, path = station_env
    assert ws.main(["build", "--zones", str(path), "--station", str(station)]) == 0
    assert "clear of every zone" in capsys.readouterr().out


def test_build_fails_when_a_zone_sits_on_the_ready_pose(station_env, capsys):
    station, path = station_env
    raw = ws.load_raw(path)
    tcp = zones.ArmKinematics("right").points(np.radians(poses.READY_DEG["right"]))["tcp"].tolist()
    raw["arms"]["right"]["keep_out"].append({"name": "blocker", "kind": "box", "center": tcp, "size": [0.2, 0.2, 0.2]})
    ws.save_raw(path, raw)
    assert ws.main(["build", "--zones", str(path), "--station", str(station)]) == 1
    assert "blocker" in capsys.readouterr().out


def test_apply_refuses_unverified_zones_then_writes_station_and_makes_them_current(station_env, tmp_path):
    station, path = station_env
    with pytest.raises(SystemExit, match="verif"):
        ws.main(["apply", "--zones", str(path), "--station", str(station)])
    raw = ws.load_raw(path)
    raw["verify"] = {"status": "ok", "worst_m": 0.003, "checked_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    ws.save_raw(path, raw)
    before = station.read_text()
    assert ws.main(["apply", "--zones", str(path), "--station", str(station)]) == 0
    new = yaml.safe_load(station.read_text())
    assert new["follower_l"]["keep_out"][0]["kind"] == "box" and new["follower_l"]["reach"]
    assert new["custom"]["driver"] == yaml.safe_load(before)["custom"]["driver"]  # the rest of the file is untouched
    assert (tmp_path / "in_force.yaml").exists()
    assert list(station.parent.glob("station.yaml.*.bak"))
    from robots import config
    config.load(station)  # Dream itself accepts the new keys


def test_status_reports_the_zones_in_force(station_env, capsys, monkeypatch):
    station, path = station_env
    shutil.copy2(path, ws.openarm_guard.zones_path())
    assert ws.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "zones in force" in out and "table" in out and "active arms" in out


# -- viewer ------------------------------------------------------------------------------------------------------------

def test_viewer_scene_builds_and_colours_a_live_pose():
    import socket

    import viewer
    import viser

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    raw = zb.build_zones(far_table_captures(), "follower", "t", dt.datetime.now(dt.timezone.utc))
    server = viser.ViserServer(port=port)
    try:
        handles = viewer.build_scene(server, raw)
        assert viewer.update_live(handles, "left", np.zeros(7)) == []  # hanging pose: clear
        tcp = zones.ArmKinematics("left").points(np.zeros(7))["tcp"].tolist()
        raw["arms"]["left"]["keep_out"].append({"name": "post", "kind": "box", "center": tcp, "size": [0.1, 0.1, 0.1]})
        handles = viewer.build_scene(viser.ViserServer(port=port + 1), raw)
        assert viewer.update_live(handles, "left", np.zeros(7))
    finally:
        server.stop()


def test_camera_panel_toggles_cameras_through_the_workers():
    import socket

    import viewer
    import viser

    class Source:
        closed = False

        def read(self):
            return np.zeros((4, 6, 3), dtype=np.uint8)

        def close(self):
            Source.closed = True

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = viser.ViserServer(port=port)
    try:
        panel = viewer.CameraPanel(server, [{"name": "head", "serial": "1"}, {"name": "wrist_l", "serial": "2"}], opener=lambda n, s: Source)
        assert set(panel.workers) == {"head", "wrist_l"} and not any(w.running for w in panel.workers.values())  # all start off
        panel.toggle("head", True)
        import time
        for _ in range(100):
            if panel.workers["head"].running:
                break
            time.sleep(0.01)
        assert panel.workers["head"].running and not panel.workers["wrist_l"].running
        panel.toggle("head", False)
        assert not panel.workers["head"].running and Source.closed
    finally:
        server.stop()


# -- round 2: the bundle carries each arm's home position -----------------------------------------------------------------

def test_bundle_has_the_frame_note_and_a_home_per_arm_defaulting_to_the_ready_pose():
    raw = zb.build_zones(capture_fixture(), source="editor", operator="k", now=NOW)
    assert "0.031" in raw["frame"] and "0.698" in raw["frame"]
    assert raw["home"]["left"] == pytest.approx(poses.ready_rad("left")) and len(raw["home"]["right"]) == 7
    parsed = zones.parse_zones(raw)
    assert parsed.source == "editor" and parsed.home["left"] == pytest.approx(poses.ready_rad("left"))


def test_a_given_home_is_kept_and_a_bad_one_is_refused():
    mine = {"left": [0.1] * 7, "right": [-0.1] * 7}
    raw = zb.build_zones(capture_fixture(), source="editor", operator="k", now=NOW, homes=mine)
    assert raw["home"] == mine
    raw["home"]["left"] = [0.0] * 6
    with pytest.raises(ValueError, match="home"):
        zones.parse_zones(raw)


def test_station_fragment_and_apply_carry_home_too(tmp_path):
    mine = {"left": [0.1] * 7, "right": [-0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1]}   # a pose inside Dream's joint ranges (elbow joint 4 runs 0 to 140 deg)
    raw = zb.build_zones(capture_fixture(), source="editor", operator="k", now=NOW, homes=mine)
    frag = zb.station_fragment(raw, homes={"left": (0.05, 0.12), "right": (0.05, -0.12)})
    assert frag["left"]["home"] == mine["left"] and frag["right"]["home"] == mine["right"]
    station = tmp_path / "station.yaml"
    station.write_text("body: custom\nfollower_l: {can_channel: can_fol_l, home: [0.0]}\nfollower_r: {can_channel: can_fol_r, home: [0.0]}\n")
    zb.apply_to_station(station, frag)
    new = yaml.safe_load(station.read_text())
    assert new["follower_l"]["home"] == mine["left"] and new["follower_r"]["home"] == mine["right"]


def test_home_pose_is_checked_against_the_zones_like_rest_and_ready():
    kin = zones.ArmKinematics("left")
    odd_home = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0]
    tcp = kin.points(np.array(odd_home))["tcp"].tolist()
    raw = zb.build_zones(capture_fixture(), source="editor", operator="k", now=NOW, homes={"left": odd_home, "right": [0.0] * 7})
    raw["arms"]["left"]["keep_out"].append({"name": "on_home", "kind": "box", "center": tcp, "size": [0.2, 0.2, 0.2]})
    found = ws_check(raw)
    assert any("home" in f and "on_home" in f for f in found)


def ws_check(raw):
    import workspace as ws
    return ws.check_free_poses(raw)


# -- the visual editor panel (no robot meshes, real viser server) ---------------------------------------------------------------

import socket  # noqa: E402
import sys  # noqa: E402

import test_collect  # noqa: E402

import designer  # noqa: E402
import editor as ed  # noqa: E402


@pytest.fixture
def panel(tmp_path):
    import viser

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = viser.ViserServer(port=port)
    p = designer.EditorPanel(server, ed.EditorState.default(), tmp_path / "out", "tester")
    yield p
    server.stop()


def test_panel_starts_clear_and_describes_it(panel):
    assert "No clash" in panel.status.content
    assert panel.f_select.value == "(none)"


def test_panel_add_select_edit_and_delete_a_box(panel):
    panel.f_name.value = "post"
    panel.add_box_clicked()
    assert [b.name for b in panel.state.boxes] == ["post"] and panel.selected == "post"
    panel.f_center.value = (0.5, 0.1, 0.6)
    panel._box_fields_changed()
    assert panel.state.boxes[0].center == (0.5, 0.1, 0.6)
    panel.f_name.value = "post"
    panel.add_box_clicked()  # a taken name gets a number instead of an error
    assert [b.name for b in panel.state.boxes] == ["post", "post 2"]
    panel.delete_clicked()
    assert [b.name for b in panel.state.boxes] == ["post"]


def test_panel_table_numbers_and_work_area_rectangle(panel):
    panel.set_table(0.45, (0.3, 0.9), (-0.4, 0.4), 0.25)
    t = panel.state.table
    assert t.top_z == 0.45 and t.x_range == (0.3, 0.9) and t.thickness == 0.25
    panel.f_area["left"][0].value, panel.f_area["left"][1].value = (0.4, 0.5), (0.0, 0.1)
    panel.fill_area("left")
    assert panel.state.arms["left"].reach
    panel.clear_area("left")
    assert not panel.state.arms["left"].reach


def test_panel_reports_a_clash_with_a_zone_on_the_ready_pose(panel):
    tcp = zones.ArmKinematics("right").points(np.radians(poses.READY_DEG["right"]))["tcp"]
    panel.state.add_box("blocker", tuple(ed.to_world("right", tcp)), (0.2, 0.2, 0.2), side="right")
    panel.redraw()
    assert "CLASH" in panel.status.content and "blocker" in panel.status.content


def test_panel_home_buttons_and_live_pose_message(panel):
    panel.use_home("left", "rest")
    assert panel.state.arms["left"].home == pytest.approx(np.radians(poses.REST_DEG["left"]))
    panel.use_home("left", "live")
    assert "No live follower" in panel.out_msg.content
    panel.read_q = lambda side: np.full(7, 0.1)
    panel.use_home("right", "live")
    assert panel.state.arms["right"].home == pytest.approx([0.1] * 7)


def test_panel_save_and_export_write_the_files(panel, tmp_path):
    panel.add_box_clicked()
    panel.save_clicked()
    yaml_path, png_path = panel.last_saved
    assert yaml_path.exists() and png_path.exists() and "Saved" in panel.out_msg.content
    keys = panel.export_clicked()
    assert keys.exists() and "follower_l" in keys.read_text()


def test_save_warns_when_there_is_a_clash(panel):
    tcp = zones.ArmKinematics("left").points(np.radians(poses.READY_DEG["left"]))["tcp"]
    panel.state.add_box("blocker", tuple(ed.to_world("left", tcp)), (0.2, 0.2, 0.2), side="left")
    panel.save_clicked()
    assert "clash" in panel.out_msg.content.lower()


def test_the_real_model_and_dreams_model_agree_once_the_base_offset_is_applied():
    """The meshes (v10 URDF in the body frame) and the zones (Dream's arm base frames) share one world: wrist positions match to 1e-5 m (the URDF writes pi/2 with 12 digits)."""
    import yourdfpy

    urdf = yourdfpy.URDF.load(str(designer.URDF), load_meshes=False, build_scene_graph=True, load_collision_meshes=False)
    rng = np.random.default_rng(11)
    for side in ("left", "right"):
        kin = zones.ArmKinematics(side)
        for _ in range(10):
            q = rng.uniform(-0.8, 0.8, 7)
            cfg = {f"openarm_{s}_joint{i + 1}": 0.0 for s in ("left", "right") for i in range(7)}
            cfg.update({f"openarm_{side}_joint{i + 1}": float(q[i]) for i in range(7)})
            urdf.update_cfg(np.array([cfg.get(n, 0.0) for n in urdf.actuated_joint_names]))
            for link, point in ((f"openarm_{side}_link7", "wrist"),):  # intermediate frames differ by up to 3 cm between the two models; the wrist is the shared point
                world = urdf.get_transform(link, "world")[:3, 3]
                assert world == pytest.approx(ed.to_world(side, kin.points(q)[point]), abs=1e-5), (side, point)


# -- region: the allowed box (round 3) ---------------------------------------------------------------------------------------

REGION = {"lo": [0.1, -0.3, -0.45], "hi": [0.6, 0.3, -0.05]}


def region_zones(region=REGION, side="left"):
    raw = zb.build_zones(capture_fixture(), source="teach", operator="k", now=NOW)
    raw["arms"][side]["region"] = region
    return raw


def test_region_is_parsed_and_checked_for_every_point():
    parsed = zones.parse_zones(region_zones())
    assert parsed.arms["left"].region == {"lo": [0.1, -0.3, -0.45], "hi": [0.6, 0.3, -0.05]} and parsed.arms["right"].region is None
    inside = {"elbow": np.array([0.3, 0.0, -0.2]), "tcp": np.array([0.5, 0.1, -0.4])}
    assert parsed.violations("left", inside, check_reach=False) == []
    outside = {"elbow": np.array([0.3, 0.0, -0.2]), "tcp": np.array([0.5, 0.1, -0.5])}  # 5 cm below the floor of the box
    found = parsed.violations("left", outside, check_reach=False)
    assert found and "tcp" in found[0] and "region" in found[0]
    assert parsed.violations("left", outside, check_reach=False, check_region=False) == []
    assert parsed.violations("right", outside, check_reach=False) == []  # no region on the right arm


def test_a_bad_region_is_refused():
    for bad in ({"lo": [0.1, 0, 0], "hi": [0.0, 1, 1]}, {"lo": [0.1, 0], "hi": [1, 1, 1]}, {"lo": [0, 0, 0]}):
        with pytest.raises(ValueError, match="region"):
            zones.parse_zones(region_zones(region=bad))


def test_pose_violations_check_the_region_only_for_home_and_live_poses():
    kin = zones.ArmKinematics("left")
    raw = region_zones(region={"lo": [0.2, -0.1, -0.1], "hi": [0.3, 0.1, 0.0]})  # a small box far from the hanging arm
    parsed = zones.parse_zones(raw)
    named = {"rest": poses.REST_DEG["left"], "home": poses.READY_DEG["left"]}
    found = zb.pose_violations(parsed, "left", named, kin)
    assert any(f.startswith("home") and "region" in f for f in found)       # home is outside this box
    assert not any(f.startswith("rest") for f in found)                      # rest hangs below the table: never held to the region


def test_station_fragment_and_apply_carry_the_region(tmp_path):
    raw = region_zones()
    frag = zb.station_fragment(raw, homes={"left": (0.05, 0.12), "right": (0.05, -0.12)})
    assert frag["left"]["region"] == REGION and "region" not in frag["right"]
    station = tmp_path / "station.yaml"
    station.write_text("body: custom\nfollower_l: {can_channel: can_fol_l, home: [0.0]}\nfollower_r: {can_channel: can_fol_r, home: [0.0]}\n")
    zb.apply_to_station(station, frag)
    new = yaml.safe_load(station.read_text())
    assert new["follower_l"]["region"] == REGION and "region" not in new["follower_r"]


# -- the teach folder of the editor panel (the leader moves the follower; the panel only listens) --------------------------------

import teach as th  # noqa: E402


def feed(panel, side, j1_list, j4=40.0):
    rest = np.radians(poses.REST_DEG[side])
    for a in j1_list:
        q = rest.copy()
        q[0], q[3] = np.radians(a if side == "left" else -a), np.radians(j4)
        panel.live_q = {side: q}
        panel.teach_tick({side: q})


def test_teach_without_a_live_source_says_so(panel):
    panel.set_home_from_follower("left")
    assert "No live" in panel.out_msg.content or "no live" in panel.out_msg.content.lower()


def test_sweeping_the_follower_builds_the_region_and_save_writes_it(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.sweep_changed("left", True)
    feed(panel, "left", range(0, 40, 3))
    panel.sweep_changed("left", False)
    panel.apply_sweep_clicked()
    region = panel.state.arms["left"].region
    assert region and region["lo"][2] < region["hi"][2] and panel.state.arms["left"].reach
    assert panel.state.arms["right"].region is None
    panel.save_clicked()
    raw = yaml.safe_load(panel.last_saved[0].read_text())
    assert raw["arms"]["left"]["region"] == region and "region" not in raw["arms"]["right"]


def test_home_from_the_follower_records_the_path_from_rest(panel):
    panel.read_q = lambda side: panel.live_q[side]
    feed(panel, "left", [0], j4=poses.REST_DEG["left"][3])  # the arm sits at its rest pose
    panel.start_path_clicked("left")
    assert panel.teach.recording_path("left")
    feed(panel, "left", range(0, 60, 4))
    panel.set_home_from_follower("left")
    arm = panel.state.arms["left"]
    assert arm.home_path and arm.home_path[0] == pytest.approx(poses.REST_DEG["left"], abs=0.5)
    assert arm.home == pytest.approx(list(panel.live_q["left"]))


def test_path_recording_refuses_to_start_away_from_rest(panel):
    panel.read_q = lambda side: panel.live_q[side]
    feed(panel, "left", [60])
    panel.start_path_clicked("left")
    assert not panel.teach.recording_path("left") and "rest" in panel.out_msg.content


def test_mark_here_adds_a_labelled_point(panel):
    panel.read_q = lambda side: panel.live_q[side]
    feed(panel, "left", [30])
    panel.f_mark.value = "far edge"
    panel.mark_clicked("left")
    assert panel.teach.marks("left")[0][0] == "far edge"


# -- round 4: the card (tab) page ------------------------------------------------------------------------------------------------

def rest_q(side):
    return np.radians(poses.REST_DEG[side])


def moving_q(side, j1, j4=None):
    q = rest_q(side).copy()
    sign = 1.0 if side == "left" else -1.0
    q[0] = q[0] + np.radians(sign * j1)
    q[3] = q[3] + np.radians(j1 * 0.8 if j4 is None else j4)
    return q


def test_page_has_the_four_cards_in_order_and_the_top_strip(panel):
    assert panel.tab_labels == ["1 Position", "2 Home", "3 Danger area", "4 Save", "5 Collect"]
    assert "No clash" in panel.status.content and "waiting" in panel.status.content.lower()


def test_position_cards_show_live_joints_and_waiting_for_the_arm_that_is_silent(panel):
    q = moving_q("left", 30)
    panel.update_position_cards({"left": q}, {"left": True, "right": False})
    left, right = panel.pos_cards["left"].content, panel.pos_cards["right"].content
    assert "LEFT" in left and f"{np.degrees(q)[0]:.1f}" in left and "hand" in left
    assert "waiting" in right.lower() and "J1" not in right
    assert "receiving" in panel.status.content.lower()


def test_set_home_records_the_path_and_the_cards_and_checklist_follow(panel):
    panel.read_q = lambda side: panel.live_q[side]
    for _ in range(4):
        panel.live_q = {"left": rest_q("left")}; panel.teach_tick(panel.live_q)
    for a in range(0, 50, 2):
        panel.live_q = {"left": moving_q("left", a)}; panel.teach_tick(panel.live_q)
    panel.set_home_clicked("left")
    arm = panel.state.arms["left"]
    assert arm.home_source == "teach" and arm.home_path and arm.home_path[0] == pytest.approx(poses.REST_DEG["left"], abs=0.5)
    assert "path from rest recorded" in panel.home_cards["left"].content
    assert "&#10004;" in panel.check_html.content and "Left home set" in panel.check_html.content


def test_set_home_without_a_path_says_so_on_the_card(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.live_q = {"right": moving_q("right", 40)}
    panel.teach_tick(panel.live_q)                  # the arm was never seen at rest or ready
    panel.set_home_clicked("right")
    assert "NO path" in panel.home_cards["right"].content


def test_set_home_without_a_live_follower_says_how_to_start_the_listener(panel):
    panel.set_home_clicked("left")
    assert "--observe" in panel.out_msg.content


def test_record_extremes_toggle_follows_the_chosen_arms_and_builds_the_region_live(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.record_side.value = "right"
    panel.record_toggle_clicked()                   # start
    assert panel.teach.sweeping("right") and not panel.teach.sweeping("left")
    for a in range(0, 60, 3):
        panel.live_q = {"right": moving_q("right", a)}; panel.teach_tick(panel.live_q)
    assert panel.state.arms["right"].region and panel.state.arms["left"].region is None   # the box grew while sweeping, no Apply press
    panel.record_toggle_clicked()                   # stop
    assert not panel.teach.sweeping("right")
    panel.record_side.value = "both"
    panel.record_toggle_clicked()
    assert panel.teach.sweeping("left") and panel.teach.sweeping("right")


def test_table_numbers_are_in_cm_and_mark_the_table_as_placed(panel):
    hanging_z = zones.ArmKinematics("left").points(np.zeros(7))["tcp"][2] + ed.BASE_OFFSET["left"][2]
    panel.f_height_cm.value, panel.f_front_cm.value, panel.f_depth_cm.value, panel.f_width_cm.value = 30.0, 40.0, 50.0, 70.0
    panel.table_cards_changed()
    t = panel.state.table
    assert t.top_z == pytest.approx(hanging_z + 0.30, abs=1e-6) and t.x_range == pytest.approx((0.40, 0.90)) and t.y_range == pytest.approx((-0.35, 0.35))
    assert panel.state.table_set


def test_danger_box_size_is_edited_in_cm(panel):
    panel.add_box_clicked()
    panel.f_size_cm.value = (20.0, 10.0, 40.0)
    panel._box_fields_changed()
    assert panel.state.boxes[0].size == pytest.approx((0.2, 0.1, 0.4))


def test_save_tab_checklist_lists_what_is_missing_and_ticks_off_what_is_done(panel):
    html = panel.check_html.content
    assert "Left home set" in html and "Table placed" in html and "&#10008;" in html
    panel.set_table(0.45, (0.3, 0.9), (-0.4, 0.4), 0.25)
    panel.state.table_set = True
    panel.refresh_checklist()
    assert panel.check_html.content.count("&#10004;") >= 2   # table placed and no clash


# -- round 5: the Danger card records points; the safe place is made from them -----------------------------------------------

def put_live(panel, side, j1, j4=None):
    panel.live_q = {side: moving_q(side, j1, j4)}
    panel.teach_tick(panel.live_q)


def test_record_point_uses_the_live_pose_of_the_chosen_arms(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.record_side.value = "left"
    put_live(panel, "left", 20)
    panel.record_point_clicked()
    assert panel.teach.point_count("left") == 1 and panel.teach.point_count("right") == 0
    panel.record_side.value = "both"
    panel.live_q = {"left": moving_q("left", 30), "right": moving_q("right", 30)}
    panel.record_point_clicked()
    assert panel.teach.point_count("left") == 2 and panel.teach.point_count("right") == 1


def test_the_safe_place_appears_from_the_second_point_and_the_card_says_what_is_missing(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.record_side.value = "left"
    put_live(panel, "left", 10); panel.record_point_clicked()
    assert panel.state.arms["left"].region is None and "at least" in panel.record_info.content
    put_live(panel, "left", 40, 70); panel.record_point_clicked()
    region = panel.state.arms["left"].region
    assert region and region["lo"][2] < region["hi"][2]
    assert "2 point" in panel.record_info.content


def test_undo_point_and_clear_update_the_safe_place(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.record_side.value = "left"
    for a in (10, 30, 50):
        put_live(panel, "left", a, a + 10); panel.record_point_clicked()
    panel.undo_point_clicked()
    assert panel.teach.point_count("left") == 2 and panel.state.arms["left"].region
    panel.clear_clicked()
    assert panel.teach.point_count("left") == 0 and panel.state.arms["left"].region is None


def test_record_point_without_a_live_arm_says_how_to_start_the_listener(panel):
    panel.record_point_clicked()
    assert "--observe" in panel.out_msg.content


def test_saved_bundle_carries_the_region_made_from_the_points(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.record_side.value = "right"
    for a, b in ((10, 20), (50, 20), (50, 80), (10, 80)):
        put_live(panel, "right", a, b); panel.record_point_clicked()
    panel.save_clicked()
    raw = yaml.safe_load(panel.last_saved[0].read_text())
    assert raw["arms"]["right"]["region"] and raw["arms"]["right"]["reach"] and "region" not in raw["arms"]["left"]


def test_the_allowed_box_always_holds_the_home_pose_so_a_save_has_no_clash(panel):
    import workspace as ws
    panel.read_q = lambda side: panel.live_q[side]
    panel.live_q = {"left": moving_q("left", 20)}
    for a in range(0, 30, 3):
        panel.live_q = {"left": moving_q("left", a)}; panel.teach_tick(panel.live_q)
    panel.set_home_clicked("left")
    panel.record_side.value = "left"
    far = moving_q("left", 5, 5)
    for q in (far, far + 0.02):
        panel.live_q = {"left": q}; panel.record_point_clicked()    # two points nowhere near the home pose
    panel.save_clicked()
    raw = yaml.safe_load(panel.last_saved[0].read_text())
    assert not [p for p in ws.check_free_poses(raw) if "left home" in p]
    assert "widened" in panel.out_msg.content


# -- round 6: preview the move from rest to home on the 3D robot ---------------------------------------------------------------------

class FakeRobot:
    def __init__(self):
        self.calls = []

    def pose(self, side, q):
        self.calls.append((side, np.array(q, dtype=float)))


@pytest.fixture
def panel_robot(tmp_path):
    import viser

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = viser.ViserServer(port=port)
    robot = FakeRobot()
    p = designer.EditorPanel(server, ed.EditorState.default(), tmp_path / "out", "tester", robot=robot)
    robot.calls.clear()
    yield p, robot
    server.stop()


def taught(side="left"):
    rest = poses.REST_DEG[side]
    sign = 1.0 if side == "left" else -1.0
    mid = [rest[0] + sign * 20, rest[1], rest[2], rest[3] + 25, rest[4], rest[5], rest[6]]
    home = [rest[0] + sign * 45, rest[1], rest[2], rest[3] + 60, rest[4], rest[5], rest[6]]
    return [list(rest), mid, home]


def test_preview_plays_the_recorded_path_from_rest_to_home_on_the_robot(panel_robot):
    panel, robot = panel_robot
    path = taught("left")
    panel.state.arms["left"].home_path = path
    panel.state.arms["left"].home = list(np.radians(path[-1]))
    panel.state.arms["left"].home_source = "teach"
    panel.pose_view = "live"
    panel.play_path(["left"], blocking=True, realtime=False)
    left = [q for s, q in robot.calls if s == "left"]
    assert len(left) > 20
    assert np.degrees(left[0]) == pytest.approx(path[0], abs=0.5) and np.degrees(left[-1]) == pytest.approx(path[-1], abs=0.5)
    assert not [1 for s, _ in robot.calls if s == "right"]            # only the chosen arm moved
    assert panel.pose_view == "live" and "home" in panel.out_msg.content.lower()


def test_preview_of_both_arms_moves_them_together(panel_robot):
    panel, robot = panel_robot
    for side in ed.SIDES:
        panel.state.arms[side].home_path = taught(side)
        panel.state.arms[side].home = list(np.radians(taught(side)[-1]))
    panel.play_path(list(ed.SIDES), blocking=True, realtime=False)
    sides = [s for s, _ in robot.calls]
    assert sides.count("left") > 10 and sides.count("right") > 10
    assert sides[:2] == ["left", "right"]                              # frames are interleaved


def test_preview_without_a_path_says_so(panel_robot):
    panel, robot = panel_robot
    panel.state.arms["right"].home_path = None
    panel.state.arms["right"].home_source = "teach"
    panel.play_path(["right"], blocking=True, realtime=False)
    assert "no path" in panel.out_msg.content.lower() and not robot.calls


def test_preview_with_the_ready_home_uses_the_built_in_path(panel_robot):
    panel, robot = panel_robot
    panel.state.arms["left"].home_source = "ready"
    panel.play_path(["left"], blocking=True, realtime=False)
    left = [q for s, q in robot.calls if s == "left"]
    assert np.degrees(left[0]) == pytest.approx(poses.REST_DEG["left"], abs=0.5) and np.degrees(left[-1]) == pytest.approx(poses.READY_DEG["left"], abs=0.5)


# -- round 7: gentle station settings written by apply ------------------------------------------------------------------------------

def test_apply_writes_the_gentle_station_settings_and_the_arm_driver(tmp_path):
    station = tmp_path / "station.yaml"
    station.write_text("body: custom\ncustom:\n  driver: \"/x/openarm_v2.py:ReadOnlyArm\"\n  ik: openarm2_faas\nmax_ee_speed_m_s: 0.6\nramp_s: 1.0\n"
                       "follower_l: {can_channel: can_fol_l, home: [0.0]}\nfollower_r: {can_channel: can_fol_r, home: [0.0]}\n")
    raw = zb.build_zones(capture_fixture(), source="editor", operator="k", now=NOW)
    zb.apply_to_station(station, zb.station_fragment(raw, homes={"left": (0.0, 0.1), "right": (0.0, -0.1)}), tuning=zb.GENTLE_STATION)
    new = yaml.safe_load(station.read_text())
    assert new["ramp_s"] == 3.0 and new["ramp_max_rad_s"] == 0.3 and new["max_ee_speed_m_s"] == 0.6
    assert new["custom"]["driver"].endswith("openarm_v2.py:Arm") and new["custom"]["ik"] == "openarm2_faas"


def test_apply_without_tuning_leaves_the_other_settings_alone(tmp_path):
    station = tmp_path / "station.yaml"
    station.write_text("body: custom\nmax_ee_speed_m_s: 0.6\nfollower_l: {can_channel: can_fol_l, home: [0.0]}\nfollower_r: {can_channel: can_fol_r, home: [0.0]}\n")
    raw = zb.build_zones(capture_fixture(), source="editor", operator="k", now=NOW)
    zb.apply_to_station(station, zb.station_fragment(raw, homes={"left": (0.0, 0.1), "right": (0.0, -0.1)}))
    assert yaml.safe_load(station.read_text())["max_ee_speed_m_s"] == 0.6


def test_reference_buttons_record_the_hand_position_of_the_live_arm(panel):
    panel.read_q = lambda side: panel.live_q[side]
    panel.live_q = {"left": moving_q("left", 30)}
    panel.record_reference_clicked("left")
    panel.live_q = {"left": moving_q("left", 40)}
    panel.record_reference_clicked("left")
    refs = panel.state.references["left"]
    assert len(refs) == 2 and refs[0] == pytest.approx(tuple(zones.ArmKinematics("left").points(moving_q("left", 30))["tcp"]), abs=1e-3)
    assert "Left reference points (2)" in panel.check_html.content and "2" in panel.out_msg.content
    panel.save_clicked()
    raw = yaml.safe_load(panel.last_saved[0].read_text())
    assert len(raw["references"]["left"]) == 2 and raw["references"]["right"] == []


def test_reference_button_without_a_live_arm_says_how_to_start_the_listener(panel):
    panel.record_reference_clicked("right")
    assert "--observe" in panel.out_msg.content


# -- language and the Collect tab ----------------------------------------------------------------------------------------
def test_language_button_switches_labels_cards_and_messages(panel):
    panel.set_language("日本語")
    assert panel.b_point.label == "この点を記録" and panel.b_home["left"].label == "左ホームをここに設定"
    assert "干渉なし" in panel.status.content and "ホーム" in panel.home_cards["left"].content
    assert panel.tab_labels[4] == "5 収集"
    assert panel.tabs._tab_labels == tuple(panel.tab_labels)  # what the browser is sent
    panel.use_home("left", "live")
    assert "ライブのフォロワー" in panel.out_msg.content
    panel.set_language("English")
    assert panel.b_point.label == "Record this point" and "No clash" in panel.status.content


def test_language_switch_keeps_the_chosen_arm_and_box_side(panel):
    panel.record_side.value = panel.record_side.options[1]  # right
    panel.set_language("日本語")
    assert panel._record_sides() == ("right",)
    panel.set_language("English")
    assert panel._record_sides() == ("right",)


def test_every_translated_key_has_both_languages():
    import i18n

    assert all(en.strip() and ja.strip() for en, ja in i18n.STRINGS.values())


@pytest.fixture
def collecting(panel, tmp_path):
    import collect
    import test_collect

    script = tmp_path / "fake.py"
    script.write_text(test_collect.FAKE)
    env = tmp_path / "x.env"
    env.write_text("TASK_SLUG=x\n")
    panel.collector = collect.CollectSession(repo=tmp_path, teleop_probe=lambda: None, argv_for=lambda o: ([sys.executable, str(script)], {}),
                                             set_file=tmp_path / "set.json")
    panel.collect_opts.env_file = env
    panel.f_collector.value = "Tester"
    panel._collect_buttons()  # setting a value from code does not fire on_update
    yield panel
    if panel.collector.active:
        panel.collector.proc.kill()


def test_collect_buttons_start_send_keys_and_stop(collecting):
    p = collecting
    assert p.b_cstart.disabled is False and p.b_cnext.disabled is True
    p.c_return.value = False  # without the automatic return, an episode saved the instant it ends cannot be discarded
    p.collect_start_clicked()
    assert p.b_cstart.disabled is True and p.b_cnext.disabled is False and p.b_crerecord.disabled is True
    test_collect.wait_for(lambda: p.collector.phase == "ready")
    p.collect_key("next")
    test_collect.wait_for(lambda: p.collector.phase == "recording")
    p.collect_tick()
    assert "RECORDING episode 3" in p.collect_card.content
    p.set_language("日本語")
    assert "録画中 エピソード3" in p.collect_card.content
    p.collect_key("stop")
    assert p.collector.wait() == 0
    p.collect_tick()
    assert p.b_cstart.disabled is False


def test_collect_start_refused_with_a_message_while_a_teleop_runs(collecting):
    collecting.collector._probe = lambda: "run_teleop"
    collecting.collect_start_clicked()
    assert "run_teleop" in collecting.out_msg.content and not collecting.collector.active
    collecting.set_language("日本語")
    collecting.collect_start_clicked()
    assert "テレオペがまだ動いています" in collecting.out_msg.content


def test_position_card_and_home_tab_show_the_hand_pitch(panel):
    wood_left = np.radians([64.42, -3.62, -1.23, 95.91, -2.48, 19.01, 21.6])
    panel.update_position_cards({"left": wood_left}, {"left": True, "right": False})
    assert "-69" in panel.pos_cards["left"].content and "-69" in panel.pitch_info.content
    panel.set_language("日本語")
    assert "手先の向き" in panel.pitch_info.content


def test_save_start_set_needs_both_arms_then_saves_and_fills_the_collect_tab(panel, monkeypatch):
    panel.f_setname.value = "new-v2"
    panel.save_start_set_clicked()
    assert "Both arms" in panel.out_msg.content
    q = np.radians([64.42, -3.62, -1.23, 95.91, -2.48, 19.01, -57.0])
    panel.update_position_cards({"left": q, "right": q}, {"left": True, "right": True})
    saved = {}
    monkeypatch.setattr(designer.startpose, "save", lambda name, base, joints, **k: saved.update(name=name, base=base, joints=joints) or "ok")
    panel.save_start_set_clicked()
    assert saved["name"] == "new-v2" and saved["joints"]["left"][6] == pytest.approx(-57.0)
    assert panel.f_poseset.value == "new-v2" and "new-v2" in panel.out_msg.content
    monkeypatch.setattr(designer.startpose, "save", lambda *a, **k: (_ for _ in ()).throw(designer.startpose.StartPoseError("start.err.exists|new-v2")))
    panel.save_start_set_clicked()
    assert "already exists" in panel.out_msg.content


def test_collect_options_carry_the_episode_time_and_pose_set(panel):
    panel.f_time.value, panel.f_poseset.value = 600, "wood-v2"
    opts = panel._collect_options()
    assert (opts.episode_time_s, opts.pose_set) == (600, "wood-v2")


def test_collect_pose_set_is_its_own_field_and_defaults_to_the_settings_file(panel):
    assert panel.f_pose.options == ("live", "home", "rest", "ready")  # the Advanced "show robot at" dropdown, a different widget
    assert panel._collect_options().pose_set in ("", collect.DEFAULT_POSE_SET)  # the default set or nothing (the settings file decides): never "live"
    panel.f_poseset.value = ""
    assert panel._collect_options().pose_set == ""


def test_collect_start_refuses_an_unknown_pose_set_before_running(collecting):
    collecting.f_poseset.value = "no-such-set"
    collecting.collect_start_clicked()
    assert "no-such-set" in collecting.out_msg.content and not collecting.collector.active


def test_collect_hold_checkbox_reaches_the_options(panel):
    assert panel._collect_options().start_pose_hold is True  # on by default
    panel.c_hold.value = False
    assert panel._collect_options().start_pose_hold is False
    panel.c_hold.value, panel.f_countdown.value = True, 5
    opts = panel._collect_options()
    assert (opts.start_pose_hold, opts.start_countdown_s) == (True, 5)


def test_discard_button_works_with_the_automatic_return_even_if_saving_at_the_end(panel):
    class Running:
        active = True
        def snapshot(self, last=12):
            return {"phase": "ready", "episode": 0, "started": 0, "rc": None, "run_dir": None, "target": None, "done": 0, "left": None, "saved": 0, "lines": [], "active": True}
        def pending(self):
            return None
        def review_pending(self):
            return None
    panel.collector = Running()
    panel.c_saveend.value, panel.c_return.value = True, True
    panel._collect_buttons()
    assert panel.b_crerecord.disabled is False
    panel.c_return.value = False
    panel._collect_buttons()
    assert panel.b_crerecord.disabled is True


def test_start_pose_set_field_starts_with_the_default_set_when_it_exists(panel, monkeypatch, tmp_path):
    expected = collect.DEFAULT_POSE_SET if (startpose.pose_sets_dir() / collect.DEFAULT_POSE_SET).is_dir() else ""
    assert panel.f_poseset.value == expected
    monkeypatch.setattr(designer.startpose, "pose_sets_dir", lambda repo=None: tmp_path)  # no such set: the field stays empty
    import viser
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = viser.ViserServer(port=port)
    try:
        assert designer.EditorPanel(server, ed.EditorState.default(), tmp_path / "out", "tester").f_poseset.value == ""
    finally:
        server.stop()


def test_collect_resume_button_offers_only_a_stopped_set(collecting, tmp_path):
    p = collecting
    p.collect_tick(force=True)
    assert p.b_cresume.disabled is True and p.resume_info.content == ""
    (tmp_path / "ds" / "meta").mkdir(parents=True)
    (tmp_path / "ds" / "meta" / "info.json").write_text('{"total_episodes": 8}')
    p.collector.set_file.write_text('{"mode": "gate", "tag": "", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % (tmp_path / "ds"))
    p.set_language("日本語")
    p.collect_tick(force=True)
    assert p.b_cresume.disabled is False and "残り **2 本**" in p.resume_info.content
    p.collect_resume_clicked()
    assert p.collector.target == 2
    test_collect.wait_for(lambda: p.collector.phase == "ready")
    p.collect_key("stop")
    assert p.collector.wait() == 0


def test_start_waits_for_an_operator_and_passes_the_typed_name(collecting):
    p = collecting
    p.f_collector.value = ""
    p._collect_buttons()
    assert p.b_cstart.disabled is True
    p.collect_start_clicked()  # a click that slips past the disabled button is refused too
    assert p.collector.phase == "idle" and p.out_msg.content == p.lang.t("col.err.operator")
    p.f_collector.value = " Suraj "
    p._collect_buttons()
    assert p.b_cstart.disabled is False and p._collect_options().operator == "Suraj"
    assert p.f_operator.value != "Suraj"  # the Advanced "operator" (who saves the zones) is a different field


def test_the_chosen_mode_and_language_are_marked_and_still_read_right(collecting):
    import designer

    p = collecting
    assert p.f_mode.options[0].startswith(designer.MARK)  # dry, chosen at start
    p.f_mode.value = p.f_mode.options[1]  # a click on gate
    assert p.f_mode.options[1].startswith(designer.MARK) and not p.f_mode.options[0].startswith(designer.MARK)
    assert p._collect_options().mode == "gate"
    p.b_lang.value = p.b_lang.options[1]  # 日本語
    assert p.lang.code != "en" and p.b_lang.options[1].startswith(designer.MARK)
    assert p.f_mode.options[1].startswith(designer.MARK) and p._collect_options().mode == "gate"  # relabelled, still gate


def test_operator_and_settings_are_frozen_while_a_run_is_going(collecting):
    import designer

    p = collecting
    fields = p._collect_fields()
    assert p.f_collector in fields and p.f_poseset in fields
    assert all(f.disabled is False for f in fields)
    p.collect_start_clicked()
    assert all(f.disabled is True for f in fields)  # one session = one operator and one set of values (instructions section 1)
    assert p.b_lang.disabled is False  # the language can still be switched
    p.f_mode.value = p.f_mode.options[1]  # a click on gate during the dry run is undone
    assert p.f_mode.options[0].startswith(designer.MARK) and p._collect_options().mode == "dry"
    test_collect.wait_for(lambda: p.collector.phase == "ready")
    p.collect_key("stop")
    assert p.collector.wait() == 0
    p.collect_tick(force=True)
    assert all(f.disabled is False for f in fields)


def test_choosing_a_task_file_shows_its_values_and_runs_with_it(collecting, tmp_path):
    import collect

    p = collecting
    cfg = tmp_path / "config"
    cfg.mkdir()
    wood = cfg / "bimanual-collect.wood.env"
    wood.write_text('TASK="Pick up the wood"\nSCENE=wood-v1\nPOSE_SET=wood-pose\nEPISODE_TIME_S=600\n')
    (cfg / "bimanual-collect.sort.env").write_text("TASK=Sort\nSCENE=sort-v1\nPOSE_SET=sort-pose\nEPISODE_TIME_S=120\n")
    p.collect_env_refresh()
    assert set(p.f_envfile.options) >= {"bimanual-collect.wood.env", "bimanual-collect.sort.env"}
    p.f_envfile.value = "bimanual-collect.wood.env"
    for value in ("Pick up the wood", "wood-v1", "wood-pose", "600"):
        assert value in p.env_info.content
    assert "wood-pose" in p.f_poseset.label and "600" in p.f_time.label  # the "empty = settings file" hints follow the chosen file
    opts = p._collect_options()
    assert opts.env_file == wood and collect.command(opts)[1]["ENV_FILE"] == str(wood)  # what command.sh will show
    p.f_envfile.value = "bimanual-collect.sort.env"
    assert "Sort" in p.env_info.content and "120" in p.env_info.content
    assert p.f_envfile in p._collect_fields()  # frozen while a run is going, like the operator


def test_a_start_pose_set_other_than_the_task_files_gets_a_yellow_warning(collecting, tmp_path):
    import cards

    p = collecting
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "bimanual-collect.wood.env").write_text("TASK=wood\nPOSE_SET=wood-pose\n")
    p.collect_env_refresh()
    p.f_envfile.value = "bimanual-collect.wood.env"
    p.f_poseset.value = "takeei-v5"
    assert cards.AMBER in p.pose_warn.content and "takeei-v5" in p.pose_warn.content and "wood-pose" in p.pose_warn.content
    p.f_poseset.value = "wood-pose"
    assert p.pose_warn.content == ""  # the same set: no warning
    p.f_poseset.value = ""
    assert p.pose_warn.content == ""  # empty uses the file's set
    p.f_poseset.value = "takeei-v5"
    p.set_language("日本語")
    assert "takeei-v5" in p.pose_warn.content and p.pose_warn.content != ""


def test_after_a_gate_the_page_waits_for_the_review_before_a_session(collecting, tmp_path):
    p = collecting
    (tmp_path / "ds" / "meta").mkdir(parents=True)
    (tmp_path / "ds" / "meta" / "info.json").write_text('{"total_episodes": 10}')
    p.collector.set_file.write_text('{"mode": "gate", "tag": "", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % (tmp_path / "ds"))
    p.collect_tick(force=True)
    assert p.review_box.visible is True and "10" in p.review_info.content
    p.f_mode.value = p.f_mode.options[2]  # session
    p.f_tag.value = "s2"
    p.collect_start_clicked()
    assert p.collector.phase == "idle" and p.out_msg.content == p.lang.t("col.err.review")
    p.c_reviewed.value = True  # ticked with no reviewer name: refused, stays unticked
    assert p.c_reviewed.value is False and p.collector.review_pending() is not None
    p.f_reviewer.value = "Kota"
    p.c_reviewed.value = True
    assert p.collector.review_pending() is None and p.review_box.visible is False


def test_an_emptied_episode_count_means_the_default_and_a_bad_one_is_refused(collecting):
    p = collecting
    p.f_episodes.value = ""
    assert p._collect_options().episodes == 0  # the script's default (gate 10, session 10)
    p.f_episodes.value = " 12 "
    assert p._collect_options().episodes == 12
    p.f_episodes.value = "1o"
    p.collect_start_clicked()
    assert "1o" in p.out_msg.content and not p.collector.active


def test_the_station_file_gives_dreams_ik_the_robots_column_for_both_arms():
    """log-040: Dream's own IK plans around the robot's centre column (`keep_out`), next to the table."""
    import openarm_body as body

    raw = zb.build_zones(capture_fixture(), source="editor", operator="k", now=NOW)
    fragment = zb.station_fragment(raw, {"left": (0.0, 0.1), "right": (0.0, -0.1)})
    for side in ("left", "right"):
        assert body.dream_keep_out(side)[0] in fragment[side]["keep_out"]


def test_a_full_width_tag_is_flagged_as_it_is_typed_and_a_stopped_set_asks_before_a_new_one(collecting, tmp_path):
    p = collecting
    p.f_tag.value = "s１"
    assert "s１" in p.tag_warn.content and p.out_msg.content == p.lang.t("col.err.tag.chars", what="s１")
    p.f_tag.value = "s2"
    assert p.tag_warn.content == ""
    (tmp_path / "ds" / "meta").mkdir(parents=True)
    (tmp_path / "ds" / "meta" / "info.json").write_text('{"total_episodes": 7}')
    p.collector.set_file.write_text('{"mode": "session", "tag": "s1", "target": 10, "dataset_root": "%s", "repo_id": "Orboh/ds"}' % (tmp_path / "ds"))
    p.collect_tick(force=True)
    assert p.c_replace.visible is True
    p.f_mode.value = p.f_mode.options[2]  # session
    p.collect_start_clicked()
    assert p.collector.phase == "idle" and "7/10" in p.out_msg.content

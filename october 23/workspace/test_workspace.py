"""Tests for the daily safety-area tools: building zones from captured points, the capture session, build/apply/verify. No hardware."""

import datetime as dt
import json

import numpy as np
import pytest
import yaml

import capture as cap
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
    mine = {"left": [0.1] * 7, "right": [-0.1] * 7}
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

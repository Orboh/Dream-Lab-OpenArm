"""Tests for the Stage 3 pieces: the table from tape measurements, the reach-over-table and unvisited-gap checks, and the report."""

import argparse
import types

import numpy as np
import pytest

import cards
import editor as ed
import stage3
import teach as th
import workspace as ws


def test_table_from_measurements_uses_the_body_frame_and_the_floor_level_base_plate():
    t = ed.table_from_measurements(13, 25, 45, 60)
    assert t.top_z == pytest.approx(0.25) and t.x_range == pytest.approx((0.13, 0.58)) and t.y_range == pytest.approx((-0.30, 0.30))
    off = ed.table_from_measurements(13, 25, 45, 60, centre_y_cm=10)
    assert off.y_range == pytest.approx((-0.20, 0.40))
    with pytest.raises(ValueError):
        ed.table_from_measurements(13, 25, 0, 60)
    left = ed.to_base("left", t.keep_out()["center"])  # in the arm's frame the top is 44.8 cm below the shoulder
    assert left[2] + t.keep_out()["size"][2] / 2 == pytest.approx(0.25 - 0.698)


def painted(state, side, cells):
    state.arms[side].reach = set(cells)


def test_reach_over_table_counts_the_cells_above_the_table_seen_from_above():
    state = ed.EditorState.default()
    state.table = ed.table_from_measurements(13, 25, 45, 60)
    assert state.reach_over_table("left") == (0, 0)
    painted(state, "left", {(0, 0), (10, 0), (11, 1), (-5, 0)})  # x = 0, 20, 22, -10 cm: only the 20 and 22 cm cells are past the front edge (13 cm)
    assert state.reach_over_table("left") == (2, 4)
    painted(state, "right", {(0, 0)})
    assert state.reach_over_table("right") == (0, 1)


def fake_points(session, side, xyz_list):
    session._recorded[side] = [{"pts": {"tcp": np.asarray(p, dtype=float)}} for p in xyz_list]


def test_a_polygon_through_far_apart_points_reports_the_unvisited_space_inside():
    session = th.TeachSession()
    assert session.reach_holes("left") == 0  # nothing recorded
    fake_points(session, "left", [(0.15, -0.30, -0.4), (0.15, 0.30, -0.4), (0.55, 0.0, -0.4)])
    assert session.reach_holes("left") > 20  # the filled triangle has a middle nobody visited
    fake_points(session, "left", [(0.15, -0.30, -0.4), (0.15, 0.30, -0.4), (0.55, 0.0, -0.4), (0.28, 0.0, -0.4), (0.40, 0.0, -0.4), (0.25, -0.15, -0.4), (0.25, 0.15, -0.4)])
    assert session.reach_holes("left") == 0  # points inside the area close the gaps


def test_checklist_asks_for_reach_on_the_table_and_no_unvisited_gaps():
    state = ed.EditorState.default()
    state.table = ed.table_from_measurements(13, 25, 45, 60)
    session = th.TeachSession()
    items = {label: (ok, hint) for label, ok, hint in cards.build_checklist(state, session)}
    assert items["Left work area reaches the table"][0] is False and "0 of 0" in items["Left work area reaches the table"][1]
    painted(state, "left", {(10, 0), (11, 0)})
    fake_points(session, "left", [(0.15, -0.30, -0.4), (0.15, 0.30, -0.4), (0.55, 0.0, -0.4)])
    items = {label: (ok, hint) for label, ok, hint in cards.build_checklist(state, session)}
    assert items["Left work area reaches the table"][0] is True
    assert items["Left work area has no large unvisited gaps"][0] is False and "15 cm" in items["Left work area has no large unvisited gaps"][1]
    from i18n import Lang
    ja = {label for label, _, _ in cards.build_checklist(state, session, Lang("ja"))}
    assert "左の作業エリアが机に届いている" in ja


def test_surface_points_must_be_three_flat_and_match_the_stated_table():
    t = ed.table_from_measurements(13, 25, 45, 60)
    good = [(13.2, -20, 25.3), (40, 0, 24.8), (57, 25, 25.0)]
    assert stage3.check_surface_points(t, good) == []
    assert "at least three" in stage3.check_surface_points(t, good[:2])[0]
    assert any("not flat" in p for p in stage3.check_surface_points(t, [(13, 0, 25), (40, 0, 28), (50, 0, 25)]))
    assert any("front edge" in p for p in stage3.check_surface_points(t, [(20, 0, 25), (40, 0, 25), (50, 0, 25)]))
    assert any("outside" in p for p in stage3.check_surface_points(t, [(13, 0, 25), (40, 0, 25), (70, 0, 25)]))


def test_report_lists_every_pose_with_its_clearance_and_flags_a_clash():
    state = ed.EditorState.default()
    state.table = ed.table_from_measurements(13, 25, 45, 60)
    text = "\n".join(stage3.report(state))
    assert "table (body frame): front edge x = 13.0 cm" in text and "rest" in text and "ready" in text and "above the table top" in text
    state.table = ed.table_from_measurements(-5, 25, 45, 60)  # a table whose near edge is under the hanging arms: the rest pose now hits it
    assert "CLASH" in "\n".join(stage3.report(state))


def test_design_arguments_place_the_table_but_leave_it_unconfirmed(capsys):
    state = ed.EditorState.default()
    args = argparse.Namespace(table_front_cm=13.0, table_top_cm=25.0, table_depth_cm=45.0, table_width_cm=60.0, table_centre_y_cm=0.0)
    assert ws.apply_table_args(state, args) and state.table.x_range[0] == pytest.approx(0.13) and state.table_set is False
    assert "table from the tape" in capsys.readouterr().out
    none = argparse.Namespace(table_front_cm=None, table_top_cm=None, table_depth_cm=45.0, table_width_cm=60.0, table_centre_y_cm=0.0)
    assert ws.apply_table_args(ed.EditorState.default(), none) is False
    with pytest.raises(SystemExit):
        ws.apply_table_args(ed.EditorState.default(), argparse.Namespace(table_front_cm=13.0, table_top_cm=None, table_depth_cm=45.0, table_width_cm=60.0, table_centre_y_cm=0.0))


# -- danger-only: name only the places the arms must stay out of ---------------------------------------------------------

def danger_state():
    state = ed.EditorState.default()
    state.table = ed.table_from_measurements(13, 25, 100, 140)  # generous on purpose: only "this is a danger zone" is said, not the table's exact size
    state.danger_only = True
    return state


def test_a_danger_only_bundle_carries_the_danger_zones_and_no_work_area_or_region():
    import datetime as dt

    state = danger_state()
    for side in ed.SIDES:
        state.use_ready(side)  # a home that is the ready pose: the built-in rest-to-ready path is its path
    state.arms["left"].region = {"lo": [0, 0, 0], "hi": [1, 1, 1]}
    state.arms["left"].reach = {(1, 1)}  # painted by mistake: danger-only must not write it
    raw = state.to_bundle("t", dt.datetime.now(dt.timezone.utc))
    assert raw["mode"] == "danger_only"
    for side in ed.SIDES:
        assert "reach" not in raw["arms"][side] and "region" not in raw["arms"][side] and raw["arms"][side]["keep_out"][0]["name"] == "table"
        assert raw["home_path"][side][0] == pytest.approx(__import__("openarm_poses").REST_DEG[side])  # the known rest-to-ready path stands in for a taught one
    again = ed.EditorState.from_bundle(raw)
    assert again.danger_only is True and again.table.x_range[0] == pytest.approx(0.13)


def test_the_station_keys_of_a_danger_only_bundle_are_keep_out_and_home_only(tmp_path):
    import datetime as dt

    import yaml
    import zonebuild as zb

    raw = danger_state().to_bundle("t", dt.datetime.now(dt.timezone.utc))
    fragment = zb.station_fragment(raw, {"left": (0.0, 0.1), "right": (0.0, -0.1)})
    for side in ed.SIDES:
        assert set(fragment[side]) == {"keep_out", "home"}
    station = tmp_path / "station.yaml"
    station.write_text(yaml.safe_dump({"follower_l": {"can_channel": "can_fol_l", "home": [0] * 7, "reach": [[0.0, 0.0]], "region": {"lo": [0, 0, 0], "hi": [1, 1, 1]}},
                                       "follower_r": {"can_channel": "can_fol_r", "home": [0] * 7, "reach": [[0.0, 0.0]]}}))
    zb.apply_to_station(station, fragment)
    written = yaml.safe_load(station.read_text())
    for key in ("follower_l", "follower_r"):  # the old painted area and region are removed, not left behind
        assert "reach" not in written[key] and "region" not in written[key] and written[key]["keep_out"]


def test_dreams_station_loader_accepts_the_danger_only_arm_sections(tmp_path):
    import datetime as dt

    import yaml
    from robots import config

    import workspace as ws_mod
    import zonebuild as zb

    raw = danger_state().to_bundle("t", dt.datetime.now(dt.timezone.utc))
    station = tmp_path / "station.yaml"
    station.write_text(ws_mod.STATION.read_text())
    zb.apply_to_station(station, zb.station_fragment(raw, {"left": (0.0, 0.1), "right": (0.0, -0.1)}), tuning=zb.GENTLE_STATION)
    loaded = config.load(str(station))
    for arm in (loaded.follower_l, loaded.follower_r):
        assert arm.reach is None and arm.region is None and arm.keep_out  # Dream: "the hand can go anywhere", outside the danger zones


def test_the_danger_only_checklist_asks_only_for_home_table_and_no_clash():
    state = danger_state()
    labels = [label for label, _, _ in cards.build_checklist(state, th.TeachSession())]
    assert not any("allowed area" in l.lower() or "reference" in l.lower() or "work area" in l.lower() or "overlap" in l.lower() for l in labels)
    assert "Table placed" in labels and "No clash" in labels and "Left home set" in labels


def test_danger_only_zones_still_guard_the_arms_and_have_no_painted_area():
    import datetime as dt

    import openarm_zones as zones

    raw = danger_state().to_bundle("t", dt.datetime.now(dt.timezone.utc))
    parsed = zones.parse_zones(raw)
    assert parsed.arms["left"].reach is None and parsed.arms["left"].region is None
    kin = zones.ArmKinematics("left", logical=True)
    low = np.radians([0, 0, 0, 0, 0, 0, 0])
    assert parsed.violations("left", kin.points(low), check_reach=False) == []  # hanging beside the near edge: free
    ready = np.radians(__import__("openarm_poses").READY_DEG["left"])
    for top_cm in range(25, 90, 5):  # raise the table until the ready pose reaches into it: the table zone, and only it, must object
        state = danger_state()
        state.table = ed.table_from_measurements(-20, top_cm, 100, 140)  # near edge behind the shoulders, so the raised arm is over it
        found = zones.parse_zones(state.to_bundle("t", dt.datetime.now(dt.timezone.utc))).violations("left", kin.points(ready), check_reach=False)
        if found:
            assert all("keep-out zone 'table'" in v for v in found)
            return
    pytest.fail("the ready pose never reached the table")


# -- lowering the danger box top (an option, off by default) -------------------------------------------------------------

def top_of(zone) -> float:
    return zone["center"][2] + zone["size"][2] / 2


def test_the_danger_box_top_is_at_the_surface_unless_lowered():
    flat = ed.table_from_measurements(13, 25, 100, 140)
    assert flat.lower_cm == 0.0 and top_of(flat.keep_out()) == pytest.approx(0.25) and flat.lower_warning() is None
    two = ed.table_from_measurements(13, 25, 100, 140, lower_cm=2.0)
    assert top_of(two.keep_out()) == pytest.approx(0.23) and two.top_z == pytest.approx(0.25) and two.lower_warning() is None  # 2 cm: the 4 cm safety balls just touch the real table
    three = ed.table_from_measurements(13, 25, 100, 140, lower_cm=3.0)
    assert "dip 1 cm" in three.lower_warning()
    for bad in (-1.0, 4.5):
        with pytest.raises(ValueError):
            ed.table_from_measurements(13, 25, 100, 140, lower_cm=bad)


def test_the_lowering_is_saved_with_the_bundle_and_restored_on_load():
    import datetime as dt

    state = danger_state()
    state.table.lower_cm = 2.0
    raw = state.to_bundle("t", dt.datetime.now(dt.timezone.utc))
    assert raw["table_danger_lowered_cm"] == 2.0
    box = next(z for z in raw["arms"]["left"]["keep_out"] if z["name"] == "table")
    assert top_of(box) == pytest.approx(0.25 - 0.02 - 0.698)  # arm base frame, box top 2 cm below the surface
    back = ed.EditorState.from_bundle(raw)
    assert back.table.lower_cm == 2.0 and back.table.top_z == pytest.approx(0.25)  # the real surface comes back, not the lowered box top
    plain = danger_state().to_bundle("t", dt.datetime.now(dt.timezone.utc))
    assert "table_danger_lowered_cm" not in plain


def test_the_page_has_a_field_for_it_that_starts_at_zero_and_warns_past_the_safe_limit(tmp_path):
    import socket

    import viser

    import designer

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = viser.ViserServer(port=port)
    try:
        panel = designer.EditorPanel(server, danger_state(), tmp_path / "out", "tester")
        assert panel.f_lower_cm.value == 0.0
        panel.f_lower_cm.value = 2.0
        panel.lower_changed()
        assert panel.state.table.lower_cm == 2.0 and panel.out_msg.content == ""
        panel.f_lower_cm.value = 3.5
        panel.lower_changed()
        assert "dip 1.5 cm" in panel.out_msg.content
        panel.set_language("日本語")
        panel.lower_changed()
        assert "めり込む" in panel.out_msg.content
    finally:
        server.stop()


def test_start_up_flag_and_report_carry_the_option(capsys):
    state = ed.EditorState.default()
    args = argparse.Namespace(table_front_cm=13.0, table_top_cm=25.0, table_depth_cm=100.0, table_width_cm=140.0, table_centre_y_cm=0.0, table_lower_cm=3.0)
    assert ws.apply_table_args(state, args) and state.table.lower_cm == 3.0
    assert "WARNING" in capsys.readouterr().out
    text = "\n".join(stage3.report(state))
    assert "lowered by 3 cm" in text and "WARNING" in text
    ok = ed.EditorState.default()
    ok.table = ed.table_from_measurements(13, 25, 100, 140)
    assert "lowered by 0 cm" in "\n".join(stage3.report(ok)) and "WARNING" not in "\n".join(stage3.report(ok))


# -- build and apply of a danger-only bundle (the same path the saved file will take in Stage 4) -----------------------------

from test_workspace import station_env  # noqa: E402,F401  (a copy of the real station file, a temporary zones folder and in-force file)


def saved_danger_bundle(tmp_path):
    import datetime as dt
    import pathlib

    import outputs

    state = danger_state()
    for side in ed.SIDES:
        state.use_ready(side)
    yaml_path, _ = outputs.save_bundle(state, "tester", pathlib.Path(tmp_path))
    return yaml_path


def test_build_of_a_danger_only_bundle_names_no_reach_and_passes(station_env, tmp_path, capsys):  # noqa: F811
    station, _ = station_env
    path = saved_danger_bundle(tmp_path)
    assert ws.main(["build", "--zones", str(path), "--station", str(station)]) == 0
    out = capsys.readouterr().out
    assert "no reach and no region" in out and "clear of every zone" in out


def test_apply_of_a_danger_only_bundle_needs_no_verification_and_removes_the_painted_area(station_env, tmp_path, capsys):  # noqa: F811
    import yaml

    station, _ = station_env
    data = yaml.safe_load(station.read_text())
    for key in ("follower_l", "follower_r"):  # give the copy an old painted area and region, as the station file had on 10-23
        data[key]["reach"], data[key]["region"] = [[0.0, 0.0]], {"lo": [-0.2, -0.1, -0.5], "hi": [0.1, 0.2, -0.1]}
    station.write_text(yaml.safe_dump(data))
    assert "reach" in yaml.safe_load(station.read_text())["follower_l"] and "region" in yaml.safe_load(station.read_text())["follower_r"]
    path = saved_danger_bundle(tmp_path)
    assert ws.main(["apply", "--zones", str(path), "--station", str(station)]) == 0
    assert "applied without a reference-point check" in capsys.readouterr().out
    new = yaml.safe_load(station.read_text())
    for key in ("follower_l", "follower_r"):
        assert "reach" not in new[key] and "region" not in new[key] and new[key]["keep_out"][0]["kind"] == "box"
    from robots import config

    loaded = config.load(station)
    assert loaded.follower_l.reach is None and loaded.follower_l.region is None  # Dream accepts it: the hand can go anywhere outside the danger zone
    assert (tmp_path / "in_force.yaml").exists()


def test_apply_records_the_mapping_the_station_file_was_written_under_and_keeps_the_old_files(station_env, tmp_path, monkeypatch):  # noqa: F811
    import openarm_mapping as mapping

    station, _ = station_env
    record = tmp_path / "model_mapping"
    monkeypatch.setenv(mapping.FILE_ENV, str(record))
    monkeypatch.setenv(mapping.ENV, "official")
    (tmp_path / "in_force.yaml").write_text("old zones in force\n")
    record.write_text("identity\n")
    path = saved_danger_bundle(tmp_path)
    assert ws.main(["apply", "--zones", str(path), "--station", str(station)]) == 0
    assert mapping.recorded() == "official"
    assert list(tmp_path.glob("model_mapping.*.bak")) and list(tmp_path.glob("in_force.yaml.*.bak"))  # the old record and the old zones are kept
    assert "old zones" in next(tmp_path.glob("in_force.yaml.*.bak")).read_text()


def test_a_normal_bundle_still_needs_verification(station_env):  # noqa: F811
    station, path = station_env
    with pytest.raises(SystemExit, match="verif"):
        ws.main(["apply", "--zones", str(path), "--station", str(station)])

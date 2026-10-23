"""Tests for the zone editor's state (no browser): table, danger boxes, work-area rectangle, home choice, clash list, bundle."""

import datetime as dt

import numpy as np
import pytest

import editor as ed
import openarm_poses as poses
import openarm_zones as zones
import zonebuild as zb

NOW = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


def test_editor_works_in_the_robot_body_frame_and_the_arm_bases_are_shifted():
    assert ed.BASE_OFFSET["left"] == pytest.approx([0.0, 0.031, 0.698]) and ed.BASE_OFFSET["right"] == pytest.approx([0.0, -0.031, 0.698])
    assert ed.to_base("left", (0.5, 0.2, 0.4)) == pytest.approx([0.5, 0.169, -0.298])
    assert ed.to_world("right", (0.5, 0.2, -0.298)) == pytest.approx([0.5, 0.169, 0.4])


def test_defaults_put_the_table_top_30_cm_above_the_hanging_hand():
    state = ed.EditorState.default()
    hanging_tcp_z = zones.ArmKinematics("left").points(np.zeros(7))["tcp"][2]  # in the arm base frame
    assert state.table.top_z == pytest.approx(hanging_tcp_z + 0.698 + 0.30, abs=0.01)  # body frame
    assert state.arms["left"].home == pytest.approx(poses.ready_rad("left"))


def test_table_box_follows_its_numbers():
    t = ed.Table(top_z=-0.3, x_range=(0.2, 0.8), y_range=(-0.3, 0.3), thickness=0.2)
    box = t.keep_out()
    assert box["center"] == pytest.approx([0.5, 0.0, -0.4]) and box["size"] == pytest.approx([0.6, 0.6, 0.2])
    assert box["name"] == "table" and box["allow_tcp_touch"]


def test_danger_boxes_can_be_added_moved_resized_and_removed():
    state = ed.EditorState.default()
    state.add_box("post", center=(0.4, 0.0, -0.2), size=(0.1, 0.1, 0.4))
    state.move_box("post", center=(0.5, 0.1, -0.2))
    state.resize_box("post", size=(0.2, 0.2, 0.4))
    box = next(b for b in state.boxes if b.name == "post")
    assert box.center == (0.5, 0.1, -0.2) and box.size == (0.2, 0.2, 0.4)
    state.remove_box("post")
    assert all(b.name != "post" for b in state.boxes)
    with pytest.raises(KeyError):
        state.remove_box("post")


def test_box_names_must_be_unique_and_sizes_positive():
    state = ed.EditorState.default()
    state.add_box("post", (0, 0, 0), (0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="already"):
        state.add_box("post", (1, 1, 1), (0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="size"):
        state.add_box("flat", (0, 0, 0), (0.1, 0.0, 0.1))
    with pytest.raises(ValueError, match="table"):
        state.add_box("table", (0, 0, 0), (0.1, 0.1, 0.1))


def test_work_area_rectangle_fills_the_two_cm_grid():
    state = ed.EditorState.default()
    state.set_work_area("left", x_range=(0.40, 0.50), y_range=(0.0, 0.10))
    cells = state.arms["left"].reach
    xs = sorted({c[0] for c in cells}); ys = sorted({c[1] for c in cells})
    assert len(cells) == len(xs) * len(ys) and 5 <= len(xs) <= 6 and 5 <= len(ys) <= 6
    state.set_work_area("left", None, None)
    assert not state.arms["left"].reach


def test_home_choices_rest_ready_live():
    state = ed.EditorState.default()
    state.use_rest("right")
    assert state.arms["right"].home == pytest.approx(np.radians(poses.REST_DEG["right"]))
    state.use_ready("right")
    assert state.arms["right"].home == pytest.approx(poses.ready_rad("right"))
    live = np.linspace(-0.2, 0.2, 7)
    state.use_pose("right", live)
    assert state.arms["right"].home == pytest.approx(list(live))
    with pytest.raises(ValueError):
        state.use_pose("right", [0.0] * 6)


def test_clashes_name_the_zone_and_the_pose_and_clear_when_the_zone_moves():
    state = ed.EditorState.default()
    assert state.clashes() == []  # default table is in front of the arms
    tcp = zones.ArmKinematics("left").points(np.radians(poses.READY_DEG["left"]))["tcp"]
    state.add_box("in_the_way", tuple(ed.to_world("left", tcp)), (0.2, 0.2, 0.2), side="left")
    found = state.clashes()
    assert found and any("in_the_way" in c and "ready" in c for c in found)
    state.move_box("in_the_way", (2.0, 2.0, 2.0))
    assert state.clashes() == []


def test_clashes_include_a_live_pose():
    state = ed.EditorState.default()
    live = np.zeros(7)
    tcp = zones.ArmKinematics("left").points(live)["tcp"]
    state.add_box("post", tuple(ed.to_world("left", tcp)), (0.1, 0.1, 0.1), side="left")
    assert any("live" in c and "post" in c for c in state.clashes(live={"left": live}))


def test_bundle_matches_the_documented_output_format_and_loads():
    state = ed.EditorState.default()
    state.add_box("torso", (-0.2, 0.0, -0.3), (0.2, 0.4, 0.6), side="both")
    state.set_work_area("left", (0.4, 0.6), (-0.1, 0.1))
    raw = state.to_bundle(operator="kota", now=NOW)
    assert raw["source"] == "editor" and raw["verify"]["status"] == "unverified"
    assert raw["frame"] == zb.FRAME_NOTE
    assert set(raw["home"]) == {"left", "right"} and len(raw["home"]["left"]) == 7
    left = raw["arms"]["left"]
    assert left["table_z"] == pytest.approx(state.table.top_z - 0.698)  # stored in the arm base frame
    assert raw["arms"]["right"]["table_z"] == pytest.approx(state.table.top_z - 0.698)
    assert [z["name"] for z in left["keep_out"]] == ["table", "torso"]
    # the same physical table is shifted by the base offset: left base is +0.031 in y, right base -0.031
    lt = left["keep_out"][0]["center"][1]; rt = raw["arms"]["right"]["keep_out"][0]["center"][1]
    assert lt - rt == pytest.approx(-0.062)
    assert left["reach"]
    zones.parse_zones(raw)  # the driver and apply read it as is


def test_round_trip_from_a_bundle():
    state = ed.EditorState.default()
    state.add_box("torso", (-0.2, 0.0, -0.3), (0.2, 0.4, 0.6), side="both")
    state.set_work_area("right", (0.4, 0.6), (-0.1, 0.1))
    raw = state.to_bundle(operator="k", now=NOW)
    again = ed.EditorState.from_bundle(raw)
    assert [b.name for b in again.boxes] == ["torso"] and again.table.top_z == pytest.approx(state.table.top_z)
    assert again.to_bundle(operator="k", now=NOW)["arms"] == raw["arms"]


# -- saving and exporting ---------------------------------------------------------------------------------------------------

import outputs  # noqa: E402
import yaml  # noqa: E402


def test_save_bundle_writes_the_yaml_and_the_layout_picture_side_by_side(tmp_path):
    state = ed.EditorState.default()
    state.add_box("torso", (-0.15, 0.0, 0.35), (0.2, 0.5, 0.7))
    yaml_path, png_path = outputs.save_bundle(state, "kota", tmp_path, now=NOW)
    assert yaml_path.name == "zones-20261023-1200.yaml" and png_path.name == "layout-20261023-1200.png"
    raw = yaml.safe_load(yaml_path.read_text())
    zones.parse_zones(raw)
    assert raw["operator"] == "kota" and raw["source"] == "editor" and png_path.stat().st_size > 1000


def test_export_station_keys_has_home_reach_and_keep_out_per_arm_and_nothing_else(tmp_path):
    state = ed.EditorState.default()
    state.set_work_area("left", (0.4, 0.6), (0.0, 0.2))
    state.add_box("torso", (-0.15, 0.0, 0.35), (0.2, 0.5, 0.7))
    raw = state.to_bundle("k", NOW)
    path = outputs.export_station_keys(raw, tmp_path / "station_keys.yaml")
    keys = yaml.safe_load(path.read_text())
    assert set(keys) == {"follower_l", "follower_r"}
    for key in keys.values():
        assert set(key) == {"home", "reach", "keep_out"} and len(key["home"]) == 7
        assert all(set(z) == {"kind", "center", "size"} for z in key["keep_out"])
    assert keys["follower_l"]["keep_out"][1]["size"] == [0.2, 0.5, 0.7]
    assert path.read_text().startswith("#")  # a header says what it is and where it goes


# -- the home pose must lie inside the allowed region ---------------------------------------------------------------------------

def test_fit_home_gives_the_home_pose_room_on_every_side_and_nothing_more():
    """The raised home is only there for the cameras to see the table: it must not sit against a wall of the allowed area."""
    state = ed.EditorState.default()
    kin = zones.ArmKinematics("left")
    home_pts = np.array(list(kin.points(np.array(state.arms["left"].home)).values()))
    room = ed.HOME_ROOM_M
    inside = {"lo": (home_pts.min(axis=0) - room - 0.05).tolist(), "hi": (home_pts.max(axis=0) + room + 0.05).tolist()}
    state.arms["left"].region = dict(inside)
    assert state.fit_home_into_region() == []                       # more than the room already: untouched
    assert state.arms["left"].region == inside
    state.arms["left"].region = {"lo": (home_pts.min(axis=0)).tolist(), "hi": (home_pts.max(axis=0)).tolist()}   # home touching every wall
    assert state.fit_home_into_region() == ["left"]
    region = state.arms["left"].region
    assert np.array(region["lo"]) == pytest.approx(home_pts.min(axis=0) - room, abs=2e-4)
    assert np.array(region["hi"]) == pytest.approx(home_pts.max(axis=0) + room, abs=2e-4)
    assert state.fit_home_into_region() == []                       # idempotent
    assert state.clashes() == []


def test_fit_home_never_shrinks_a_region_it_widens_on_one_side_only():
    state = ed.EditorState.default()
    home_pts = np.array(list(zones.ArmKinematics("left").points(np.array(state.arms["left"].home)).values()))
    big = {"lo": (home_pts.min(axis=0) - 1.0).tolist(), "hi": (home_pts.max(axis=0) - 0.02).tolist()}   # far on the low side, 2 cm short on the high side
    state.arms["left"].region = dict(big)
    state.fit_home_into_region()
    assert state.arms["left"].region["lo"] == big["lo"]
    assert state.arms["left"].region["hi"] == pytest.approx((home_pts.max(axis=0) + ed.HOME_ROOM_M).tolist(), abs=2e-4)


def test_an_arm_without_a_region_is_left_alone_by_fit_home():
    state = ed.EditorState.default()
    assert state.fit_home_into_region() == [] and state.arms["right"].region is None


# -- reference points (for the daily verify) ---------------------------------------------------------------------------------------

def test_reference_points_are_kept_in_the_state_and_the_bundle_and_round_trip():
    state = ed.EditorState.default()
    state.add_reference("left", (0.40, 0.05, -0.30)); state.add_reference("left", (0.50, 0.10, -0.30))
    raw = state.to_bundle("k", NOW)
    assert raw["references"]["left"] == [[0.4, 0.05, -0.3], [0.5, 0.1, -0.3]] and raw["references"]["right"] == []
    zones.parse_zones(raw)
    again = ed.EditorState.from_bundle(raw)
    assert again.references["left"] == [(0.4, 0.05, -0.3), (0.5, 0.1, -0.3)]


# -- the two allowed areas must not run into each other ---------------------------------------------------------------------------

def test_region_overlap_is_zero_for_separate_areas_and_the_depth_when_they_run_into_each_other():
    state = ed.EditorState.default()
    state.arms["left"].region = {"lo": [-0.1, 0.0, -0.5], "hi": [0.1, 0.2, -0.2]}      # arm base frame: world y 0.031 .. 0.231
    state.arms["right"].region = {"lo": [-0.1, -0.2, -0.5], "hi": [0.1, 0.0, -0.2]}    # world y -0.231 .. -0.031
    assert state.region_overlap_cm() == 0.0
    state.arms["right"].region = {"lo": [-0.1, -0.2, -0.5], "hi": [0.1, 0.1, -0.2]}    # reaches world y +0.069: 3.8 cm into the left one
    assert state.region_overlap_cm() == pytest.approx(3.8, abs=0.05)
    state.arms["right"].region = None
    assert state.region_overlap_cm() == 0.0

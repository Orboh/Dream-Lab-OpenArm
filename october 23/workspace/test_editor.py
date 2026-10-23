"""Tests for the zone editor's state (no browser): table, danger boxes, work-area rectangle, home choice, clash list, bundle."""

import datetime as dt

import numpy as np
import pytest

import editor as ed
import openarm_poses as poses
import openarm_zones as zones
import zonebuild as zb

NOW = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


def test_defaults_put_the_table_top_30_cm_above_the_hanging_hand():
    state = ed.EditorState.default()
    hanging_tcp_z = zones.ArmKinematics("left").points(np.zeros(7))["tcp"][2]
    assert state.table.top_z == pytest.approx(hanging_tcp_z + 0.30, abs=0.01)
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
    state.add_box("in_the_way", tuple(tcp), (0.2, 0.2, 0.2), side="left")
    found = state.clashes()
    assert found and any("in_the_way" in c and "ready" in c for c in found)
    state.move_box("in_the_way", (2.0, 2.0, 2.0))
    assert state.clashes() == []


def test_clashes_include_a_live_pose():
    state = ed.EditorState.default()
    live = np.zeros(7)
    tcp = zones.ArmKinematics("left").points(live)["tcp"]
    state.add_box("post", tuple(tcp), (0.1, 0.1, 0.1), side="left")
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
    assert left["table_z"] == pytest.approx(state.table.top_z)
    assert [z["name"] for z in left["keep_out"]] == ["table", "torso"]
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

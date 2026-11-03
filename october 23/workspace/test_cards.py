"""Tests for the card HTML and the checklist: they are plain strings and lists, so no browser is needed."""

import datetime as dt

import numpy as np
import pytest

import cards
import editor as ed
import openarm_poses as poses
import teach as th

NOW = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


def test_position_card_shows_the_side_the_joints_and_the_hand():
    html = cards.position_card("left", [61.5, -3.3, -1.6, 93.7, -2.6, 17.5, 26.0], 0.0, (1.2, 4.7, -40.0), fresh=True)
    assert "LEFT" in html and "61.5" in html and "93.7" in html and "J7" in html
    assert "x 1.2" in html and "y 4.7" in html and "z -40.0" in html
    assert "receiving" in html.lower()


def test_position_card_says_waiting_when_there_is_no_fresh_data():
    html = cards.position_card("right", None, None, None, fresh=False)
    assert "RIGHT" in html and "waiting" in html.lower() and "J1" not in html


def test_status_strip_lights_and_first_clash():
    ok = cards.status_strip({"left": True, "right": True}, None)
    assert "left" in ok and "right" in ok and "No clash" in ok
    bad = cards.status_strip({"left": True, "right": False}, "left ready: left hand is 1.0 cm from keep-out zone 'post'")
    assert "waiting" in bad.lower() and "post" in bad and "No clash" not in bad


@pytest.mark.parametrize("home_set, path, text", [(False, False, "not set"), (True, True, "path"), (True, False, "NO path")])
def test_home_card_states(home_set, path, text):
    html = cards.home_card("left", home_set=home_set, has_path=path, home_deg=[57.8, -3.4, -1.5, 93.6, -2.3, -2.0, -37.6] if home_set else None)
    assert text in html
    if home_set:
        assert "57.8" in html


def test_checklist_shows_a_tick_or_a_cross_with_a_hint():
    html = cards.checklist([("Left home set", True, ""), ("Right home set", False, "press Set RIGHT home here")])
    assert html.count("&#10004;") == 1 and html.count("&#10008;") == 1 and "press Set RIGHT home here" in html


def test_build_checklist_follows_the_state_and_the_session():
    state = ed.EditorState.default()
    session = th.TeachSession()
    items = {label: ok for label, ok, _ in cards.build_checklist(state, session)}
    assert not items["Left home set"] and not items["Right home set"] and not items["Table placed"] and not items["Left allowed area"]
    assert items["No clash"]
    state.arms["left"].home_source = "teach"
    state.arms["left"].home_path = poses.path_to_ready("left")
    state.arms["left"].region = {"lo": [0, 0, 0], "hi": [1, 1, 1]}
    state.table_set = True
    items = {label: ok for label, ok, _ in cards.build_checklist(state, session)}
    assert items["Left home set"] and items["Left path recorded"] and items["Left allowed area"] and items["Table placed"]
    assert not items["Right home set"]


def test_ready_pose_home_counts_as_having_a_path():
    state = ed.EditorState.default()
    state.arms["right"].home_source = "ready"
    items = {label: ok for label, ok, _ in cards.build_checklist(state, th.TeachSession())}
    assert items["Right path recorded"]  # the built-in rest -> ready path covers it


def test_checklist_asks_for_two_reference_points_per_arm():
    state = ed.EditorState.default()
    items = {label: ok for label, ok, _ in cards.build_checklist(state, th.TeachSession())}
    assert not items["Left reference points (2)"] and not items["Right reference points (2)"]
    state.add_reference("left", (0.4, 0, -0.3)); state.add_reference("left", (0.5, 0, -0.3))
    items = {label: ok for label, ok, _ in cards.build_checklist(state, th.TeachSession())}
    assert items["Left reference points (2)"] and not items["Right reference points (2)"]


def test_checklist_warns_when_the_two_allowed_areas_overlap():
    state = ed.EditorState.default()
    state.arms["left"].region = {"lo": [-0.1, 0.0, -0.5], "hi": [0.1, 0.2, -0.2]}
    state.arms["right"].region = {"lo": [-0.1, -0.2, -0.5], "hi": [0.1, 0.0, -0.2]}
    items = {label: ok for label, ok, _ in cards.build_checklist(state, th.TeachSession())}
    assert items["The two allowed areas do not overlap"]
    state.arms["right"].region = {"lo": [-0.1, -0.2, -0.5], "hi": [0.1, 0.1, -0.2]}
    items = {label: (ok, hint) for label, ok, hint in cards.build_checklist(state, th.TeachSession())}
    ok, hint = items["The two allowed areas do not overlap"]
    assert not ok and "3.8" in hint


def test_collect_card_shows_where_a_finished_run_saved_or_that_nothing_was():
    snap = {"phase": "finished", "rc": 0, "episode": 2, "started": 2, "run_dir": "/c/run", "target": 2, "done": 2, "left": 0, "saved": 2, "lines": [], "active": False,
            "saved_to": {"run_dir": "/c/run", "dataset_root": "/hf/Orboh/ds_20261028_132329", "episodes": 2}}
    html = cards.collect_card(snap)
    assert "Saved to" in html and "/hf/Orboh/ds_20261028_132329" in html and "2 episodes" in html and "No dataset" not in html
    snap["saved_to"] = {"run_dir": "/c/run", "dataset_root": None, "episodes": None}
    assert "No dataset was saved" in cards.collect_card(snap)
    assert "Saved to" not in cards.collect_card({**snap, "phase": "recording"})  # only once the run is over

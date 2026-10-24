"""The raise and park move: a recorded wandering path is shortened (and checked against the danger zones), preflight refuses a move `dreamcli robot down` cannot wait for, and the
strict clearance rule of the driver's own move. Offline: the driver's planner, no hardware."""

import datetime as dt

import numpy as np
import pytest
import yaml

import cards
import editor as ed
import openarm_poses as poses
import preflight as pf
import teach as th
import zonebuild as zb
from test_preflight import Probes, failed, run, setup  # noqa: F401  (a consistent station, zones in force and bundle)

NOW = dt.datetime.now(dt.timezone.utc)


def wandering(side: str, detours: int = 6, amplitude: float = 6.0) -> list[list[float]]:
    """The built-in rest-to-ready path with small back-and-forth wiggles of joint 1 added on every segment: what a path taught by hand looks like (the hand is never steady)."""
    base = poses.path_to_ready(side)
    out = [list(base[0])]
    for a, b in zip(base[:-1], base[1:]):
        mid = (np.asarray(a) + np.asarray(b)) / 2
        for k in range(detours):
            shifted = mid.copy()
            shifted[0] += amplitude * (1 if k % 2 == 0 else -1)
            out.append([float(v) for v in shifted])
        out.append([float(v) for v in b])
    return out


def state_with_path(side_path: dict[str, list[list[float]]], tolerance: float = 0.0, lower_cm: float = -5.0) -> ed.EditorState:
    state = ed.EditorState.default()
    state.table = ed.table_from_measurements(13, 29.3, 100, 140, lower_cm=lower_cm)
    state.danger_only = True
    state.park_tolerance_deg = tolerance
    for side in ed.SIDES:
        state.use_ready(side)
        state.arms[side].home_path = side_path.get(side) or poses.path_to_ready(side)
    return state


def test_simplify_path_keeps_the_ends_and_cuts_the_detours():
    path = wandering("left")
    short = poses.simplify_path(path, 8.0)
    assert short[0] == path[0] and short[-1] == path[-1] and len(short) < len(path) // 2
    assert poses.simplify_path(path[:2], 8.0) == [list(path[0]), list(path[1])]


def test_a_wandering_path_takes_far_longer_than_its_shortened_version():
    path = wandering("left")
    assert zb.path_seconds(path) > 60.0  # too long for a stop that waits 60 s
    assert zb.path_seconds(poses.simplify_path(path, 8.0)) < zb.path_seconds(path) * 0.6


def test_saving_with_a_tolerance_shortens_the_home_path_and_says_by_how_much():
    path = wandering("left")
    state = state_with_path({"left": path}, tolerance=8.0)
    raw = state.to_bundle("t", NOW)
    kept = raw["home_path"]["left"]
    assert len(kept) < len(path) and kept[0] == path[0] and kept[-1] == path[-1]
    note = state.last_shortening["left"]
    assert note["shortened"] and note["seconds_after"] < note["seconds_before"] and note["points_after"] == len(kept)
    untouched = state_with_path({"left": path}, tolerance=0.0).to_bundle("t", NOW)
    assert len(untouched["home_path"]["left"]) == len(path)  # tolerance 0: the path is saved as recorded


def test_a_shortcut_that_comes_closer_to_a_zone_than_the_recorded_path_is_not_taken(monkeypatch):
    path = wandering("left")
    slacks = iter([4.0, 0.5, 4.0, 4.0])  # left: clear before (4 cm spare), the shortcut leaves 0.5 cm... then right arm values
    calls = {"n": 0}

    def fake(raw, side, p):
        calls["n"] += 1
        return 4.0 if len(p) == len(path) or side == "right" else -3.0   # the shorter left path would touch a zone

    monkeypatch.setattr(zb, "taught_path_slack_cm", fake)
    state = state_with_path({"left": path}, tolerance=8.0)
    raw = state.to_bundle("t", NOW)
    assert raw["home_path"]["left"] == [list(map(float, p)) for p in path]
    assert state.last_shortening["left"]["shortened"] is False


def test_the_strict_rule_forbids_the_grasp_point_touching_a_zone_on_the_drivers_own_move():
    import openarm_zones as zones

    kin = zones.ArmKinematics("left", logical=True)
    ready = poses.READY_DEG["left"]
    tcp = kin.points(np.radians(ready))["tcp"]
    box = {"name": "table", "kind": "box", "allow_tcp_touch": True, "center": [float(tcp[0]), float(tcp[1]), float(tcp[2]) - 0.005 - 0.10], "size": [0.04, 0.04, 0.20]}  # its top 5 mm under the grasp point
    raw = {"version": 1, "captured_at": NOW.isoformat(), "source": "t", "arms": {"left": {"reach": None, "keep_out": [box]}, "right": {"reach": None, "keep_out": []}}}
    parsed = zones.parse_zones(raw)
    assert parsed.violations("left", kin.points(np.radians(ready)), check_reach=False, check_region=False) == []   # the guard for Dream's targets lets a grasp point touch the table
    path = poses.path_to_ready("left")[-2:]
    problems = zb.taught_path_problems(raw, "left", path)
    assert problems and "tcp" in problems[0] and "keep-out zone 'table'" in problems[0]   # the driver's own move may not
    assert zb.taught_path_slack_cm(raw, "left", path) < 0
    assert raw["arms"]["left"]["keep_out"][0]["allow_tcp_touch"] is True   # the check works on a copy: the zones themselves are untouched


def test_the_save_card_has_an_item_for_the_path_and_ticks_it_for_a_clear_one():
    clear = {label: ok for label, ok, _ in cards.build_checklist(state_with_path({}), th.TeachSession())}
    assert clear["The path from rest to home keeps clear of the danger zones"] is True


def zones_in_force_with_path(setup, tmp_path, path_by_side):
    station, in_force, zdir, raw = setup
    data = yaml.safe_load(in_force.read_text())
    data["home_path"] = {side: path_by_side.get(side) or poses.path_to_ready(side) for side in ed.SIDES}
    in_force.write_text(yaml.safe_dump(data))


def test_preflight_refuses_a_raise_and_park_that_takes_too_long_for_an_active_arm(setup, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(tmp_path / "active.json"))
    import openarm_lock as lock

    lock.set_active("both")
    zones_in_force_with_path(setup, tmp_path, {"left": wandering("left")})
    bad = [c for c in run(setup) if c.label == pf.LABEL_PARK][0]
    assert bad.ok is False and "left" in bad.detail and "limit 45 s" in bad.detail


def test_preflight_ignores_the_path_of_a_locked_arm(setup, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(tmp_path / "active.json"))
    (tmp_path / "active.json").write_text('{"active": ["right"], "set_at": "2026-10-24T00:00:00+09:00"}')
    zones_in_force_with_path(setup, tmp_path, {"left": wandering("left")})
    assert pf.LABEL_PARK not in failed(run(setup))  # the left arm is locked: it is not raised or parked


def test_preflight_passes_the_normal_setup(setup):  # noqa: F811
    labels = failed(run(setup))
    assert pf.LABEL_PARK not in labels and pf.LABEL_PATH_CLEAR not in labels

"""Tests for the zones, the guard and the one-arm lock. No hardware, no CAN.

Run from this folder with Dream's Python:  env -u PYTHONPATH -u AMENT_PREFIX_PATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../.venv/bin/python -m pytest -q
"""

import datetime as dt
import json
import math
import time
import types

import numpy as np
import pytest
import yaml

import openarm_guard as guard
import openarm_lock as lock
import openarm_poses as poses
import openarm_v2 as drv
import openarm_zones as zones
from robots import errors

NOW = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


def zones_dict(captured=NOW, left_keep=None, left_reach=None, right_keep=None):
    return {
        "version": 1, "captured_at": captured.isoformat(), "source": "follower",
        "arms": {
            "left": {"reach": left_reach, "keep_out": left_keep or []},
            "right": {"reach": None, "keep_out": right_keep or []},
        },
    }


def write_zones(path, **kw):
    path.write_text(yaml.safe_dump(zones_dict(**kw)))
    return path


# -- zones: schema and geometry ---------------------------------------------------------------------------------------

def test_parse_zones_accepts_a_valid_file_and_refuses_bad_ones():
    ok = zones.parse_zones(zones_dict(left_keep=[{"kind": "box", "center": [0, 0, 0], "size": [1, 1, 1]}]))
    assert ok.arms["left"].keep_out[0]["kind"] == "box" and ok.arms["left"].reach is None
    for bad in (
        {"captured_at": NOW.isoformat()},
        zones_dict(left_keep=[{"kind": "sphere", "center": [0, 0, 0]}]),
        zones_dict(left_keep=[{"kind": "box", "center": [0, 0], "size": [1, 1, 1]}]),
        zones_dict(left_keep=[{"kind": "box", "center": [0, 0, 0], "size": [1, 0, 1]}]),
        zones_dict(left_keep=[{"kind": "capsule", "center": [0, 0, 0], "radius": -1, "height": 1}]),
    ):
        with pytest.raises(ValueError):
            zones.parse_zones(bad)


def test_box_and_capsule_distances():
    box_c, box_s = np.zeros(3), np.array([0.2, 0.2, 0.2])
    assert zones.distance_to_box(np.array([0.0, 0.0, 0.0]), box_c, box_s) == pytest.approx(-0.1)  # inside: the depth
    assert zones.distance_to_box(np.array([0.4, 0.0, 0.0]), box_c, box_s) == pytest.approx(0.3)
    assert zones.distance_to_capsule(np.array([0.3, 0, 0]), np.zeros(3), 0.1, 0.4) == pytest.approx(0.2)
    assert zones.distance_to_capsule(np.array([0.0, 0, 0.5]), np.zeros(3), 0.1, 0.4) == pytest.approx(0.2)
    assert zones.distance_to_capsule(np.zeros(3), np.zeros(3), 0.1, 0.4) < 0


def test_reach_distance_uses_the_nearest_painted_point():
    cells = np.array([[0.0, 0.0], [0.1, 0.0]])
    assert zones.distance_to_reach(np.array([0.1, 0.03]), cells) == pytest.approx(0.03)


def test_violations_name_the_point_and_the_zone():
    z = zones.parse_zones(zones_dict(left_keep=[{"name": "torso", "kind": "box", "center": [0, 0, 0], "size": [0.2, 0.2, 0.2]}]))
    points = {"elbow": np.array([0.15, 0.0, 0.0]), "tcp": np.array([1.0, 1.0, 1.0])}
    found = z.violations("left", points)
    assert len(found) == 1 and "elbow" in found[0] and "torso" in found[0]
    assert z.violations("right", points) == []  # zones are per arm


def test_grasp_point_may_touch_an_allow_touch_zone_but_links_may_not():
    table = {"name": "table", "kind": "box", "center": [0, 0, -0.05], "size": [1, 1, 0.1], "allow_tcp_touch": True}
    z = zones.parse_zones(zones_dict(left_keep=[table]))
    assert z.violations("left", {"tcp": np.array([0.0, 0.0, 0.003])}) == []        # 3 mm above the top: touching
    assert z.violations("left", {"tcp": np.array([0.0, 0.0, -0.02])})               # 2 cm into the table
    assert z.violations("left", {"wrist": np.array([0.0, 0.0, 0.05])})              # a link sphere (4 cm + 2 cm) too close
    assert z.violations("left", {"wrist": np.array([0.0, 0.0, 0.09])}) == []


def test_hand_outside_the_painted_area_is_a_violation():
    z = zones.parse_zones(zones_dict(left_reach=[[0.0, 0.0], [0.04, 0.0]]))
    assert z.violations("left", {"tcp": np.array([0.05, 0.02, 0.0])}) == []
    assert z.violations("left", {"tcp": np.array([0.30, 0.0, 0.0])})


def test_age_hours():
    z = zones.parse_zones(zones_dict(captured=NOW - dt.timedelta(hours=5)))
    assert z.age_hours(NOW) == pytest.approx(5.0)


# -- forward kinematics ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("side, y", [("left", 0.1224), ("right", -0.1224)])
def test_zero_pose_hangs_straight_down(side, y):
    points = zones.ArmKinematics(side).points(np.zeros(7))
    assert points["tcp"] == pytest.approx([-0.0016, y, -0.595], abs=1e-3)


def test_forward_kinematics_moves_the_hand_when_a_joint_moves():
    k = zones.ArmKinematics("left")
    q = np.zeros(7)
    q[3] = math.radians(90)  # elbow
    assert np.linalg.norm(k.points(q)["tcp"] - k.points(np.zeros(7))["tcp"]) > 0.2


# -- guard (policy) ----------------------------------------------------------------------------------------------------

def make_guard(monkeypatch, tmp_path, required="1", **kw):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", required)
    path = write_zones(tmp_path / "zones.yaml", **kw)
    monkeypatch.setenv("OPENARM_ZONES", str(path))
    return guard.Guard("left", clock=lambda: NOW), path


def test_guard_off_passes_everything(monkeypatch, tmp_path):
    g, _ = make_guard(monkeypatch, tmp_path, required="0")
    g.check(np.zeros(7))  # no error


def test_guard_refuses_when_there_is_no_zones_file(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "1")
    monkeypatch.setenv("OPENARM_ZONES", str(tmp_path / "missing.yaml"))
    with pytest.raises(errors.ArmError, match="daily"):
        guard.Guard("left", clock=lambda: NOW).check(np.zeros(7))


def test_guard_refuses_stale_zones(monkeypatch, tmp_path):
    g, _ = make_guard(monkeypatch, tmp_path, captured=NOW - dt.timedelta(hours=30))
    with pytest.raises(errors.ArmError, match="30"):
        g.check(np.zeros(7))


def test_guard_passes_a_clean_pose_and_refuses_a_zone_hit(monkeypatch, tmp_path):
    g, _ = make_guard(monkeypatch, tmp_path)
    g.check(np.zeros(7))  # hanging pose, no zones in the way
    tcp = zones.ArmKinematics("left").points(np.zeros(7))["tcp"].tolist()
    hit = [{"name": "post", "kind": "box", "center": tcp, "size": [0.1, 0.1, 0.1]}]
    g2, _ = make_guard(monkeypatch, tmp_path, left_keep=hit)
    with pytest.raises(errors.ArmError, match="post"):
        g2.check(np.zeros(7))


def test_guard_reloads_a_changed_zones_file(monkeypatch, tmp_path):
    g, path = make_guard(monkeypatch, tmp_path)
    g.check(np.zeros(7))
    tcp = zones.ArmKinematics("left").points(np.zeros(7))["tcp"].tolist()
    time.sleep(0.01)
    path.write_text(yaml.safe_dump(zones_dict(left_keep=[{"name": "new", "kind": "box", "center": tcp, "size": [0.1, 0.1, 0.1]}])))
    import os
    os.utime(path, (time.time() + 5, time.time() + 5))
    with pytest.raises(errors.ArmError, match="new"):
        g.check(np.zeros(7))


# -- arm lock ----------------------------------------------------------------------------------------------------------

def test_lock_defaults_to_both_arms(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(tmp_path / "none.json"))
    assert lock.active_arms() == {"left", "right"}


def test_lock_set_and_read(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(tmp_path / "a.json"))
    lock.set_active("right")
    assert lock.active_arms() == {"right"}
    lock.set_active("both")
    assert lock.active_arms() == {"left", "right"}
    with pytest.raises(ValueError):
        lock.set_active("middle")


def test_lock_with_a_broken_file_keeps_both_arms_locked_out_safely(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    path.write_text("{not json")
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(path))
    assert lock.active_arms() == set()  # unreadable lock: no arm moves


# -- driver: guard and lock in Arm.write ------------------------------------------------------------------------------

from test_openarm_v2 import LOGICAL, OFFSETS, make_arm  # noqa: E402,F401  (fixture reuse)


def ready_pose_rad():
    return np.radians([LOGICAL[n] for n in drv.JOINTS])


def test_write_is_refused_without_zones_when_required(make_arm, monkeypatch):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "1")
    arm, bus = make_arm(drv.Arm)
    before = dict(bus.mit_targets)
    with pytest.raises(errors.ArmError, match="daily"):
        arm.write(ready_pose_rad(), 0.5)
    assert bus.mit_targets == before  # nothing was sent


def test_write_is_refused_into_a_keep_out_zone_and_passes_outside(make_arm, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "1")
    now = dt.datetime.now(dt.timezone.utc)
    q = ready_pose_rad()
    tcp = zones.ArmKinematics("left").points(q)["tcp"].tolist()
    path = write_zones(tmp_path / "zones.yaml", captured=now,
                       left_keep=[{"name": "post", "kind": "box", "center": tcp, "size": [0.1, 0.1, 0.1]}])
    monkeypatch.setenv("OPENARM_ZONES", str(path))
    arm, bus = make_arm(drv.Arm)
    before = dict(bus.mit_targets)
    with pytest.raises(errors.ArmError, match="post"):
        arm.write(q, 0.5)
    assert bus.mit_targets == before
    write_zones(path, captured=now)  # same file without the zone
    import os
    os.utime(path, (time.time() + 5, time.time() + 5))
    arm.write(q + np.radians(2.0), 0.5)  # now allowed (a little away from where the arm is)
    assert bus.mit_targets != before


def test_a_locked_arm_holds_and_ignores_targets(make_arm, monkeypatch, tmp_path):
    arm, bus = make_arm(drv.Arm)
    lock.set_active("right")  # the left arm is not active
    held = {n: bus.native[n] for n in drv.MOTORS}
    target = ready_pose_rad() + np.radians(5.0)
    arm.write(target, 0.2)
    for name in drv.MOTORS:
        assert bus.mit_targets[name] == pytest.approx(held[name], abs=0.05)  # still holding where it was
    lock.set_active("both")
    arm.write(target, 0.2)  # active again: now it takes the target
    assert bus.mit_targets["joint_1"] == pytest.approx(LOGICAL["joint_1"] + 5.0 + OFFSETS["joint_1"], abs=0.1)


def test_guard_does_not_enforce_the_painted_area(monkeypatch, tmp_path):
    far = [[5.0, 5.0]]  # a painted area nowhere near the hanging hand
    g, _ = make_guard(monkeypatch, tmp_path, left_reach=far)
    g.check(np.zeros(7))  # allowed: Dream clips to the area, and Home/Stop moves cross it


def test_guard_refuses_a_pose_outside_the_allowed_region(monkeypatch, tmp_path):
    q = np.zeros(7)
    tcp = zones.ArmKinematics("left").points(q)["tcp"]
    inside = {"lo": (tcp - 0.8).tolist(), "hi": (tcp + 0.8).tolist()}
    g, _ = make_guard(monkeypatch, tmp_path)
    raw = zones_dict()
    raw["arms"]["left"]["region"] = inside
    path = tmp_path / "zones.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("OPENARM_ZONES", str(path))
    guard.Guard("left", clock=lambda: NOW).check(q)  # inside: fine
    raw["arms"]["left"]["region"] = {"lo": (tcp + 0.1).tolist(), "hi": (tcp + 0.5).tolist()}  # the hand is below its floor
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(errors.ArmError, match="region"):
        guard.Guard("left", clock=lambda: NOW).check(q)


# -- a recorded home path (from the teach session) replaces the built-in ready path -------------------------------------------------

def teach_path(side="left"):
    rest = list(poses.REST_DEG[side])
    mid = [rest[0] + (25.0 if side == "left" else -25.0), rest[1], rest[2], rest[3] + 20.0, rest[4], rest[5], rest[6]]
    home = [rest[0] + (40.0 if side == "left" else -40.0), rest[1], rest[2], rest[3] + 55.0, rest[4], rest[5], rest[6]]
    return [rest, mid, home]


def write_zones_with_path(tmp_path, monkeypatch, side="left", path=None):
    path = path or teach_path(side)
    raw = zones_dict(captured=dt.datetime.now(dt.timezone.utc))
    raw["home"] = {"left": [0.0] * 7, "right": [0.0] * 7}
    raw["home"][side] = [math.radians(v) for v in path[-1]]
    raw["home_path"] = {side: path}
    file = tmp_path / "zones.yaml"
    file.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("OPENARM_ZONES", str(file))
    return path


def test_zones_parse_a_home_path_and_refuse_a_short_or_wrong_one():
    raw = zones_dict()
    raw["home"] = {"left": [0.0] * 7, "right": [0.0] * 7}
    raw["home_path"] = {"left": teach_path("left")}
    assert zones.parse_zones(raw).home_path["left"][0] == pytest.approx(poses.REST_DEG["left"])
    for bad in ([teach_path()[0]], [[0.0] * 6, [1.0] * 6]):
        raw["home_path"] = {"left": bad}
        with pytest.raises(ValueError, match="home_path"):
            zones.parse_zones(raw)


def test_arm_raises_to_a_taught_home_along_the_recorded_path_and_parks_back(make_arm, monkeypatch, tmp_path):
    from test_openarm_v2 import as_logical, joints_of
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    path = write_zones_with_path(tmp_path, monkeypatch)
    arm, bus = make_arm(drv.Arm, logical=as_logical(poses.REST_DEG["left"]), follow=True, home=[math.radians(v) for v in path[-1]])
    assert joints_of(bus) == pytest.approx(path[-1], abs=0.2)  # at the taught home, not at the built-in ready pose
    arm.torque_off()
    assert joints_of(bus) == pytest.approx(poses.REST_DEG["left"], abs=0.2) and not any(bus.enabled.values())


def test_a_home_that_matches_neither_path_leaves_the_arm_where_it_is(make_arm, monkeypatch, tmp_path):
    from test_openarm_v2 import as_logical, joints_of
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    write_zones_with_path(tmp_path, monkeypatch)
    arm, bus = make_arm(drv.Arm, logical=as_logical(poses.REST_DEG["left"]), follow=True, home=[0.5] * 7)
    assert joints_of(bus) == pytest.approx(poses.REST_DEG["left"], abs=0.05)

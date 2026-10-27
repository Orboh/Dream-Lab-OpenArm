"""Tests for openarm_gripper.py: the gripper's hand housing and fingers attached to the guard's arm model (log-036). No hardware.

Run from this folder with Dream's Python:  ../../.venv/bin/python -m pytest -q test_openarm_gripper.py
"""

import numpy as np
import pytest

import openarm_gripper as grip
import openarm_zones as zones
from test_openarm_v2 import make_arm  # noqa: F401  (the fake bus fixture)


def test_the_boxes_come_from_the_official_meshes():
    """The numbers measured from the official OpenArm v10 meshes (cm, official hand_tcp frame; log-036)."""
    assert grip.HAND_BOX_M[0] == pytest.approx((-0.028, -0.084, -0.090)) and grip.HAND_BOX_M[1] == pytest.approx((0.029, 0.084, -0.059))
    assert grip.FINGER_TIP_PAST_TCP_M == pytest.approx(0.016)


def test_points_in_the_tcp_frame_put_the_fingertips_3_7_cm_past_dreams_grasp_point():
    pts = grip.points_in_tcp_frame("left", grip.FINGER_OPEN_M)
    deepest = min(p[2] for p in pts.values())                         # Dream's tcp frame: the gripper points along -z
    assert deepest == pytest.approx(-(0.0211 + 0.016), abs=0.001)
    assert len(pts) == 24 and all(name.startswith("grip:") for name in pts)


def test_opening_the_fingers_widens_them_and_leaves_the_housing():
    closed, opened = grip.points_in_tcp_frame("left", 0.0), grip.points_in_tcp_frame("left", grip.FINGER_OPEN_M)
    width = lambda pts, part: np.ptp([p[:2] for n, p in pts.items() if part in n], axis=0).max()   # noqa: E731
    assert width(opened, "finger") > width(closed, "finger") + 0.08
    assert np.allclose([p for n, p in closed.items() if "hand" in n], [p for n, p in opened.items() if "hand" in n])


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_arm_model_adds_the_gripper_points_where_the_hand_is(side):
    k = zones.ArmKinematics(side)
    q = np.radians([10, -5, 20, 60, 10, -10, 30])
    pts = k.points(q)
    assert sum(n.startswith("grip:") for n in pts) == 24
    tool = k.tool_pose(q)
    for name, p in grip.points_in_tcp_frame(side, grip.FINGER_OPEN_M).items():
        assert pts[name] == pytest.approx((tool @ np.append(p, 1.0))[:3])


# -- rule C (user, 2026-10-27, log-036): fingers may touch the table box like the grasp point, the hand housing keeps 2 cm ----------------

def _zones_with_table(top_z: float) -> zones.Zones:
    table = {"name": "table", "kind": "box", "allow_tcp_touch": True, "center": [0.6, 0.0, top_z - 0.15], "size": [1.0, 1.4, 0.3]}
    post = {"name": "post", "kind": "box", "center": [0.6, 0.6, top_z + 0.2], "size": [0.05, 0.05, 0.4]}
    return zones.parse_zones({"version": 1, "captured_at": "2026-10-27T12:00:00+09:00", "source": "test",
                              "arms": {"left": {"reach": None, "keep_out": [table, post]}, "right": {"reach": None, "keep_out": []}}})


def _place(z_of_point: float, name: str) -> dict[str, np.ndarray]:
    return {name: np.array([0.6, 0.0, z_of_point])}


def test_a_fingertip_may_touch_the_table_box_like_the_grasp_point():
    z = _zones_with_table(-0.40)
    assert z.violations("left", _place(-0.40 - 0.004, "grip:finger_l:3"), check_reach=False) == []      # 0.4 cm into the box top: allowed
    assert z.violations("left", _place(-0.40 - 0.006, "grip:finger_l:3"), check_reach=False) != []      # past the 0.5 cm touch tolerance


def test_the_hand_housing_keeps_2_cm_from_the_table_box():
    z = _zones_with_table(-0.40)
    assert z.violations("left", _place(-0.40 + 0.021, "grip:hand:0"), check_reach=False) == []
    assert z.violations("left", _place(-0.40 + 0.019, "grip:hand:0"), check_reach=False) != []


def test_fingers_keep_2_cm_from_any_zone_that_is_not_the_table():
    z = _zones_with_table(-0.40)
    near_post = {"grip:finger_l:3": np.array([0.6, 0.6 - 0.025 - 0.015, -0.40 + 0.2])}                 # 1.5 cm from the post's side
    assert z.violations("left", near_post, check_reach=False) != []


# -- the gripper's real opening (log-040): measured or commanded during a task, closed for the driver's own raise and park ---------------

def test_the_arm_model_takes_the_finger_opening():
    k = zones.ArmKinematics("left")
    q = np.radians([10, -5, 20, 60, 10, -10, 30])
    opened, closed = k.points(q), k.points(q, opening_m=0.0)
    gap = lambda pts: np.linalg.norm(pts["grip:finger_l:3"] - pts["grip:finger_r:3"])   # noqa: E731
    assert gap(opened) - gap(closed) == pytest.approx(2 * grip.FINGER_OPEN_M, abs=0.002)
    assert np.allclose(opened["grip:hand:0"], closed["grip:hand:0"])


def _box_beside_the_open_finger(q):
    """A small post 1 cm outside the open left fingertip: open fingers hit it, closed ones keep 4 cm from it."""
    k = zones.ArmKinematics("left")
    pts = k.points(q)
    tip, inner = pts["grip:finger_l:3"], pts["grip:finger_r:3"]
    out = (tip - inner) / np.linalg.norm(tip - inner)
    center = (tip + 0.02 * out).tolist()
    return {"name": "post", "kind": "box", "center": [float(v) for v in center], "size": [0.02, 0.02, 0.02]}


def test_the_driver_judges_a_target_with_the_gripper_as_it_is(make_arm, monkeypatch, tmp_path):
    import datetime as dt

    import openarm_v2 as drv
    from test_openarm_guard import write_zones

    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "1")
    q = np.radians([10, -5, 20, 60, 10, -10, 30])
    path = write_zones(tmp_path / "zones.yaml", captured=dt.datetime.now(dt.timezone.utc), left_keep=[_box_beside_the_open_finger(q)])
    monkeypatch.setenv("OPENARM_ZONES", str(path))
    start = {f"joint_{i + 1}": float(v) for i, v in enumerate(np.degrees(q))}
    arm, bus = make_arm(drv.Arm, logical={**start, "gripper": 0.0})            # gripper measured closed
    arm.write(q, 0.0)                                                          # and asked to stay closed: the open fingers would hit the post
    assert np.allclose(arm._safe_q, q)                                         # passed unchanged: the closed fingers keep clear
    from test_openarm_v2 import OFFSETS

    arm.write(q, 1.0)                                                          # asked to open there: the open fingers would be in the post
    assert np.allclose(arm._safe_q, q)                                         # no error, the arm stays (log-042)
    assert abs(bus.mit_targets["gripper"] - OFFSETS["gripper"] - 0.0) < 2.0    # and the gripper stays closed instead of opening into the post


def test_raise_and_park_close_the_gripper_first(make_arm, monkeypatch):
    import openarm_poses as poses
    import openarm_v2 as drv
    from test_openarm_v2 import OFFSETS, as_logical

    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    arm, bus = make_arm(drv.Arm, logical={**as_logical(poses.READY_DEG["left"]), "gripper": -40.0}, follow=True)
    arm.torque_off()
    grip_targets = [t for name, t, _ in bus.history if name == "gripper"]
    assert grip_targets and abs(grip_targets[-1] - OFFSETS["gripper"] - 0.0) < 1.0    # closed (0 deg) while parking


def test_closing_the_gripper_for_the_park_is_not_taken_for_holding_an_object(make_arm, monkeypatch, caplog):
    """log-041, 13:58:23: Dream's Home leaves the grippers open; the park then closes them, and the stall monitor, still comparing with Dream's open
    target, logged 'gripper holds an object (it led its angle by 62 deg)' and slowed the closing."""
    import openarm_poses as poses
    import openarm_v2 as drv
    from test_openarm_v2 import as_logical

    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    arm, bus = make_arm(drv.Arm, logical={**as_logical(poses.READY_DEG["left"]), "gripper": -60.0}, follow=True, smooth=True)
    arm.write(np.radians(poses.READY_DEG["left"]), 1.0)          # Dream's last target: the gripper open
    for _ in range(50):
        arm._tick(0.01)
    clock = iter(np.arange(0.0, 1e4, 0.02))                    # the park as on the robot: real time goes on while the arm moves
    monkeypatch.setattr(arm, "_time", lambda: float(next(clock)))
    arm.torque_off()
    assert not any("holds an object" in r.message for r in caplog.records)


def test_held_at_the_table_the_gripper_does_not_open_further_and_the_run_goes_on(make_arm, monkeypatch, tmp_path):
    """log-042, 14:03:32: held at the table edge, Dream asked the gripper to open; the wider fingers put the held pose itself past the edge, the guard
    called it "already there" and ended the run. Now the gripper keeps the opening it had while the arm is held."""
    import datetime as dt

    import openarm_v2 as drv
    from test_openarm_guard import write_zones
    from test_openarm_v2 import OFFSETS

    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "1")
    q = np.radians([-60, 0, 0, 0, 0, -20, 0])                                     # arm reaching down: the gripper is lowest, and opening lowers a fingertip by 2.8 cm
    k = zones.ArmKinematics("left")
    closed_low = float(min(p[2] for n, p in k.points(q, 0.0).items() if n.startswith("grip:")))
    table = {"name": "table", "kind": "box", "allow_tcp_touch": True, "center": [0.4, 0.12, closed_low - 0.003 - 0.15], "size": [0.6, 0.6, 0.3]}
    path = write_zones(tmp_path / "zones.yaml", captured=dt.datetime.now(dt.timezone.utc), left_keep=[table])
    monkeypatch.setenv("OPENARM_ZONES", str(path))
    start = {f"joint_{i + 1}": float(v) for i, v in enumerate(np.degrees(q))}
    arm, bus = make_arm(drv.Arm, logical={**start, "gripper": 0.0})             # closed, just above the table
    arm.write(q, 0.0)
    arm.write(q, 1.0)                                                           # Dream asks it to open there: no error
    assert np.allclose(arm._safe_q, q, atol=1e-6)                               # the joints stay where they are
    assert abs(bus.mit_targets["gripper"] - OFFSETS["gripper"] - 0.0) < 2.0    # and the gripper stays closed

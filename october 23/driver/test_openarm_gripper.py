"""Tests for openarm_gripper.py: the gripper's hand housing and fingers attached to the guard's arm model (log-036). No hardware.

Run from this folder with Dream's Python:  ../../.venv/bin/python -m pytest -q test_openarm_gripper.py
"""

import numpy as np
import pytest

import openarm_gripper as grip
import openarm_zones as zones


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

"""Tests for openarm_body.py: the robot's own body as keep-out zones for both arms, always on (log-039). No hardware.

Run from this folder with Dream's Python:  ../../.venv/bin/python -m pytest -q test_openarm_body.py
"""

import numpy as np
import pytest

import openarm_body as body
import openarm_mapping as mapping
import openarm_pair as pair
import openarm_zones as zones

LEFT_HOME = np.radians([10.2, -2.0, 11.9, 74.1, -0.5, -3.2, -16.5])     # the homes of 2026-10-27 (log-031)
RIGHT_HOME = np.radians([-7.4, 0.4, 7.2, 75.2, 1.0, 4.5, 19.1])


@pytest.fixture(autouse=True)
def official(monkeypatch):
    monkeypatch.setenv(mapping.ENV, "official")


def _empty_zones() -> zones.Zones:
    return zones.parse_zones({"version": 1, "captured_at": "2026-10-27T13:00:00+09:00", "source": "test",
                              "arms": {"left": {"reach": None, "keep_out": []}, "right": {"reach": None, "keep_out": []}}})


def test_the_body_boxes_come_from_the_official_mesh_and_sit_in_each_arms_frame():
    names = [b["name"] for b in body.keep_out("left")]
    assert names == ["body_column", "body_top"]                                            # the base plate is left out: the arms rest just above it
    column = body.keep_out("left")[0]
    assert column["center"][2] == pytest.approx((0.05 + 0.60) / 2 - 0.698)              # 5 to 60 cm above the plate, in the arm frame
    assert column["center"][1] - body.keep_out("right")[0]["center"][1] == pytest.approx(-0.062)   # the arm frames are 6.2 cm apart
    assert body.BASE_OFFSET_M["left"] == pytest.approx(pair.BASE_OFFSET_M["left"])


def test_a_point_inside_the_column_breaks_the_body_zone_with_no_bundle_zones_at_all():
    z = _empty_zones()
    inside = np.array([0.0, -0.031, 0.30 - 0.698])                                      # the column at 30 cm above the plate, left arm frame
    found = z.violations("left", {"grip:hand:0": inside}, check_reach=False)
    assert found and "body_column" in found[0]


@pytest.mark.parametrize("q_left, q_right", [(np.zeros(7), np.zeros(7)), (LEFT_HOME, RIGHT_HOME)])
def test_rest_and_the_10_27_homes_keep_clear_of_the_body(q_left, q_right):
    z = _empty_zones()
    for side, q in (("left", q_left), ("right", q_right)):
        pts = zones.ArmKinematics(side, logical=True).points(q)
        assert [v for v in z.violations(side, pts, check_reach=False) if "body" in v] == [], side


def test_a_hand_turned_in_against_the_column_is_refused():
    z = _empty_zones()
    q = np.radians([0.0, 0.0, 90.0, 90.0, 0.0, 0.0, 0.0])                                  # elbow bent, upper arm turned in: the hand swings toward the middle
    pts = zones.ArmKinematics("left", logical=True).points(q)
    assert any("body" in v for v in z.violations("left", pts, check_reach=False))


def test_dream_gets_the_column_in_its_own_format_but_not_the_upper_body():
    """log-040: Dream's IK lays 4 cm spheres on every moving link from joint 1, about 9.4 cm from the centre line; the upper body box reaches 8 cm out
    there, so Dream would see its own shoulder inside it. Only the column goes to Dream; the driver's guard keeps both."""
    for side in ("left", "right"):
        boxes = body.dream_keep_out(side)
        assert len(boxes) == 1 and set(boxes[0]) == {"kind", "center", "size"} and boxes[0]["kind"] == "box"
        column = [b for b in body.keep_out(side) if b["name"] == "body_column"][0]
        assert boxes[0]["center"] == column["center"] and boxes[0]["size"] == column["size"]

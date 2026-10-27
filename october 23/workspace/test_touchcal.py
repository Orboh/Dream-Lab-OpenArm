"""Tests for the touch calibration (no CAN): the mesh method reproduces log-025, and a moving arm is not taken as resting."""

import json
import pathlib

import numpy as np
import pytest

import touchcal

LOG = pathlib.Path(__file__).resolve().parent.parent / "log"


@pytest.fixture(scope="module")
def urdf():
    return touchcal.load_urdf()


def test_the_right_arm_touch_of_log_025_gives_the_same_table_top(urdf):
    q = json.loads((LOG / "stage6-touch-calibration-right-arm-20261024.json").read_text())["q_rad"]
    top, resting = touchcal.table_top(touchcal.lowest_points(urdf, "right", q))
    assert top == pytest.approx(-41.8, abs=0.1)
    assert {"openarm_right_link5", "openarm_right_link6"} <= set(resting) and all(p.startswith("openarm_right_") for p in resting)


def test_a_steady_reading_is_steady_and_a_moving_one_is_not(urdf, tmp_path):
    still = lambda: np.zeros(7)  # noqa: E731
    ticks = iter(range(1000))
    moving = lambda: np.full(7, next(ticks) * 0.01)  # noqa: E731
    out = tmp_path / "touch.json"
    assert touchcal.main(["right", "--out", str(out)], readers={"right": still}) == 0
    assert touchcal.main(["right", "--out", str(out)], readers={"right": moving}) == 1
    records = json.loads(out.read_text())
    assert [r["steady"] for r in records] == [True, False]

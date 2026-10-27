"""Tests for the website's "Go to home" button (no arms, no CAN): it refuses on a failed check, raises the allowed arms, holds until Park, and always switches off."""

import pathlib
import threading
import time

import numpy as np
import pytest
import yaml

import homemove
import preflight as pf

HOME_DEG = [10.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0]


class FakeArm:
    """An arm that "raises itself" while opening and records being switched off."""

    opened: list[str] = []
    off: list[str] = []

    def __init__(self, cfg, side, pose_deg=None, fail=False):
        if fail:
            raise RuntimeError("CAN write failed")
        self.side = side
        self.pose = list(HOME_DEG if pose_deg is None else pose_deg)
        FakeArm.opened.append(side)

    def _joint_pose(self):
        return self.pose

    def torque_off(self):
        FakeArm.off.append(self.side)


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    FakeArm.opened, FakeArm.off = [], []
    monkeypatch.setenv("OPENARM_MODEL_MAPPING", "identity")
    monkeypatch.setattr(homemove.openarm_lock, "active_arms", lambda: {"left", "right"})
    station = tmp_path / "station.yaml"
    arm = {"can_channel": "x", "gripper_closed_deg": 0, "gripper_open_deg": 60, "home": [float(x) for x in np.radians(HOME_DEG)]}
    station.write_text(yaml.safe_dump({"follower_l": arm, "follower_r": arm}))
    return station


def passing():
    return [pf.Check(True, pf.LABEL_CAN), pf.Check(False, pf.LABEL_CAMERAS_FREE, "held by this page")]


def wait_for(mover, state, timeout=2.0):
    end = time.monotonic() + timeout
    while mover.state != state and time.monotonic() < end:
        time.sleep(0.01)
    assert mover.state == state, mover.messages


def make(station: pathlib.Path, checks=passing, opener=FakeArm):
    return homemove.HomeMover(station, opener=opener, checks=checks)


def test_a_failed_check_refuses_and_nothing_opens(fresh):
    mover = make(fresh, checks=lambda: [pf.Check(False, pf.LABEL_FREE, "still running: run_teleop")])
    mover.go(["left", "right"])
    wait_for(mover, "idle")
    assert FakeArm.opened == [] and "run_teleop" in mover.text()


def test_the_page_own_cameras_do_not_block_the_move(fresh):
    mover = make(fresh)
    mover.go(["right"])
    wait_for(mover, "at home")
    assert FakeArm.opened == ["right"] and FakeArm.off == []


def test_both_arms_hold_at_home_until_park_then_switch_off(fresh):
    mover = make(fresh)
    mover.go(["left", "right"])
    wait_for(mover, "at home")
    assert FakeArm.opened == ["left", "right"] and FakeArm.off == []
    mover.park()
    wait_for(mover, "idle")
    assert sorted(FakeArm.off) == ["left", "right"]


def test_a_locked_arm_is_not_moved(fresh, monkeypatch):
    monkeypatch.setattr(homemove.openarm_lock, "active_arms", lambda: {"right"})
    mover = make(fresh)
    mover.go(["left", "right"])
    wait_for(mover, "at home")
    assert FakeArm.opened == ["right"] and "left: locked" in mover.text()


def test_an_arm_that_did_not_reach_home_is_switched_off_at_once(fresh):
    mover = make(fresh, opener=lambda cfg, side: FakeArm(cfg, side, pose_deg=[60.0] * 7))
    mover.go(["right"])
    wait_for(mover, "idle")
    assert FakeArm.off == ["right"] and "NOT at the home" in mover.text()


def test_an_arm_failing_to_open_switches_off_the_ones_already_up(fresh):
    opener = lambda cfg, side: FakeArm(cfg, side, fail=(side == "right"))  # noqa: E731
    mover = make(fresh, opener=opener)
    mover.go(["left", "right"])
    wait_for(mover, "idle")
    assert FakeArm.off == ["left"] and "CAN write failed" in mover.text()


def test_a_second_go_while_busy_is_refused(fresh):
    gate = threading.Event()
    mover = make(fresh, checks=lambda: (gate.wait(2.0), passing())[1])
    assert mover.go(["right"]) is True
    assert mover.go(["right"]) is False
    gate.set()
    wait_for(mover, "at home")
    mover.park()
    wait_for(mover, "idle")


def test_close_parks_arms_left_at_home(fresh):
    mover = make(fresh)
    mover.go(["right"])
    wait_for(mover, "at home")
    mover.close()
    assert FakeArm.off == ["right"]

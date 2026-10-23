"""Tests for the camera-view move: offline, the driver's arm is a fake."""

import types

import numpy as np
import pytest
import yaml

import camerapose as cp
import openarm_lock as lock

HOME = {"left": [0.8, -0.1, 0.1, 1.1, 0.0, 0.0, -1.2], "right": [-0.8, 0.0, 0.1, 1.0, -0.1, 0.0, 1.4]}


class FakeArm:
    def __init__(self, cfg, side, at_rest=True):
        self.side, self.cfg, self.off = side, cfg, False
        self.pose = np.degrees(cfg.home) + 1.5 if at_rest else np.degrees(cfg.home) + 60.0   # raised itself while opening, or did not move

    def _joint_pose(self):
        return list(self.pose)

    def torque_off(self):
        self.off = True


@pytest.fixture
def station(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENARM_ACTIVE_ARMS", str(tmp_path / "active.json"))
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump({KEY: {"can_channel": f"can_fol_{side[0]}", "gripper_closed_deg": 0.9, "gripper_open_deg": -66.0, "home": HOME[side]}
                                    for side, KEY in cp.KEYS.items()}))
    return path


def test_arm_config_reads_the_link_gripper_range_and_home(station):
    cfg = cp.arm_config(station, "right")
    assert cfg.can_channel == "can_fol_r" and cfg.gripper_open_deg == -66.0 and cfg.home == HOME["right"]


def test_each_arm_is_raised_reported_held_and_then_parked_and_switched_off(station):
    opened, held, said = [], [], []
    def opener(cfg, side):
        opened.append(FakeArm(cfg, side)); return opened[-1]
    code = cp.run(station, ["left", "right"], hold=lambda: held.append(len(opened)), opener=opener, say=said.append)
    assert code == 0 and [a.side for a in opened] == ["left", "right"] and all(a.off for a in opened)
    assert held == [1, 2]                                           # held once per arm, after that arm opened and before the next
    assert any("camera-view pose" in s for s in said) and said[-1] == "done"


def test_an_arm_that_did_not_start_at_rest_is_reported_and_still_switched_off_safely(station):
    arms, held, said = [], [], []
    def opener(cfg, side):
        arms.append(FakeArm(cfg, side, at_rest=False)); return arms[-1]
    code = cp.run(station, ["right"], hold=lambda: held.append(1), opener=opener, say=said.append)
    assert code == 1 and held == [] and arms[0].off and any("NOT at the home" in s and "hand-guide" in s for s in said)


def test_a_locked_arm_is_skipped_and_never_opened(station):
    lock.set_active("right")
    opened, said = [], []
    code = cp.run(station, ["left", "right"], hold=lambda: None, opener=lambda c, s: opened.append(s) or FakeArm(c, s), say=said.append)
    assert code == 0 and opened == ["right"] and any(s.startswith("left: locked") for s in said)


def test_the_arm_is_switched_off_even_when_the_hold_is_interrupted(station):
    arms = []
    def opener(cfg, side):
        arms.append(FakeArm(cfg, side)); return arms[-1]
    def interrupted():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        cp.run(station, ["left"], hold=interrupted, opener=opener, say=lambda s: None)
    assert arms[0].off

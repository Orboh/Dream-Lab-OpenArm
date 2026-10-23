"""Tests for the passive CAN observer: it decodes the follower's own state frames and never sends anything."""

import json
import math
import threading
import time
import types

import numpy as np
import pytest

import canobserver as co
import openarm_v2 as drv
from robots import damiao, errors, robstride

OFFSETS = {"joint_1": 2.0, "joint_2": -1.0, "joint_3": 3.0, "joint_4": -4.0, "joint_5": 5.0, "joint_6": -6.0, "joint_7": 7.0, "gripper": 0.0}
LOGICAL = {"joint_1": -10.0, "joint_2": -20.0, "joint_3": 5.0, "joint_4": 30.0, "joint_5": 1.0, "joint_6": 2.0, "joint_7": 3.0, "gripper": -32.5}


def state_frame(name, native_deg, status=1):
    pmax, vmax, tmax = damiao.MOTOR_LIMITS[drv.MODELS[name]]
    q = robstride.float_to_uint(math.radians(native_deg), -pmax, pmax, 16)
    dq = robstride.float_to_uint(0.0, -vmax, vmax, 12)
    tau = robstride.float_to_uint(0.0, -tmax, tmax, 12)
    data = bytes([(status << 4) | drv.CAN_IDS[name], q >> 8, q & 0xFF, dq >> 4, ((dq & 0xF) << 4) | (tau >> 8), tau & 0xFF, 30, 30])
    return types.SimpleNamespace(arbitration_id=0x10 + drv.CAN_IDS[name], data=data)


class ListenOnlyBus:
    """A fake link: frames can be fed in; sending raises, so a test proves the observer transmits nothing."""

    def __init__(self):
        self.queue, self.closed = [], False
        self.lock = threading.Lock()

    def feed(self, *frames):
        with self.lock:
            self.queue.extend(frames)

    def recv(self, timeout=None):
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
        time.sleep(min(timeout or 0.01, 0.005))
        return None

    def send(self, *a, **k):
        raise AssertionError("the observer must never transmit")

    def shutdown(self):
        self.closed = True


def feed_pose(bus, logical=LOGICAL):
    bus.feed(*(state_frame(n, logical[n] + OFFSETS[n]) for n in drv.MOTORS))


@pytest.fixture
def observer(tmp_path, monkeypatch):
    (tmp_path / "follower_left.json").write_text(json.dumps({n: {"homing_offset": o} for n, o in OFFSETS.items()}))
    monkeypatch.setattr(drv, "CALIBRATION_DIR", tmp_path)
    bus = ListenOnlyBus()
    obs = co.PassiveArm("left", opener=lambda channel: bus, max_age_s=0.3)
    obs.start()
    yield obs, bus
    obs.stop()


def wait_until(condition, seconds=2.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_decodes_state_frames_into_logical_degrees_and_model_radians(observer):
    obs, bus = observer
    feed_pose(bus)
    assert wait_until(lambda: obs.has_all())
    deg = obs.logical_deg()
    assert [deg[n] for n in drv.MOTORS] == pytest.approx([LOGICAL[n] for n in drv.MOTORS], abs=0.05)
    assert np.degrees(obs.q_rad()) == pytest.approx([LOGICAL[n] for n in drv.JOINTS], abs=0.05)


def test_ignores_command_frames_and_register_replies(observer):
    obs, bus = observer
    bus.feed(types.SimpleNamespace(arbitration_id=1, data=bytes(8)),                       # a command to motor 1 (looped back)
             types.SimpleNamespace(arbitration_id=0x11, data=bytes([0x01, 1, 0x33, 0, 0, 0, 0, 0])),  # a register reply, not a state
             types.SimpleNamespace(arbitration_id=0x7FF, data=bytes(8)))
    feed_pose(bus)
    assert wait_until(lambda: obs.has_all())
    assert obs.logical_deg()["joint_1"] == pytest.approx(LOGICAL["joint_1"], abs=0.05)


def test_it_never_transmits_and_closes_the_link_on_stop(observer):
    obs, bus = observer
    feed_pose(bus)
    wait_until(lambda: obs.has_all())
    obs.stop()
    assert bus.closed  # send() would have raised inside the reader thread: no error was recorded
    assert obs.error is None


def test_stale_or_missing_data_is_an_error(observer):
    obs, bus = observer
    with pytest.raises(errors.ArmError, match="no frames"):
        obs.logical_deg()
    bus.feed(state_frame("joint_1", 0.0))
    assert wait_until(lambda: "joint_1" in obs.motors_seen())
    with pytest.raises(errors.ArmError, match="joint_2"):
        obs.logical_deg()
    feed_pose(bus)
    assert wait_until(lambda: obs.has_all())
    time.sleep(0.5)  # nothing new: the teleop stopped
    with pytest.raises(errors.ArmError, match="stale|old"):
        obs.logical_deg()


def test_status_text_tells_whether_the_teleop_is_running(observer):
    obs, bus = observer
    assert "no frames" in obs.status()
    feed_pose(bus)
    assert wait_until(lambda: obs.has_all())
    assert "receiving" in obs.status()

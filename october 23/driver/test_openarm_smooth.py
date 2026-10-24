"""Tests for the smooth, motor-safe motion of `Arm`: bounded speed and acceleration, Dream's linear Home, torque budget, soft limits, soft start, temperature and gripper alarms,
and the control thread. A fake CAN bus, no hardware."""

import math
import threading
import time

import numpy as np
import pytest

import openarm_motion as motion
import openarm_poses as poses
import openarm_v2 as drv
from robots import damiao, errors
from test_openarm_v2 import LOGICAL, OFFSETS, FakeBus, as_logical, joints_of, make_arm  # noqa: F401  (fixtures)


def smooth_arm(make_arm, **kw):
    return make_arm(drv.Arm, smooth=True, follow=True, **kw)


def targets_of(bus, name):
    return np.array([t - OFFSETS[name] for n, t, _ in bus.history if n == name])


def test_a_big_step_target_moves_smoothly_within_the_limits_and_settles(make_arm):
    arm, bus = smooth_arm(make_arm)
    bus.history.clear()
    target = np.radians([LOGICAL[n] for n in drv.JOINTS]) + np.radians([6.0, 0, 0, 0, 15.0, 0, 18.0])   # an 18 deg wrist step, 6 deg shoulder step
    arm.write(target, 0.5)
    assert arm.settle(max_s=20.0)
    j7 = targets_of(bus, "joint_7")
    speed = np.abs(np.diff(j7)) / 0.01
    v5 = (j7[5:] - j7[:-5]) / 0.05                      # speed over 50 ms windows: the 16-bit MIT position step (0.022 deg) would show up as noise on a 10 ms difference
    assert speed.max() <= motion.GENTLE_VEL_DPS[6] * 1.05 and np.abs(np.diff(v5[::5])).max() / 0.05 <= motion.GENTLE_ACC_DPS2[6] * 1.1
    assert joints_of(bus) == pytest.approx(list(np.degrees(target)), abs=0.2)


def test_dreams_linear_home_move_is_shaped_and_keeps_settling_after_dream_stops_writing(make_arm):
    """Dream: 120 linear steps, 25 ms apart, then nothing. The driver turns it into a soft, bounded move that still arrives."""
    arm, bus = smooth_arm(make_arm, sync_dt=0.025)
    bus.history.clear()
    start = np.radians([LOGICAL[n] for n in drv.JOINTS])
    home = start + np.radians([40.0, 0, 0, 30.0, 0, 0, -50.0])
    for step in range(1, 121):
        arm.write(start + (home - start) * step / 120, 0.5)
    assert arm.settle(max_s=20.0)                                    # no more writes: the driver finishes by itself
    assert joints_of(bus) == pytest.approx(list(np.degrees(home)), abs=0.2)
    for name in ("joint_1", "joint_4", "joint_7"):
        series = targets_of(bus, name)
        dt = 0.025 if len(series) <= 125 else 0.01
        assert np.abs(np.diff(series)).max() < 40.0 * 0.025 + 1e-6 or name == "joint_7"   # never faster than the limit per tick
    assert np.abs(np.diff(targets_of(bus, "joint_1"))).max() / 0.025 <= 40.0 * 1.05


def test_a_blocked_joint_never_builds_a_lead_beyond_the_torque_budget(make_arm):
    arm, bus = make_arm(drv.Arm, smooth=True, follow=True, frozen=("joint_4",))
    bus.history.clear()
    target = np.radians([LOGICAL[n] for n in drv.JOINTS]) + np.radians([0, 0, 0, 15.0, 0, 0, 0])   # joint 4 is asked 15 deg away but cannot move
    arm.write(target, 0.5)
    for _ in range(300):
        arm._tick(0.01)
    lead = np.abs(targets_of(bus, "joint_4") - LOGICAL["joint_4"])
    assert lead.max() <= motion.build_limits(drv.KP, drv.MODELS).lead_deg[3] + 0.05      # about 3.3 deg
    assert lead.max() > 1.0                                                              # it did push, within the budget


def test_targets_stay_inside_the_soft_limits(make_arm):
    low, high = drv.LIMITS_DEG["left"]["joint_4"]
    arm, bus = smooth_arm(make_arm, logical={"joint_4": high - 5.0})
    bus.history.clear()
    target = np.radians([arm._deg[n] for n in drv.JOINTS]); target[3] = math.radians(high + 5.0)   # past the hard limit
    arm.write(target, 0.5)
    arm.settle(max_s=30.0)
    series = targets_of(bus, "joint_4")
    assert series.max() <= high - 3.0 + 1e-6 and series.max() > high - 5.0 + 1.0        # it went up toward the limit and stopped 3 deg short


def test_the_gains_ramp_up_from_a_share_to_full_at_start(make_arm):
    arm, bus = make_arm(drv.Arm, smooth=True, follow=True)
    kp = [k for n, _, k in bus.history if n == "joint_1"]
    assert kp[0] == pytest.approx(drv.KP["joint_1"] * drv.SOFT_START_GAIN0, rel=0.05)
    assert kp[-1] == pytest.approx(drv.KP["joint_1"], rel=0.05) and all(b >= a - 1e-6 for a, b in zip(kp, kp[1:]))


def test_a_hot_motor_makes_the_arm_hold_and_raise_an_alarm_then_recover(make_arm, caplog):
    arm, bus = smooth_arm(make_arm, temps={"joint_2": 62})
    arm._tick(0.01)
    assert arm.alarm() is None and any("62" in r.message for r in caplog.records)           # a warning only
    bus.temps["joint_2"] = 72
    arm._tick(0.01)
    assert "joint_2" in arm.alarm() and "72" in arm.alarm()
    bus.history.clear()
    arm.write(np.radians([LOGICAL[n] for n in drv.JOINTS]) + np.radians(5.0), 0.5)
    for _ in range(100):
        arm._tick(0.01)
    assert np.abs(targets_of(bus, "joint_1") - LOGICAL["joint_1"]).max() < 0.2            # held: it did not move toward the new target
    bus.temps["joint_2"] = 60
    arm._tick(0.01)
    assert arm.alarm() is None


def test_a_motor_fault_raises_an_alarm_with_dreams_text(make_arm):
    arm, bus = smooth_arm(make_arm)
    bus.fault["joint_3"] = 0xA
    arm._tick(0.01)
    assert "joint_3" in arm.alarm() and "overcurrent" in arm.alarm()


def test_a_gripper_that_squeezes_against_something_is_relaxed_and_reported(make_arm):
    arm, bus = make_arm(drv.Arm, smooth=True, follow=True, frozen=("gripper",), ends=(0.0, -65.0))
    bus.history.clear()
    target = np.radians([LOGICAL[n] for n in drv.JOINTS])
    arm.write(target, 0.2)                         # shut the gripper toward -13 deg while it cannot move from -32.5
    for _ in range(400):                           # 4 seconds
        arm._tick(0.01)
    grip = targets_of(bus, "gripper")
    assert "gripper" in (arm.alarm() or "")
    assert abs(grip[-1] - LOGICAL["gripper"]) < GRIPPER_RELAXED_TOL                      # relaxed to where it is
    assert np.abs(grip - LOGICAL["gripper"]).max() <= motion.GRIPPER_LEAD_DEG + 0.05    # and never led it by more than the squeeze limit


GRIPPER_RELAXED_TOL = 0.5


def test_the_raise_to_the_taught_home_is_smooth_and_the_park_returns_the_same_way(make_arm, monkeypatch, tmp_path):
    from test_openarm_guard import teach_path, write_zones_with_path
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    path = write_zones_with_path(tmp_path, monkeypatch)
    arm, bus = smooth_arm(make_arm, logical=as_logical(poses.REST_DEG["left"]), home=[math.radians(v) for v in path[-1]])
    assert joints_of(bus) == pytest.approx(path[-1], abs=0.3)
    for name, vmax in (("joint_1", 40.0), ("joint_4", 40.0), ("joint_7", 70.0)):
        series = targets_of(bus, name)
        assert np.abs(np.diff(series)).max() / drv.SAMPLE_DT_S <= vmax * 1.05
    arm.torque_off()
    assert joints_of(bus) == pytest.approx(poses.REST_DEG["left"], abs=0.3) and not any(bus.enabled.values())


def test_the_control_thread_follows_targets_keeps_sending_and_reads_without_querying(make_arm):
    arm, bus = make_arm(drv.Arm, smooth=True, follow=True)
    arm._start_thread()
    try:
        before = len(bus.history)
        target = np.radians([LOGICAL[n] for n in drv.JOINTS]) + np.radians([4.0, 0, 0, 0, 10.0, 0, 0])
        arm.write(target, 0.5)
        end = time.monotonic() + 8.0
        while time.monotonic() < end and abs(joints_of(bus)[4] - (LOGICAL["joint_5"] + 10.0)) > 0.3:
            time.sleep(0.05)
        assert joints_of(bus)[4] == pytest.approx(LOGICAL["joint_5"] + 10.0, abs=0.3)
        queries_before = sum(1 for arb, _ in bus.sent if arb == damiao.PARAM_ID)
        arm.read(); arm.read()
        assert sum(1 for arb, _ in bus.sent if arb == damiao.PARAM_ID) == queries_before      # read() uses the thread's replies, not the bus
        count = len(bus.history)
        time.sleep(0.3)
        assert len(bus.history) > count                                                       # still sending: the keep-alive
    finally:
        arm.close()
    frames = len(bus.history)
    time.sleep(0.2)
    assert len(bus.history) == frames and bus.closed                                          # close() stops it


def test_torque_off_with_the_control_thread_parks_stops_the_thread_and_switches_off(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    arm, bus = make_arm(drv.Arm, smooth=True, follow=True, logical=as_logical(poses.READY_DEG["left"]))
    arm._start_thread()
    time.sleep(0.2)
    arm.torque_off()
    assert not any(bus.enabled.values()) and bus.closed and arm._thread is None
    assert joints_of(bus) == pytest.approx(poses.REST_DEG["left"], abs=0.5)
    frames = len(bus.history)
    time.sleep(0.2)
    assert len(bus.history) == frames

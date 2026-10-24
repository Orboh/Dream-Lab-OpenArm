"""Our `Arm` and `ReadOnlyArm` behind Dream's own `FollowerArm` wrapper, the object the station and control loop really call. Fake bus: no hardware."""

import numpy as np
import pytest

import openarm_v2 as drv
from robots import errors, followers
from test_openarm_v2 import LOGICAL, make_arm  # noqa: F401  (make_arm is a fixture)


def wrapped(arm, side="left"):
    return followers.FollowerArm(side, arm, home_rad=np.zeros(7))


def test_read_through_the_wrapper_carries_temperatures_and_no_gripper_effort(make_arm):
    arm, _ = make_arm(drv.Arm, temps={"joint_3": 41})
    state = wrapped(arm).read()
    assert state.positions_rad.shape == (7,) and state.motor_temps_c.shape == (8,)
    assert state.motor_temps_c.max() == 41
    assert state.gripper_effort is None  # Dream's grip force cap stays off for this driver


def test_write_hold_and_disengage_through_the_wrapper(make_arm):
    arm, bus = make_arm(drv.Arm)
    follower = wrapped(arm)
    follower.hold()  # nothing written yet: holds the measured pose
    assert all(bus.enabled.values())
    pose = np.radians([LOGICAL[n] for n in drv.JOINTS])
    follower.write(followers.types.ArmCommand(positions_rad=pose, gripper=0.5))
    follower.disengage()
    assert not any(bus.enabled.values())


def test_a_hot_motor_stops_the_run_through_the_wrapper_alarm(make_arm):
    """Dream's loop calls `alarm()` every tick and raises SafetyStop on it: the heat limit lives in our driver, as the station file's motor_temp keys do not apply to a custom driver."""
    arm, _ = make_arm(drv.Arm, temps={"joint_2": 75}, smooth=True)
    follower = wrapped(arm)
    arm._tick(0.01)
    assert "75 C" in (follower.alarm() or "") and follower.alarm().startswith("arm 'left':")


def test_close_for_reopen_switches_the_arm_off_and_it_reads_dead_after(make_arm):
    arm, bus = make_arm(drv.Arm)
    follower = wrapped(arm)
    arm.write(np.radians([LOGICAL[n] for n in drv.JOINTS]), 0.5)
    assert drv.Arm.drops_when_limp and follower.is_alive()
    follower.close_for_reopen()
    assert not any(bus.enabled.values()) and bus.closed
    assert not follower.is_alive()
    with pytest.raises(errors.ArmError, match="closed"):
        follower.read()


def test_read_only_arm_reads_and_refuses_writes_through_the_wrapper(make_arm):
    arm, bus = make_arm(drv.ReadOnlyArm)
    follower = wrapped(arm)
    assert follower.read().motor_temps_c is not None
    with pytest.raises(errors.ArmError, match="read only"):
        follower.write(followers.types.ArmCommand(positions_rad=np.zeros(7), gripper=0.5))
    follower.close()
    assert not any(bus.enabled.values())


def test_a_dead_control_thread_reads_dead_with_its_reason_and_close_for_reopen_does_not_move_it(make_arm):
    arm, bus = make_arm(drv.Arm, smooth=True)
    follower = wrapped(arm)
    assert follower.is_alive() and follower.why_stopped() == "its driver stopped answering"
    arm._alarms["thread"] = "arm left: the control thread stopped (OSError)"
    assert not follower.is_alive() and follower.holds_dead()
    assert "control thread stopped" in follower.why_stopped()
    with pytest.raises(errors.ArmError, match="stopped"):
        follower.read()
    sent = len(bus.sent)
    follower.close_for_reopen()  # a dead arm is closed, not switched off (that would drop it)
    assert len(bus.sent) == sent and bus.closed


def test_silent_motors_under_a_running_thread_read_dead(make_arm):
    arm, _ = make_arm(drv.Arm, smooth=True, follow=True)
    arm._start_thread()
    try:
        arm._deg_time -= 1.0
        assert not arm.is_alive() and "no reply" in arm.why_stopped()
    finally:
        arm._stop_thread()

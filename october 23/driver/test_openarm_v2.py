"""Tests for openarm_v2.py against a fake CAN bus: no hardware, nothing moves.

Run from this folder with Dream's Python:  ../../.venv/bin/python -m pytest -q test_openarm_v2.py
"""

import json
import math
import types

import numpy as np
import pytest

import openarm_poses as poses
import openarm_v2 as drv
from robots import damiao, errors, robstride


class FakeBus:
    """Answers like the arm's motors: state queries, enable/disable and MIT frames, each with a state frame."""

    def __init__(self, native_deg: dict[str, float], temp: int = 30):
        self.native = dict(native_deg)
        self.enabled = {n: False for n in drv.MOTORS}
        self.sent: list[tuple[int, bytes]] = []
        self.fd_flags: list[bool] = []
        self.history: list[tuple[str, float, float]] = []   # (motor, MIT target in degrees, kp) of every MIT frame
        self.temps: dict[str, int] = {}
        self.mit_targets: dict[str, float] = {}
        self.queue: list = []
        self.silent: set[str] = set()
        self.fault: dict[str, int] = {}
        self.stuck_on: set[str] = set()
        self.temp = temp
        self.closed = False
        self.follow = False  # True: a motor reaches each MIT target at once
        self.frozen: set[str] = set()  # motors that never move (a stuck joint)
        self.ids = {i: n for n, i in drv.CAN_IDS.items()}

    def _reply(self, name: str) -> None:
        if name in self.silent:
            return
        pmax, vmax, tmax = damiao.MOTOR_LIMITS[drv.MODELS[name]]
        q = robstride.float_to_uint(math.radians(self.native[name]), -pmax, pmax, 16)
        dq = robstride.float_to_uint(0.0, -vmax, vmax, 12)
        tau = robstride.float_to_uint(0.0, -tmax, tmax, 12)
        status = self.fault.get(name, damiao.STATUS_ENABLED if self.enabled[name] else damiao.STATUS_DISABLED)
        data = bytes([(status << 4) | drv.CAN_IDS[name], q >> 8, q & 0xFF, dq >> 4, ((dq & 0xF) << 4) | (tau >> 8),
                      tau & 0xFF, self.temps.get(name, self.temp), self.temps.get(name, self.temp)])
        self.queue.append(types.SimpleNamespace(arbitration_id=0x10 + drv.CAN_IDS[name], data=data))

    def send(self, message, timeout=None) -> None:
        arb, data = message.arbitration_id, bytes(message.data)
        self.sent.append((arb, data))
        self.fd_flags.append(bool(message.is_fd))
        if arb == damiao.PARAM_ID:
            self._reply(self.ids[data[0]])
        elif data[:7] == b"\xff" * 7:
            name = self.ids[arb]
            self._reply(name)  # like the real motors, the reply carries the status from before the command
            if data[7] == damiao.CMD_ENABLE:
                self.enabled[name] = True
            elif data[7] == damiao.CMD_DISABLE and name not in self.stuck_on:
                self.enabled[name] = False
        else:
            name = self.ids[arb]
            pmax = damiao.MOTOR_LIMITS[drv.MODELS[name]][0]
            q = (data[0] << 8) | data[1]
            self.mit_targets[name] = math.degrees(robstride.uint_to_float(q, -pmax, pmax, 16))
            kp = robstride.uint_to_float(((data[3] & 0xF) << 8) | data[4], 0.0, 500.0, 12)
            self.history.append((name, self.mit_targets[name], kp))
            if self.follow and name not in self.frozen:
                self.native[name] = self.mit_targets[name]
            self._reply(name)

    def recv(self, timeout=None):
        return self.queue.pop(0) if self.queue else None

    def shutdown(self) -> None:
        self.closed = True


OFFSETS = {"joint_1": 2.0, "joint_2": -1.0, "joint_3": 3.0, "joint_4": -4.0, "joint_5": 5.0, "joint_6": -6.0,
           "joint_7": 7.0, "gripper": 0.0}
LOGICAL = {"joint_1": -10.0, "joint_2": -20.0, "joint_3": 5.0, "joint_4": 30.0, "joint_5": 1.0, "joint_6": 2.0,
           "joint_7": 3.0, "gripper": -32.5}


@pytest.fixture
def make_arm(tmp_path, monkeypatch):
    """Builds an arm of a class on a fake bus whose motors sit at LOGICAL (native = logical + offset)."""
    for side in ("left", "right"):
        (tmp_path / f"follower_{side}.json").write_text(
            json.dumps({n: {"homing_offset": o} for n, o in OFFSETS.items()}))
    monkeypatch.setattr(drv, "CALIBRATION_DIR", tmp_path)

    def build(cls, side="left", ends=(0.0, -65.0), channel="can_fol_l", logical=None, follow=False, home=None, frozen=(), smooth=False, sync_dt=0.01, temps=None):
        start = {**LOGICAL, **(logical or {})}
        bus = FakeBus({n: start[n] + OFFSETS[n] for n in drv.MOTORS})
        bus.follow = follow
        bus.frozen = set(frozen)
        bus.temps = dict(temps or {})
        def opener(opened_channel):
            bus.opened_channel = opened_channel
            return bus

        monkeypatch.setattr(cls, "open_can_bus", staticmethod(opener))
        cfg = types.SimpleNamespace(can_channel=channel, gripper_closed_deg=ends[0], gripper_open_deg=ends[1])
        cfg.smooth, cfg.sync, cfg.sync_dt = smooth, True, sync_dt
        if home is not None:
            cfg.home = home
        return cls(cfg, side), bus

    return build


def test_read_applies_homing_offsets_and_maps_gripper(make_arm):
    arm, _ = make_arm(drv.ReadOnlyArm)
    reading = arm.read()
    assert np.allclose(np.degrees(reading.positions_rad), [LOGICAL[n] for n in drv.JOINTS], atol=0.05)
    assert reading.gripper == pytest.approx(0.5, abs=0.01)  # -32.5 deg of 0..-65
    assert arm.gripper_deg() == pytest.approx(-32.5, abs=0.05)


def test_read_only_arm_sends_only_state_queries_and_refuses_write(make_arm):
    arm, bus = make_arm(drv.ReadOnlyArm)
    arm.read()
    assert all(arb == damiao.PARAM_ID for arb, _ in bus.sent)  # no enable, no MIT frame
    assert not any(bus.enabled.values())
    with pytest.raises(errors.ArmError):
        arm.write(np.zeros(7), 0.5)
    arm.close()
    assert bus.closed
    assert all(arb == damiao.PARAM_ID for arb, _ in bus.sent)


def test_side_comes_from_the_channel_while_setup_finds_arms(make_arm):
    arm, _ = make_arm(drv.ReadOnlyArm, side="adapter-7", channel="can_fol_r")
    assert arm.side == "right"
    with pytest.raises(errors.ArmError):
        make_arm(drv.ReadOnlyArm, side="adapter-7", channel="can0")


def test_missing_calibration_refuses_to_open(make_arm, monkeypatch, tmp_path):
    monkeypatch.setattr(drv, "CALIBRATION_DIR", tmp_path / "nowhere")
    with pytest.raises(errors.ArmError, match="calibration"):
        make_arm(drv.ReadOnlyArm)


def test_a_silent_motor_is_an_error(make_arm):
    arm, bus = make_arm(drv.ReadOnlyArm)
    bus.silent.add("joint_3")
    bus.queue.clear()
    arm.bus.motors["joint_3"].state = None
    with pytest.raises(errors.ArmError):
        arm.read()


def test_a_reading_far_outside_limits_is_refused(make_arm):
    arm, bus = make_arm(drv.ReadOnlyArm)
    bus.native["joint_4"] = 160.0 + OFFSETS["joint_4"]  # 160 deg: past 140 but inside the 30 deg slack
    arm.read()
    bus.native["joint_4"] = 200.0 + OFFSETS["joint_4"]  # past 140 + 30
    with pytest.raises(errors.ArmError, match="wrong zero"):
        arm.read()


def test_a_fault_status_is_reported(make_arm):
    arm, bus = make_arm(drv.ReadOnlyArm)
    bus.fault["joint_2"] = 0xA
    with pytest.raises(errors.ArmError, match="overcurrent"):
        arm.read()


def test_arm_switches_on_holding_where_it_is(make_arm):
    arm, bus = make_arm(drv.Arm)
    assert all(bus.enabled.values())
    for name in drv.MOTORS:  # first target of each motor is its own position
        assert bus.mit_targets[name] == pytest.approx(LOGICAL[name] + OFFSETS[name], abs=0.05)


def test_arm_never_enables_when_the_first_reading_is_bad(make_arm, monkeypatch):
    make_arm(drv.ReadOnlyArm)  # installs the calibration files
    bus = FakeBus({n: LOGICAL[n] + OFFSETS[n] for n in drv.MOTORS})
    bus.native["joint_1"] = 168.0 + OFFSETS["joint_1"]  # left joint_1 is -200..80: 168 is past 80 + 30
    monkeypatch.setattr(drv.Arm, "open_can_bus", staticmethod(lambda _c: bus))
    cfg = types.SimpleNamespace(can_channel="can_fol_l", gripper_closed_deg=0.0, gripper_open_deg=-65.0)
    with pytest.raises(errors.ArmError):
        drv.Arm(cfg, "left")
    assert not any(bus.enabled.values())
    assert not any(data[:7] == b"\xff" * 7 for _, data in bus.sent)  # not even an enable frame was sent
    assert bus.closed


def test_write_sends_native_targets_through_the_offsets(make_arm):
    arm, bus = make_arm(drv.Arm)
    target = np.radians([LOGICAL[n] + 5.0 for n in drv.JOINTS])
    arm.write(target, 0.5)
    for name in drv.JOINTS:
        assert bus.mit_targets[name] == pytest.approx(LOGICAL[name] + 5.0 + OFFSETS[name], abs=0.05)
    assert bus.mit_targets["gripper"] == pytest.approx(-32.5, abs=0.05)


def test_write_refuses_a_big_jump_and_sends_nothing(make_arm):
    arm, bus = make_arm(drv.Arm)
    before = dict(bus.mit_targets)
    target = np.radians([LOGICAL[n] + (60.0 if n == "joint_1" else 0.0) for n in drv.JOINTS])
    with pytest.raises(errors.ArmError, match="joint_1"):
        arm.write(target, 0.5)
    assert bus.mit_targets == before


def test_write_clamps_to_the_limits(make_arm):
    arm, bus = make_arm(drv.Arm)
    here = np.radians([LOGICAL[n] for n in drv.JOINTS])
    low, _ = drv.LIMITS_DEG["left"]["joint_4"]
    arm.write(here - np.eye(7)[3] * math.radians(15.0), 0.5)  # 30 - 15 = 15, inside
    arm.write(np.radians([LOGICAL[n] if n != "joint_4" else 10.0 for n in drv.JOINTS]), 0.5)
    assert bus.mit_targets["joint_4"] >= low + OFFSETS["joint_4"] - 1e-6


def test_write_refuses_while_the_gripper_range_is_unmeasured(make_arm):
    arm, bus = make_arm(drv.Arm, ends=(math.nan, math.nan))
    before = dict(bus.mit_targets)
    with pytest.raises(errors.ArmError, match="not measured"):
        arm.write(np.radians([LOGICAL[n] for n in drv.JOINTS]), 0.5)
    assert bus.mit_targets == before
    assert arm.read().gripper == pytest.approx(0.5, abs=0.01)  # reading falls back to the fork's range


def test_disengage_switches_off_and_the_next_write_reengages_and_is_dropped(make_arm):
    arm, bus = make_arm(drv.Arm)
    assert arm.disengage()
    assert not any(bus.enabled.values())
    bus.native["joint_1"] += 8.0  # a hand moved the arm
    arm.write(np.radians([LOGICAL[n] for n in drv.JOINTS]), 0.5)
    assert all(bus.enabled.values())
    assert bus.mit_targets["joint_1"] == pytest.approx(LOGICAL["joint_1"] + 8.0 + OFFSETS["joint_1"], abs=0.05)


def test_torque_off_confirms_every_motor_and_raises_if_one_stays_on(make_arm):
    arm, bus = make_arm(drv.Arm)
    bus.stuck_on.add("joint_2")
    with pytest.raises(errors.ArmError, match="joint_2"):
        arm.torque_off()
    assert bus.closed
    arm2, bus2 = make_arm(drv.Arm)
    arm2.torque_off()
    assert not any(bus2.enabled.values()) and bus2.closed


def test_close_does_not_switch_motors_off(make_arm):
    arm, bus = make_arm(drv.Arm)
    sent = len(bus.sent)
    arm.close()
    assert bus.closed and len(bus.sent) == sent and all(bus.enabled.values())


def test_every_frame_is_a_can_fd_frame(make_arm):
    arm, bus = make_arm(drv.Arm)
    arm.write(np.radians([LOGICAL[n] for n in drv.JOINTS]), 0.5)
    arm.disengage()
    assert bus.fd_flags and all(bus.fd_flags)


@pytest.mark.parametrize("address, link, side", [("left", "can_fol_l", "left"), ("right", "can_fol_r", "right"),
                                                 ("Left", "can_fol_l", "left"), (" RIGHT ", "can_fol_r", "right")])
def test_left_and_right_are_accepted_as_addresses(make_arm, address, link, side):
    arm, bus = make_arm(drv.ReadOnlyArm, side="typed-" + address, channel=address)
    assert arm.side == side
    assert bus.opened_channel == link  # the real CAN link is what gets opened
    arm.read()


def test_the_side_argument_still_decides_when_the_channel_is_a_link_name(make_arm):
    arm, bus = make_arm(drv.ReadOnlyArm, side="right", channel="can_fol_r")
    assert arm.side == "right" and bus.opened_channel == "can_fol_r"


def test_a_channel_that_is_neither_alias_nor_link_name_is_refused(make_arm):
    with pytest.raises(errors.ArmError):
        make_arm(drv.ReadOnlyArm, side="adapter-9", channel="sideways")


# -- rest / ready poses and the path between them ---------------------------------------------------------------------

def as_logical(pose):
    return {n: v for n, v in zip(drv.JOINTS, pose)}


def joints_of(bus):
    return [bus.native[n] - OFFSETS[n] for n in drv.JOINTS]


def test_densify_ends_on_the_last_point_and_respects_the_speed():
    points = [[0.0] * 7, [30.0, 0, 0, 0, 0, 0, 0], [30.0, 0, 0, 60.0, 0, 0, 0]]
    samples = poses.densify(points, max_dps=30.0, dt=0.02)
    assert samples[-1] == pytest.approx(points[-1])
    full = [points[0]] + samples
    peak = max(abs(b - a) for p, q in zip(full, full[1:]) for a, b in zip(p, q)) / 0.02
    assert peak <= 1.5 * 30.0 * 1.35  # easing peaks near 1.5x the target; allow rounding


def test_nearest_picks_the_closest_path_point():
    path = poses.path_to_ready("left")
    assert poses.nearest(path, poses.REST_DEG["left"]) == (0, 0.0)
    index, distance = poses.nearest(path, [x + 2.0 for x in poses.READY_DEG["left"]])
    assert index == len(path) - 1 and distance == pytest.approx(2.0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ready_and_rest_are_inside_the_joint_limits(side):
    for pose in (poses.REST_DEG[side], poses.READY_DEG[side], *poses.WAYPOINTS_DEG[side]):
        for name, angle in zip(drv.JOINTS, pose):
            low, high = drv.LIMITS_DEG[side][name]
            assert low <= angle <= high, (side, name, angle)


def test_arm_goes_to_ready_at_start_when_it_rests_and_home_is_ready(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    arm, bus = make_arm(drv.Arm, logical=as_logical(poses.REST_DEG["left"]), follow=True,
                        home=poses.ready_rad("left"))
    assert joints_of(bus) == pytest.approx(poses.READY_DEG["left"], abs=0.2)
    assert bus.native["gripper"] - OFFSETS["gripper"] == pytest.approx(LOGICAL["gripper"], abs=0.05)  # gripper untouched
    assert all(bus.enabled.values())


def test_arm_stays_put_when_home_is_not_the_ready_pose(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    arm, bus = make_arm(drv.Arm, logical=as_logical(poses.REST_DEG["left"]), follow=True, home=[0.0] * 7)
    assert joints_of(bus) == pytest.approx(poses.REST_DEG["left"], abs=0.05)


def test_arm_stays_put_when_it_does_not_start_at_rest(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    far = {**as_logical(poses.REST_DEG["left"]), "joint_1": 40.0}
    arm, bus = make_arm(drv.Arm, logical=far, follow=True, home=poses.ready_rad("left"))
    assert bus.native["joint_1"] - OFFSETS["joint_1"] == pytest.approx(40.0, abs=0.05)


def test_ready_move_stops_when_a_joint_lags(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    with pytest.raises(errors.ArmError, match="behind"):  # joint_4 never moves although it is told to
        make_arm(drv.Arm, logical=as_logical(poses.REST_DEG["left"]), follow=True, home=poses.ready_rad("left"),
                 frozen=("joint_4",))


def test_torque_off_parks_at_rest_along_the_path_then_switches_off(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    arm, bus = make_arm(drv.Arm, logical=as_logical(poses.READY_DEG["left"]), follow=True)
    arm.torque_off()
    assert joints_of(bus) == pytest.approx(poses.REST_DEG["left"], abs=0.2)
    assert not any(bus.enabled.values()) and bus.closed


def test_torque_off_away_from_the_path_only_switches_off(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    far = {**as_logical(poses.REST_DEG["left"]), "joint_1": 40.0, "joint_4": 60.0}
    arm, bus = make_arm(drv.Arm, logical=far, follow=True)
    arm.torque_off()
    assert bus.native["joint_1"] - OFFSETS["joint_1"] == pytest.approx(40.0, abs=0.05)  # no parking move
    assert not any(bus.enabled.values())


def test_torque_off_at_rest_does_not_move(make_arm, monkeypatch):
    monkeypatch.setattr(drv, "CONTROL_PERIOD_S", 0.0)
    arm, bus = make_arm(drv.Arm, logical=as_logical(poses.REST_DEG["left"]), follow=True)
    frames = len(bus.sent)
    arm.torque_off()
    assert joints_of(bus) == pytest.approx(poses.REST_DEG["left"], abs=0.05)
    assert not any(bus.enabled.values())


# -- leader arms (read-only, for defining the work area) ------------------------------------------------------------------

def test_leader_reader_uses_the_leader_calibration_and_link(tmp_path, monkeypatch):
    leader_offsets = {n: o + 10.0 for n, o in OFFSETS.items()}
    for side in ("left", "right"):
        (tmp_path / f"leader_{side}.json").write_text(json.dumps({n: {"homing_offset": o} for n, o in leader_offsets.items()}))
    monkeypatch.setenv("OPENARM_LEADER_CALIBRATION_DIR", str(tmp_path))
    bus = FakeBus({n: LOGICAL[n] + leader_offsets[n] for n in drv.MOTORS})  # a leader sitting at LOGICAL by its own calibration

    def opener(channel):
        bus.opened_channel = channel
        return bus

    monkeypatch.setattr(drv.ReadOnlyLeader, "open_can_bus", staticmethod(opener))
    cfg = types.SimpleNamespace(can_channel="left", gripper_closed_deg=0.0, gripper_open_deg=-65.0)
    leader = drv.ReadOnlyLeader(cfg, "left")
    assert bus.opened_channel == "can_lead_l"
    assert np.degrees(leader.read().positions_rad) == pytest.approx([LOGICAL[n] for n in drv.JOINTS], abs=0.05)
    with pytest.raises(errors.ArmError):
        leader.write(np.zeros(7), 0.5)
    assert all(arb == damiao.PARAM_ID for arb, _ in bus.sent)  # only state queries


def test_leader_reader_refuses_when_its_calibration_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENARM_LEADER_CALIBRATION_DIR", str(tmp_path / "nowhere"))
    cfg = types.SimpleNamespace(can_channel="left", gripper_closed_deg=0.0, gripper_open_deg=-65.0)
    with pytest.raises(errors.ArmError, match="calibration"):
        drv.ReadOnlyLeader(cfg, "left")

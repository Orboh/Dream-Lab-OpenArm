"""Enactic OpenArm as a Dream robot of your own: one arm, seven joints and a gripper, Damiao motors on SocketCAN.

Two classes, as Dream's examples have:

- #ReadOnlyArm opens the CAN link and only asks the motors for their state. It switches no motor on, so the arm is
  limp, and its write() refuses. Use it for the dry run and for checking the angles it reads.
- #Arm switches each motor on holding the pose it reads, then follows the targets Dream sends.

Neither class uses the LeRobot fork's connect() or disconnect(): the fork's connect() runs a startup alignment that
moves both arms. This driver talks to the motors through Dream's own Damiao bus (`robots.damiao`), with one change:
OpenArm runs CAN-FD (the fork's `use_can_fd` defaults to True, links at 1 Mbit / 5 Mbit), so #FdBus opens the link with
`fd=True` and sends FD frames, where Dream's bus sends classic ones. The CAN links must be configured for FD
(`sudo bash scripts/canup.sh` in openarm-orboh does it); a link brought up as classic CAN will not work.

Angles. The motors report degrees in their own frame; the fork's calibration file for the arm
(`follower_left.json` / `follower_right.json`, the `homing_offset` of each joint) turns them into the arm's logical
angle, whose zero is the arm hanging down. The driver reads those offsets and refuses to open without them, since
without them every angle would be wrong. Logical degrees then become radians in Dream's arm model through MODEL_SIGN
and MODEL_ZERO_RAD below.

NOT VERIFIED: MODEL_SIGN and MODEL_ZERO_RAD are the identity, a starting point. Whether Dream's `openarm2_faas` model
counts every joint the same way as the fork's logical angle has not been checked on the arm. Check it with
#ReadOnlyArm before using #Arm: move each joint by hand (the arm is limp) and watch that the radians Dream logs
move the way the joint does, and that a known pose reads as the arm model says.

No gravity feed-forward: held at the fork's gains the arm sags a little, and with the motors off it falls, so
`drops_when_limp` is True: support the arm before disengage(). torque_off() first parks a raised arm at rest along the path.

@see ReadOnlyArm, Arm
"""

import json
import logging
import math
import os
import pathlib
import threading
import time
from typing import Any

import numpy as np
import numpy.typing as npt

import openarm_guard
import openarm_motion as motion
import openarm_lock
import openarm_poses as poses
import openarm_zones as zones_module
from robots import config, damiao, errors, followers, robstride

_log = logging.getLogger(__name__)

## The fork's calibration files: `follower_<side>.json`, a `homing_offset` in degrees per joint.
CALIBRATION_DIR = pathlib.Path(
    os.environ.get("OPENARM_CALIBRATION_DIR", "~/.cache/huggingface/lerobot/calibration/robots/openarm_follower")
).expanduser()
CALIBRATION_IDS = {"left": "follower_left", "right": "follower_right"}

JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
GRIPPER = "gripper"
MOTORS = JOINTS + (GRIPPER,)
## CAN ids 1-8; motor `i` replies from 0x10 + i, which is what Dream's bus expects.
CAN_IDS = dict(zip(MOTORS, range(1, 9), strict=True))
## The fork's motor table: DM8009 on the shoulders, DM4340 on joints 3-4, DM4310 on the wrist and the gripper.
MODELS = {
    "joint_1": "openarm_dm8009",
    "joint_2": "openarm_dm8009",
    "joint_3": "openarm_dm4340",
    "joint_4": "openarm_dm4340",
    "joint_5": "openarm_dm4310",
    "joint_6": "openarm_dm4310",
    "joint_7": "openarm_dm4310",
    "gripper": "openarm_dm4310",
}
## `(PMAX rad, VMAX rad/s, TMAX N.m)` per model, from the fork's `motors/damiao/tables.py`. Dream's table has only the
## MakerMods Metal models, so these are added to it when this file is imported.
damiao.MOTOR_LIMITS.update(
    {
        "openarm_dm8009": (12.5, 45.0, 54.0),
        "openarm_dm4340": (12.5, 8.0, 28.0),
        "openarm_dm4310": (12.5, 30.0, 10.0),
    }
)

## MIT gains per motor, the fork's `position_kp` / `position_kd`.
## The gripper's gain is the lowered value of 2026-09-21 (kp 25 held a squeeze long enough to latch the coil over-temperature protection).
KP = dict(zip(MOTORS, (240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 12.0), strict=True))
KD = dict(zip(MOTORS, (3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2), strict=True))

## Logical joint limits, degrees, the fork's per-side defaults (the arm's physical range).
LIMITS_DEG = {
    "left": {
        "joint_1": (-200.0, 80.0),
        "joint_2": (-190.0, 10.0),
        "joint_3": (-90.0, 90.0),
        "joint_4": (0.0, 140.0),
        "joint_5": (-90.0, 90.0),
        "joint_6": (-45.0, 45.0),
        "joint_7": (-90.0, 90.0),
        "gripper": (-65.0, 0.0),
    },
    "right": {
        "joint_1": (-80.0, 200.0),
        "joint_2": (-10.0, 190.0),
        "joint_3": (-90.0, 90.0),
        "joint_4": (0.0, 140.0),
        "joint_5": (-90.0, 90.0),
        "joint_6": (-45.0, 45.0),
        "joint_7": (-90.0, 90.0),
        "gripper": (-65.0, 0.0),
    },
}
## A reading this far past a limit, degrees, means a wrong zero or a stale frame (the fork's start tolerance).
LIMIT_SLACK_DEG = 30.0
## write() refuses a joint target this far from where the arm is, degrees: the loop rate-caps its targets, so a
## jump this big means a wrong angle mapping, not a command.
MAX_STEP_DEG = 20.0
## The gripper's logical degrees closed and open when setup has not measured them yet (the fork's range; open is
## negative). Used for reading only: write() refuses until `gripper_closed_deg` and `gripper_open_deg` are set.
FALLBACK_GRIPPER_DEG = (0.0, -65.0)

## Joint i's model angle = MODEL_SIGN[i] * logical radians + MODEL_ZERO_RAD[i]. Identity until checked on the arm.
MODEL_SIGN = np.ones(len(JOINTS))
MODEL_ZERO_RAD = np.zeros(len(JOINTS))

## Seconds a frame may wait in the adapter's send queue (Dream's bus uses 0.1 too).
SEND_TIMEOUT_S = 0.1

## Short addresses the wizard may be given instead of a link name. It cannot tell the two channels of one adapter apart,
## so each follower can be typed as `left` or `right`; the real SocketCAN link is what gets opened.
LINK_ALIASES = {"left": "can_fol_l", "right": "can_fol_r"}

## Seconds between the frames of a path move (50 Hz) and the spacing its samples are cut at.
CONTROL_PERIOD_S = 0.02
SAMPLE_DT_S = 0.02
## Joint speed of the ready and park moves, degrees per second (the fork's own start alignment uses 40).
FOLLOW_MAX_DPS = 25.0
## A joint this far behind its path target, degrees, stops the move (stuck, blocked, or a wrong mapping).
FOLLOW_LAG_DEG = 8.0
## At start the arm goes to the ready pose only if it is within this of the rest pose, degrees.
REST_TOLERANCE_DEG = 12.0
## On torque_off the arm is parked only if it is within this of the rest-to-ready path, degrees.
PARK_TOLERANCE_DEG = 20.0
## The config's home counts as the ready pose within this, degrees.
READY_HOME_TOLERANCE_DEG = 2.0

## Smooth motion: the limits table (speed, acceleration, torque budget) and the monitors' thresholds. "Gentle" is the chosen style.
LIMITS = motion.build_limits(KP, MODELS)
CONTROL_PERIOD = 0.01   # the control thread's tick, seconds (100 Hz)
## A motor this hot, degrees C, is logged; at the stop temperature the arm holds and `alarm()` says why (so Dream ends the run); below RESUME it may move again.
TEMP_WARN_C, TEMP_STOP_C, TEMP_RESUME_C = 60.0, 70.0, 65.0
## A gripper that leads its measured angle by more than this, degrees, for this long, seconds, is blocked or squeezing: its target is relaxed and `alarm()` says so.
GRIPPER_STALL_DEG, GRIPPER_STALL_S = 4.0, 2.0
## Soft start: the gains start at this share and reach full over this many seconds.
SOFT_START_GAIN0, SOFT_START_S = 0.4, 0.5
## Soft limits: a target stays this far inside a joint's limits, degrees (the gripper's margin is smaller).
GRIPPER_LIMIT_MARGIN_DEG = 1.0

## Tries to confirm a motor switched off: leaving one energized is not acceptable.
DISABLE_TRIES = 5


def open_fd_bus(channel: str) -> Any:
    """Opens a SocketCAN link with CAN-FD enabled, so FD frames can be sent and received.

    @param channel The link name, e.g. `can_fol_l`; it must already be up in FD mode.
    @return The python-can bus.
    """
    return robstride.python_can().interface.Bus(channel=channel, interface="socketcan", fd=True)


class FdBus(damiao.DamiaoBus):
    """Dream's Damiao bus sending CAN-FD frames, as the fork does (`is_fd=True`, no bit-rate switch)."""

    def _send(self, arbitration_id: int, data: bytes) -> None:
        """Sends one standard-id FD frame."""
        message = robstride.python_can().Message(
            arbitration_id=arbitration_id, data=data, is_extended_id=False, is_fd=True
        )
        self._require().send(message, timeout=SEND_TIMEOUT_S)


def resolve_channel(channel: object, aliases: dict[str, str] | None = None) -> str:
    """The SocketCAN link behind an address: `left` / `right` (any case) become `can_fol_l` / `can_fol_r`.

    @param channel What the config's `can_channel` holds.
    @param aliases The short names and their links; #LINK_ALIASES (the followers) by default.
    @return The link name; anything that is not an alias is returned as written, without surrounding spaces.
    """
    text = str(channel).strip()
    return (aliases or LINK_ALIASES).get(text.lower(), text)


def side_of(cfg: config.ArmConfig, side: str, aliases: dict[str, str] | None = None) -> str:
    """The arm's side, from `side` or, while setup is still finding the arms, from the adapter's name.

    @param cfg The arm's config; `can_channel` ending in `_l` or `_r`, or `left` / `right`, names the side.
    @param side `"left"`, `"right"`, or the adapter's id while setup finds the arms.
    @return `"left"` or `"right"`.
    @throws ArmError The side is unknown and the channel does not say.
    """
    if side in LIMITS_DEG:
        return side
    channel = resolve_channel(cfg.can_channel, aliases)
    for suffix, name in (("_l", "left"), ("_r", "right")):
        if channel.endswith(suffix):
            return name
    raise errors.ArmError(
        f"cannot tell which arm {channel!r} is (side {side!r}): name the CAN link like can_fol_l / can_fol_r, or type left / right"
    )


def homing_offsets(side: str, path: pathlib.Path | None = None) -> dict[str, float]:
    """The fork's homing offsets for an arm, degrees per motor.

    @param side `"left"` or `"right"`.
    @param path The calibration file; the follower's by default.
    @return The offset of each of #MOTORS.
    @throws ArmError The calibration file is missing or lacks a motor.
    """
    path = path or CALIBRATION_DIR / f"{CALIBRATION_IDS[side]}.json"
    try:
        data = json.loads(path.read_text())
        return {name: float(data[name]["homing_offset"]) for name in MOTORS}
    except (OSError, KeyError, ValueError) as exc:
        raise errors.ArmError(
            f"arm {side}: no usable calibration at {path} ({exc!r}); the angles would be wrong without its homing "
            "offsets. Set OPENARM_CALIBRATION_DIR if it lives elsewhere"
        ) from exc


class ReadOnlyArm(followers.Driver):
    """One OpenArm arm, opened to be read: no motor is switched on and write() refuses. The dry run's driver."""

    ## No gravity compensation: with the motors off the arm falls.
    drops_when_limp = True

    ## How the CAN link is opened; tests put a fake one here.
    open_can_bus = staticmethod(open_fd_bus)
    ## The short addresses (`left` / `right`) and the links they stand for.
    link_aliases = LINK_ALIASES

    def _load_offsets(self) -> dict[str, float]:
        """This arm's homing offsets (the follower's calibration file)."""
        return homing_offsets(self.side)

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Opens the arm's CAN link and asks nothing of the motors yet.

        @param cfg `can_channel` is the SocketCAN link (`can_fol_l`, `can_fol_r`); `gripper_closed_deg` and
            `gripper_open_deg` are the gripper's logical degrees shut and open once setup has measured them.
        @param side `"left"` or `"right"`.
        @throws ArmError The side is unknown, or its calibration is missing.
        """
        self.side = side_of(cfg, side, self.link_aliases)
        self._offsets = self._load_offsets()
        self._ends = (float(cfg.gripper_closed_deg), float(cfg.gripper_open_deg))
        self._deg: dict[str, float] | None = None
        self._limp = False
        motors = [damiao.Motor(name=n, can_id=CAN_IDS[n], model=MODELS[n]) for n in MOTORS]
        self.bus = FdBus(resolve_channel(cfg.can_channel, self.link_aliases), motors, opener=self.open_can_bus, clock=time.monotonic)
        self.bus.open()

    # -- reading ------------------------------------------------------------------------------------------------

    def _gripper_ends(self) -> tuple[float, float]:
        """The gripper's logical degrees shut and open: measured by setup, else the fork's range."""
        return self._ends if not any(math.isnan(e) for e in self._ends) else FALLBACK_GRIPPER_DEG

    def _logical(self) -> dict[str, float]:
        """Asks every motor for its state and returns each logical angle, degrees; refuses a fault or a bad zero."""
        native = self.bus.refresh_all(list(MOTORS))
        logical = {n: native[n] - self._offsets[n] for n in MOTORS}
        for name in MOTORS:
            state = self.bus.motors[name].state
            if state is not None and state.status in damiao.FAULT_STATUSES:
                raise errors.ArmError(f"arm {self.side} {name}: {damiao.FAULT_TEXT.get(state.status, state.status)}")
            low, high = LIMITS_DEG[self.side][name]
            if not low - LIMIT_SLACK_DEG <= logical[name] <= high + LIMIT_SLACK_DEG:
                raise errors.ArmError(
                    f"arm {self.side} {name} reads {logical[name]:.1f} deg, outside {low}..{high}: a wrong zero or a "
                    "wrong calibration file; nothing was sent"
                )
        self._deg = logical
        return logical

    def read(self) -> followers.Reading:
        """The joints in radians in Dream's arm model, the gripper from 0 shut to 1 open.

        @return The #robots.followers.Reading.
        @throws ArmError A motor is silent or faulted, or a joint reads outside its limits.
        """
        deg = self._logical()
        states = [self.bus.motors[n].state for n in MOTORS]
        positions = MODEL_SIGN * np.radians([deg[n] for n in JOINTS]) + MODEL_ZERO_RAD
        velocities = MODEL_SIGN * np.radians([states[i].velocity_deg_s for i in range(len(JOINTS))])
        closed, opened = self._gripper_ends()
        grip = float(np.clip((deg[GRIPPER] - closed) / (opened - closed), 0.0, 1.0))
        temps = np.array([max(s.temp_mos_c, s.temp_rotor_c) for s in states], dtype=np.float64)
        return followers.Reading(positions, velocities, grip, motor_temps_c=temps)

    def gripper_deg(self) -> float | None:
        """The gripper's logical angle as of the last read(), degrees, for setup to record shut and open by hand.

        @return Degrees, or None before the first read.
        """
        return None if self._deg is None else self._deg[GRIPPER]

    # -- writing ------------------------------------------------------------------------------------------------

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Refuses: this arm was opened read only.

        @param positions_rad Unused.
        @param gripper Unused.
        @throws ArmError Always.
        """
        raise errors.ArmError(f"arm {self.side} was opened read only (ReadOnlyArm): it takes no target")

    def disengage(self) -> bool:
        """Nothing to switch off.

        @return True.
        """
        return True

    def close(self) -> None:
        """Closes the CAN link and sends nothing."""
        self.bus.close()

    def torque_off(self) -> None:
        """Nothing is on; closes the CAN link."""
        self.bus.close()


## The leader (teleoperation) arms: same motors and ids, their own links and calibration.
LEADER_LINK_ALIASES = {"left": "can_lead_l", "right": "can_lead_r"}
LEADER_CALIBRATION_IDS = {"left": "leader_left", "right": "leader_right"}


class ReadOnlyLeader(ReadOnlyArm):
    """A leader arm opened only to be read, by hand: used to define the work area. Never switches a motor on; write() refuses."""

    link_aliases = LEADER_LINK_ALIASES

    def _load_offsets(self) -> dict[str, float]:
        """The leader's homing offsets (`leader_<side>.json` in the fork's teleoperator calibration folder)."""
        directory = pathlib.Path(
            os.environ.get(
                "OPENARM_LEADER_CALIBRATION_DIR",
                "~/.cache/huggingface/lerobot/calibration/teleoperators/openarm_leader",
            )
        ).expanduser()
        return homing_offsets(self.side, directory / f"{LEADER_CALIBRATION_IDS[self.side]}.json")


class Arm(ReadOnlyArm):
    """One OpenArm arm, switched on and holding where it is, then following Dream's targets."""

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Opens the link, reads the arm, then switches each motor on holding the pose it read.

        @param cfg As #ReadOnlyArm.
        @param side `"left"` or `"right"`.
        @throws ArmError A motor is silent, faulted, or reads outside its limits; no motor is switched on then. Or a
            motor does not confirm switching on, and those already on are switched off again.
        """
        super().__init__(cfg, side)
        self._guard = openarm_guard.Guard(self.side)
        self._lock = openarm_lock.ArmLock(self.side)
        self._held: dict[str, float] | None = None  # the pose a locked arm holds
        home = getattr(cfg, "home", None)
        self._home_deg = None if home is None else np.degrees(MODEL_SIGN * (np.asarray(home, dtype=float) - MODEL_ZERO_RAD))
        try:
            self._logical()  # refuses a bad reading before any motor is on
            self._switch_on()
            if self._should_go_ready():
                self.go_ready()
        except BaseException:
            self.bus.close()
            raise

    def _native_of(self, name: str, logical_deg: float) -> float:
        """A logical angle in the motor's own frame, degrees."""
        return logical_deg + self._offsets[name]

    def _hold(self, name: str, logical_deg: float) -> None:
        """One MIT frame holding `name` at `logical_deg` with the fork's gains."""
        self.bus.mit(name, KP[name], KD[name], self._native_of(name, logical_deg), 0.0, 0.0)

    def _switch_on(self) -> None:
        """Enables each motor and sends it, at once, the angle it reads, so its first target is where it is."""
        assert self._deg is not None
        enabled: list[str] = []
        try:
            for name in MOTORS:
                state = self.bus.command(name, damiao.CMD_ENABLE)
                enabled.append(name)
                self._hold(name, self._deg[name])
                state = self.bus.motors[name].state or state
                if state is None or state.status != damiao.STATUS_ENABLED:
                    raise errors.ArmError(f"arm {self.side} {name} did not confirm switching on")
        except BaseException:
            self._disable(enabled)
            raise
        self._limp = False

    def _disable(self, names: list[str]) -> None:
        """Switches `names` off, retrying until each confirms; raises if one stays on.

        @throws ArmError A motor still reports energized after #DISABLE_TRIES tries.
        """
        stuck = []
        for name in names:
            for _ in range(DISABLE_TRIES):
                # The reply to the disable frame can still carry the status from before it: confirm with a fresh query.
                self.bus.command(name, damiao.CMD_DISABLE)
                try:
                    if self.bus.read(name).status == damiao.STATUS_DISABLED:
                        break
                except damiao.MotorSilent:
                    pass
            else:
                stuck.append(name)
        if stuck:
            raise errors.ArmError(f"arm {self.side}: {stuck} did not confirm switching off: cut the arm's power")

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Sends the joints and the gripper as one batch of MIT frames.

        The first target after disengage() switches the arm back on holding where the hand left it and is dropped.

        @param positions_rad Seven joint targets in Dream's arm model, radians.
        @param gripper 0 shut to 1 open.
        @throws ArmError The gripper's range was not measured, a target is far from where the arm is, or a motor is
            faulted.
        """
        closed, opened = self._ends
        if math.isnan(closed) or math.isnan(opened):
            raise errors.ArmError(
                f"arm {self.side}: the gripper's shut and open angles are not measured; run the gripper step of "
                "`dreamcli setup` (they become gripper_closed_deg and gripper_open_deg)"
            )
        if self._limp:
            self._logical()
            self._switch_on()
            return  # the arm now holds where the hand left it; this target is dropped
        if not self._lock.is_active():
            self._hold_locked()
            return  # this arm is locked (arms.py): it keeps its pose whatever it is asked
        self._held = None
        self._guard.check(positions_rad)
        here = self._deg if self._deg is not None else self._logical()
        logical = np.degrees(MODEL_SIGN * (np.asarray(positions_rad, dtype=np.float64) - MODEL_ZERO_RAD))
        targets = dict(zip(JOINTS, logical.tolist(), strict=True))
        targets[GRIPPER] = closed + float(np.clip(gripper, 0.0, 1.0)) * (opened - closed)
        commands: dict[str, tuple[float, float, float, float, float]] = {}
        for name in MOTORS:
            low, high = LIMITS_DEG[self.side][name]
            target = min(max(targets[name], low), high)
            if abs(target - here[name]) > MAX_STEP_DEG:
                raise errors.ArmError(
                    f"arm {self.side} {name}: target {target:.1f} deg is {abs(target - here[name]):.1f} deg from the "
                    f"arm ({here[name]:.1f} deg); nothing was sent. A wrong angle mapping looks like this"
                )
            commands[name] = (KP[name], KD[name], self._native_of(name, target), 0.0, 0.0)
        self.bus.mit_batch(commands)
        for name in MOTORS:  # the replies carry where each motor is now
            state = self.bus.motors[name].state
            if state is not None:
                here[name] = state.position_deg - self._offsets[name]

    def _hold_locked(self) -> None:
        """Keeps a locked arm where it was when the lock first showed, motors alive."""
        if self._held is None:
            self._held = dict(self._logical())
        self.bus.mit_batch({n: (KP[n], KD[n], self._native_of(n, self._held[n]), 0.0, 0.0) for n in MOTORS})

    def disengage(self) -> bool:
        """Switches every motor off where it is, so a hand can move the arm; the arm falls unless supported.

        @return True once every motor confirmed.
        @throws ArmError A motor did not confirm switching off.
        """
        self._disable(list(MOTORS))
        self._limp = True
        return True

    def close(self) -> None:
        """Closes the CAN link without a move; motors keep their last target only while their own CAN timeout allows."""
        self.bus.close()

    # -- ready and rest poses -----------------------------------------------------------------------------------

    def _joint_pose(self) -> list[float]:
        """The seven joints as last read, logical degrees."""
        return [self._deg[n] for n in JOINTS]

    def _taught_path(self) -> list[list[float]] | None:
        """The recorded rest-to-home path in the zones in force (degrees), when it ends at the config's home."""
        if self._home_deg is None or len(self._home_deg) != len(JOINTS):
            return None
        try:
            taught = zones_module.load_zones(openarm_guard.zones_path()).home_path
        except (OSError, ValueError):
            return None
        path = (taught or {}).get(self.side)
        if path and float(np.max(np.abs(self._home_deg - np.asarray(path[-1])))) <= READY_HOME_TOLERANCE_DEG:
            return path
        return None

    def _home_path(self) -> list[list[float]] | None:
        """The path from rest to the config's home: the taught one, else the built-in ready path when home is the ready pose, else None."""
        taught = self._taught_path()
        if taught is not None:
            return taught
        if self._home_deg is not None and len(self._home_deg) == len(JOINTS) and \
                float(np.max(np.abs(self._home_deg - np.asarray(poses.READY_DEG[self.side])))) <= READY_HOME_TOLERANCE_DEG:
            return poses.path_to_ready(self.side)
        return None

    def _should_go_ready(self) -> bool:
        """True when a path to the config's home is known and the arm sits at rest: then the station start raises it along that path."""
        path = self._home_path()
        if path is None:
            return False
        index, distance = poses.nearest(path, self._joint_pose())
        return index == 0 and distance <= REST_TOLERANCE_DEG

    def _follow(self, points: list[list[float]]) -> None:
        """Moves the joints along `points` (degrees, the first is where the arm is) at #FOLLOW_MAX_DPS, the gripper held.

        @throws ArmError A motor leaves the enabled state, or a joint lags its target by more than #FOLLOW_LAG_DEG.
        """
        grip = self._deg[GRIPPER]
        for sample in poses.densify(points, FOLLOW_MAX_DPS, SAMPLE_DT_S):
            targets = {}
            for name, angle in zip(JOINTS, sample, strict=True):
                low, high = LIMITS_DEG[self.side][name]
                targets[name] = min(max(angle, low), high)
            targets[GRIPPER] = grip
            self.bus.mit_batch({n: (KP[n], KD[n], self._native_of(n, a), 0.0, 0.0) for n, a in targets.items()})
            for name in MOTORS:
                state = self.bus.motors[name].state
                if state.status != damiao.STATUS_ENABLED:
                    raise errors.ArmError(f"arm {self.side} {name} left the enabled state (status {state.status:#x})")
                behind = abs(state.position_deg - self._offsets[name] - targets[name])
                if name != GRIPPER and behind > FOLLOW_LAG_DEG:
                    raise errors.ArmError(
                        f"arm {self.side} {name} is {behind:.1f} deg behind its target: stopped (blocked, or a wrong "
                        "angle mapping)"
                    )
            time.sleep(CONTROL_PERIOD_S)
        self._logical()

    def go_ready(self) -> None:
        """Raises the arm from rest to the ready pose along the recorded path: back first, then up.

        @throws ArmError As #_follow.
        """
        _log.info("arm %s: rest -> ready along %d waypoints", self.side, len(poses.WAYPOINTS_DEG[self.side]))
        path = self._home_path() or poses.path_to_ready(self.side)
        self._follow([self._joint_pose(), *path[1:]])

    def _park(self) -> None:
        """Brings a raised arm back to the rest pose along the same path, so it does not fall when switched off.

        Does nothing when the arm is limp, already at rest, or too far from the path to follow it.
        """
        if self._limp or self._deg is None:
            return
        self._logical()
        path = self._home_path() or poses.path_to_ready(self.side)
        here = self._joint_pose()
        index, distance = poses.nearest(path, here)
        if distance > PARK_TOLERANCE_DEG:
            _log.warning("arm %s: %.0f deg from the rest-to-ready path, not parking", self.side, distance)
            return
        if index == 0:
            return
        _log.info("arm %s: parking at rest along the path before switching off", self.side)
        self._follow([here, *reversed(path[:index])])

    def torque_off(self) -> None:
        """Parks a raised arm at rest along the recorded path, then switches every motor off and closes the link.

        A failed park is logged and the motors are switched off where the arm is: it then falls unless supported.

        @throws ArmError A motor did not confirm switching off (the link is closed either way).
        """
        try:
            try:
                self._park()
            except Exception as exc:
                _log.warning("arm %s: parking failed (%s); switching off where it is", self.side, exc)
            self._disable(list(MOTORS))
        finally:
            self.bus.close()


def describe(arm: ReadOnlyArm) -> dict[str, Any]:
    """One read of `arm` as plain numbers, for a quick look: model radians, logical degrees and the gripper.

    @param arm An opened arm.
    @return `{"radians": [...], "gripper": float}`.
    """
    reading = arm.read()
    return {"radians": [round(float(q), 4) for q in reading.positions_rad], "gripper": round(reading.gripper, 3)}

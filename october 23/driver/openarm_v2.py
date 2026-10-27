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
without them every angle would be wrong. Logical degrees then become radians in Dream's arm model through the mapping of
`openarm_mapping` (`OPENARM_MODEL_MAPPING`: `identity` by default, `official` after the 2026-10-24 check, log-018).

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

import openarm_gripper as gripper
import openarm_guard
import openarm_mapping as mapping
import openarm_motion as motion
import openarm_pair as pair
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
## The gripper is the exception: Dream opens and closes it in big steps (a 30 deg jump between two targets ended the first live run on 10-24), and its angle is measured, so a jump is no sign of a wrong
## mapping. A target further than this from the last one is approached in steps of this size; the tracker (80 deg/s) sets the real speed.
GRIPPER_MAX_STEP_DEG = 20.0
## A raise or park move that plans longer than this is logged as a warning: `dreamcli robot down` gives up after 60 s and kills the station (10-24: the motors were left on).
LONG_MOVE_WARN_S = 45.0
## The gripper's logical degrees closed and open when setup has not measured them yet (the fork's range; open is
## negative). Used for reading only: write() refuses until `gripper_closed_deg` and `gripper_open_deg` are set.
FALLBACK_GRIPPER_DEG = (0.0, -65.0)

## The identity mapping, kept for reference: joint i's model angle = MODEL_SIGN[i] * logical radians + MODEL_ZERO_RAD[i]. The arms use `openarm_mapping.get(side)` (identity unless OPENARM_MODEL_MAPPING says otherwise).
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
## Within this of the rest-to-ready path, degrees, torque_off parks along the path; farther away it first goes back to the path along a checked line (log-040).
PARK_TOLERANCE_DEG = 20.0
## Far from its path with no clear way back, the arm lets go over this long, seconds, in at least this many steps (log-040).
RELEASE_S, RELEASE_STEPS = 2.0, 40
## The config's home counts as the ready pose within this, degrees.
READY_HOME_TOLERANCE_DEG = mapping.HOME_MATCH_DEG

## Smooth motion: the limits table (speed, acceleration, torque budget) and the monitors' thresholds. "Gentle" is the chosen style.
LIMITS = motion.build_limits(KP, MODELS)
CONTROL_PERIOD = 0.01   # the control thread's tick, seconds (100 Hz)
## A motor this hot, degrees C, is logged; at the stop temperature the arm holds and `alarm()` says why (so Dream ends the run); below RESUME it may move again.
TEMP_WARN_C, TEMP_STOP_C, TEMP_RESUME_C = 60.0, 70.0, 65.0
## A gripper that leads its measured angle by more than this, degrees, for this long, seconds, is blocked or squeezing (a normal grasp): it then holds with
## #GRIPPER_HOLD_LEAD_DEG instead of the full squeeze, which halves the coil current (the 2026-09-21 over-temperature), and the run goes on: no alarm
## (it used to let go and raise a sticky alarm that ended every grasp, log-028 run 2; log-031). Opening the gripper ends the hold; heat still stops the run.
GRIPPER_STALL_DEG, GRIPPER_STALL_S = 4.0, 2.0
GRIPPER_HOLD_LEAD_DEG = 4.0
## ...and it holds an object only if it also moved less than this in those seconds, degrees: a slow gripper still on its way is no grasp (log-043).
GRIPPER_STALL_MOVE_DEG = 1.0
## Soft start: the gains start at this share and reach full over this many seconds.
SOFT_START_GAIN0, SOFT_START_S = 0.4, 0.5
## Soft limits: a target stays this far inside a joint's limits, degrees (the gripper's margin is smaller).
SAFE_Q_FRESH_S = 0.5  # how long the last target sent serves as the starting point for cutting the next one back
## ...and after that, while the measured arm is still within this of it, degrees (an arm sags or lags a little below its target under gravity; log-037).
SAFE_Q_NEAR_DEG = 5.0
## A guard warning of one kind is logged at most once in this many seconds (log-040).
WARN_EVERY_S = 2.0
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
        self._sign, self._zero = mapping.get(self.side)  # logical -> Dream's model: sign * logical + zero
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
        positions = self._sign * np.radians([deg[n] for n in JOINTS]) + self._zero
        velocities = self._sign * np.radians([states[i].velocity_deg_s for i in range(len(JOINTS))])
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


def targets_clamped_from(offsets: dict[str, float], commands: dict[str, tuple]) -> dict[str, float]:
    """The logical target (degrees) inside each command tuple `(kp, kd, native position, vel, torque)`."""
    return {name: commands[name][2] - offsets[name] for name in commands}


class Arm(ReadOnlyArm):
    """One OpenArm arm, switched on and holding where it is, then following Dream's targets."""

    ## Class defaults: smooth, motor-safe motion on, run by a control thread. A test arm can switch either off per arm (`cfg.smooth`, `cfg.sync`).
    smooth_default = True
    sync_default = False

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Opens the link, reads the arm, switches each motor on holding the pose it read (the gains ramp up softly), raises the arm to its home if it rests, then starts the control thread.

        @param cfg As #ReadOnlyArm; optional `smooth`, `sync` (no thread: every write ticks once, for tests), `sync_dt` (that tick's seconds).
        @param side `"left"` or `"right"`.
        @throws ArmError A motor is silent, faulted, or reads outside its limits; no motor is switched on then. Or a
            motor does not confirm switching on, and those already on are switched off again.
        """
        super().__init__(cfg, side)
        self._guard = openarm_guard.Guard(self.side)
        self._lock = openarm_lock.ArmLock(self.side)
        self._held: dict[str, float] | None = None  # the pose a locked arm holds
        self._safe_q: np.ndarray | None = None      # the last target the guard let through, Dream's model, radians
        self._safe_time = 0.0
        self._tally = motion.RequestTally()         # what Dream asks of this arm, for the log
        self._at_edge = False                       # the last target was cut back to the edge of the allowed space
        self._smooth = bool(getattr(cfg, "smooth", type(self).smooth_default))
        self._sync = bool(getattr(cfg, "sync", type(self).sync_default))
        self._sync_dt = float(getattr(cfg, "sync_dt", CONTROL_PERIOD))
        self._cl = threading.RLock()                       # guards the tracker and the target
        self._tracker: motion.SmoothTracker | None = None
        self._target: np.ndarray | None = None             # the latest target per motor, logical degrees, in MOTORS order
        self._alarms: dict[str, str] = {}                  # heat, fault, gripper
        self._hot = False
        self._stall_since: float | None = None
        self._last_request: np.ndarray | None = None   # Dream's previous request (logical degrees) and when, for the jump check (log-036)
        self._last_request_time = 0.0
        self._ok_opening: float | None = None   # the finger opening the guard last accepted, metres (log-042)
        self._grip_cap: float | None = None     # the gripper share it may not exceed while held at an edge, or None
        self._warn_last: dict[str, tuple[float, int]] = {}   # per kind of warning: when it was last logged and how many were held back since
        self._at_pair_edge = False   # True while a target is cut short by the other arm (one warning per stretch)
        self._stall_from = 0.0   # the gripper's angle when the current stall count began (log-043)
        self._grip_hold = 0.0   # +1 or -1 while the gripper holds an object (the way it squeezes), 0 otherwise
        self._soft_t0: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._deg_time = 0.0
        self._sim_t = 0.0                                  # simulated seconds of an inline arm (tests): the monitors' clock when there is no thread
        self._warned: dict[str, float] = {}
        home = getattr(cfg, "home", None)
        self._home_deg = None if home is None else np.degrees(self._sign * (np.asarray(home, dtype=float) - self._zero))
        try:
            self._logical()  # refuses a bad reading before any motor is on
            self._switch_on()
            if not self._lock.is_active():
                _log.info("arm %s: locked (arms.py), so it stays where it is: not raised to ready", self.side)
            elif self._should_go_ready():
                self.go_ready()
            if self._smooth and not self._sync:
                self._start_thread()
        except BaseException:
            self._stop_thread()
            self.bus.close()
            raise

    def _native_of(self, name: str, logical_deg: float) -> float:
        """A logical angle in the motor's own frame, degrees."""
        return logical_deg + self._offsets[name]

    def _hold(self, name: str, logical_deg: float, gain: float = 1.0) -> None:
        """One MIT frame holding `name` at `logical_deg` with the fork's gains (times `gain`)."""
        self.bus.mit(name, KP[name] * gain, KD[name], self._native_of(name, logical_deg), 0.0, 0.0)

    def _switch_on(self) -> None:
        """Enables each motor and sends it, at once, the angle it reads, so its first target is where it is; with smooth motion the gains then ramp up softly."""
        assert self._deg is not None
        enabled: list[str] = []
        first = SOFT_START_GAIN0 if self._smooth else 1.0
        try:
            for name in MOTORS:
                state = self.bus.command(name, damiao.CMD_ENABLE)
                enabled.append(name)
                self._hold(name, self._deg[name], first)
                state = self.bus.motors[name].state or state
                if state is None or state.status != damiao.STATUS_ENABLED:
                    raise errors.ArmError(f"arm {self.side} {name} did not confirm switching on")
            if self._smooth:
                self._soft_start()
        except BaseException:
            self._disable(enabled)
            raise
        self._limp = False

    def _soft_start(self, steps: int = 25) -> None:
        """Ramps every motor's gain from #SOFT_START_GAIN0 to full over #SOFT_START_S seconds, holding where the arm is."""
        for k in range(steps + 1):
            gain = SOFT_START_GAIN0 + (1.0 - SOFT_START_GAIN0) * k / steps
            self.bus.mit_batch({n: (KP[n] * gain, KD[n], self._native_of(n, self._deg[n]), 0.0, 0.0) for n in MOTORS})
            if CONTROL_PERIOD_S > 0:
                time.sleep(SOFT_START_S / steps)

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
        self._note_request(positions_rad)
        if self._limp:
            self._safe_q = None
            self._stop_thread()
            self._logical()
            self._switch_on()
            self._start_control()
            return  # the arm now holds where the hand left it; this target is dropped
        if not self._lock.is_active():
            if self._smooth:
                self._lock_target()
                return
            self._hold_locked()
            return  # this arm is locked (arms.py): it keeps its pose whatever it is asked
        self._held = None
        here = self._deg if self._deg is not None else self._logical()
        before = here
        if self._smooth and self._target is not None:
            before = dict(zip(MOTORS, self._target.tolist(), strict=True))   # the tracker keeps the arm behind Dream's targets: the step is from the last target
        # the wrong-mapping check looks at what Dream asked, before a safety cut shortens the step and hides it (log-035), and compares it with Dream's
        # previous request while that is fresh, not with the target sent: an arm held at an edge while Dream goes on is no jump (log-036, run 1)
        requested = np.degrees(self._sign * (np.asarray(positions_rad, dtype=np.float64) - self._zero))
        now = self._time()
        fresh = self._last_request is not None and now - self._last_request_time <= SAFE_Q_FRESH_S
        reference, what = (dict(zip(JOINTS, self._last_request.tolist(), strict=True)), "last request") if fresh else \
            (before, "last target" if before is not here else "arm")
        for name, value in zip(JOINTS, requested.tolist(), strict=True):
            low, high = LIMITS_DEG[self.side][name]
            self._refuse_a_jump(name, min(max(value, low), high), reference, what)
        self._last_request, self._last_request_time = requested.copy(), now
        positions_rad = self._limited(positions_rad, here, float(np.clip(gripper, 0.0, 1.0)))
        if self._grip_cap is not None:
            gripper = min(float(gripper), self._grip_cap)   # held at an edge: not wider than the guard accepted (log-042)
        logical = np.degrees(self._sign * (np.asarray(positions_rad, dtype=np.float64) - self._zero))
        targets = dict(zip(JOINTS, logical.tolist(), strict=True))
        targets[GRIPPER] = closed + float(np.clip(gripper, 0.0, 1.0)) * (opened - closed)
        commands: dict[str, tuple[float, float, float, float, float]] = {}
        for name in MOTORS:
            low, high = LIMITS_DEG[self.side][name]
            target = min(max(targets[name], low), high)
            if name == GRIPPER and abs(target - before[name]) > GRIPPER_MAX_STEP_DEG:
                self._note_gripper_step(target, before[name])
                target = before[name] + math.copysign(GRIPPER_MAX_STEP_DEG, target - before[name])
            elif not self._smooth:
                # without smoothing a target is sent as it is, so its step from the last one is checked too; with smoothing the tracker limits the speed,
                # and after a hold the first clear target may lie far from the held one (log-040; Dream's own steps are checked above, before any cut)
                self._refuse_a_jump(name, target, before, "last target" if before is not here else "arm")
            commands[name] = (KP[name], KD[name], self._native_of(name, target), 0.0, 0.0)
        if self._smooth:
            self._ensure_tracker()
            with self._cl:
                clamped = targets_clamped_from(self._offsets, commands)
                self._target[:] = [clamped[name] for name in MOTORS]
            if self._sync:
                self._tick(self._sync_dt)
            return
        self.bus.mit_batch(commands)
        for name in MOTORS:  # the replies carry where each motor is now
            state = self.bus.motors[name].state
            if state is not None:
                here[name] = state.position_deg - self._offsets[name]

    def _note_gripper_step(self, target: float, last: float) -> None:
        """Says (at most every 5 s) that a gripper target was approached in steps instead of at once."""
        now = time.monotonic()
        if now - getattr(self, "_grip_noted", 0.0) >= 5.0:
            self._grip_noted = now
            _log.info("arm %s: the gripper was asked %.0f deg in one step (%.1f to %.1f deg): approached in steps of %.0f deg, the tracker sets the speed", self.side,
                      abs(target - last), last, target, GRIPPER_MAX_STEP_DEG)

    def _refuse_a_jump(self, name: str, target: float, before: dict[str, float], what: str) -> None:
        """Refuses a joint target more than #MAX_STEP_DEG from `before` (`what`: the last request, the last target or the arm): a wrong angle mapping looks like this.

        @throws ArmError
        """
        if abs(target - before[name]) > MAX_STEP_DEG:
            raise errors.ArmError(
                f"arm {self.side} {name}: target {target:.1f} deg is {abs(target - before[name]):.1f} deg from the "
                f"{what} ({before[name]:.1f} deg); nothing was sent. A wrong angle mapping looks like this"
            )

    def _note_request(self, positions_rad: np.ndarray) -> None:
        """Logs what Dream asks of this arm (net move of the run, joint, travel), and says when the arm is locked and so holds still."""
        for kind, net, joint, path in self._tally.update(np.degrees(np.asarray(positions_rad, dtype=float)), self._time()):
            _log.info("arm %s: Dream %s: %.0f deg from where this run began (joint %d), %.0f deg of joint travel in all%s", self.side, kind, net,
                      joint + 1, path, "; this arm is LOCKED, so it holds still" if not self._lock.is_active() else "")

    def _opening_m(self, asked_share: float | None = None) -> float:
        """The fingers' travel the checks count, metres: the measured opening, or the one asked for if wider (log-040)."""
        closed, opened = self._gripper_ends()
        measured = (self._deg[GRIPPER] - closed) / (opened - closed) if self._deg is not None and opened != closed else 1.0
        share = max(float(measured), asked_share if asked_share is not None else 0.0)
        return float(np.clip(share, 0.0, 1.0)) * gripper.FINGER_OPEN_M

    def _warn_now(self, kind: str, now: float, message: str, *args) -> None:
        """Logs a warning of `kind` at most once every #WARN_EVERY_S, with how many were held back since (an arm going in and out of a hold, log-040)."""
        last, held = self._warn_last.get(kind, (-1e9, 0))
        if now - last >= WARN_EVERY_S:
            _log.warning(message + (f" ({held} more since the last one)" if held else ""), *args)
            self._warn_last[kind] = (now, 0)
        else:
            self._warn_last[kind] = (last, held + 1)

    def _limited(self, positions_rad: np.ndarray, here: dict[str, float], asked_gripper: float | None = None) -> np.ndarray:
        """Cuts a target back to the edge of the allowed space (the guard) instead of refusing it, and says so once per stretch outside.

        The reference is the last target sent a moment ago, else the measured pose (an arm already outside is still refused).
        """
        now = self._time()
        measured = self._zero + self._sign * np.radians([here[n] for n in JOINTS])
        near = self._safe_q is not None and float(np.max(np.abs(np.degrees(self._sign * (np.asarray(self._safe_q) - measured))))) <= SAFE_Q_NEAR_DEG
        if self._safe_q is not None and (now - self._safe_time <= SAFE_Q_FRESH_S or near):
            prev = self._safe_q   # also after a pause while the arm is still near it: an arm sagging just past an edge is held, not stopped (log-037)
        else:
            prev = measured
        opening = self._opening_m(asked_gripper)
        self._grip_cap = None
        try:
            limited, found = self._guard.limit(prev, positions_rad, opening)
        except errors.ArmError:
            # held at an edge, a wider gripper can put the held pose itself past it (log-042): keep the opening the guard accepted last, and the gripper
            # does not open further than that, while the joints stay held; if that opening is not clear either, the arm really is inside and it stops
            if self._ok_opening is None or self._ok_opening >= opening:
                raise
            opening = self._ok_opening
            limited, found = self._guard.limit(prev, positions_rad, opening)
            self._grip_cap = opening / gripper.FINGER_OPEN_M
            self._warn_now("grip", now, "arm %s: held at an edge, the gripper keeps its opening (it would reach past the edge if it opened further)", self.side)
        self._ok_opening = opening
        # the other arm (log-034): cut where the two arms would come within pair.MIN_GAP_M of each other
        kept, crossing = pair.cut(self.side, self._sign * (np.asarray(prev) - self._zero), self._sign * (np.asarray(limited) - self._zero), pair.others(self.side),
                                  opening_m=opening)
        if crossing is not None:
            limited = self._zero + self._sign * kept
            self._warn_now("pair", now, "arm %s: stopped where it keeps %.0f cm from the other arm (Dream asked for a pose with: %s); the target stays out of reach "
                           "while the other arm is there", self.side, pair.MIN_GAP_M * 100, crossing)   # the numbers are the request's, not the arm's (log-041)
        self._at_pair_edge = crossing is not None
        pair.publish(self.side, "target", self._sign * (np.asarray(limited) - self._zero), opening)
        self._safe_q, self._safe_time = limited, now
        if found:
            self._warn_now("edge", now, "arm %s: stopped at the edge of the allowed space (Dream asked for a pose with: %s); the target stays out of reach until it "
                           "comes back inside", self.side, found[0])   # the numbers are the request's, not the arm's (log-041)
        self._at_edge = bool(found)
        return limited

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
        self._stop_thread()
        self._disable(list(MOTORS))
        self._limp = True
        return True

    def close(self) -> None:
        """Stops the control thread and closes the CAN link without a move; motors keep their last target only while their own CAN timeout allows."""
        self._stop_thread()
        self.bus.close()

    # -- smooth, motor-safe motion ------------------------------------------------------------------------------

    def _logical(self) -> dict[str, float]:
        """Every motor's logical angle: from the control thread's latest replies when it runs (it owns the bus), else asked directly."""
        if self._thread is not None:
            if self._time() - self._deg_time > 0.5:
                raise errors.ArmError(f"arm {self.side}: no recent reply from the motors (the control thread is not getting answers)")
            return dict(self._deg)
        logical = super()._logical()
        self._deg_time = time.monotonic()
        return logical

    def _vector(self) -> np.ndarray:
        """The motors' measured logical angles as a vector in MOTORS order."""
        return np.array([self._deg[n] for n in MOTORS], dtype=float)

    def _ensure_tracker(self) -> None:
        if self._tracker is None:
            self._tracker = motion.SmoothTracker(LIMITS, self._vector())
            self._target = self._vector().copy()

    def _start_control(self) -> None:
        """(Re)starts smooth control from where the arm is now: a fresh tracker, the target here, and the thread unless this arm runs inline."""
        if not self._smooth:
            return
        self._tracker, self._target = None, None
        self._ensure_tracker()
        if not self._sync:
            self._start_thread()

    def _start_thread(self) -> None:
        """Starts the control thread: from now on it alone talks to the motors, and `write()` only sets the target."""
        if self._thread is not None:
            return
        self._ensure_tracker()
        self._sync = False
        self._stop.clear()
        self._deg_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"arm-{self.side}", daemon=True)
        self._thread.start()

    def _stop_thread(self) -> None:
        """Stops the control thread (it sends nothing after this); the arm keeps its last motor target."""
        thread, self._thread = self._thread, None
        if thread is not None:
            self._stop.set()
            thread.join(timeout=2.0)

    def _run(self) -> None:
        last = time.monotonic()
        next_tick = last
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                self._tick(min(max(now - last, 0.001), 0.05))
                last = now
                next_tick += CONTROL_PERIOD
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except Exception as exc:  # a dead thread must be loud
            self._alarms["thread"] = f"arm {self.side}: the control thread stopped ({exc!r}); the arm holds its last target"
            _log.exception("arm %s: control thread stopped", self.side)

    def _time(self) -> float:
        """The monitors' clock: real time with the thread, simulated seconds for an arm that ticks inline."""
        return self._sim_t if self._sync else time.monotonic()

    def _gain(self) -> float:
        """The soft-start gain share: below 1 for #SOFT_START_S after a restart of smooth control, else 1."""
        if self._soft_t0 is None:
            return 1.0
        share = (self._time() - self._soft_t0) / SOFT_START_S
        if share >= 1.0:
            self._soft_t0 = None
            return 1.0
        return SOFT_START_GAIN0 + (1.0 - SOFT_START_GAIN0) * share

    def _tick(self, dt: float) -> None:
        """One control step: the tracker moves toward the target, the torque budget and soft limits shape the setpoint, one MIT batch goes out, the replies are read and watched."""
        self._ensure_tracker()
        self._sim_t += dt
        with self._cl:
            target = self._tracker.x.copy() if self._hot else self._target.copy()
            setpoint = self._tracker.step(target, dt).copy()
        sent, over = self._send_setpoint(setpoint)
        if over.any():  # anti wind-up: a blocked joint's tracker does not run away from the arm
            with self._cl:
                self._tracker.x = np.where(over, sent, self._tracker.x)
                self._tracker.v = np.where(over, 0.0, self._tracker.v)

    def _send_setpoint(self, setpoint: np.ndarray, gain: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Sends one batch: the setpoint (logical degrees, MOTORS order) limited to the torque budget around the measured angle and kept inside the soft limits.

        @return `(sent, over)`: what was sent and which joints were held back by the torque budget.
        """
        measured = self._vector()
        lead = LIMITS.lead_deg
        if self._grip_hold:  # holding an object: the lighter grip
            lead = lead.copy()
            lead[-1] = GRIPPER_HOLD_LEAD_DEG
        over = np.abs(setpoint - measured) > lead
        sent = measured + np.clip(setpoint - measured, -lead, lead)
        for i, name in enumerate(MOTORS):
            low, high = LIMITS_DEG[self.side][name]
            margin = GRIPPER_LIMIT_MARGIN_DEG if name == GRIPPER else motion.SOFT_LIMIT_MARGIN_DEG
            sent[i] = min(max(sent[i], low + margin), high - margin)
        gain = self._gain() if gain is None else gain
        self.bus.mit_batch({n: (KP[n] * gain, KD[n], self._native_of(n, float(sent[i])), 0.0, 0.0) for i, n in enumerate(MOTORS)})
        self._read_back()
        return sent, over

    def _read_back(self) -> None:
        """Takes the motors' positions from the replies to the last batch and runs the monitors."""
        for name in MOTORS:
            state = self.bus.motors[name].state
            if state is not None:
                self._deg[name] = state.position_deg - self._offsets[name]
        self._deg_time = self._time()
        pair.publish(self.side, "measured", np.radians([self._deg[n] for n in JOINTS]), self._opening_m())
        self._monitor()

    def _monitor(self) -> None:
        """Faults, temperature and gripper stall: sets the alarms; a hot motor makes the arm hold."""
        now = self._time()
        faults, hottest = [], ("", 0.0)
        for name in MOTORS:
            state = self.bus.motors[name].state
            if state is None:
                continue
            if state.status in damiao.FAULT_STATUSES:
                faults.append(f"arm {self.side} {name}: {damiao.FAULT_TEXT.get(state.status, state.status)}")
            temp = max(state.temp_mos_c, state.temp_rotor_c)
            if temp > hottest[1]:
                hottest = (name, temp)
            if temp >= TEMP_WARN_C and now - self._warned.get(name, -1e9) > 30.0:
                self._warned[name] = now
                _log.warning("arm %s %s is at %.0f C (warns at %.0f, holds at %.0f)", self.side, name, temp, TEMP_WARN_C, TEMP_STOP_C)
        if faults:
            self._alarms["fault"] = faults[0]
        else:
            self._alarms.pop("fault", None)
        if hottest[1] >= TEMP_STOP_C:
            self._hot = True
            self._alarms["heat"] = f"arm {self.side} {hottest[0]} is at {hottest[1]:.0f} C (limit {TEMP_STOP_C:.0f}): the arm holds still; stop the run and let it cool"
        elif self._hot and hottest[1] < TEMP_RESUME_C:
            self._hot = False
            self._alarms.pop("heat", None)
        if self._target is not None and self._tracker is not None:
            lead = float(self._target[-1]) - self._deg[GRIPPER]   # signed: the way the gripper is told to move
            if self._grip_hold and (abs(lead) <= GRIPPER_HOLD_LEAD_DEG or np.sign(lead) != self._grip_hold):
                self._grip_hold = 0.0  # told to open (or the object is gone): the full lead again
                self._stall_since = None
            elif not self._grip_hold and abs(lead) > GRIPPER_STALL_DEG:
                if self._stall_since is None or abs(self._deg[GRIPPER] - self._stall_from) > GRIPPER_STALL_MOVE_DEG:
                    # still moving: a slow gripper on its way, not one held by an object (log-043); the 2 s count from here
                    self._stall_since, self._stall_from = now, self._deg[GRIPPER]
                elif now - self._stall_since > GRIPPER_STALL_S:
                    self._grip_hold = float(np.sign(lead))
                    _log.warning("arm %s gripper holds an object (it led its angle by %.0f deg for %g s): lighter grip, %g deg lead", self.side, abs(lead), GRIPPER_STALL_S, GRIPPER_HOLD_LEAD_DEG)
            elif not self._grip_hold:
                self._stall_since = None

    def _dead_reason(self) -> str | None:
        """Why the arm no longer answers (its control thread died, or the motors have been silent for over 0.5 s), or None while it does."""
        if "thread" in self._alarms:
            return self._alarms["thread"]
        if self._thread is not None and self._time() - self._deg_time > 0.5:
            return f"arm {self.side}: no reply from the motors for over 0.5 s (power or CAN cable?)"
        return None

    def is_alive(self) -> bool:
        """Whether the arm still answers; False tells the station to report it and offer a Reset (it may still hold under power: support it)."""
        return self._dead_reason() is None

    def why_stopped(self) -> str:
        """The reason #is_alive is False, or empty."""
        return self._dead_reason() or ""

    def alarm(self) -> str | None:
        """Why the run must stop although the arm answers (a hot motor, a fault, a dead control thread; a gripper holding an object is not one: it keeps a lighter grip), or None."""
        return next(iter(self._alarms.values()), None) if self._alarms else None

    def settle(self, max_s: float = 10.0, dt: float = CONTROL_PERIOD) -> bool:
        """Runs control ticks inline until the arm sits on its target (arms run without the thread: tests, offline moves).

        @return Whether it settled within `max_s` seconds of simulated time.
        """
        self._ensure_tracker()
        for _ in range(int(max_s / dt)):
            if self._tracker.settled(self._target) and not self._hot:
                return True
            self._tick(dt)
        return self._tracker.settled(self._target)

    def _lock_target(self) -> None:
        """A locked arm holds the pose it had when the lock first showed: its target becomes that pose."""
        self._ensure_tracker()
        if self._held is None:
            self._held = dict(self._logical())
        with self._cl:
            self._target[:] = [self._held[n] for n in MOTORS]
        if self._sync:
            self._tick(self._sync_dt)

    def _follow_smooth(self, points: list[list[float]], grip_deg: float | None = None) -> None:
        """Moves the joints along `points` (degrees, the first is where the arm is) on a smooth path within the limits, the gripper held (or moved to `grip_deg`).

        The control thread, if it runs, is stopped for the move and restarted afterwards from the end pose.

        @throws ArmError A motor leaves the enabled state, or a joint lags the path by more than #FOLLOW_LAG_DEG.
        """
        was_running = self._thread is not None
        self._stop_thread()
        grip = self._deg[GRIPPER] if grip_deg is None else grip_deg
        path = motion.smooth_path(np.array(points, dtype=float), LIMITS, SAMPLE_DT_S)
        peak_v = float(np.max(np.abs(np.diff(path.samples, axis=0)) / SAMPLE_DT_S)) if len(path.samples) > 1 else 0.0
        _log.info("arm %s: smooth move of %.1f s, peak joint speed %.0f deg/s", self.side, path.duration, peak_v)
        if path.duration > LONG_MOVE_WARN_S:
            _log.warning("arm %s: this move takes %.0f s; `dreamcli robot down` waits only 60 s for it, then kills the station and leaves the motors on: shorten the home path", self.side, path.duration)
        for sample in path.samples[1:]:
            sent, _ = self._send_setpoint(np.append(sample, grip), 1.0)
            for name in MOTORS:
                state = self.bus.motors[name].state
                if state.status != damiao.STATUS_ENABLED:
                    raise errors.ArmError(f"arm {self.side} {name} left the enabled state (status {state.status:#x})")
            for i, name in enumerate(JOINTS):
                behind = abs(self._deg[name] - float(sample[i]))
                if behind > FOLLOW_LAG_DEG:
                    raise errors.ArmError(f"arm {self.side} {name} is {behind:.1f} deg behind its path: stopped (blocked, or a wrong angle mapping)")
            time.sleep(CONTROL_PERIOD_S)
        self._logical() if self._thread is None else None
        self._start_control_after_move(was_running)

    def _start_control_after_move(self, was_running: bool) -> None:
        """Fresh smooth control from the end pose; the thread comes back if it was running."""
        self._tracker, self._target = None, None
        self._ensure_tracker()
        if was_running:
            self._start_thread()

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

    def _set_grip_target(self, grip_deg: float) -> None:
        """Makes `grip_deg` the gripper's target for the stall monitor too, and ends any hold: the driver's own move closing the gripper is no grasp (log-041)."""
        if self._target is not None:
            with self._cl:
                self._target[-1] = grip_deg
        self._grip_hold, self._stall_since = 0.0, None

    def _follow(self, points: list[list[float]], grip_deg: float | None = None) -> None:
        """Moves the joints along `points` (degrees, the first is where the arm is) at #FOLLOW_MAX_DPS, the gripper held (or moved to `grip_deg`).

        @throws ArmError A motor leaves the enabled state, or a joint lags its target by more than #FOLLOW_LAG_DEG.
        """
        if grip_deg is not None:
            self._set_grip_target(grip_deg)
        if self._smooth:
            self._follow_smooth(points, grip_deg)
            return
        grip = self._deg[GRIPPER] if grip_deg is None else grip_deg
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
        self._follow([self._joint_pose(), *path[1:]], self._gripper_ends()[0])   # the gripper closed: the preflight checks this path that way (log-040)

    def _park(self) -> None:
        """Brings a raised arm back to the rest pose along the same path, so it does not fall when switched off.

        Does nothing when the arm is limp or already at rest. Far from the path it first goes back to the nearest point a checked straight line reaches;
        with no clear way back it lets go slowly (#_release_slowly) instead of dropping (log-040).
        """
        if self._limp or self._deg is None:
            return
        self._logical()
        path = self._home_path() or poses.path_to_ready(self.side)
        here = self._joint_pose()
        index, distance = poses.nearest(path, here)
        if distance > PARK_TOLERANCE_DEG:
            # far from its path (log-040): Dream's contract for an arm that drops when limp is to park it first, so go back along a checked line
            back = self._way_back(path, here)
            if back is None:
                _log.warning("arm %s: %.0f deg from the rest-to-ready path and no clear way back to it: letting go slowly over %g s", self.side, distance, RELEASE_S)
                self._release_slowly()
                return
            _log.warning("arm %s: %.0f deg from the rest-to-ready path: back to its point %d along a checked straight line, then parking", self.side, distance, back)
            self._follow([here, *reversed(path[:back + 1])], self._gripper_ends()[0])
            return
        if index == 0:
            return
        _log.info("arm %s: parking at rest along the path before switching off", self.side)
        self._follow([here, *reversed(path[:index])], self._gripper_ends()[0])   # closed, as checked by the preflight (log-040)

    def _line_clear(self, here_deg: list[float], there_deg: list[float]) -> bool:
        """Whether the straight joint line from `here_deg` to `there_deg` (logical degrees) keeps clear of the zones, the body and the other arm."""
        here_model = self._zero + self._sign * np.radians(here_deg)
        there_model = self._zero + self._sign * np.radians(there_deg)
        try:
            if self._guard.way_problem(here_model, there_model, self._opening_m()) is not None:
                return False
        except errors.ArmError:   # no zones, or stale ones: no way back can be checked
            return False
        return pair.way_problem(self.side, np.radians(here_deg), np.radians(there_deg), pair.others(self.side), opening_m=self._opening_m()) is None

    def _way_back(self, path: list[list[float]], here: list[float]) -> int | None:
        """The index of the nearest point of `path` that a checked straight line reaches from `here`, or None."""
        order = sorted(range(len(path)), key=lambda k: float(np.max(np.abs(np.asarray(path[k]) - np.asarray(here)))))
        return next((k for k in order if self._line_clear(here, path[k])), None)

    def _release_slowly(self) -> None:
        """Lowers every motor's stiffness to nothing over #RELEASE_S while the damping stays, so the arm settles instead of falling; then the caller switches off."""
        self._stop_thread()
        steps = max(RELEASE_STEPS, int(RELEASE_S / CONTROL_PERIOD_S) if CONTROL_PERIOD_S > 0 else RELEASE_STEPS)
        for i in range(steps):
            share = 1.0 - (i + 1) / steps
            batch = {}
            for name in MOTORS:
                state = self.bus.motors[name].state
                where = state.position_deg - self._offsets[name] if state is not None else self._deg[name]
                batch[name] = (KP[name] * share, KD[name], self._native_of(name, where), 0.0, 0.0)
            self.bus.mit_batch(batch)
            time.sleep(CONTROL_PERIOD_S)

    def torque_off(self) -> None:
        """Parks a raised arm at rest along the recorded path, then switches every motor off and closes the link.

        A failed park is logged and the arm lets go slowly (#_release_slowly) before the motors switch off, rather than falling at once (log-042).
        Far from the path see #_park.

        @throws ArmError A motor did not confirm switching off (the link is closed either way).
        """
        try:
            try:
                self._park()
            except Exception as exc:
                _log.warning("arm %s: parking failed (%s); letting go slowly over %g s, then switching off", self.side, exc, RELEASE_S)
                try:
                    self._release_slowly()   # an arm that drops when limp must not fall where the park stopped (log-042)
                except Exception as release_exc:
                    _log.warning("arm %s: the slow release failed too (%s); switching off where it is", self.side, release_exc)
            self._stop_thread()   # the control thread must not share the bus with the switch-off
            self._disable(list(MOTORS))
        finally:
            self._stop_thread()
            self.bus.close()


class DryRunArm(ReadOnlyArm):
    """A dry-run arm: it reads like #ReadOnlyArm (state queries only, every motor stays off) and ACCEPTS Dream's targets without sending them anywhere.

    For each arm it logs what Dream asks (the net move of the run, as #Arm does), the target's hand position against the measured one, and what the guard would have done with it
    (cut it at a zone edge, or refuse it). The arm never moves, so Dream keeps seeing the same pictures and asks again from the same pose: the log shows its plan, not a motion.
    Select it with `custom.driver: .../openarm_v2.py:DryRunArm` in a copy of the station file; it needs no gripper range and sends no frame but the queries.
    """

    NOTE_EVERY_S = 1.0  # between target lines while Dream keeps asking

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        super().__init__(cfg, side)
        self._guard = openarm_guard.Guard(self.side)
        self._lock = openarm_lock.ArmLock(self.side)
        self._tally = motion.RequestTally()
        self._kin = zones_module.ArmKinematics(self.side)
        self._noted = 0.0
        self._cut = False

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Takes a target, records it and its guard verdict, and sends nothing.

        @param positions_rad Seven joint targets in Dream's arm model, radians.
        @param gripper 0 shut to 1 open.
        """
        q = np.asarray(positions_rad, dtype=np.float64)
        now = time.monotonic()
        locked = "" if self._lock.is_active() else "; this arm is LOCKED, so a live run would hold it still"
        for kind, net, joint, path in self._tally.update(np.degrees(q), now):
            _log.info("arm %s: DRY RUN, Dream %s: %.0f deg from where this run began (joint %d), %.0f deg of joint travel in all%s", self.side, kind, net, joint + 1, path, locked)
        verdict = self._verdict(q)
        cut = verdict is not None
        if now - self._noted >= self.NOTE_EVERY_S or cut != self._cut:
            self._noted, self._cut = now, cut
            here = self._deg if self._deg is not None else self._logical()
            measured = self._sign * np.radians([here[n] for n in JOINTS]) + self._zero
            want, have = self._kin.points(q)["tcp"], self._kin.points(measured)["tcp"]
            far = np.degrees(np.abs(self._sign * (q - self._zero) - np.radians([here[n] for n in JOINTS])))
            _log.info("arm %s: DRY RUN target: hand %s cm (measured %s cm), farthest joint %.0f deg from the measured pose (joint %d), gripper %.2f; guard: %s%s",
                      self.side, np.round(want * 100).astype(int).tolist(), np.round(have * 100).astype(int).tolist(), float(far.max()), int(far.argmax()) + 1, gripper,
                      "would stop it: " + verdict if cut else "allows it", locked)

    def _verdict(self, q: np.ndarray) -> str | None:
        """Why the guard would not let this target through as it is (None when it would)."""
        here = self._deg if self._deg is not None else self._logical()
        prev = self._zero + self._sign * np.radians([here[n] for n in JOINTS])
        try:
            _, found = self._guard.limit(prev, q)
        except errors.ArmError as exc:
            return f"refused ({exc})"
        return found[0] if found else None

    def disengage(self) -> bool:
        """Nothing is on. @return True."""
        return True


def describe(arm: ReadOnlyArm) -> dict[str, Any]:
    """One read of `arm` as plain numbers, for a quick look: model radians, logical degrees and the gripper.

    @param arm An opened arm.
    @return `{"radians": [...], "gripper": float}`.
    """
    reading = arm.read()
    return {"radians": [round(float(q), 4) for q in reading.positions_rad], "gripper": round(reading.gripper, 3)}

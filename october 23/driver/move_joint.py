"""Moves one joint of one follower arm by a few degrees and back, everything else held where it is.

  cd "october 23/driver"
  env -u PYTHONPATH -u AMENT_PREFIX_PATH ../../.venv/bin/python move_joint.py --side left --joint 6 --delta 10

The arm is switched on holding its own position (nothing jumps), the joint follows a smooth move out, holds, then
smoothly returns, and the arm is switched off again, so it ends limp: support it if it could fall. It aborts and switches
off if a motor leaves the enabled state, another joint drifts more than 3 deg, or the moving joint is more than 4 deg
from its target. `--dry-run` only reads the arm and prints the plan; no motor is switched on.
"""

import argparse
import math
import sys
import time
import types

import openarm_v2 as drv
from robots import damiao

CHANNELS = {"left": "can_fol_l", "right": "can_fol_r"}
PERIOD_S = 0.02
MAX_DELTA_DEG = 15.0
LIMIT_MARGIN_DEG = 10.0
DRIFT_LIMIT_DEG = 3.0
TRACKING_LIMIT_DEG = 4.0


def smooth(x: float) -> float:
    """Smoothstep, 0 to 1 with zero slope at both ends."""
    return x * x * (3 - 2 * x)


def run(arm: drv.Arm, joint: str, delta: float, move_s: float, hold_s: float) -> None:
    """Moves `joint` by `delta` degrees, holds, returns; every other motor holds its starting position."""
    base = dict(arm._deg)
    peak = 0.0
    for duration, start, end in ((move_s, 0, 1), (hold_s, 1, 1), (move_s, 1, 0), (1.0, 0, 0)):
        began = time.monotonic()
        while (elapsed := time.monotonic() - began) < duration:
            frac = start + (end - start) * smooth(min(elapsed / duration, 1.0))
            want = {n: base[n] + (delta * frac if n == joint else 0.0) for n in drv.MOTORS}
            arm.bus.mit_batch({n: (drv.KP[n], drv.KD[n], arm._native_of(n, want[n]), 0.0, 0.0) for n in drv.MOTORS})
            for n in drv.MOTORS:
                state = arm.bus.motors[n].state
                if state.status != damiao.STATUS_ENABLED:
                    raise RuntimeError(f"{n} left the enabled state (status {state.status:#x})")
                error = abs(state.position_deg - arm._offsets[n] - want[n])
                if n != joint and error > DRIFT_LIMIT_DEG:
                    raise RuntimeError(f"{n} drifted {error:.1f} deg while holding")
                if n == joint:
                    peak = max(peak, error)
                    if error > TRACKING_LIMIT_DEG:
                        raise RuntimeError(f"{n} is {error:.1f} deg from its target")
            time.sleep(PERIOD_S)
    print(f"done: {joint} {base[joint]:.2f} -> back to {arm.bus.motors[joint].state.position_deg - arm._offsets[joint]:.2f} deg; "
          f"worst tracking error {peak:.2f} deg")


def main() -> int:
    """Parses the arguments, checks the plan against the limits and runs it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--side", choices=CHANNELS, required=True)
    parser.add_argument("--joint", required=True, help="1-7 or joint_1..joint_7 (the gripper is not moved by this tool)")
    parser.add_argument("--delta", type=float, required=True, help=f"degrees out and back, at most {MAX_DELTA_DEG}")
    parser.add_argument("--move-s", type=float, default=3.0, help="seconds for each leg (default 3)")
    parser.add_argument("--hold-s", type=float, default=2.0, help="seconds held at the far end (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="read the arm and print the plan; switch nothing on")
    args = parser.parse_args()

    joint = args.joint if args.joint.startswith("joint_") else f"joint_{args.joint}"
    if joint not in drv.JOINTS:
        parser.error(f"unknown joint {args.joint!r}")
    if abs(args.delta) > MAX_DELTA_DEG:
        parser.error(f"--delta is limited to {MAX_DELTA_DEG} deg")
    if args.move_s < 1.0:
        parser.error("--move-s must be at least 1 second")

    cfg = types.SimpleNamespace(can_channel=CHANNELS[args.side], gripper_closed_deg=math.nan, gripper_open_deg=math.nan)
    arm = None
    try:
        arm = drv.ReadOnlyArm(cfg, args.side) if args.dry_run else drv.Arm(cfg, args.side)
        here = arm.bus.read(joint).position_deg - arm._offsets[joint] if args.dry_run else arm._deg[joint]
        low, high = drv.LIMITS_DEG[args.side][joint]
        far = here + args.delta
        if not low + LIMIT_MARGIN_DEG <= far <= high - LIMIT_MARGIN_DEG:
            raise RuntimeError(f"{joint} would go to {far:.1f} deg; limits {low}..{high} need {LIMIT_MARGIN_DEG} deg margin")
        print(f"{args.side} {joint}: {here:.2f} deg -> {far:.2f} deg and back "
              f"({args.move_s}s out, {args.hold_s}s hold, {args.move_s}s back)")
        if args.dry_run:
            print("dry run: nothing was switched on")
        else:
            run(arm, joint, args.delta, args.move_s, args.hold_s)
        return 0
    except Exception as exc:
        print("ABORTED:", exc)
        return 1
    finally:
        if arm is not None:
            try:
                arm.torque_off()
                print(f"{args.side} arm switched off" if not args.dry_run else "link closed")
            except Exception as exc:
                print("SWITCH-OFF PROBLEM:", exc, "- cut the arm's power if it is still stiff")


if __name__ == "__main__":
    sys.exit(main())

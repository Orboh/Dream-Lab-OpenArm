"""Moves each arm to the camera-view pose (the taught home) so the wrist cameras see the work table, holds it, then parks the arm and switches it off.

Run it with the station stopped and the space clear: the real arms move. Opening an arm with the driver already raises it from rest along the recorded
path (back first, then up); this command adds the report (how far the arm ended from the home), the hold for you to look at the camera views, and the
safe park. An arm that does not start at rest is not moved (it would have no path); hand-guide it to the rest pose first. An arm the one-arm lock does
not allow is skipped.
"""

import math
import pathlib
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import yaml

import openarm_lock
import openarm_mapping as mapping

SIDES = ("left", "right")
KEYS = {"left": "follower_l", "right": "follower_r"}
DONE_WITHIN_DEG = 6.0   # an arm ending within this of the home counts as there (the soft limit margin and the end of the path leave a few degrees)


def arm_config(station: pathlib.Path, side: str) -> types.SimpleNamespace:
    """The driver's settings for one arm, from the station file (CAN link, gripper range, home in radians)."""
    follower = yaml.safe_load(station.read_text())[KEYS[side]]
    return types.SimpleNamespace(can_channel=follower["can_channel"], gripper_closed_deg=follower["gripper_closed_deg"],
                                 gripper_open_deg=follower["gripper_open_deg"], home=list(follower["home"]))


def run(station: pathlib.Path, sides: list[str], hold: Callable[[], None], opener: Callable[[Any, str], Any], say: Callable[[str], None] = print) -> int:
    """Raises each arm to the home, reports, holds, and parks it.

    @param station The station file.
    @param sides The arms to move, in order.
    @param hold Waits while you look at the camera views (a timer, or the Enter key).
    @param opener Opens the driver's arm: `opener(cfg, side)`; the arm raises itself from rest while opening.
    @param say Where the report goes.
    @return 0 when every requested arm reached the home (or was skipped because it is locked), 1 otherwise.
    """
    code = 0
    for side in sides:
        if side not in openarm_lock.active_arms():
            say(f"{side}: locked (arms.py), so it is left where it is")
            continue
        cfg = arm_config(station, side)
        say(f"{side}: opening (it raises itself from rest to the home along the recorded path if it hangs at rest)")
        arm = opener(cfg, side)
        try:
            home_deg = np.degrees(mapping.to_logical(side, cfg.home))  # the station's home is in Dream's model angles; _joint_pose() is logical
            gap = float(np.abs(np.asarray(arm._joint_pose(), dtype=float) - home_deg).max())
            if gap <= DONE_WITHIN_DEG:
                say(f"{side}: at the camera-view pose (largest joint {gap:.1f} deg from the home): look at the camera views now")
                hold()
            else:
                code = 1
                say(f"{side}: NOT at the home (largest joint {gap:.0f} deg away): it did not start at rest, so it was not moved; hand-guide it to the rest pose and run again")
        finally:
            say(f"{side}: parking at rest along the path, then switching off")
            arm.torque_off()
            say(f"{side}: switched off")
    say("done" if code == 0 else "stopped with a problem (see above)")
    return code

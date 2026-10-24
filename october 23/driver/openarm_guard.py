"""The driver's hard guard: refuses a joint target whose arm would enter a danger zone, or any target on stale or missing zones.

The painted work area (`reach`) is not enforced here: Dream clips its hand targets to it, and the station's Home and Stop moves cross it.

Reads the zones file (`~/.dreamlabs/zones.yaml`, or `OPENARM_ZONES`). Settings by environment variable:
- `OPENARM_ZONES_REQUIRED`: `0` turns the guard off (tests and bench work); anything else keeps it on (default on).
- `OPENARM_ZONES_MAX_AGE_H`: hours a zones file stays valid, default 24 (the zones are captured every day).
A refusal raises `ArmError`; the station then stops the run and the arms hold.
`limit()` is what `Arm.write` uses: a target past the edge of the allowed space is cut back to the edge (the arm stops there, the run goes on); stale zones and an arm already outside still raise.
"""

import datetime as dt
import os
import pathlib
from collections.abc import Callable

import numpy as np

import openarm_zones as zones
from robots import errors

DEFAULT_MAX_AGE_H = 24.0
MEASURED_SLACK_M = 0.02  # an arm that stopped at the edge lags its target and sits a little outside the region: still where it may move back from
CUT_STEPS = 16  # halvings of the way to the edge: the stop is within 1/65000 of the move short of it


def zones_path() -> pathlib.Path:
    """The zones file's path."""
    return pathlib.Path(os.environ.get("OPENARM_ZONES", "~/.dreamlabs/zones.yaml")).expanduser()


def required() -> bool:
    """Whether the guard is on."""
    return os.environ.get("OPENARM_ZONES_REQUIRED", "1") != "0"


class Guard:
    """The guard of one arm."""

    def __init__(self, side: str, clock: Callable[[], dt.datetime] | None = None) -> None:
        """@param side `"left"` or `"right"`. @param clock Now, timezone-aware; the system clock by default."""
        self.side = side
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self._kinematics: zones.ArmKinematics | None = None
        self._zones: zones.Zones | None = None
        self._stamp: tuple[str, int, int] | None = None

    def _load(self) -> zones.Zones:
        path = zones_path()
        try:
            info = path.stat()
            stamp = (str(path), info.st_mtime_ns, info.st_size)
        except OSError as exc:
            raise errors.ArmError(
                f"no safety zones at {path} ({exc.strerror}): run the daily workspace capture "
                "(workspace.py capture, then apply) before the arms move"
            ) from exc
        if stamp != self._stamp:
            try:
                self._zones = zones.load_zones(path)
            except (OSError, ValueError) as exc:
                raise errors.ArmError(f"the safety zones at {path} cannot be used: {exc}") from exc
            self._stamp = stamp
        assert self._zones is not None
        return self._zones

    def _ready(self) -> tuple[zones.Zones, zones.ArmKinematics] | None:
        """The loaded zones and kinematics, or None when the guard is off.

        @throws ArmError There are no zones or they are stale.
        """
        if not required():
            return None
        loaded = self._load()
        age = loaded.age_hours(self._clock())
        limit = float(os.environ.get("OPENARM_ZONES_MAX_AGE_H", DEFAULT_MAX_AGE_H))
        if age > limit:
            raise errors.ArmError(
                f"the safety zones are {age:.0f} hours old (limit {limit:g}): run the daily workspace capture and apply it"
            )
        if self._kinematics is None:
            self._kinematics = zones.ArmKinematics(self.side)
        return loaded, self._kinematics

    @staticmethod
    def _violations(loaded: zones.Zones, kinematics: zones.ArmKinematics, side: str, q_rad: np.ndarray) -> list[str]:
        return loaded.violations(side, kinematics.points(np.asarray(q_rad, dtype=float)), check_reach=False)

    def check(self, q_rad: np.ndarray) -> None:
        """Refuses a target pose.

        @param q_rad Seven joint targets, radians, in Dream's model.
        @throws ArmError There are no zones or they are stale, or the pose enters a danger zone or leaves the work area.
        """
        ready = self._ready()
        if ready is None:
            return
        found = self._violations(*ready, self.side, q_rad)
        if found:
            raise errors.ArmError("safety zone: " + "; ".join(found[:3]) + " — nothing was sent")

    def limit(self, prev_rad: np.ndarray, q_rad: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Cuts a target back to the edge of the allowed space instead of refusing it.

        The arm moves along the straight joint-space line from `prev_rad` towards `q_rad` and stops at the last point that is still allowed.
        The same stale or missing zones still raise, and so does an arm that is already well outside (more than 2 cm past the region, or in a keep-out zone); one a few millimetres out, because it lags its target at the edge, may still be moved back and never gets a target further out.

        @param prev_rad Where the arm is or was last sent: a pose that must itself be allowed.
        @param q_rad The target, seven joint angles in Dream's model, radians.
        @return `(pose to send, violations of the original target)`; the pose is `q_rad` itself and the list empty when the target is allowed.
        @throws ArmError There are no zones or they are stale, or `prev_rad` already breaks one.
        """
        ready = self._ready()
        q_rad = np.asarray(q_rad, dtype=float)
        if ready is None:
            return q_rad, []
        found = self._violations(*ready, self.side, q_rad)
        if not found:
            return q_rad, []
        prev_rad = np.asarray(prev_rad, dtype=float)
        already = ready[0].violations(self.side, ready[1].points(prev_rad), check_reach=False, region_slack=MEASURED_SLACK_M)
        if already:
            raise errors.ArmError("safety zone: " + "; ".join(already[:3]) + " — the arm is already there, nothing was sent")
        low, high = 0.0, 1.0   # `low` is allowed, `high` is not
        for _ in range(CUT_STEPS):
            mid = (low + high) / 2
            if self._violations(*ready, self.side, prev_rad + mid * (q_rad - prev_rad)):
                high = mid
            else:
                low = mid
        return prev_rad + low * (q_rad - prev_rad), found

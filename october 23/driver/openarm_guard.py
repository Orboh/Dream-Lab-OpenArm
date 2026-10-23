"""The driver's hard guard: refuses a joint target whose arm would enter a danger zone, or any target on stale or missing zones.

The painted work area (`reach`) is not enforced here: Dream clips its hand targets to it, and the station's Home and Stop moves cross it.

Reads the zones file (`~/.dreamlabs/zones.yaml`, or `OPENARM_ZONES`). Settings by environment variable:
- `OPENARM_ZONES_REQUIRED`: `0` turns the guard off (tests and bench work); anything else keeps it on (default on).
- `OPENARM_ZONES_MAX_AGE_H`: hours a zones file stays valid, default 24 (the zones are captured every day).
A refusal raises `ArmError`; the station then stops the run and the arms hold.
"""

import datetime as dt
import os
import pathlib
from collections.abc import Callable

import numpy as np

import openarm_zones as zones
from robots import errors

DEFAULT_MAX_AGE_H = 24.0


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

    def check(self, q_rad: np.ndarray) -> None:
        """Refuses a target pose.

        @param q_rad Seven joint targets, radians, in Dream's model.
        @throws ArmError There are no zones or they are stale, or the pose enters a danger zone or leaves the work area.
        """
        if not required():
            return
        loaded = self._load()
        age = loaded.age_hours(self._clock())
        limit = float(os.environ.get("OPENARM_ZONES_MAX_AGE_H", DEFAULT_MAX_AGE_H))
        if age > limit:
            raise errors.ArmError(
                f"the safety zones are {age:.0f} hours old (limit {limit:g}): run the daily workspace capture and apply it"
            )
        if self._kinematics is None:
            self._kinematics = zones.ArmKinematics(self.side)
        found = loaded.violations(self.side, self._kinematics.points(np.asarray(q_rad, dtype=float)), check_reach=False)
        if found:
            raise errors.ArmError("safety zone: " + "; ".join(found[:3]) + " — nothing was sent")

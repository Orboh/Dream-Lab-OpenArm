"""Which arms may move: a small file the driver re-reads, switched by `arms.py`. Pure Python.

`~/.dreamlabs/active_arms.json` (or the path in `OPENARM_ACTIVE_ARMS`) holds `{"active": ["left", "right"], "set_at": "..."}`. A missing file
means both arms. A file that cannot be read means no arm: a lock nobody can read must not let an arm move.
"""

import datetime as dt
import json
import os
import pathlib

SIDES = ("left", "right")


def lock_path() -> pathlib.Path:
    """The lock file's path."""
    return pathlib.Path(os.environ.get("OPENARM_ACTIVE_ARMS", "~/.dreamlabs/active_arms.json")).expanduser()


def active_arms() -> set[str]:
    """The arms that may move now.

    @return A subset of `{"left", "right"}`: both when there is no lock file, none when it cannot be read.
    """
    path = lock_path()
    if not path.exists():
        return set(SIDES)
    try:
        return {side for side in json.loads(path.read_text())["active"] if side in SIDES}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def set_active(which: str) -> set[str]:
    """Chooses which arms may move.

    @param which `"left"`, `"right"` or `"both"`.
    @return The active arms.
    @throws ValueError `which` is none of those.
    """
    if which not in (*SIDES, "both"):
        raise ValueError(f"arms: choose left, right or both, not {which!r}")
    active = list(SIDES) if which == "both" else [which]
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"active": active, "set_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}))
    return set(active)


class ArmLock:
    """The lock as one arm's driver reads it: the small file is read on every call, so a change takes effect on the next command.

    (A modification-time cache is not used: two writes within the filesystem's timestamp tick would look unchanged.)
    """

    def __init__(self, side: str) -> None:
        self.side = side

    def is_active(self) -> bool:
        """Whether this arm may move now."""
        return self.side in active_arms()

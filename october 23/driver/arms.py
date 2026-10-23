"""Chooses which arm may move, for the driver's lock:  python arms.py right | left | both | status

The other arm keeps holding its pose whatever Dream-1 asks. Takes effect at once; no station restart.
"""

import sys

import openarm_lock as lock


def main(argv: list[str]) -> int:
    """Runs the command; returns the exit code."""
    if len(argv) != 2 or argv[1] not in ("left", "right", "both", "status"):
        print(__doc__)
        return 2
    if argv[1] != "status":
        lock.set_active(argv[1])
    active = sorted(lock.active_arms())
    print("active arms:", ", ".join(active) if active else "none (lock file unreadable)", f"[{lock.lock_path()}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

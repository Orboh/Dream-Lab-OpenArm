"""The website's "Go to home" button: raises the allowed arms to the home (the camera-view pose), holds them there until "Park", then parks and switches off.

The same moves as `workspace.py camera-pose`, but both arms hold together and the hold ends with a button instead of the Enter key. The real arms MOVE.
Before anything opens, the preflight runs (teleop or station stopped, zones fresh, home path clear, guard on, ...). Only the two camera checks are left out:
this page holds the cameras itself, and moving to the home does not use them. An arm that does not end near the home, or any error, switches every
opened arm off at once. Opening an arm blocks while it raises, so Park takes effect when the raise has finished: the hardware stop is the emergency stop.
"""

import pathlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

import camerapose as camp
import openarm_lock
import openarm_mapping as mapping
import preflight as pf

SKIPPED = (pf.LABEL_CAMERAS, pf.LABEL_CAMERAS_FREE)  # this page holds the cameras; the move does not need them
MAX_LINES = 12


class HomeMover:
    """Runs the move on its own thread; the page reads `state` and `text()` and calls `go`, `park`, `close`."""

    def __init__(self, station: pathlib.Path, opener: Callable[[Any, str], Any] | None = None,
                 checks: Callable[[], list[pf.Check]] | None = None) -> None:
        """@param station The station file. @param opener Opens the driver's arm, `opener(cfg, side)` (the real `openarm_v2.Arm` by default).
        @param checks Runs the preflight (the real one against `station` by default)."""
        self.station = pathlib.Path(station)
        self._opener = opener
        self._checks = checks
        self.state = "idle"  # idle, checking, moving, at home, parking
        self.messages: list[str] = []
        self._arms: dict[str, Any] = {}
        self._park = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def text(self) -> str:
        """The latest report lines, newest last."""
        return "\n".join(self.messages[-MAX_LINES:])

    def _say(self, line: str) -> None:
        self.messages.append(line)

    def go(self, sides: list[str]) -> bool:
        """Starts the move. @param sides The arms to raise. @return False when a move is already running (nothing new starts)."""
        with self._lock:
            if self.state != "idle":
                return False
            self.state = "checking"
        self.messages = []
        self._park.clear()
        self._thread = threading.Thread(target=self._run, args=(list(sides),), name="home-move", daemon=True)
        self._thread.start()
        return True

    def park(self) -> None:
        """Ends the hold: the arms park at rest along the path and switch off."""
        self._park.set()

    def close(self) -> None:
        """For the page closing: parks whatever is up and waits for it."""
        self._park.set()
        if self._thread is not None:
            self._thread.join(timeout=120.0)

    def _run(self, sides: list[str]) -> None:
        try:
            if not self._preflight():
                return
            self.state = "moving"
            if not self._raise(sides):
                return
            if not self._arms:
                self._say("no arm was moved")
                return
            self.state = "at home"
            self._say("at the home: look at the camera views, then press Park")
            self._park.wait()
        except Exception as exc:  # a failed raise must not leave an arm powered
            self._say(f"stopped: {exc}")
        finally:
            self._switch_off()
            self.state = "idle"

    def _preflight(self) -> bool:
        if self._checks is None:
            import openarm_guard

            checks = pf.run_preflight(self.station, openarm_guard.zones_path(), pathlib.Path(__file__).resolve().parent / "zones")
        else:
            checks = self._checks()
        failed = [c for c in checks if not c.ok and c.label not in SKIPPED]
        for c in failed:
            self._say(f"check failed: {c.label}" + (f" ({c.detail})" if c.detail else ""))
        if failed:
            self._say("nothing moved: fix the checks above and press Go to home again")
        return not failed

    def _raise(self, sides: list[str]) -> bool:
        """Opens each allowed arm (it raises itself from rest along the recorded path). @return False when an arm did not reach the home."""
        opener = self._opener
        if opener is None:
            import openarm_v2 as drv

            opener = drv.Arm
        active = openarm_lock.active_arms()
        for side in sides:
            if side not in active:
                self._say(f"{side}: locked (arms.py), so it is left where it is")
                continue
            if self._park.is_set():
                self._say(f"{side}: not raised (Park was pressed)")
                continue
            cfg = camp.arm_config(self.station, side)
            self._say(f"{side}: raising from rest to the home along the recorded path")
            arm = opener(cfg, side)
            self._arms[side] = arm
            home_deg = np.degrees(mapping.to_logical(side, cfg.home))  # the station's home is in Dream's model angles; _joint_pose() is logical
            gap = float(np.abs(np.asarray(arm._joint_pose(), dtype=float) - home_deg).max())
            if gap > camp.DONE_WITHIN_DEG:
                self._say(f"{side}: NOT at the home (largest joint {gap:.0f} deg away): it did not start at rest; switching off")
                return False
            self._say(f"{side}: at the home (largest joint {gap:.1f} deg from it)")
        return True

    def _switch_off(self) -> None:
        if self._arms:
            self.state = "parking"
        for side, arm in list(self._arms.items()):
            self._say(f"{side}: parking at rest along the path, then switching off")
            try:
                arm.torque_off()
                self._say(f"{side}: switched off")
            except Exception as exc:
                self._say(f"{side}: switching off FAILED ({exc}): use the emergency stop")
        self._arms = {}

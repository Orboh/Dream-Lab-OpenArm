"""The interactive capture session: live hand positions, painting the work area, touching the table, marking dangers. No hardware here:
the hand positions come from a callable, so the session is tested with a fake and run on the arms (or the leaders) by workspace.py.
"""

import select
import sys
import time
from collections.abc import Callable

import numpy as np

import zonebuild as zb

HELP = """commands (type, Enter):
  paint on | paint off     paint the work area while you move the hand over it
  table                    record a point on the table top (touch it with the follower's gripper tip)
  hazard NAME              record a point of a danger zone called NAME (repeat for its corners)
  ref                      record a reference point for the daily check (follower only)
  undo                     remove the last recorded point
  arm left | arm right     choose which arm records points
  status | help | save | quit"""


class CaptureSession:
    """What is being captured, and the commands that change it."""

    def __init__(self, side: str, read_tcp: Callable[[str], np.ndarray], ground_truth: bool = True) -> None:
        """@param side The arm that records points first. @param read_tcp Side to its grasp point `[x, y, z]` now (base frame, metres).
        @param ground_truth True for the follower (it may record table and reference points); False for a leader."""
        self.side = side
        self._read = read_tcp
        self.ground_truth = ground_truth
        self.painting = False
        self.captures = {"left": zb.ArmCapture(), "right": zb.ArmCapture()}
        self._history: list[tuple[str, str, str | None]] = []  # (side, kind, hazard name)

    def tick(self) -> None:
        """Adds the current hand position to the work area while painting is on."""
        if self.painting:
            self.captures[self.side].reach.add(zb.cell(self._read(self.side)))

    def _point(self) -> tuple[float, float, float]:
        return tuple(round(float(v), 4) for v in self._read(self.side))  # type: ignore[return-value]

    def command(self, line: str) -> tuple[str | None, str]:
        """Runs one typed command.

        @param line The text typed.
        @return `(action, message)`: `action` is `"save"` or `"quit"` when the session should end, else None.
        """
        words = line.strip().split()
        if not words:
            return None, ""
        cmd, args = words[0].lower(), words[1:]
        capture = self.captures[self.side]
        if cmd in ("save", "quit"):
            return cmd, ""
        if cmd == "help":
            return None, HELP
        if cmd == "paint":
            self.painting = (args[:1] == ["on"]) if args else not self.painting
            return None, f"painting {'ON' if self.painting else 'off'} for the {self.side} arm"
        if cmd == "arm" and args and args[0] in zb.SIDES:
            self.side = args[0]
            return None, f"recording for the {self.side} arm"
        if cmd in ("table", "ref") and not self.ground_truth:
            return None, f"'{cmd}' needs the follower: the table and reference points must be touched with the follower arm, not a leader"
        if cmd == "table":
            capture.table_points.append(self._point())
            self._history.append((self.side, "table", None))
            return None, f"table point {len(capture.table_points)} at {capture.table_points[-1]}"
        if cmd == "ref":
            capture.references.append(self._point())
            self._history.append((self.side, "ref", None))
            return None, f"reference point {len(capture.references)} at {capture.references[-1]}"
        if cmd == "hazard" and args:
            capture.hazards.setdefault(args[0], []).append(self._point())
            self._history.append((self.side, "hazard", args[0]))
            return None, f"hazard '{args[0]}' now has {len(capture.hazards[args[0]])} point(s)"
        if cmd == "undo":
            if not self._history:
                return None, "nothing to undo"
            side, kind, name = self._history.pop()
            target = self.captures[side]
            if kind == "table":
                target.table_points.pop()
            elif kind == "ref":
                target.references.pop()
            else:
                target.hazards[name].pop()
                if not target.hazards[name]:
                    del target.hazards[name]
            return None, f"removed the last {kind} point of the {side} arm"
        if cmd == "status":
            return None, self.status_line()
        return None, f"unknown command '{cmd}' (type help)"

    def status_line(self) -> str:
        """One line: the arm, its hand, and what has been recorded."""
        c = self.captures[self.side]
        x, y, z = (self._read(self.side) * 100)
        hazards = ", ".join(f"{n}({len(p)})" for n, p in c.hazards.items()) or "none"
        return (f"{self.side}: hand x {x:6.1f} y {y:6.1f} z {z:6.1f} cm | paint {'ON ' if self.painting else 'off'} {len(c.reach)} cells | "
                f"table pts {len(c.table_points)} | refs {len(c.references)} | hazards {hazards}")


def run_interactive(session: CaptureSession, rate_hz: float = 20.0) -> str:
    """Runs the session in a terminal: paints at `rate_hz`, shows the live line, reads commands without blocking.

    @param session The session.
    @param rate_hz Painting and display rate.
    @return `"save"` or `"quit"`.
    """
    print(HELP)
    last_print = 0.0
    while True:
        session.tick()
        now = time.monotonic()
        if now - last_print > 0.25:
            sys.stdout.write("\r" + session.status_line() + "   ")
            sys.stdout.flush()
            last_print = now
        ready, _, _ = select.select([sys.stdin], [], [], 1.0 / rate_hz)
        if ready:
            line = sys.stdin.readline()
            if not line:
                return "quit"
            action, message = session.command(line)
            if message:
                print("\n" + message)
            if action:
                return action

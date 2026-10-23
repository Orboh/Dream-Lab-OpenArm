"""A receive-only view of one follower arm while your own teleop runs: it listens to the CAN link and NEVER sends a frame.

The teleop (`scripts/run_teleop_bimanual.sh`) polls the motors; every motor answers with a state frame from `0x10 + its id`, and a SocketCAN link hands all frames on it to every
listener on this computer (your dmlog logger works the same way). This class decodes those state frames into the arm's logical joint angles, so the safety-area tools can show
and record where the follower is without touching the bus.
"""

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import openarm_v2 as drv
from robots import damiao, errors

REPLY_BASE = damiao.REPLY_OFFSET


class PassiveArm:
    """Listens to one follower link and keeps each motor's latest position."""

    def __init__(self, side: str, channel: str | None = None, opener: Callable[[str], Any] | None = None, max_age_s: float = 0.5,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """@param side `"left"` or `"right"`. @param channel The SocketCAN link (`can_fol_l` / `can_fol_r` by default).
        @param opener Opens the link (FD socketcan by default; tests pass a fake). @param max_age_s Frames older than this count as stale."""
        self.side = side
        self.channel = channel or drv.LINK_ALIASES[side]
        self._open = opener or drv.open_fd_bus
        self._max_age = max_age_s
        self._clock = clock
        self._offsets = drv.homing_offsets(side)
        self._native: dict[str, tuple[float, float]] = {}  # motor -> (native degrees, time received)
        self._ids = {REPLY_BASE + drv.CAN_IDS[name]: name for name in drv.MOTORS}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bus: Any = None
        self.error: Exception | None = None

    def start(self) -> None:
        """Opens the link and starts listening."""
        self._bus = self._open(self.channel)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"observer-{self.side}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops listening and closes the link."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                message = self._bus.recv(timeout=0.05)
                if message is None:
                    continue
                name = self._ids.get(message.arbitration_id)
                data = bytes(message.data)
                if name is None or len(data) < 8 or not damiao.is_state_frame(data, drv.CAN_IDS[name]):
                    continue
                state = damiao.decode_state(data, drv.MODELS[name], self._clock())
                with self._lock:
                    self._native[name] = (state.position_deg, self._clock())
        except Exception as exc:  # a dead link must not be silent
            self.error = exc

    def motors_seen(self) -> set[str]:
        """The motors a state frame has been heard from."""
        with self._lock:
            return set(self._native)

    def has_all(self) -> bool:
        """Whether every motor has been heard."""
        return self.motors_seen() == set(drv.MOTORS)

    def logical_deg(self) -> dict[str, float]:
        """Every motor's logical angle, degrees (native minus the calibration's homing offset).

        @throws ArmError No frames, a motor never heard, or frames older than `max_age_s` (the teleop stopped).
        """
        if self.error is not None:
            raise errors.ArmError(f"arm {self.side}: the listener stopped ({self.error!r})")
        with self._lock:
            seen = dict(self._native)
        if not seen:
            raise errors.ArmError(f"arm {self.side}: no frames on {self.channel}: is the teleop running?")
        missing = [name for name in drv.MOTORS if name not in seen]
        if missing:
            raise errors.ArmError(f"arm {self.side}: no state heard yet from {missing}")
        now = self._clock()
        old = [name for name, (_, t) in seen.items() if now - t > self._max_age]
        if old:
            raise errors.ArmError(f"arm {self.side}: the frames from {old} are stale (older than {self._max_age:g} s): the teleop stopped?")
        return {name: seen[name][0] - self._offsets[name] for name in drv.MOTORS}

    def q_rad(self) -> np.ndarray:
        """The seven arm joints in radians in Dream's model (identity mapping from the logical angles)."""
        deg = self.logical_deg()
        return drv.MODEL_SIGN * np.radians([deg[n] for n in drv.JOINTS]) + drv.MODEL_ZERO_RAD

    def status(self) -> str:
        """One line for the page: whether frames are arriving."""
        try:
            self.logical_deg()
            return f"{self.side}: receiving frames from {self.channel}"
        except errors.ArmError as exc:
            return str(exc).replace(f"arm {self.side}: ", f"{self.side}: ")

"""Camera feeds for the viewer, switched on and off from the browser. The worker is plain Python (tested without cameras); the viewer wires it to checkboxes.

Cameras are the RealSense devices listed in `~/.dreamlabs/station.yaml` (`cameras` with a `serial`), opened through the project's
`orboh.cameras.sources.RealSenseSource`. They start OFF: a camera only opens when its box is ticked, and is released when it is unticked.
The station must be stopped (it holds the cameras while it runs).
"""

import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import yaml

WIDTH, HEIGHT, FPS = 640, 360, 15


class CameraWorker:
    """Reads one camera on its own thread while it is on."""

    def __init__(self, name: str, open_source: Callable[[], Any], on_frame: Callable[[np.ndarray | None], None],
                 on_status: Callable[[str], None], period: float = 0.1) -> None:
        """@param name The camera's role. @param open_source Opens it (an object with read() and close()). @param on_frame Gets each RGB frame, or None when off.
        @param on_status Gets a short text. @param period Seconds between reads."""
        self.name = name
        self._open = open_source
        self._on_frame = on_frame
        self._on_status = on_status
        self._period = period
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the camera is open and being read."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Opens the camera and starts reading; does nothing if it is already on."""
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops reading and releases the camera."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            source = self._open()
        except Exception as exc:
            self._on_status(f"{self.name}: cannot open ({exc})")
            self._on_frame(None)
            return
        self._on_status(f"{self.name}: on")
        try:
            while not self._stop.is_set():
                frame = source.read()
                if frame is not None:
                    self._on_frame(frame)
                time.sleep(self._period)
        except Exception as exc:
            self._on_status(f"{self.name}: stopped ({exc})")
        else:
            self._on_status("off")
        finally:
            try:
                source.close()
            except Exception:
                pass
            self._on_frame(None)


def cameras_from_station(station: pathlib.Path) -> list[dict[str, str]]:
    """The station file's RealSense cameras (`name`, `serial`); nothing if the file is missing or has none."""
    try:
        data = yaml.safe_load(pathlib.Path(station).read_text()) or {}
    except OSError:
        return []
    return [{"name": str(c["name"]), "serial": str(c["serial"])} for c in data.get("cameras", []) if isinstance(c, dict) and "serial" in c]


def realsense_opener(name: str, serial: str) -> Callable[[], Any]:
    """An opener for a RealSense by serial, 640x360 at 15 fps, through the project's camera source."""
    def opener() -> Any:
        from orboh.cameras.sources import CameraDevice, RealSenseSource

        return RealSenseSource(CameraDevice(serial=serial, name=name), WIDTH, HEIGHT, FPS)

    return opener

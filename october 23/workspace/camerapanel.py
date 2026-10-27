"""Camera feeds for the viewer, switched on and off from the browser. The worker is plain Python (tested without cameras); the viewer wires it to checkboxes.

Cameras are the RealSense devices listed in `~/.dreamlabs/station.yaml` (`cameras` with a `serial`), opened through the project's
`orboh.cameras.sources.RealSenseSource`. They start OFF: a camera only opens when its box is ticked, and is released when it is unticked.
The station must be stopped (it holds the cameras while it runs).
"""

import base64
import html
import io
import pathlib
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import yaml
from PIL import Image

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


# The camera strip above the robot view (left, center, right). It follows the 3D view's box through a CSS anchor, so it keeps the view's width when the
# panel is resized; browsers without anchors get the left 60% of the screen. While it shows, the 3D view moves down by the strip's height.
STRIP_CSS = """<style>
:root{--cam-h:clamp(110px,22vh,260px);}
div:has(> canvas){anchor-name:--view3d;}
body:has(.cam-tile) div:has(> canvas){top:var(--cam-h) !important;}
.cam-bar{position:fixed;top:0;left:0;width:60vw;height:var(--cam-h);background:#16181c;z-index:4;display:none;}
body:has(.cam-tile) .cam-bar{display:block;}
.cam-tile{position:fixed;top:0;height:var(--cam-h);width:calc(20vw - 6px);z-index:4;display:flex;align-items:center;justify-content:center;
  background:#000;color:#bbb;font:600 14px sans-serif;overflow:hidden;}
.cam-tile img{width:100%;height:100%;object-fit:contain;}
.cam-tile span{position:absolute;left:6px;top:4px;padding:1px 6px;border-radius:4px;background:rgba(0,0,0,.55);color:#fff;font-size:12px;}
.cam-left{left:3px;} .cam-center{left:calc(20vw + 3px);} .cam-right{left:calc(40vw + 3px);}
@supports (anchor-name:--a){
  .cam-bar{position-anchor:--view3d;left:anchor(left);right:anchor(right);width:auto;}
  .cam-tile{position-anchor:--view3d;width:calc(anchor-size(width) / 3 - 6px);}
  .cam-left{left:calc(anchor(left) + 3px);} .cam-center{left:anchor(center);translate:-50% 0;} .cam-right{left:auto;right:calc(anchor(right) + 3px);}
}
</style><div class="cam-bar"></div>"""


def strip_slot(name: str) -> str:
    """The camera's place in the strip: the left wrist `"left"`, the right wrist `"right"`, anything else (the head) `"center"`."""
    low = name.lower()
    if low.endswith(("_l", "-l")) or "left" in low:
        return "left"
    if low.endswith(("_r", "-r")) or "right" in low:
        return "right"
    return "center"


def strip_tile_html(name: str, frame: np.ndarray | None, quality: int = 70) -> str:
    """One tile of the strip: the RGB frame embedded as a JPEG data URL, or a black "off" tile for None."""
    label = f"<span>{html.escape(name)}</span>"
    if frame is None:
        return f'<div class="cam-tile cam-{strip_slot(name)}">{label}{html.escape(name)}: off</div>'
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<div class="cam-tile cam-{strip_slot(name)}"><img src="{url}">{label}</div>'


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

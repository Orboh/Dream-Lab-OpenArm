"""3D view of the daily safety area in a browser (viser, http://localhost:<port>). Reads the same zones file the driver and the station use.

Shows, per arm: the table slab and danger zones (red), the 2 cm + link-radius margin around them (amber wireframe), the painted work area (green),
the arm's skeleton with the 4 cm collision spheres Dream's IK uses, and the rest and ready poses for context. With a live source the skeleton follows
the real arm (limp, moved by hand) and turns red when it would break a zone. The two arm bases are drawn 6 cm apart with the same orientation: an
assumption (check it by touching one physical point with both arms), not a measurement.
"""

import threading
import time
from collections.abc import Callable

import numpy as np
import viser

import camerapanel
import openarm_poses as poses
import openarm_zones as zones

BASE_OFFSET = {"left": np.array([0.0, 0.0305, 0.0]), "right": np.array([0.0, -0.0305, 0.0])}
GREEN, RED, AMBER, BLUE, GREY = (60, 180, 75), (220, 50, 47), (255, 191, 0), (60, 120, 220), (150, 150, 150)
CHAIN = ("elbow", "wrist_pitch", "wrist", "hand", "tcp")
SPHERES = ("elbow", "wrist_pitch", "wrist", "forearm", "hand")


def _skeleton(points: dict[str, np.ndarray], offset: np.ndarray) -> np.ndarray:
    chain = [offset] + [points[name] + offset for name in CHAIN]
    return np.array([[chain[i], chain[i + 1]] for i in range(len(chain) - 1)])


def build_scene(server: viser.ViserServer, raw: dict) -> dict:
    """Draws the zones and the context poses; returns the handles the live update moves.

    @param server The viser server.
    @param raw The zones mapping.
    @return Handles per arm for the live skeleton.
    """
    parsed = zones.parse_zones(raw)
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)
    server.scene.add_grid("/floor", width=1.6, height=1.6, plane="xy", cell_size=0.1, section_size=0.5,
                          position=(0.3, 0.0, min(a["table_z"] for a in raw["arms"].values()) - 0.32))
    handles: dict = {}
    for side in ("left", "right"):
        offset = BASE_OFFSET[side]
        for zone in parsed.arms[side].keep_out:
            if zone["kind"] != "box":
                continue
            center = np.array(zone["center"]) + offset
            table = zone.get("allow_tcp_touch")
            server.scene.add_box(f"/{side}/zone/{zone['name']}", BLUE if table else RED, tuple(zone["size"]), opacity=0.25 if table else 0.35, position=tuple(center))
            grow = 2 * (zones.LINK_RADIUS_M + zones.CLEARANCE_M)
            server.scene.add_box(f"/{side}/margin/{zone['name']}", AMBER, tuple(np.array(zone["size"]) + grow), wireframe=True, position=tuple(center))
        if parsed.arms[side].reach is not None:
            z = raw["arms"][side].get("table_z", 0.0) + 0.004
            cells = np.array([[x + offset[0], y + offset[1], z] for x, y in parsed.arms[side].reach])
            server.scene.add_point_cloud(f"/{side}/work_area", cells, GREEN, point_size=0.012, point_shape="square")
        kin = zones.ArmKinematics(side)
        for name, pose, color in (("rest", poses.REST_DEG[side], GREY), ("ready", poses.READY_DEG[side], BLUE)):
            pts = kin.points(np.radians(pose))
            server.scene.add_line_segments(f"/{side}/{name}", _skeleton(pts, offset), color, thickness=0.006)
            server.scene.add_label(f"/{side}/{name}_label", f"{side} {name}", position=tuple(pts["tcp"] + offset + [0, 0, 0.03]))
        live_pts = kin.points(np.zeros(7))
        handles[side] = {
            "kin": kin, "offset": offset,
            "line": server.scene.add_line_segments(f"/{side}/live", _skeleton(live_pts, offset), GREEN, thickness=0.012, visible=False),
            "spheres": {n: server.scene.add_icosphere(f"/{side}/live_{n}", zones.LINK_RADIUS_M, GREEN, opacity=0.35, visible=False) for n in SPHERES},
            "tcp": server.scene.add_icosphere(f"/{side}/live_tcp", 0.012, GREEN, visible=False),
        }
    handles["parsed"] = parsed
    return handles


def update_live(handles: dict, side: str, q_rad: np.ndarray) -> list[str]:
    """Moves one arm's live skeleton to `q_rad` and colours it by whether the zones allow the pose.

    @param handles From #build_scene().
    @param side `"left"` or `"right"`.
    @param q_rad Seven joint angles, radians, in Dream's model.
    @return The violations (empty when the pose is allowed).
    """
    h = handles[side]
    pts = h["kin"].points(q_rad)
    bad = handles["parsed"].violations(side, pts, check_reach=False)
    color = np.array(RED if bad else GREEN, dtype=np.uint8)
    h["line"].points = _skeleton(pts, h["offset"])
    h["line"].colors = color
    h["line"].visible = True
    for name, sphere in h["spheres"].items():
        sphere.position = tuple(pts[name] + h["offset"])
        sphere.color = color
        sphere.visible = True
    h["tcp"].position = tuple(pts["tcp"] + h["offset"])
    h["tcp"].color = color
    h["tcp"].visible = True
    return bad


class CameraPanel:
    """Checkboxes and pictures for the cameras: tick a box to switch a camera on, untick to release it. All start off."""

    BLANK = np.full((90, 160, 3), 40, dtype=np.uint8)

    def __init__(self, server: viser.ViserServer, cameras: list[dict[str, str]], opener=camerapanel.realsense_opener) -> None:
        """@param server The viser server. @param cameras `name`/`serial` pairs. @param opener Builds a camera opener from name and serial."""
        self.workers: dict[str, camerapanel.CameraWorker] = {}
        self._status: dict[str, object] = {}
        with server.gui.add_folder("Cameras"):
            if not cameras:
                server.gui.add_markdown("No RealSense cameras in the station file.")
            for cam in cameras:
                name = cam["name"]
                box = server.gui.add_checkbox(f"{name} camera on", False)
                image = server.gui.add_image(self.BLANK, label=name, jpeg_quality=70)
                status = server.gui.add_markdown(f"{name}: off")
                self._status[name] = status
                self.workers[name] = camerapanel.CameraWorker(
                    name, opener(name, cam["serial"]),
                    lambda frame, image=image: setattr(image, "image", self.BLANK if frame is None else frame),
                    lambda text, status=status: setattr(status, "content", text), period=0.1)
                box.on_update(lambda event, name=name, box=box: self.toggle(name, box.value))

    def toggle(self, name: str, on: bool) -> None:
        """Switches one camera on or off."""
        worker = self.workers[name]
        worker.start() if on else worker.stop()

    def close(self) -> None:
        """Releases every camera."""
        for worker in self.workers.values():
            worker.stop()


def serve(raw: dict, port: int = 8080, read_q: Callable[[str], np.ndarray] | None = None, stop: threading.Event | None = None,
          cameras: list[dict[str, str]] | None = None) -> None:
    """Starts the viewer and, with `read_q`, follows the real arms until `stop` is set (Ctrl-C in the terminal).

    @param raw The zones mapping.
    @param port The browser port.
    @param read_q Side to the arm's joints in radians now, or None for a static view.
    @param stop Set to end the loop.
    @param cameras RealSense `name`/`serial` pairs to offer as switchable feeds; None for no camera panel.
    """
    server = viser.ViserServer(port=port)
    handles = build_scene(server, raw)
    panel = CameraPanel(server, cameras) if cameras is not None else None
    status = server.gui.add_markdown("Move an arm by hand: the skeleton follows and turns red inside a danger zone." if read_q else "Static view: zones, rest and ready poses.")
    print(f"viewer: open http://localhost:{port}   (Ctrl-C to stop)")
    stop = stop or threading.Event()
    try:
        while not stop.is_set():
            if read_q is not None:
                lines = []
                for side in ("left", "right"):
                    try:
                        lines += update_live(handles, side, read_q(side))
                    except KeyError:
                        pass
                status.content = "**All clear**" if not lines else "**DANGER**  \n" + "  \n".join(lines[:6])
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        server.stop()

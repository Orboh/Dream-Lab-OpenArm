"""The layout picture: a top view and a side view of both arms, the table, the danger boxes and the work area, drawn with PIL.

Reads a zones bundle (arm base frames) and draws everything in the robot body frame: x forward, y left, z up, metres. The top view has x going up the page and
y to the left; the side view has x to the right and z up. Arm skeletons are drawn at the rest pose (grey), the ready pose (blue) and the bundle's home pose (green).
"""

import pathlib
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

import editor as ed
import openarm_zones as zones

SIZE = (1500, 820)
BACKGROUND = (250, 250, 250)
BLUE_FILL, BLUE_LINE = (150, 185, 235), (40, 90, 200)
RED_FILL, RED_LINE = (235, 140, 135), (200, 40, 40)
AMBER = (240, 170, 0)
GREEN = (50, 160, 70)
GREY = (140, 140, 140)
INK = (30, 30, 30)

# panel geometry: (left, top, width, height) in pixels
TOP_PANEL = (40, 70, 640, 700)
SIDE_PANEL = (780, 70, 680, 700)
TOP_X, TOP_Y = (-0.35, 1.0), (-0.65, 0.65)      # metres: x forward (up the page), y left (to the left)
SIDE_X, SIDE_Z = (-0.35, 1.0), (-0.05, 1.0)      # metres: x to the right, z up
CHAIN = ("elbow", "wrist_pitch", "wrist", "hand", "tcp")


def project_top(x: float, y: float) -> tuple[int, int]:
    """Pixel of a body-frame point in the top view."""
    left, top, w, h = TOP_PANEL
    return int(left + (TOP_Y[1] - y) / (TOP_Y[1] - TOP_Y[0]) * w), int(top + (TOP_X[1] - x) / (TOP_X[1] - TOP_X[0]) * h)


def project_side(x: float, z: float) -> tuple[int, int]:
    """Pixel of a body-frame point in the side view."""
    left, top, w, h = SIDE_PANEL
    return int(left + (x - SIDE_X[0]) / (SIDE_X[1] - SIDE_X[0]) * w), int(top + (SIDE_Z[1] - z) / (SIDE_Z[1] - SIDE_Z[0]) * h)


def _rect(draw: ImageDraw.ImageDraw, project, a: tuple[float, float], b: tuple[float, float], fill, line, width: int = 2) -> None:
    (x0, y0), (x1, y1) = project(*a), project(*b)
    draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill, outline=line, width=width)


def _skeleton(points: dict[str, np.ndarray], side: str) -> np.ndarray:
    chain = [np.zeros(3)] + [points[n] for n in CHAIN]
    return np.array([ed.to_world(side, p) for p in chain])


def render_layout(raw: dict[str, Any], out: pathlib.Path) -> pathlib.Path:
    """Draws the bundle and writes a PNG.

    @param raw The zones bundle (as `EditorState.to_bundle` or `build_zones` makes it).
    @param out The PNG path; its folder is created.
    @return `out`.
    """
    parsed = zones.parse_zones(raw)
    img = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw.text((40, 14), f"Safety area  |  captured {raw['captured_at']}  |  source {raw['source']}  |  verify {raw['verify']['status']}", fill=INK)
    draw.text((40, 34), "body frame, metres: x forward, y left, z up. Zones are stored per arm in its base frame = body frame shifted by (0, +/-0.031, 0.698)", fill=GREY)
    draw.text((TOP_PANEL[0], TOP_PANEL[1] - 18), "TOP VIEW (x up the page, y to the left)", fill=INK)
    draw.text((SIDE_PANEL[0], SIDE_PANEL[1] - 18), "SIDE VIEW (x to the right, z up)", fill=INK)
    for panel in (TOP_PANEL, SIDE_PANEL):
        draw.rectangle([panel[0], panel[1], panel[0] + panel[2], panel[1] + panel[3]], outline=GREY)
    margin = zones.LINK_RADIUS_M + zones.CLEARANCE_M
    drawn: set[tuple[str, tuple[float, ...]]] = set()
    for side in ed.SIDES:
        for zone in parsed.arms[side].keep_out:
            if zone["kind"] != "box":
                continue
            c = np.array(ed.to_world(side, zone["center"]))
            s = np.array(zone["size"])
            key = (zone["name"], tuple(np.round(c, 3)))
            if key in drawn:
                continue
            drawn.add(key)
            table = zone.get("allow_tcp_touch")
            fill, line = (BLUE_FILL, BLUE_LINE) if table else (RED_FILL, RED_LINE)
            lo, hi = c - s / 2, c + s / 2
            _rect(draw, project_top, (lo[0] - margin, lo[1] - margin), (hi[0] + margin, hi[1] + margin), None, AMBER, 1)
            _rect(draw, project_side, (lo[0] - margin, lo[2] - margin), (hi[0] + margin, hi[2] + margin), None, AMBER, 1)
            _rect(draw, project_top, (lo[0], lo[1]), (hi[0], hi[1]), fill, line)
            _rect(draw, project_side, (lo[0], lo[2]), (hi[0], hi[2]), fill, line)
            draw.text(project_top(hi[0], hi[1] - 0.02), zone["name"], fill=INK)
            draw.text(project_side(lo[0] + 0.01, hi[2] - 0.02), zone["name"], fill=INK)
        region = parsed.arms[side].region
        if region is not None:
            lo, hi = np.array(ed.to_world(side, region["lo"])), np.array(ed.to_world(side, region["hi"]))
            _rect(draw, project_top, (lo[0], lo[1]), (hi[0], hi[1]), None, GREEN, 3)
            _rect(draw, project_side, (lo[0], lo[2]), (hi[0], hi[2]), None, GREEN, 3)
            draw.text(project_top(hi[0], hi[1] - 0.02), f"{side} allowed region", fill=GREEN)
        if parsed.arms[side].reach is not None:
            base = ed.BASE_OFFSET[side]
            for x, y in parsed.arms[side].reach:
                px, py = project_top(x + base[0], y + base[1])
                draw.rectangle([px - 2, py - 2, px + 2, py + 2], fill=GREEN)
    for side in ed.SIDES:
        kin = zones.ArmKinematics(side)
        home = parsed.home[side] if parsed.home else None
        ready = np.radians(zones_pose(side, "ready"))
        home_is_ready = home is not None and np.allclose(home, ready, atol=0.01)
        for pose, color, label in ((np.radians(zones_pose(side, "rest")), GREY, "rest"),
                                   (ready, GREEN if home_is_ready else BLUE_LINE, "ready = home" if home_is_ready else "ready"),
                                   (None if home is None or home_is_ready else np.asarray(home), GREEN, "home")):
            if pose is None:
                continue
            pts = _skeleton(kin.points(pose), side)
            draw.line([project_top(p[0], p[1]) for p in pts], fill=color, width=3)
            draw.line([project_side(p[0], p[2]) for p in pts], fill=color, width=3)
            draw.text(project_side(pts[-1][0] + 0.01, pts[-1][2]), f"{side} {label}", fill=color)
    # scale bar: 0.5 m in the side view
    x0, y0 = project_side(SIDE_X[0] + 0.05, SIDE_Z[0] + 0.03)
    x1, _ = project_side(SIDE_X[0] + 0.55, SIDE_Z[0] + 0.03)
    draw.line([x0, y0, x1, y0], fill=INK, width=3)
    draw.text((x0, y0 + 6), "0.5 m", fill=INK)
    legend = [(BLUE_FILL, "table"), (RED_FILL, "danger box"), (AMBER, "margin (link radius + 2 cm)"), (GREEN, "work area / region / home"), (GREY, "rest"), (BLUE_LINE, "ready")]
    for i, (color, text) in enumerate(legend):
        draw.rectangle([40 + i * 230, 790, 56 + i * 230, 804], fill=color, outline=INK)
        draw.text((62 + i * 230, 790), text, fill=INK)
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG")
    return out


def zones_pose(side: str, name: str) -> list[float]:
    """The rest or ready pose in degrees."""
    import openarm_poses as poses

    return poses.REST_DEG[side] if name == "rest" else poses.READY_DEG[side]

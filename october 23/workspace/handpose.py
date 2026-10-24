"""Which way the hand points, from the real URDF (the one the page draws): the angle of the gripper's approach direction above the horizontal.

0 deg is level, +90 straight up, -90 straight down (a hanging arm). The wrist camera looks along that direction, so this is the number to make level
when the camera must see the table. The URDF has no meshes loaded here: only its joints and links.
"""

import math
import pathlib

import numpy as np
import yourdfpy

URDF = pathlib.Path(__file__).resolve().parent.parent / "urdf" / "openarm_bimanual.urdf"
_urdf: yourdfpy.URDF | None = None


def _model() -> yourdfpy.URDF:
    global _urdf
    if _urdf is None:
        _urdf = yourdfpy.URDF.load(str(URDF), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)
    return _urdf


def approach_elevation_deg(side: str, q_rad) -> float:
    """The hand's approach direction above the horizontal, in degrees.

    @param side `"left"` or `"right"`.
    @param q_rad Seven joint angles, radians (the fork's logical angles, as the page's robot is posed).
    """
    model = _model()
    model.update_cfg({f"openarm_{side}_joint{i + 1}": float(v) for i, v in enumerate(q_rad)})
    approach = model.get_transform(f"openarm_{side}_hand_tcp")[:3, 2]  # the tcp frame's z: down when the arm hangs, the gripper's forward direction
    return math.degrees(math.asin(float(np.clip(approach[2], -1.0, 1.0))))

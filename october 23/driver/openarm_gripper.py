"""The gripper's hand housing and two fingers, as points attached to the guard's arm model (log-036). Nothing here talks to a motor.

Dream's arm model (`dream_ik` openarm2_faas) ends at a frame with the grasp point 15.9 cm below it, and has no gripper. The official OpenArm v10 URDF
(`october 23/urdf/openarm_bimanual.urdf` with the `openarm_description` meshes) has the hand and both fingers. Its `hand_tcp` frame sits at a fixed place
in Dream's grasp-point frame (#OFFICIAL_TCP_IN_DREAM_TCP: the same over 30 random poses of each arm to 0.00 cm and 0.000 in rotation, under the official
joint mapping; 2.1 cm further out along the gripper than Dream's grasp point).

Each part is the axis-aligned box of its mesh vertices in the official `hand_tcp` frame (measured from the meshes, log-036), and counts by its 8 corners.
The box encloses the mesh, so for a flat surface (the table) the lowest corner is never above the lowest point of the part.
"""

import numpy as np

## The hand housing, metres, official hand_tcp frame (x, y, z; z points out along the fingers): lower and upper corner.
HAND_BOX_M = ((-0.028, -0.084, -0.090), (0.029, 0.084, -0.059))
## A finger: its x and z range, metres, official hand_tcp frame, and its width across the opening (y).
FINGER_X_M, FINGER_Z_M, FINGER_WIDTH_M = (-0.030, 0.031), (-0.069, 0.016), 0.034
## How far the fingertips reach past the official grasp point, metres.
FINGER_TIP_PAST_TCP_M = FINGER_Z_M[1]
## Each finger's travel when fully open, metres (the URDF's `finger_joint1` limit).
FINGER_OPEN_M = 0.044

## The official hand_tcp frame in Dream's grasp-point frame, per arm (4x4): Dream's tcp frame looks along -z, the official one along +z.
OFFICIAL_TCP_IN_DREAM_TCP = {
    "left": np.array([[0.0, 1.0, 0.0, 0.0016], [1.0, 0.0, 0.0, 0.0001], [0.0, 0.0, -1.0, -0.0211], [0.0, 0.0, 0.0, 1.0]]),
    "right": np.array([[0.0, -1.0, 0.0, 0.0016], [-1.0, 0.0, 0.0, -0.0001], [0.0, 0.0, -1.0, -0.0211], [0.0, 0.0, 0.0, 1.0]]),
}


def _corners(lo, hi) -> list[np.ndarray]:
    return [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def boxes(opening_m: float = FINGER_OPEN_M) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """The three parts' boxes in the official hand_tcp frame, metres. @param opening_m Each finger's travel (0 closed, #FINGER_OPEN_M fully open)."""
    o = float(np.clip(opening_m, 0.0, FINGER_OPEN_M))
    return {"hand": HAND_BOX_M,
            "finger_l": ((FINGER_X_M[0], o, FINGER_Z_M[0]), (FINGER_X_M[1], o + FINGER_WIDTH_M, FINGER_Z_M[1])),
            "finger_r": ((FINGER_X_M[0], -o - FINGER_WIDTH_M, FINGER_Z_M[0]), (FINGER_X_M[1], -o, FINGER_Z_M[1]))}


def points_in_tcp_frame(side: str, opening_m: float = FINGER_OPEN_M) -> dict[str, np.ndarray]:
    """The 24 box corners (`grip:<part>:<n>`) in Dream's grasp-point frame of `side`, metres."""
    rel = OFFICIAL_TCP_IN_DREAM_TCP[side]
    out = {}
    for part, (lo, hi) in boxes(opening_m).items():
        for n, corner in enumerate(_corners(lo, hi)):
            out[f"grip:{part}:{n}"] = (rel @ np.append(corner, 1.0))[:3]
    return out


## Surface samples of the three boxes for the arm-to-arm check (log-039): this far apart, metres, each counted as a ball of #SAMPLE_RADIUS_M so that
## no point of a box surface is further than that from a sample (half the diagonal of a sample square).
SAMPLE_SPACING_M = 0.02
SAMPLE_RADIUS_M = SAMPLE_SPACING_M * np.sqrt(2.0) / 2.0 + 0.001


def surface_points(side: str, opening_m: float = FINGER_OPEN_M, spacing_m: float = SAMPLE_SPACING_M) -> np.ndarray:
    """Points on the surfaces of the hand housing and both fingers, `[n, 3]` metres in Dream's grasp-point frame of `side` (every `spacing_m`)."""
    rel = OFFICIAL_TCP_IN_DREAM_TCP[side]
    out = []
    for lo, hi in boxes(opening_m).values():
        axes = [np.linspace(lo[i], hi[i], max(2, int(np.ceil((hi[i] - lo[i]) / spacing_m)) + 1)) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        on_face = np.zeros(len(grid), dtype=bool)
        for i in range(3):
            on_face |= np.isclose(grid[:, i], lo[i]) | np.isclose(grid[:, i], hi[i])
        out.append(grid[on_face])
    pts = np.vstack(out)
    return pts @ rel[:3, :3].T + rel[:3, 3]

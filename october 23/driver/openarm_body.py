"""The robot's own body as keep-out zones for both arms, always on, whatever bundle is in force (log-039). Nothing here talks to a motor.

Measured from the official OpenArm v10 mesh of `openarm_body_link0` (`october 23/urdf`, `openarm_description`), body frame (x forward, y left, z up from the
base plate), as boxes around the mesh in height bands: the centre column, and the upper body with the shoulder hub and head mount (the base plate is left out, see #BODY_BOXES_M).
Each arm's own shoulder housing (link0, link1) is not a zone: the arm's points start at the elbow.
"""

## Each arm's base frame in the robot's body frame, metres (log-013; equal to `openarm_pair.BASE_OFFSET_M`).
BASE_OFFSET_M = {"left": (0.0, 0.031, 0.698), "right": (0.0, -0.031, 0.698)}

## The body's boxes, body frame, metres: name, lower corner, upper corner. The base plate (z 0 to 5 cm, x -15.5 to 9.5, y +-9.5 cm) is left out on purpose:
## the arms hang at rest just above it by design (the open fingers would count 1.6 cm from it), and a working arm does not reach down there.
BODY_BOXES_M = (
    ("body_column", (-0.052, -0.041, 0.050), (0.025, 0.041, 0.600)),
    ("body_top", (-0.085, -0.080, 0.600), (0.065, 0.080, 0.773)),
)


def keep_out(side: str) -> list[dict]:
    """The body boxes as keep-out zones in the arm frame of `side` (the zones file's format: `kind: box`, `center`, `size`)."""
    off = BASE_OFFSET_M[side]
    out = []
    for name, lo, hi in BODY_BOXES_M:
        out.append({"name": name, "kind": "box",
                    "center": [round((lo[i] + hi[i]) / 2.0 - off[i], 4) for i in range(3)],
                    "size": [round(hi[i] - lo[i], 4) for i in range(3)]})
    return out


## The body boxes Dream's own IK can take as `keep_out` (log-040): only the column. Dream lays its 4 cm spheres on every moving link from joint 1 on,
## about 9.4 cm from the centre line at the shoulder, inside the upper body box: Dream would see its own shoulder in it. The driver's guard keeps both.
DREAM_KEEP_OUT = ("body_column",)


def dream_keep_out(side: str) -> list[dict]:
    """The body boxes for Dream's station file, in exactly the form it takes (`kind`, `center`, `size`), in the arm frame of `side`."""
    return [{"kind": b["kind"], "center": b["center"], "size": b["size"]} for b in keep_out(side) if b["name"] in DREAM_KEEP_OUT]

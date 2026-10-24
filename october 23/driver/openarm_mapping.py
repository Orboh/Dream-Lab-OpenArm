"""The joint mapping between the fork's logical angles (the real arm, zero = hanging; the convention of the official OpenArm v10 URDF) and Dream's `openarm2_faas` model.

Dream's model angle of joint i = sign[i] * logical[i] + zero[i] (radians). Two tables:
- `identity` (the default and what has always run): the model angle is the logical angle.
- `official`: the mapping that makes Dream's model put the hand where the official v10 URDF does, over random poses (mean grasp-point residual 0.2 cm; with `identity` it is 16.6 cm).
  Left: joint 5 + 90 deg, joints 6 and 7 reversed. Right: joint 5 - 90 deg, joints 6 and 7 reversed. Found by `workspace/stage2.py mapping` (log-018).
Choose with the environment variable `OPENARM_MODEL_MAPPING=official`, or by the record `~/.dreamlabs/model_mapping` that `workspace.py apply` writes together with the station file (the station file's `home` is in
Dream's model angles, so the mapping it was written under must be the one the driver uses later, whatever shell starts the station). The variable wins over the record; neither set means `identity`, today's behaviour.
Nothing here talks to a motor.

Known limit of `official`: Dream's joint 5 range is +-90 deg, centred on ITS zero, so a quarter-turn offset leaves only half of the arm's joint 5 range inside Dream's range
(left: logical -90..0, right: logical 0..+90), and the logical zero sits at the model's limit. #outside_model_limits() lists the poses that fall outside.
"""

import math
import os
import pathlib

import numpy as np

SIDES = ("left", "right")
JOINTS = 7
ENV = "OPENARM_MODEL_MAPPING"
FILE_ENV = "OPENARM_MAPPING_FILE"  # where the record lives (tests point it elsewhere)
_HALF_PI = math.pi / 2


def _table(j5: float) -> tuple[np.ndarray, np.ndarray]:
    sign, zero = np.ones(JOINTS), np.zeros(JOINTS)
    sign[5] = sign[6] = -1.0
    zero[4] = j5
    return sign, zero


TABLES: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
    "identity": {side: (np.ones(JOINTS), np.zeros(JOINTS)) for side in SIDES},
    "official": {"left": _table(+_HALF_PI), "right": _table(-_HALF_PI)},
}


def record_path() -> pathlib.Path:
    """The file that records the mapping the station file was written under."""
    return pathlib.Path(os.environ.get(FILE_ENV) or "~/.dreamlabs/model_mapping").expanduser()


def recorded() -> str | None:
    """The mapping named by the record file, or None when there is no record. @throws ValueError The record names an unknown table."""
    try:
        name = record_path().read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        return None
    if name not in TABLES:
        raise ValueError(f"{record_path()}: {name!r} is not one of {sorted(TABLES)}")
    return name


def write_record(name: str) -> pathlib.Path:
    """Records `name` (a table) as the mapping the station file is written under. @throws ValueError Unknown table."""
    if name not in TABLES:
        raise ValueError(f"unknown mapping {name!r}: choose one of {sorted(TABLES)}")
    path = record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name + "\n", encoding="utf-8")
    return path


def active() -> str:
    """The mapping in use: the environment variable if set, else the record file, else `identity`. @throws ValueError The name is not a table."""
    name = os.environ.get(ENV, "").strip() or recorded() or "identity"
    if name not in TABLES:
        raise ValueError(f"{ENV}={name!r}: choose one of {sorted(TABLES)}")
    return name


## Dream's joint limits per side, degrees (equal to the fork's; the driver's LIMITS_DEG is tested against this).
MODEL_LIMITS_DEG = {
    "left": [(-200.0, 80.0), (-190.0, 10.0), (-90.0, 90.0), (0.0, 140.0), (-90.0, 90.0), (-45.0, 45.0), (-90.0, 90.0)],
    "right": [(-80.0, 200.0), (-10.0, 190.0), (-90.0, 90.0), (0.0, 140.0), (-90.0, 90.0), (-45.0, 45.0), (-90.0, 90.0)],
}


def clip_to_model_limits(side: str, model_rad, margin_deg: float = 0.5) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Moves model angles that are outside Dream's joint limits (or closer than `margin_deg` to one) to `margin_deg` inside it.

    @return `(clipped model angles in radians, [(joint number, degrees moved)])`; nothing moves for a pose that fits.
    """
    model = np.degrees(np.asarray(model_rad, dtype=float))
    out, moved = model.copy(), []
    for i, (low, high) in enumerate(MODEL_LIMITS_DEG[side]):
        inside = min(max(model[i], low + margin_deg), high - margin_deg)
        if abs(inside - model[i]) > 1e-9:
            out[i] = inside
            moved.append((i + 1, round(float(inside - model[i]), 2)))
    return (np.radians(out) if moved else np.asarray(model_rad, dtype=float)), moved  # unchanged exactly when nothing moves (no degree round trip)


def get(side: str, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`(sign, zero_rad)` of one arm. @param name A table name, or None for the environment's."""
    return TABLES[name or active()][side]


def to_model(side: str, logical_rad, name: str | None = None) -> np.ndarray:
    """Logical joint angles (radians, shape `[..., 7]`) to Dream's model angles."""
    sign, zero = get(side, name)
    return sign * np.asarray(logical_rad, dtype=float) + zero


def to_logical(side: str, model_rad, name: str | None = None) -> np.ndarray:
    """Dream's model angles (radians, shape `[..., 7]`) back to logical angles (the sign is +-1, so it is its own inverse)."""
    sign, zero = get(side, name)
    return sign * (np.asarray(model_rad, dtype=float) - zero)


def outside_model_limits(side: str, logical_deg, limits_deg: dict[str, tuple[float, float]], name: str | None = None) -> list[tuple[int, float]]:
    """Joints whose Dream model angle falls outside the model's limits for a pose.

    @param limits_deg `{"joint_1": (low, high), ...}` in degrees (the model's limits equal the fork's: log-010).
    @return `(joint number, degrees outside)` for each such joint, empty when the pose fits.
    """
    model = np.degrees(to_model(side, np.radians(np.asarray(logical_deg, dtype=float)), name))
    out = []
    for i in range(JOINTS):
        low, high = limits_deg[f"joint_{i + 1}"]
        if model[i] < low - 1e-9:
            out.append((i + 1, round(float(low - model[i]), 2)))
        elif model[i] > high + 1e-9:
            out.append((i + 1, round(float(model[i] - high), 2)))
    return out

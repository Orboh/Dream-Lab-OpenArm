"""Writes the recorder's start pose files (`start_pose/left.yaml`, `right.yaml`) from the station: every episode starts where the Live View starts.

The start pose is the end of the taught rest-to-home path in the zones in force (`~/.dreamlabs/zones.yaml`, logical degrees),
which the driver checks equals the station's home; the waypoints are that path before it, the same way the driver raises the arm.
The earlier wood recordings started at pose set kitakyushu-takeei-v5, 20 to 47 deg away from today's home in joints 1, 4 and 7 (log-001).
Nothing here talks to a motor.

Run:  env -u PYTHONPATH ../../.venv/bin/python make_start_pose.py
"""

import pathlib
import sys

import numpy as np
import yaml

import config

sys.path.insert(0, str(config.DRIVER_DIR))
import openarm_guard  # noqa: E402
import openarm_mapping as mapping  # noqa: E402
import openarm_zones as zones  # noqa: E402

OUT_DIR = pathlib.Path(__file__).resolve().parent / "start_pose"
## The station's home and the taught path's end must agree to this, degrees (the driver's own `HOME_MATCH_DEG`).
HOME_MATCH_DEG = mapping.HOME_MATCH_DEG
## The recorder's alignment speed, degrees per second (as in the earlier pose sets).
ALIGN_MAX_SPEED_DPS = 40.0


def _joints(q_deg) -> dict[str, float]:
    return {f"joint_{i + 1}": round(float(v), 2) for i, v in enumerate(q_deg)}


def pose_file(side: str, path_deg: list[list[float]], home_logical_deg: np.ndarray) -> str:
    """The recorder's pose file text for one arm: the taught path as waypoints, its end as the start pose, gripper at the calibration zero."""
    gap = float(np.max(np.abs(np.asarray(path_deg[-1]) - home_logical_deg)))
    if gap > HOME_MATCH_DEG:
        raise ValueError(f"arm {side}: the taught path ends {gap:.1f} deg from the station's home; refresh the zones first")
    body = {"align": {"max_speed_dps": ALIGN_MAX_SPEED_DPS}, "waypoints": [_joints(p) for p in path_deg[:-1]],
            **_joints(path_deg[-1]), "gripper": 0.0}
    head = (f"# OpenArm start pose -- side={side}, degrees (logical, follower). Written by Finetuning dreamlab/pipeline/make_start_pose.py.\n"
            f"# Start pose = the station's home (the end of the taught path in zones.yaml, where the Live View starts). Waypoints = that path (the driver's raise).\n"
            f"# The gripper at the calibration zero (the same rule as the earlier pose sets).\n")
    return head + yaml.safe_dump(body, sort_keys=False, allow_unicode=True)


def main() -> None:
    """Writes both files and prints the start pose of each arm."""
    station = yaml.safe_load(config.STATION_FILE.read_text())
    taught = zones.load_zones(openarm_guard.zones_path()).home_path or {}
    OUT_DIR.mkdir(exist_ok=True)
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        if side not in taught:
            sys.exit(f"no taught path for arm {side} in {openarm_guard.zones_path()}")
        home = np.degrees(mapping.to_logical(side, station[key]["home"], config.MAPPING))
        (OUT_DIR / f"{side}.yaml").write_text(pose_file(side, taught[side], home))
        print(f"{side}: start {np.round(taught[side][-1], 1).tolist()} deg, {len(taught[side]) - 1} waypoints -> {OUT_DIR / (side + '.yaml')}")


if __name__ == "__main__":
    main()

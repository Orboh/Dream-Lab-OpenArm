"""Write a small synthetic LeRobot v3.0 demo dataset, so the pipeline runs before real demos exist.

The motion is a gentle reach-down / close-grippers / lift with both arms, generated
in canonical units from the sim home pose and written back in robot units.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from orboh.robot.mapping import JointMapping
from orboh.types import NUM_JOINTS, joint_index, min_jerk

FPS = 30


def _bimanual_grasp(home: np.ndarray, reach: float) -> np.ndarray:
    open_ = home.copy()
    open_[[joint_index("left_gripper"), joint_index("right_gripper")]] = 1.0
    down = open_.copy()
    for side, s in (("left", 1), ("right", -1)):
        down[joint_index(f"{side}_joint1")] += -s * reach * 0.5
        down[joint_index(f"{side}_joint4")] -= reach
    closed = down.copy()
    closed[[joint_index("left_gripper"), joint_index("right_gripper")]] = 0.2
    lift = closed.copy()
    for side in ("left", "right"):
        lift[joint_index(f"{side}_joint4")] += reach * 0.6
    keys = [home, open_, down, closed, lift, home]
    seconds = [1.0, 2.5, 1.0, 2.0, 2.5]
    parts = [min_jerk(a, b, int(s * FPS))[:-1] for a, b, s in zip(keys, keys[1:], seconds)]
    return np.vstack(parts + [keys[-1][None]])


def write_sample_dataset(root: str | Path, mapping: JointMapping, home: np.ndarray, episodes: int = 3) -> Path:
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    names = mapping.keys
    frames, ep_rows, offset = [], [], 0
    tasks = ["pick up the wooden block with both arms", "lift the box with both arms", "grasp the bottle"]
    for ep in range(episodes):
        act = _bimanual_grasp(home, reach=0.25 + 0.1 * ep)
        state = np.vstack([act[:1], act[:-1]])  # follower lags leader by one frame
        for t in range(len(act)):
            frames.append({
                "action": [mapping.from_canonical(act[t])[k] for k in names],
                "observation.state": [mapping.from_canonical(state[t])[k] for k in names],
                "timestamp": t / FPS, "frame_index": t, "episode_index": ep,
                "index": offset + t, "task_index": ep % len(tasks),
            })
        ep_rows.append({
            "episode_index": ep, "tasks": [tasks[ep % len(tasks)]], "length": len(act),
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": offset, "dataset_to_index": offset + len(act),
        })
        offset += len(act)
    pq.write_table(pa.Table.from_pylist(frames), root / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pylist(ep_rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = {
        "codebase_version": "v3.0", "robot_type": "bi_openarm_follower", "fps": FPS,
        "total_episodes": episodes, "total_frames": offset,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [NUM_JOINTS], "names": names},
            "observation.state": {"dtype": "float32", "shape": [NUM_JOINTS], "names": names},
        },
        "orboh_synthetic": True,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return root

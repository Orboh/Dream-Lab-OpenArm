"""Read teleop demonstrations from a LeRobot Dataset v3.0 directory.

Layout (v3.0):
  meta/info.json                         fps, features, data_path / video_path templates
  meta/episodes/chunk-XXX/file-XXX.parquet   one row per episode
  data/chunk-XXX/file-XXX.parquet        frames of many episodes (episode_index column)
  videos/<camera key>/chunk-XXX/file-XXX.mp4
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from orboh.robot.mapping import JointMapping
from orboh.types import Trajectory


@dataclass
class DemoEpisode:
    index: int
    task: str
    trajectory: Trajectory  # leader (teleop) actions, canonical
    states: np.ndarray | None  # follower observation.state, canonical, (T, 16)
    videos: dict[str, dict] = field(default_factory=dict)  # camera key -> {path, from_s, to_s}


class LeRobotDemoDataset:
    def __init__(self, root: str | Path, mapping: JointMapping):
        self.root = Path(root)
        self.mapping = mapping
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"no LeRobot dataset at {self.root} (missing meta/info.json)")
        self.info = json.loads(info_path.read_text())
        self.fps = float(self.info["fps"])
        feats = self.info.get("features", {})
        self.action_names = feats.get("action", {}).get("names")
        self.state_names = feats.get("observation.state", {}).get("names")
        self.video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
        if not self.action_names:
            raise ValueError("dataset has no action feature names; cannot map joints")

    def _episode_rows(self) -> list[dict]:
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        rows: list[dict] = []
        for f in files:
            rows += pq.read_table(f).to_pylist()
        return sorted(rows, key=lambda r: r["episode_index"])

    def list_episodes(self) -> list[dict]:
        out = []
        for r in self._episode_rows():
            tasks = r.get("tasks") or []
            out.append({
                "index": int(r["episode_index"]),
                "task": tasks[0] if tasks else "",
                "length": int(r.get("length", 0)),
                "duration_s": round(int(r.get("length", 0)) / self.fps, 1),
            })
        return out

    def load(self, index: int) -> DemoEpisode:
        row = next((r for r in self._episode_rows() if int(r["episode_index"]) == index), None)
        if row is None:
            raise KeyError(f"episode {index} not found")
        data_file = self.root / self.info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        table = pq.read_table(data_file)
        table = table.filter(pc.equal(table["episode_index"], index))
        if "frame_index" in table.column_names:
            table = table.sort_by("frame_index")
        actions = np.asarray(table["action"].to_pylist(), dtype=float)
        traj = np.stack([self.mapping.vector_to_canonical(a, self.action_names) for a in actions])
        states = None
        if self.state_names and "observation.state" in table.column_names:
            raw = np.asarray(table["observation.state"].to_pylist(), dtype=float)
            states = np.stack([self.mapping.vector_to_canonical(s, self.state_names) for s in raw])
        videos = {}
        for key in self.video_keys:
            prefix = f"videos/{key}/"
            if prefix + "chunk_index" not in row:
                continue
            path = self.root / self.info["video_path"].format(
                video_key=key, chunk_index=int(row[prefix + "chunk_index"]), file_index=int(row[prefix + "file_index"]))
            videos[key] = {"path": str(path), "from_s": row.get(prefix + "from_timestamp", 0.0),
                           "to_s": row.get(prefix + "to_timestamp")}
        tasks = row.get("tasks") or []
        return DemoEpisode(index, tasks[0] if tasks else "", Trajectory(traj, self.fps), states, videos)

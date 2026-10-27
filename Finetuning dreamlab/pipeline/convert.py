"""Converts our recordings (LeRobot v3.0, `lerobot-record`) into sessions Dream can fine-tune on (LeRobot v2.1 in the body `koutaueda/openarm2-faas`).

Per frame, our 16 values (right then left: 7 joints and the gripper, logical degrees) become:
- joints: Dream's model radians through the driver's own mapping (`openarm_mapping`, `official`), the same as `OpenArm.read()`;
- grip: 0 closed to 1 open from the station's gripper ends (`gripper_closed_deg`, `gripper_open_deg`), clipped like the driver;
- gripper drive: grip times the jaw's open angle (`dream_ik.specs` `openarm2_faas`: left +0.785, right -0.785 rad);
- hand pose: forward kinematics with Dream's own arm model and tool point (`dream_ik.solver.ChunkIK.fk`), the code that
  computes the poses at run time, so every pose is in the training frame; rot6 is the first two columns of the rotation.
The state is the measured follower, the action the leader's command, each converted the same way.

Per episode, every camera's frames are cut out of the v3.0 file (one MP4 for all episodes) and written as one H.264 MP4,
frame for frame with the parquet. The chest camera becomes the body's head camera (`observation.images.head_left`).

Run:  env -u PYTHONPATH ../../.venv/bin/python convert.py <recording folder> [...]   (default: every folder in dataset/raw)
"""

import argparse
import fractions
import json
import pathlib
import re
import shutil
import sys
from collections.abc import Callable

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

import config

sys.path.insert(0, str(config.DRIVER_DIR))
import openarm_mapping as mapping  # noqa: E402

## Our recorder's vector: right then left, each 7 joints then the gripper.
SIDES = ("right", "left")
JOINTS = 7
## A session folder name Dream accepts (`validation.SESSION_NAME_RE`).
SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
## The names written into `info.json`, one per value, in the body's order.
STATE_NAMES = ([f"right_arm_{i}" for i in range(1, 8)] + ["right_gripper"] + [f"left_arm_{i}" for i in range(1, 8)] + ["left_gripper"]
               + [f"right_eef_{a}" for a in ("x", "y", "z")] + [f"right_eef_rot6_{i}" for i in range(6)] + ["right_grip"]
               + [f"left_eef_{a}" for a in ("x", "y", "z")] + [f"left_eef_rot6_{i}" for i in range(6)] + ["left_grip"])
ACTION_NAMES = ([f"right_eef_{a}" for a in ("x", "y", "z")] + [f"right_eef_rot6_{i}" for i in range(6)] + ["right_grip"]
                + [f"left_eef_{a}" for a in ("x", "y", "z")] + [f"left_eef_rot6_{i}" for i in range(6)] + ["left_grip"]
                + [f"left_arm_{i}" for i in range(1, 8)] + [f"right_arm_{i}" for i in range(1, 8)])

Fk = Callable[[np.ndarray, str], np.ndarray]


def dream_fk() -> Fk:
    """Dream's own forward kinematics of `openarm2_faas`: `(q [n, 7] model radians, side) -> [n, 4, 4]` tool poses in that arm's base frame."""
    from dream_ik.solver import ChunkIK
    from dream_ik.specs import SERVE

    ik = ChunkIK(config.IK_ARM, SERVE.tool, float(config.FPS))
    return lambda q, side: ik.fk(np.asarray(q, dtype=np.float64), side)


def gripper_ends(station_file: pathlib.Path = config.STATION_FILE) -> dict[str, tuple[float, float]]:
    """Each arm's gripper degrees `(closed, open)` from the station file, the ends the driver reads the grip with."""
    station = yaml.safe_load(station_file.read_text())
    out = {}
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        arm = station[key]
        out[side] = (float(arm["gripper_closed_deg"]), float(arm["gripper_open_deg"]))
    return out


def split_sides(rows_deg: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`[n, 16]` recorder rows to `{side: (joints [n, 7] deg, gripper [n] deg)}`."""
    rows_deg = np.asarray(rows_deg, dtype=np.float64)
    out = {}
    for i, side in enumerate(SIDES):
        start = i * (JOINTS + 1)
        out[side] = (rows_deg[:, start:start + JOINTS], rows_deg[:, start + JOINTS])
    return out


def side_parts(joints_deg: np.ndarray, gripper_deg: np.ndarray, side: str, ends: tuple[float, float], fk: Fk) -> dict[str, np.ndarray]:
    """One arm's parts in Dream's units: `arm` [n, 7] rad, `gripper` [n, 1] jaw drive rad, `grip` [n, 1] 0..1, `xyz` [n, 3] m, `rot6` [n, 6]."""
    q = mapping.to_model(side, np.radians(joints_deg), config.MAPPING)
    closed, opened = ends
    grip = np.clip((np.asarray(gripper_deg) - closed) / (opened - closed), 0.0, 1.0)[:, None]
    poses = fk(q, side)
    rot = poses[:, :3, :3]
    return {"arm": q, "gripper": grip * config.JAW_OPEN_RAD[side], "grip": grip, "xyz": poses[:, :3, 3],
            "rot6": np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=1)}


def to_body(rows_deg: np.ndarray, ends: dict[str, tuple[float, float]], fk: Fk, kind: str) -> np.ndarray:
    """Recorder rows `[n, 16]` (degrees) to the body's `state` `[n, 36]` or `action` `[n, 34]` (`kind`), float32, in the spans of #config.EMBODIMENT_CONFIG."""
    parts = {side: side_parts(j, g, side, ends[side], fk) for side, (j, g) in split_sides(rows_deg).items()}
    spans = config.EMBODIMENT_CONFIG[kind]
    out = np.zeros((len(rows_deg), max(e for _, e in spans.values())), dtype=np.float32)
    for name, (start, end) in spans.items():
        side, part = name.split("_", 1)
        out[:, start:end] = parts[side][part.replace("eef_", "")]
    return out


def session_name(source: pathlib.Path) -> str:
    """The session folder's name: the recording's own, cut from the front to Dream's 64 characters when longer."""
    name = source.name[-64:].lstrip("._-")
    if not SESSION_NAME_RE.match(name):
        raise ValueError(f"{source.name}: no session name Dream accepts can be made from it")
    return name


def _episodes(src: pathlib.Path) -> list[dict]:
    files = sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    rows = [r for f in files for r in pq.read_table(f).to_pylist()]
    return sorted(rows, key=lambda r: r["episode_index"])


def _data(src: pathlib.Path) -> pa.Table:
    files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    return pa.concat_tables([pq.read_table(f) for f in files])


def cut_video(src_file: pathlib.Path, start_s: float, frames: int, fps: int, out_file: pathlib.Path) -> int:
    """Writes `frames` frames from `start_s` of `src_file` as an H.264 MP4 at `fps`; returns the frames written (fewer only when the source ends)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(src_file)) as inp, av.open(str(out_file), "w") as out:
        stream = inp.streams.video[0]
        enc = out.add_stream("libx264", rate=fps)
        enc.width, enc.height, enc.pix_fmt = stream.codec_context.width, stream.codec_context.height, "yuv420p"
        enc.options = {"crf": "20", "g": "2"}
        inp.seek(max(0, int((start_s - 1.0) / stream.time_base)), stream=stream, backward=True)
        half = 0.5 / fps
        for frame in inp.decode(stream):
            if frame.time is None or frame.time < start_s - half:
                continue
            new = frame.reformat(format="yuv420p")
            new.pts, new.time_base = written, fractions.Fraction(1, fps)
            out.mux(enc.encode(new))
            written += 1
            if written == frames:
                break
        out.mux(enc.encode(None))
    return written


def convert(src: pathlib.Path, dst_root: pathlib.Path = config.DREAM_DIR, fk: Fk | None = None,
            ends: dict[str, tuple[float, float]] | None = None, videos: bool = True) -> pathlib.Path:
    """Converts one recording folder into one Dream session under `dst_root`; returns the session folder.

    @param src A LeRobot v3.0 recording of `bi_openarm_follower` (16 values, cameras chest/wrist_right/wrist_left).
    @param fk Forward kinematics; Dream's own when None.
    @param ends Gripper ends per arm; the station file's when None.
    @param videos False writes the tables only (for tests).
    """
    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{src}: codebase_version {info.get('codebase_version')!r}, expected v3.0")
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [2 * (JOINTS + 1)]:
            raise ValueError(f"{src}: {key} has shape {info['features'][key]['shape']}, expected [16]")
    cams = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    if videos and sorted(config.CAMERA_KEYS) != sorted(cams):
        raise ValueError(f"{src}: cameras {cams}, expected {sorted(config.CAMERA_KEYS)}")
    fk = fk or dream_fk()
    ends = ends or gripper_ends()
    fps = int(info["fps"])
    episodes = _episodes(src)
    if not episodes:
        raise ValueError(f"{src}: no episodes")
    data = _data(src)
    dst = dst_root / session_name(src)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    tasks: dict[str, int] = {}
    ep_lines, total, short = [], 0, []
    for new_index, ep in enumerate(episodes):
        rows = data.filter(pc.equal(data["episode_index"], ep["episode_index"]))
        if rows.num_rows == 0:
            raise ValueError(f"{src}: episode {ep['episode_index']} has no rows")
        n = rows.num_rows
        state = to_body(np.stack(rows["observation.state"].to_numpy(zero_copy_only=False)), ends, fk, "state")
        action = to_body(np.stack(rows["action"].to_numpy(zero_copy_only=False)), ends, fk, "action")
        task = ep["tasks"][0]
        task_index = tasks.setdefault(task, len(tasks))
        table = pa.table({
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), state.shape[1])),
            "action": pa.array(list(action), type=pa.list_(pa.float32(), action.shape[1])),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / fps),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, new_index, dtype=np.int64)),
            "index": pa.array(np.arange(total, total + n, dtype=np.int64)),
            "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        })
        out = dst / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        if videos:
            for key in cams:
                start = float(ep[f"videos/{key}/from_timestamp"])
                file = src / info["video_path"].format(video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"],
                                                       file_index=ep[f"videos/{key}/file_index"])
                got = cut_video(file, start, n, fps, dst / "videos" / "chunk-000" / config.CAMERA_KEYS[key] / f"episode_{new_index:06d}.mp4")
                if got < n - 1:
                    raise ValueError(f"{src}: episode {ep['episode_index']} {key}: {got} frames for {n} rows")
        if n < 48 * fps // 30 + 1:
            short.append(ep["episode_index"])
        ep_lines.append({"episode_index": new_index, "tasks": [task], "length": n, "source_episode_index": ep["episode_index"]})
        total += n

    features = {
        "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if videos:
        for key in cams:
            h, w, c = info["features"][key]["shape"]
            features[config.CAMERA_KEYS[key]] = {
                "dtype": "video", "shape": [h, w, c], "names": ["height", "width", "channels"],
                "info": {"video.height": h, "video.width": w, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                         "video.is_depth_map": False, "video.fps": fps, "video.channels": c, "has_audio": False}}
    new_info = {
        "codebase_version": "v2.1", "robot_type": info.get("robot_type", "bi_openarm_follower"),
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "total_videos": len(episodes) * (len(cams) if videos else 0), "total_chunks": 1, "chunks_size": 1000, "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    (dst / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_lines))
    (dst / "meta" / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items()))
    (dst / "meta" / "source.json").write_text(json.dumps({
        "source": str(src.resolve()), "source_codebase_version": "v3.0", "mapping": config.MAPPING, "gripper_ends_deg": ends,
        "ik_arm": config.IK_ARM, "jaw_open_rad": config.JAW_OPEN_RAD, "short_episodes": short}, indent=2))
    return dst


def main() -> None:
    """Converts the folders named, or every recording in `dataset/raw`."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="*", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=config.DREAM_DIR)
    args = ap.parse_args()
    sources = args.sources or sorted(p for p in config.RAW_DIR.iterdir() if (p / "meta" / "info.json").exists())
    if not sources:
        sys.exit(f"nothing to convert: no recording in {config.RAW_DIR}")
    fk = dream_fk()
    for src in sources:
        dst = convert(src, args.out, fk=fk)
        n = json.loads((dst / "meta" / "info.json").read_text())["total_episodes"]
        print(f"{src.name}: {n} episodes -> {dst}")


if __name__ == "__main__":
    main()

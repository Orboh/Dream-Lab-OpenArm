"""Tests for the fine-tuning pipeline's converter and start pose. No hardware, no Dream service.

Run from this folder:  env -u PYTHONPATH ../../.venv/bin/python -m pytest -q test_pipeline.py
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import config
import convert
import make_start_pose

ENDS = {"left": (0.9, -66.0), "right": (0.2, -64.8)}      # the station's gripper ends (2026-10-27)


def fake_fk(q, side):
    """A stand-in for Dream's FK: the position is the first three joints, the rotation the identity (1 for the left arm's x, -1 for the right's)."""
    q = np.asarray(q)
    out = np.tile(np.eye(4), (len(q), 1, 1))
    out[:, :3, 3] = q[:, :3]
    out[:, 0, 0] = 1.0 if side == "left" else -1.0
    return out


def _row(right_deg, right_grip_deg, left_deg, left_grip_deg) -> np.ndarray:
    return np.array([[*right_deg, right_grip_deg, *left_deg, left_grip_deg]], dtype=np.float64)


def test_joints_go_through_the_drivers_official_mapping():
    """Joint 5 shifts by -90 (right) / +90 deg (left), joints 6 and 7 turn the other way: Dream's model angles, as the driver sends them."""
    row = _row([10, 20, 30, 40, 5, 6, 7], 0.2, [-10, -20, -30, 40, -5, -6, -7], 0.9)
    state = convert.to_body(row, ENDS, fake_fk, "state")[0]
    right, left = np.degrees(state[0:7]), np.degrees(state[8:15])
    assert right == pytest.approx([10, 20, 30, 40, 5 - 90, -6, -7], abs=1e-4)
    assert left == pytest.approx([-10, -20, -30, 40, -5 + 90, 6, 7], abs=1e-4)


def test_grip_is_0_closed_to_1_open_and_the_drive_turns_each_jaw_its_own_way():
    row = _row([0] * 7, -64.8, [0] * 7, (0.9 - 66.0) / 2)
    state = convert.to_body(row, ENDS, fake_fk, "state")[0]
    assert state[25] == pytest.approx(1.0) and state[35] == pytest.approx(0.5)            # right_grip, left_grip
    assert state[7] == pytest.approx(-0.785) and state[15] == pytest.approx(0.3925)      # right_gripper, left_gripper (jaw drive, rad)


def test_grip_past_the_ends_is_clipped_like_the_driver():
    row = _row([0] * 7, 5.0, [0] * 7, -80.0)
    state = convert.to_body(row, ENDS, fake_fk, "state")[0]
    assert state[25] == 0.0 and state[35] == 1.0


def test_every_part_lands_in_its_span_of_the_registered_body():
    row = _row([10, 20, 30, 0, 0, 0, 0], 0.2, [1, 2, 3, 0, 0, 0, 0], 0.9)
    state = convert.to_body(row, ENDS, fake_fk, "state")[0]
    action = convert.to_body(row, ENDS, fake_fk, "action")[0]
    assert state.shape == (36,) and action.shape == (34,)
    assert state[16:19] == pytest.approx(np.radians([10, 20, 30]))      # right_eef_xyz (fake FK: the first three joints)
    assert state[26:29] == pytest.approx(np.radians([1, 2, 3]))         # left_eef_xyz
    assert state[19] == -1.0 and state[29] == 1.0                       # rot6 starts with the first column
    assert action[0:3] == pytest.approx(state[16:19]) and action[10:13] == pytest.approx(state[26:29])
    assert action[20:27] == pytest.approx(state[8:15]) and action[27:34] == pytest.approx(state[0:7])   # left_arm, then right_arm


def _v3_recording(root, lengths, fps=15):
    """A LeRobot v3.0 recording on disk, tables only: one data file and one episodes file, as `lerobot-record` writes them."""
    src = root / "bi_openarm_dream_wood_block_15fps_20261028_100000"
    (src / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (src / "data" / "chunk-000").mkdir(parents=True)
    n = sum(lengths)
    vec = pa.list_(pa.float32(), 16)
    rows = np.zeros((n, 16), dtype=np.float32)
    rows[:, 0] = np.arange(n)                                            # right joint 1 counts the frames, to follow them
    pq.write_table(pa.table({
        "action": pa.array(list(rows), type=vec), "observation.state": pa.array(list(rows), type=vec),
        "timestamp": pa.array(np.concatenate([np.arange(k) / fps for k in lengths]).astype(np.float32)),
        "frame_index": pa.array(np.concatenate([np.arange(k) for k in lengths])),
        "episode_index": pa.array(np.repeat(np.arange(len(lengths)), lengths)),
        "index": pa.array(np.arange(n)), "task_index": pa.array(np.zeros(n, dtype=np.int64))}),
        src / "data" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"episode_index": list(range(len(lengths))), "tasks": [[config.TASK]] * len(lengths), "length": lengths}),
                   src / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    feat = {"dtype": "float32", "shape": [16], "names": [f"v{i}" for i in range(16)]}
    (src / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": fps, "robot_type": "bi_openarm_follower",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"action": feat, "observation.state": feat}}))
    return src


def test_a_v3_recording_becomes_a_v21_session_dreams_table_checks_accept(tmp_path):
    from shared.datasets import validation

    src = _v3_recording(tmp_path, [40, 30])
    dst = convert.convert(src, tmp_path / "dream", fk=fake_fk, ends=ENDS, videos=False)
    validation.check_tables(dst.name, dst)                               # Dream's own checks of meta and parquets
    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 2 and info["total_frames"] == 70 and info["fps"] == 15
    second = pq.read_table(dst / "data" / "chunk-000" / "episode_000001.parquet")
    assert second["index"].to_pylist()[0] == 40 and second["frame_index"].to_pylist()[0] == 0
    first_state = second["observation.state"].to_pylist()[0]
    assert np.degrees(first_state[0]) == pytest.approx(40, abs=1e-3)    # the episode's own rows, in order
    assert json.loads((dst / "meta" / "tasks.jsonl").read_text()) == {"task_index": 0, "task": config.TASK}


def test_an_episode_shorter_than_one_training_sample_is_listed(tmp_path):
    """Dream refuses an episode under 1.6 s (25 frames at 15 fps); the converter names it in source.json before the upload does."""
    src = _v3_recording(tmp_path, [40, 20])
    dst = convert.convert(src, tmp_path / "dream", fk=fake_fk, ends=ENDS, videos=False)
    assert json.loads((dst / "meta" / "source.json").read_text())["short_episodes"] == [1]


def test_a_recording_of_another_shape_is_refused(tmp_path):
    src = _v3_recording(tmp_path, [40])
    info = json.loads((src / "meta" / "info.json").read_text())
    info["features"]["action"]["shape"] = [14]
    (src / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="expected \\[16\\]"):
        convert.convert(src, tmp_path / "dream", fk=fake_fk, ends=ENDS, videos=False)


def test_the_start_pose_must_be_the_stations_home():
    path = [[0.0] * 7, [-7.4, 0.4, 7.2, 75.2, 1.0, 4.5, 19.1]]
    text = make_start_pose.pose_file("right", path, np.array(path[-1]))
    assert "joint_4: 75.2" in text and "gripper: 0.0" in text
    with pytest.raises(ValueError, match="from the station's home"):
        make_start_pose.pose_file("right", path, np.array(path[-1]) + [0, 0, 0, 5.0, 0, 0, 0])

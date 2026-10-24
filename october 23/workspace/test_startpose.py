"""Tests for handpose.py and startpose.py: how level the hand is, and the start-pose set written from the arms' pose."""

import math
import pathlib
import subprocess

import pytest

import handpose
import startpose

BASE_YAML = """# OpenArm start pose -- side=left
align:
  max_speed_dps: 40.0
waypoints:
  - joint_1: 13.84
    joint_2: -0.86
    joint_3: -1.27
    joint_4: 25.55
    joint_5: 0.74
    joint_6: 1.79
    joint_7: 1.04
joint_1: 64.42
joint_2: -3.62
joint_3: -1.23
joint_4: 95.91
joint_5: -2.48
joint_6: 19.01
joint_7: 21.6   # settles here
gripper: 0.0
"""


def rad(values):
    return [math.radians(v) for v in values]


def test_hand_pitch_hanging_level_and_down():
    assert handpose.approach_elevation_deg("left", [0.0] * 7) == pytest.approx(-90.0, abs=0.5)  # a hanging arm points straight down
    v2 = rad([64.42, -3.62, -1.23, 95.91, -2.48, 19.01, -68.77])
    wood = rad([64.42, -3.62, -1.23, 95.91, -2.48, 19.01, 21.6])
    assert handpose.approach_elevation_deg("left", v2) == pytest.approx(11.5, abs=0.5)
    assert handpose.approach_elevation_deg("left", wood) == pytest.approx(-69.1, abs=0.5)  # the start pose that looks at the floor


def test_pose_text_replaces_only_the_final_joints_and_keeps_the_rest():
    out = startpose.pose_text(BASE_YAML, [10, 20, 30, 40, 50, 60, 70], "note")
    assert out.startswith("# note\n# OpenArm start pose")
    assert "joint_7: 70.00" in out and "joint_1: 10.00" in out and "gripper: 0.0" in out
    assert "  - joint_1: 13.84" in out and "    joint_7: 1.04" in out  # the waypoint is untouched


def test_next_name_counts_up_and_skips_taken(tmp_path):
    (tmp_path / "config" / "pose_sets" / "x-v2").mkdir(parents=True)
    assert startpose.next_name("x-v1", tmp_path) == "x-v3"
    assert startpose.next_name("plain", tmp_path) == "plain-v2"


def make_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    base = tmp_path / "config" / "pose_sets" / "base-v1"
    base.mkdir(parents=True)
    for side in startpose.SIDES:
        (base / f"{side}.yaml").write_text(BASE_YAML)
    (base / "set.yaml").write_text("name: base-v1\nstatus: frozen\nscene: some-scene\nmirror: false\n")
    return tmp_path


def test_save_runs_pose_set_new_with_the_new_poses(tmp_path):
    repo, seen = make_repo(tmp_path), {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        tmp = pathlib.Path(argv[argv.index("--poses-dir") + 1])
        seen["left"] = (tmp / "left.yaml").read_text()
        return subprocess.CompletedProcess(argv, 0, stdout="[pose-set] created", stderr="")

    joints = {"left": [1, 2, 3, 4, 5, 6, 7], "right": [7, 6, 5, 4, 3, 2, 1]}
    assert startpose.save("base-v2", "base-v1", joints, repo, runner=fake_run) == "[pose-set] created"
    assert seen["argv"][2:4] == ["new", "base-v2"]
    assert "--scene" in seen["argv"] and "some-scene" in seen["argv"] and "--mirror" in seen["argv"]
    assert "joint_7: 7.00" in seen["left"] and "saved from" not in seen["left"] and "エディタ画面から保存" in seen["left"]


@pytest.mark.parametrize("name,base,code", [("bad name!", "base-v1", "start.err.name"), ("base-v2", "nope", "start.err.base"), ("base-v1", "base-v1", "start.err.exists")])
def test_save_refuses_bad_names_missing_base_and_existing_set(tmp_path, name, base, code):
    repo = make_repo(tmp_path)
    with pytest.raises(startpose.StartPoseError, match=code):
        startpose.save(name, base, {"left": [0] * 7, "right": [0] * 7}, repo, runner=lambda *a, **k: None)


def test_save_reports_a_failing_tool(tmp_path):
    repo = make_repo(tmp_path)
    failing = lambda argv, **k: subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")  # noqa: E731
    with pytest.raises(startpose.StartPoseError, match="start.err.tool.*boom"):
        startpose.save("base-v2", "base-v1", {"left": [0] * 7, "right": [0] * 7}, repo, runner=failing)

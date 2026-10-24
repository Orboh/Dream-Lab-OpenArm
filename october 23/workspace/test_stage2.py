"""Tests for stage2.py: the two models' disagreement, the mapping search and the physical-test verdict (on measurements invented from each model)."""

import math
import pathlib

import numpy as np
import pytest

import stage2

pytestmark = pytest.mark.skipif(not pathlib.Path(stage2.DREAM_URDF.format(side="left")).expanduser().exists(), reason="Dream's dream_ik assets are not installed here")


@pytest.fixture(scope="module")
def models():
    return stage2.Models()


def test_hanging_arm_agrees_but_the_start_pose_does_not(models):
    hanging = stage2.compare(models, *stage2.POSES["rest (left)"])
    start = stage2.compare(models, *stage2.POSES["start v2 (left)"])
    assert hanging["gap_cm"] < 5  # only the 2 cm tool-length difference
    assert start["gap_cm"] > 15  # the models put the hand 20+ cm apart for the same joint values
    assert start["dream_pitch_deg"] < 0 < start["v10_pitch_deg"]


def test_mapping_search_finds_the_joint5_quarter_turn_and_not_identity(models):
    left = stage2.mapping(models, "left", joints=(4,), top=2, n=12)
    right = stage2.mapping(models, "right", joints=(4,), top=2, n=12)
    assert left[0]["identity"]["mean_cm"] > 10 and left[1]["mean_cm"] < 1 and left[1]["offsets_deg"] == [-90]
    assert right[0]["identity"]["mean_cm"] > 10 and right[1]["mean_cm"] < 1 and right[1]["offsets_deg"] == [90]


def entries_from(models, side, hypothesis):
    poses = [stage2.POSES[n][1] for n in ("rest (left)", "start v2 (left)", "start v5 aligned (left)", "after hold released (left)")]
    pick = (lambda q: models.dream(side, np.radians(q))[0]) if hypothesis == "identity" else (lambda q: models.v10(side, np.radians(q))[0])
    return [{"q_deg": q, "tcp_cm": (pick(q) * 100).tolist()} for q in poses]


def test_judge_names_the_model_that_made_the_measurements(models):
    assert stage2.judge(models, "left", entries_from(models, "left", "v10"))["verdict"] == "v10"
    assert stage2.judge(models, "left", entries_from(models, "left", "identity"))["verdict"] == "identity"


def test_judge_says_when_nothing_fits_or_the_poses_do_not_discriminate(models):
    wrong = entries_from(models, "left", "v10")
    for e in wrong[1:]:
        e["tcp_cm"][2] += 15.0
    assert stage2.judge(models, "left", wrong)["verdict"].startswith("neither")
    near = [{"q_deg": stage2.POSES["rest (left)"][1], "tcp_cm": [0, 0, 0]}, {"q_deg": stage2.POSES["hanging (log-010, left)"][1], "tcp_cm": [0, 0, 0]}]
    assert stage2.judge(models, "left", near)["verdict"].startswith("both")


def test_plan_lists_the_poses_with_their_gap(models):
    text = stage2.plan_text(models, "left")
    assert "start v2 (left)" in text and "gap" in text and "right" not in text.split("\n", 1)[1]
    assert "tape measure" in stage2.PROTOCOL and math.isfinite(stage2.TOLERANCE_CM)


def test_add_entry_records_the_median_joints_and_the_measured_point(tmp_path):
    path = tmp_path / "rec.json"
    reads = iter([np.radians([10, 0, 0, 0, 0, 0, 0]), np.radians([12, 0, 0, 0, 0, 0, 0]), np.radians([11, 0, 0, 0, 0, 0, 0])])
    out = stage2.add_entry("left", [5, 6, -40], lambda side: next(reads), path, samples=3)
    again = stage2.add_entry("left", [9, 6, -30], lambda side: np.radians([20, 0, 0, 0, 0, 0, 0]), path, samples=1)
    saved = __import__("json").loads(path.read_text())["entries"]["left"]
    assert out["q_deg"][0] == 11.0 and out["tcp_cm"] == [5, 6, -40] and out["count"] == 1 and again["count"] == 2 and len(saved) == 2


def test_discrepancies_find_only_the_wrist_axes_and_no_limit_differences(models):
    for side in ("left", "right"):
        rows = stage2.discrepancies(models, side)
        assert [r["joint"] for r in rows if r["axis_differs"]] == [6, 7]
        assert not any(r["limits_differ"] for r in rows)
        for r in rows:  # a position gap exists only along the joint's own axis (J2, J4, J6), so it does not change the kinematics
            if r["pos_gap_cm"] > 1:
                gap = np.array(r["dream"]["pos_cm"]) - np.array(r["v10"]["pos_cm"])
                assert abs(np.dot(gap / np.linalg.norm(gap), r["v10"]["axis"])) > 0.99

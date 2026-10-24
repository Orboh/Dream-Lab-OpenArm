"""Tests for openarm_mapping.py and for the driver reading and writing through it. A fake CAN bus: nothing moves.

Run from this folder with Dream's Python:  ../../.venv/bin/python -m pytest -q test_openarm_mapping.py
"""

import math

import numpy as np
import pytest

import openarm_mapping as mapping
import openarm_v2 as drv
from test_openarm_v2 import LOGICAL, OFFSETS, make_arm  # noqa: F401  (the fake bus fixture)

LIMITS = drv.LIMITS_DEG


@pytest.fixture(autouse=True)
def default_mapping(monkeypatch):
    monkeypatch.delenv(mapping.ENV, raising=False)


def test_identity_is_the_default_and_a_bad_name_is_refused(monkeypatch):
    assert mapping.active() == "identity"
    q = np.radians([10, -20, 30, 40, 50, 20, -60])
    assert np.allclose(mapping.to_model("left", q), q)
    monkeypatch.setenv(mapping.ENV, "official")
    assert mapping.active() == "official" and not np.allclose(mapping.to_model("left", q), q)
    monkeypatch.setenv(mapping.ENV, "nonsense")
    with pytest.raises(ValueError, match="OPENARM_MODEL_MAPPING"):
        mapping.active()


@pytest.mark.parametrize("side,j5", [("left", 90.0), ("right", -90.0)])
def test_official_mapping_turns_joint_5_a_quarter_and_reverses_the_wrist(side, j5):
    q = np.radians([10, -20, 30, 40, 5, 20, -60])
    model = np.degrees(mapping.to_model(side, q, "official"))
    assert model[:4] == pytest.approx([10, -20, 30, 40]) and model[4] == pytest.approx(5 + j5) and model[5] == pytest.approx(-20) and model[6] == pytest.approx(60)
    assert np.allclose(mapping.to_logical(side, mapping.to_model(side, q, "official"), "official"), q)  # exact round trip, also for a batch
    batch = np.radians(np.random.default_rng(0).uniform(-80, 80, size=(5, 7)))
    assert np.allclose(mapping.to_logical(side, mapping.to_model(side, batch, "official"), "official"), batch)


def test_the_poses_in_use_fit_dream_limits_except_the_left_rest_pose_by_about_2_degrees():
    start = {"left": [[64.42, -3.62, -1.23, 95.91, -2.48, 19.01, -68.77], [59.04, -2.33, -4.82, 90.23, -1.21, -1.86, -75.24], [64.42, -3.62, -1.23, 95.91, -2.48, 19.01, 21.6]],
             "right": [[-59.61, 1.23, 7.53, 87.17, 2.62, -1.23, 66.72], [-58.06, 0.38, 4.80, 101.24, 2.44, 1.68, 58.13], [-59.61, 1.23, 7.53, 87.17, 2.62, -1.23, -15.0]]}
    for side, poses in start.items():
        for pose in poses:
            assert mapping.outside_model_limits(side, pose, LIMITS[side], "official") == [], (side, pose)
    assert mapping.outside_model_limits("left", [-0.5, -0.8, -4.3, 4.0, 2.1, -0.5, 6.4], LIMITS["left"], "official") == [(5, 2.1)]  # the rest pose: J5 +2.1 maps to 92.1
    assert mapping.outside_model_limits("right", [0.3, 0.7, -1.4, 2.6, 1.3, -6.5, -3.8], LIMITS["right"], "official") == []
    assert mapping.outside_model_limits("left", [0] * 7, LIMITS["left"], "official") == []  # logical zero lands exactly on the limit, not past it
    assert mapping.outside_model_limits("left", [0, 0, 0, 0, 30, 0, 0], LIMITS["left"], "official") == [(5, 30.0)]  # half of the joint 5 range is outside Dream's range


def test_arm_reads_dreams_model_angles_through_the_mapping(make_arm, monkeypatch):
    monkeypatch.setenv(mapping.ENV, "official")
    arm, _ = make_arm(drv.ReadOnlyArm, side="left")
    got = np.degrees(arm.read().positions_rad)
    want = [LOGICAL["joint_1"], LOGICAL["joint_2"], LOGICAL["joint_3"], LOGICAL["joint_4"], LOGICAL["joint_5"] + 90, -LOGICAL["joint_6"], -LOGICAL["joint_7"]]
    assert np.allclose(got, want, atol=0.05)
    right, _ = make_arm(drv.ReadOnlyArm, side="right", channel="can_fol_r")
    assert np.degrees(right.read().positions_rad)[4] == pytest.approx(LOGICAL["joint_5"] - 90, abs=0.05)


def test_arm_writes_logical_native_targets_from_dreams_model_angles(make_arm, monkeypatch):
    monkeypatch.setenv(mapping.ENV, "official")
    arm, bus = make_arm(drv.Arm)
    logical = {n: LOGICAL[n] + (5.0 if n in drv.JOINTS else 0.0) for n in LOGICAL}
    model = mapping.to_model("left", np.radians([logical[n] for n in drv.JOINTS]))
    arm.write(model, 0.5)
    for name in drv.JOINTS:  # the motors get the logical angle plus the calibration offset, whatever the model angle was
        assert bus.mit_targets[name] == pytest.approx(logical[name] + OFFSETS[name], abs=0.05)


def test_the_configs_home_is_taken_as_dreams_model_angles(make_arm, monkeypatch):
    monkeypatch.setenv(mapping.ENV, "official")
    logical_home = np.radians([-5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    arm, _ = make_arm(drv.Arm, home=list(mapping.to_model("left", logical_home)))
    assert np.allclose(arm._home_deg, np.degrees(logical_home), atol=1e-6)


def test_identity_mapping_leaves_the_driver_exactly_as_before(make_arm):
    arm, _ = make_arm(drv.ReadOnlyArm)
    assert np.allclose(np.degrees(arm.read().positions_rad), [LOGICAL[n] for n in drv.JOINTS], atol=0.05)
    assert math.isclose(float(np.sum(arm._sign)), 7.0) and not np.any(arm._zero)


def test_the_record_file_sets_the_mapping_unless_the_variable_says_otherwise(tmp_path, monkeypatch):
    record = tmp_path / "model_mapping"
    monkeypatch.setenv(mapping.FILE_ENV, str(record))
    assert mapping.recorded() is None and mapping.active() == "identity"  # no record, no variable
    mapping.write_record("official")
    assert record.read_text() == "official\n" and mapping.recorded() == "official" and mapping.active() == "official"
    monkeypatch.setenv(mapping.ENV, "identity")
    assert mapping.active() == "identity"  # the variable wins over the record
    monkeypatch.delenv(mapping.ENV)
    record.write_text("nonsense\n")
    with pytest.raises(ValueError, match="nonsense"):
        mapping.recorded()
    with pytest.raises(ValueError):
        mapping.write_record("nonsense")


def test_an_arm_follows_the_record_when_no_variable_is_set(make_arm, tmp_path, monkeypatch):
    monkeypatch.setenv(mapping.FILE_ENV, str(tmp_path / "model_mapping"))
    mapping.write_record("official")
    arm, _ = make_arm(drv.ReadOnlyArm)
    assert np.degrees(arm.read().positions_rad)[4] == pytest.approx(LOGICAL["joint_5"] + 90, abs=0.05)

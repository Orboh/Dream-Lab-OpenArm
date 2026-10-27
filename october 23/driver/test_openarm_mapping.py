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


# -- the dry-run arm: accepts Dream's targets, sends nothing, says what the guard would do ---------------------------------

def frames_that_move_a_motor(bus) -> list:
    """Enable frames and MIT command frames among everything sent (the state queries are parameter reads and are not these)."""
    return [m for m in bus.sent if m[1][:7] == b"\xff" * 7 or m[0] in {drv.CAN_IDS[n] for n in drv.MOTORS}]


def test_the_dry_run_arm_takes_targets_and_sends_nothing_that_could_move_a_motor(make_arm, caplog):
    import logging

    arm, bus = make_arm(drv.DryRunArm)
    arm.read()
    sent_before = len(bus.sent)
    caplog.set_level(logging.INFO)
    target = np.radians([LOGICAL[n] + 40.0 for n in drv.JOINTS])  # a big move: far more than the real arm's step limit would allow
    for _ in range(3):
        arm.write(target, 0.5)  # no ArmError, unlike ReadOnlyArm
    assert len(bus.sent) == sent_before and not frames_that_move_a_motor(bus) and not any(bus.enabled.values()) and not bus.mit_targets
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "DRY RUN target" in text and "farthest joint 40 deg" in text and "guard: allows it" in text
    assert arm.disengage() is True


def test_the_dry_run_arm_logs_a_run_and_says_when_the_guard_would_stop_or_refuse_it(make_arm, tmp_path, monkeypatch, caplog):
    import logging

    import test_openarm_guard as g

    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "1")
    monkeypatch.setenv("OPENARM_ZONES", str(tmp_path / "zones.yaml"))
    arm, _ = make_arm(drv.DryRunArm)
    arm.NOTE_EVERY_S = 0.0  # a line for every target
    arm.read()
    caplog.set_level(logging.INFO)
    arm.write(np.radians([LOGICAL[n] for n in drv.JOINTS]), 0.5)
    assert "refused" in caplog.text and "no safety zones" in caplog.text  # no zones file: the guard would refuse, the dry run only says so
    caplog.clear()
    box = {"kind": "box", "center": [0.0, 0.0, -0.3], "size": [3.0, 3.0, 3.0], "name": "everything"}  # a zone around the whole arm: nothing may be inside it
    g.write_zones(tmp_path / "zones.yaml", captured=g.dt.datetime.now(g.dt.timezone.utc), left_keep=[box])
    arm._guard._stamp = None
    arm.write(np.radians([LOGICAL[n] + 10.0 for n in drv.JOINTS]), 0.5)
    assert "guard: " in caplog.text and ("refused" in caplog.text or "would stop it" in caplog.text)


# -- the converter for poses we keep (a home and the end of its path), 2026-10-27, log-031 -------------------------------------------

def test_our_range_of_each_joint_follows_from_dreams_range():
    assert mapping.logical_limits_deg("left", "official")[4] == pytest.approx((-180.0, 0.0))   # Dream -90..90 = ours -180..0
    assert mapping.logical_limits_deg("right", "official")[4] == pytest.approx((0.0, 180.0))  # Dream -90..90 = ours 0..180
    assert mapping.logical_limits_deg("left", "official")[5] == pytest.approx((-45.0, 45.0))  # a reversed joint keeps a symmetric range
    assert mapping.logical_limits_deg("left", "identity") == [tuple(r) for r in mapping.MODEL_LIMITS_DEG["left"]]


def test_a_home_past_dreams_range_is_moved_just_inside_in_our_angles():
    home = np.radians([10.2, -2.0, 11.9, 74.1, 2.03, -3.2, -16.5])   # the left home of 2026-10-27: J5 +2.03 is Dream's +92.03
    fitted, moved = mapping.fit_logical("left", home, "official")
    assert np.degrees(fitted)[4] == pytest.approx(-0.5) and moved == [(5, -2.53)]
    assert np.allclose(np.degrees(fitted)[[0, 1, 2, 3, 5, 6]], np.degrees(home)[[0, 1, 2, 3, 5, 6]])  # only J5 moves
    right = np.radians([-7.4, 0.4, 7.2, 75.2, -4.73, 4.5, 19.1])
    assert np.degrees(mapping.fit_logical("right", right, "official")[0])[4] == pytest.approx(0.5)
    inside = np.radians([10, -20, 30, 40, -5, 20, -60])
    same, none = mapping.fit_logical("left", inside, "official")
    assert none == [] and same is not None and np.array_equal(same, inside)  # a pose that fits is returned exactly


def test_the_driver_rule_home_vs_path_end_is_measured_in_our_angles():
    station_home_model = mapping.to_model("left", np.radians([10.2, -2.0, 11.9, 74.1, -0.5, -3.2, -16.5]), "official")
    assert mapping.home_path_gap_deg("left", station_home_model, [10.2, -2.0, 11.9, 74.1, 2.03, -3.2, -16.5], "official") == pytest.approx(2.53)
    assert mapping.home_path_gap_deg("left", station_home_model, [10.2, -2.0, 11.9, 74.1, -0.5, -3.2, -16.5], "official") == pytest.approx(0.0, abs=1e-9)
    assert drv.READY_HOME_TOLERANCE_DEG == mapping.HOME_MATCH_DEG  # the driver and the checks use one number

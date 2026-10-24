"""The joint mapping in the workspace tools: logical angles in, Dream's model in the station file, and agreement with the official v10 URDF. Offline."""

import pathlib

import numpy as np
import pytest

import camerapose as cp
import openarm_mapping as mapping
import openarm_zones as zones
import stage2
import zonebuild as zb
from test_camerapose import FakeArm, station  # noqa: F401  (the station-file fixture)

needs_dream = pytest.mark.skipif(not pathlib.Path(stage2.DREAM_URDF.format(side="left")).expanduser().exists(), reason="Dream's dream_ik assets are not installed here")


@pytest.fixture(autouse=True)
def default_mapping(monkeypatch):
    monkeypatch.delenv(mapping.ENV, raising=False)


def test_logical_kinematics_map_first_and_the_default_is_unchanged(monkeypatch):
    q = np.radians([30, -10, 5, 60, 8, 12, -40])
    plain, logical = zones.ArmKinematics("left"), zones.ArmKinematics("left", logical=True)
    assert np.allclose(plain.points(q)["tcp"], logical.points(q)["tcp"])  # identity: the flag changes nothing
    monkeypatch.setenv(mapping.ENV, "official")
    assert np.allclose(logical.points(q)["tcp"], plain.points(mapping.to_model("left", q))["tcp"])
    assert not np.allclose(logical.points(q)["tcp"], plain.points(q)["tcp"])


@needs_dream
@pytest.mark.parametrize("side", ["left", "right"])
def test_official_mapping_makes_dreams_model_agree_with_the_official_urdf_and_identity_does_not(side):
    models = stage2.Models()
    rng = np.random.default_rng(3)
    lim = [(-200, 80), (-190, 10), (-90, 90), (0, 140), (-90, 90), (-45, 45), (-90, 90)]
    if side == "right":
        lim[0], lim[1] = (-80, 200), (-10, 190)
    poses = [np.radians([rng.uniform(a, b) for a, b in lim]) for _ in range(25)]
    p0d, _ = models.dream(side, np.zeros(7))
    p0v, R0v = models.v10(side, np.zeros(7))
    tool = R0v.T @ (p0d - p0v)  # the 2 cm tool-length difference between the two models, absorbed at the zero pose

    def mean_gap(name):
        gaps = []
        for q in poses:
            pd, _ = models.dream(side, mapping.to_model(side, q, name))
            pv, Rv = models.v10(side, q)
            gaps.append(np.linalg.norm(pd - (pv + Rv @ tool)))
        return float(np.mean(gaps)) * 100

    assert mean_gap("official") < 1.0
    assert mean_gap("identity") > 10.0


def raw_bundle():
    return {"arms": {s: {"reach": [], "keep_out": []} for s in zb.SIDES}, "home": {"left": [0.8, -0.1, 0.1, 1.1, -0.2, 0.3, -1.2], "right": [-0.8, 0.0, 0.1, 1.0, 0.2, 0.1, 1.4]}}  # joint 5 on the side that stays inside Dream's range under `official`


def test_the_station_file_gets_the_home_in_dreams_model_angles(monkeypatch):
    homes = {"left": (0.1, 0.1), "right": (0.1, -0.1)}
    raw = raw_bundle()
    same = zb.station_fragment(raw, homes)
    assert same["left"]["home"] == pytest.approx(raw["home"]["left"])  # identity: as before
    monkeypatch.setenv(mapping.ENV, "official")
    fragment = zb.station_fragment(raw, homes)
    for side in zb.SIDES:
        assert np.allclose(fragment[side]["home"], mapping.to_model(side, raw["home"][side]))
        assert np.allclose(mapping.to_logical(side, fragment[side]["home"]), raw["home"][side])


def test_camera_view_move_compares_in_logical_angles_under_the_official_mapping(station, monkeypatch):  # noqa: F811
    monkeypatch.setenv(mapping.ENV, "official")

    class Arm(FakeArm):
        def __init__(self, cfg, side):
            super().__init__(cfg, side)
            self.pose = np.degrees(mapping.to_logical(side, cfg.home)) + 1.5  # the arm's own (logical) angles, 1.5 deg from the home

    said = []
    assert cp.run(station, ["left"], hold=lambda: None, opener=lambda cfg, side: Arm(cfg, side), say=said.append) == 0
    assert any("at the camera-view pose" in line for line in said)


def test_a_home_outside_dreams_joint_range_is_moved_just_inside_it_and_reported(monkeypatch):
    monkeypatch.setenv(mapping.ENV, "official")
    raw = raw_bundle()
    raw["home"]["left"] = [0.1, -0.03, 0.06, 1.43, 0.012, 0.17, 0.34]  # left joint 5 = +0.7 deg logical -> 90.7 deg in Dream's model, past its +90 limit
    raw["mode"] = "danger_only"
    homes = {"left": (0.1, 0.1), "right": (0.1, -0.1)}
    moved = zb.home_adjustments(raw)
    assert list(moved) == ["left"] and moved["left"][0][0] == 5 and moved["left"][0][1] == pytest.approx(-1.19, abs=0.02)
    written = zb.station_fragment(raw, homes)["left"]["home"]
    assert np.degrees(written[4]) == pytest.approx(89.5)  # 0.5 deg inside the limit
    assert zb.home_adjustments({**raw_bundle(), "mode": "danger_only"}) == {}  # a home that fits is left alone
    monkeypatch.delenv(mapping.ENV)
    assert zb.home_adjustments(raw) == {}  # identity: joint 5 stays at 0.7 deg


def test_the_models_limits_equal_the_drivers():
    import openarm_v2 as drv

    for side in ("left", "right"):
        for i, (low, high) in enumerate(mapping.MODEL_LIMITS_DEG[side]):
            assert drv.LIMITS_DEG[side][f"joint_{i + 1}"] == (low, high)

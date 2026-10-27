"""Tests for the pre-session checks: every check passes on a consistent setup and fails, by name, when one thing is wrong. System probes are fakes: no CAN, no cameras."""

import datetime as dt
import shutil

import numpy as np
import pytest
import yaml

import editor as ed
import openarm_poses as poses
import preflight as pf
import workspace as ws
import zonebuild as zb

GOOD_LINK = "can_fol: <NOARP,UP,LOWER_UP,ECHO>\n can <FD> state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 100\n bitrate 1000000 dbitrate 5000000"


class Probes(pf.Probes):
    def __init__(self, links=None, procs=None, cams=None, holders=None):
        self._links = links or {"can_fol_l": GOOD_LINK, "can_fol_r": GOOD_LINK}
        self._procs = procs if procs is not None else ["/usr/bin/python something-else"]
        self._holders = holders or []
        self._cams = cams if cams is not None else {"344422070898", "409122272149", "352122273417"}

    def can_links(self):
        return self._links

    def processes(self):
        return self._procs

    def cameras(self):
        return self._cams

    def camera_holders(self):
        return self._holders


def make_bundle(now, home_path=True, region=True):
    state = ed.EditorState.default()
    for side in ed.SIDES:
        state.arms[side].home = [float(v) for v in np.radians(poses.READY_DEG[side])]
        state.arms[side].home_source = "ready"
        if home_path:
            state.arms[side].home_path = poses.path_to_ready(side)
        if region:
            state.arms[side].region = {"lo": [-0.5, -0.5, -0.9], "hi": [0.9, 0.5, 0.2]}
    state.table_set = True
    return state.to_bundle("t", now)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "1")
    now = dt.datetime.now(dt.timezone.utc)
    zdir = tmp_path / "zones"; zdir.mkdir()
    raw = make_bundle(now)
    bundle = zdir / "zones-20261023-1700.yaml"
    ws.save_raw(bundle, raw)
    in_force = tmp_path / "in_force.yaml"
    shutil.copy2(bundle, in_force)
    station = tmp_path / "station.yaml"
    station.write_text(ws.STATION.read_text())
    zb.apply_to_station(station, zb.station_fragment(raw, {"left": (0.0, 0.1), "right": (0.0, -0.1)}), tuning=zb.GENTLE_STATION)
    return station, in_force, zdir, raw


def run(setup, **kw):
    station, in_force, zdir, _ = setup
    return pf.run_preflight(station, in_force, zdir, probes=kw.pop("probes", Probes()), **kw)


def failed(checks):
    return {c.label for c in checks if not c.ok}


def test_a_consistent_setup_passes_every_check(setup):
    checks = run(setup)
    assert failed(checks) == set(), [c for c in checks if not c.ok]
    assert any("active arms" in c.label.lower() for c in checks)          # the lock state is shown


def test_a_can_link_that_is_down_or_not_fd_fails(setup):
    bad = {"can_fol_l": "can_fol_l: <NOARP,ECHO>\n can state STOPPED", "can_fol_r": GOOD_LINK}
    assert pf.LABEL_CAN in failed(run(setup, probes=Probes(links=bad)))
    classic = {"can_fol_l": GOOD_LINK.replace("<FD>", "").replace("dbitrate 5000000", ""), "can_fol_r": GOOD_LINK}
    assert pf.LABEL_CAN in failed(run(setup, probes=Probes(links=classic)))


@pytest.mark.parametrize("proc", ["python -m lerobot-teleoperate --robot.type=bi_openarm_follower", "bash scripts/run_teleop_bimanual.sh", "python -m robots.station --config x", "dreamcli setup"])
def test_another_program_holding_the_arms_fails(setup, proc):
    assert pf.LABEL_FREE in failed(run(setup, probes=Probes(procs=[proc])))


def test_a_missing_camera_fails(setup):
    assert pf.LABEL_CAMERAS in failed(run(setup, probes=Probes(cams={"344422070898"})))


def test_busy_cameras_name_the_program_holding_them(setup):
    checks = run(setup, probes=Probes(holders=["252062: python workspace.py design --observe --cameras --port 8080"]))
    busy = [c for c in checks if c.label == pf.LABEL_CAMERAS_FREE][0]
    assert not busy.ok and "workspace.py design" in busy.detail and pf.LABEL_CAMERAS_FREE in failed(checks)


def test_stale_zones_in_force_fail(setup):
    station, in_force, zdir, raw = setup
    old = make_bundle(dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=30))
    ws.save_raw(in_force, old); ws.save_raw(zdir / "zones-20261023-1700.yaml", old)
    assert pf.LABEL_FRESH in failed(run(setup))


def test_zones_in_force_that_are_not_the_newest_bundle_fail(setup):
    station, in_force, zdir, raw = setup
    newer = make_bundle(dt.datetime.now(dt.timezone.utc)); newer["operator"] = "someone else"
    ws.save_raw(zdir / "zones-20261023-1800.yaml", newer)
    assert pf.LABEL_NEWEST in failed(run(setup))


def test_a_station_home_that_differs_from_the_bundle_fails(setup):
    station, in_force, zdir, raw = setup
    data = yaml.safe_load(station.read_text()); data["follower_l"]["home"][0] += 0.2
    station.write_text(yaml.safe_dump(data))
    assert pf.LABEL_HOME in failed(run(setup))


def test_no_home_path_or_a_path_that_does_not_end_at_the_home_fails(setup):
    station, in_force, zdir, raw = setup
    nopath = make_bundle(dt.datetime.now(dt.timezone.utc), home_path=False)
    ws.save_raw(in_force, nopath); ws.save_raw(zdir / "zones-20261023-1700.yaml", nopath)
    assert pf.LABEL_PATH in failed(run(setup))


def test_a_region_that_leaves_the_home_outside_fails(setup):
    station, in_force, zdir, raw = setup
    tight = make_bundle(dt.datetime.now(dt.timezone.utc))
    tight["arms"]["left"]["region"] = {"lo": [0.5, 0.5, 0.5], "hi": [0.6, 0.6, 0.6]}
    ws.save_raw(in_force, tight); ws.save_raw(zdir / "zones-20261023-1700.yaml", tight)
    assert pf.LABEL_CLEAR in failed(run(setup))


def test_a_station_that_does_not_use_the_arm_driver_fails(setup):
    station, in_force, zdir, raw = setup
    data = yaml.safe_load(station.read_text()); data["custom"]["driver"] = data["custom"]["driver"].replace(":Arm", ":ReadOnlyArm")
    station.write_text(yaml.safe_dump(data))
    assert pf.LABEL_DRIVER in failed(run(setup))


def test_the_guard_switched_off_fails(setup, monkeypatch):
    monkeypatch.setenv("OPENARM_ZONES_REQUIRED", "0")
    assert pf.LABEL_GUARD in failed(run(setup))


def test_report_lists_ticks_and_crosses_and_start_refuses_on_a_failure(setup, capsys):
    checks = run(setup, probes=Probes(cams=set()))
    text = pf.format_report(checks)
    assert "[PASS]" in text and "[FAIL]" in text and pf.LABEL_CAMERAS in text
    station, in_force, zdir, _ = setup
    ran = []
    code = ws.main(["start", "--station", str(station), "--in-force", str(in_force), "--zones-dir", str(zdir), "--dry"], probes=Probes(cams=set()), runner=lambda cmd: ran.append(cmd) or 0)
    assert code == 1 and not ran                                          # a failed preflight never starts the station
    code = ws.main(["start", "--station", str(station), "--in-force", str(in_force), "--zones-dir", str(zdir)], probes=Probes(), runner=lambda cmd: ran.append(cmd) or 0)
    assert code == 0 and ran and ran[0][-2:] == ["robot", "up"]


# -- camera-pose goes through the same checks as start --------------------------------------------------------------------------------

def test_camera_pose_refuses_to_move_anything_while_a_check_fails(setup):
    station, in_force, zdir, _ = setup
    opened = []
    code = ws.main(["camera-pose", "--arm", "left", "--hold", "0.01", "--station", str(station), "--in-force", str(in_force),
                    "--zones-dir", str(zdir)], probes=Probes(procs=["/x/dream-env/bin/python -m robots.station --config x"]),
                   opener=lambda cfg, side: opened.append(side))
    assert code == 1 and opened == []


def test_the_station_home_is_compared_in_dreams_model_angles_under_the_official_mapping(setup, monkeypatch):
    import openarm_mapping as mapping

    station, in_force, zdir, raw = setup  # the station file was written under the identity mapping
    monkeypatch.setenv(mapping.ENV, "official")
    assert pf.LABEL_HOME in failed(run(setup))  # a station home written without the conversion is a quarter turn off the bundle's home
    data = yaml.safe_load(station.read_text())
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        data[key]["home"] = [float(v) for v in mapping.to_model(side, raw["home"][side])]
    station.write_text(yaml.safe_dump(data))
    assert pf.LABEL_HOME not in failed(run(setup))


def test_preflight_fails_when_the_mapping_in_use_differs_from_the_one_the_station_file_was_written_under(setup, tmp_path, monkeypatch):
    import openarm_mapping as mapping

    monkeypatch.setenv(mapping.FILE_ENV, str(tmp_path / "model_mapping"))
    assert pf.LABEL_MAPPING not in failed(run(setup))  # no record, identity in use: consistent
    mapping.write_record("official")
    assert pf.LABEL_MAPPING not in failed(run(setup))  # the record decides when no variable is set
    monkeypatch.setenv(mapping.ENV, "identity")
    bad = [c for c in run(setup) if c.label == pf.LABEL_MAPPING][0]
    assert bad.ok is False and "'official'" in bad.detail and "'identity'" in bad.detail


def test_a_home_that_apply_moved_inside_dreams_joint_range_still_counts_as_the_bundles_home(setup, monkeypatch):
    import openarm_mapping as mapping

    station, in_force, zdir, raw = setup
    monkeypatch.setenv(mapping.ENV, "official")
    monkeypatch.setenv(mapping.FILE_ENV, "/nonexistent/x")
    data = yaml.safe_load(station.read_text())
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        model, _ = mapping.clip_to_model_limits(side, mapping.to_model(side, raw["home"][side]))
        data[key]["home"] = [float(v) for v in model]
    station.write_text(yaml.safe_dump(data))
    assert pf.LABEL_HOME not in failed(run(setup))
    data["follower_l"]["home"][0] += 0.2  # a real difference is still caught
    station.write_text(yaml.safe_dump(data))
    assert pf.LABEL_HOME in failed(run(setup))


def test_a_path_that_ends_further_than_the_drivers_2_degrees_from_the_station_home_fails(setup, monkeypatch):
    """log-031: `apply` moved the left home's J5 inside Dream's range but not the path's end; the driver then ignored the path and did not raise the arm."""
    import openarm_mapping as mapping

    station, in_force, zdir, raw = setup
    monkeypatch.setenv(mapping.ENV, "official")
    monkeypatch.setenv(mapping.FILE_ENV, "/nonexistent/x")
    home_deg = [10.2, -2.0, 11.9, 74.1, 2.03, -3.2, -16.5]
    raw["home"]["left"] = [float(v) for v in np.radians(home_deg)]
    raw["home_path"]["left"] = [raw["home_path"]["left"][0], home_deg]
    for path in (zdir / "zones-20261023-1700.yaml", in_force):
        ws.save_raw(path, raw)
    zb.apply_to_station(station, zb.station_fragment(raw, {"left": (0.0, 0.1), "right": (0.0, -0.1)}), tuning=zb.GENTLE_STATION)
    bad = [c for c in run(setup) if c.label == pf.LABEL_PATH_END][0]
    assert bad.ok is False and "left" in bad.detail and "2.5" in bad.detail
    raw["home_path"]["left"][-1][4] = -0.5   # the end fitted like the home: the driver's rule holds
    for path in (zdir / "zones-20261023-1700.yaml", in_force):
        ws.save_raw(path, raw)
    assert pf.LABEL_PATH_END not in failed(run(setup))


def test_the_two_arms_are_checked_against_each_other_along_every_raise_park_and_dreams_home(setup):
    """Both reviews of goal.md (2026-10-27, log-033): nothing checked one arm against the other."""
    station, in_force, zdir, raw = setup
    assert pf.LABEL_ARMS not in failed(run(setup))                     # ready homes and the built-in paths: clear
    across = [45.0, 0.0, 60.0, 120.0, 0.0, 0.0, 0.0]                   # the left hand swung over to the right arm's side
    raw["home_path"]["left"] = [raw["home_path"]["left"][0], across, raw["home_path"]["left"][-1]]
    for path in (zdir / "zones-20261023-1700.yaml", in_force):
        ws.save_raw(path, raw)
    bad = [c for c in run(setup) if c.label == pf.LABEL_ARMS][0]
    assert bad.ok is False and "left hand" in bad.detail and "left raise/park" in bad.detail

"""Tests for the teach session: home from the follower's pose, the swept extremes as the allowed region, the recorded path from rest to home. No hardware."""

import datetime as dt

import numpy as np
import pytest

import editor as ed
import openarm_poses as poses
import openarm_zones as zones
import teach as th

NOW = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)
REST = np.radians(poses.REST_DEG["left"])


def pose_with(j1=0.0, j4=0.0):
    q = REST.copy()
    q[0], q[3] = np.radians(j1), np.radians(j4)
    return q


def test_nothing_is_recorded_until_sweeping_is_on():
    s = th.TeachSession()
    s.observe("left", pose_with())
    assert s.region("left") is None and not s.reach_cells("left")
    s.set_sweep("left", True)
    for i in range(30):
        s.observe("left", pose_with(j1=i, j4=i * 2))
    assert s.sweep_count("left") == 30 and s.sweep_count("right") == 0


def test_region_is_the_bounding_box_of_every_swept_point_plus_the_margin():
    s = th.TeachSession(margin_m=0.02)
    s.set_sweep("left", True)
    kin = zones.ArmKinematics("left")
    poses_swept = [pose_with(j1=a, j4=b) for a in (0, 20, 40) for b in (10, 40, 70, 90)]
    for q in poses_swept:
        s.observe("left", q)
    pts = np.array([p for q in poses_swept for p in kin.points(q).values()])
    box = s.region("left")
    assert box["lo"] == pytest.approx((pts.min(axis=0) - 0.02).tolist(), abs=1e-4)
    assert box["hi"] == pytest.approx((pts.max(axis=0) + 0.02).tolist(), abs=1e-4)


def test_region_needs_enough_samples():
    s = th.TeachSession()
    s.set_sweep("left", True)
    for _ in range(3):
        s.observe("left", pose_with())
    assert s.region("left") is None


def test_reach_cells_cover_the_swept_hand_positions_on_the_two_cm_grid():
    s = th.TeachSession()
    s.set_sweep("left", True)
    kin = zones.ArmKinematics("left")
    for i in range(20):
        s.observe("left", pose_with(j1=i * 2, j4=40))
    cells = s.reach_cells("left")
    tcp = kin.points(pose_with(j1=10, j4=40))["tcp"]
    assert (round(tcp[0] / 0.02), round(tcp[1] / 0.02)) in cells


def test_clear_sweep_forgets_the_points():
    s = th.TeachSession()
    s.set_sweep("left", True)
    for i in range(30):
        s.observe("left", pose_with(j1=i))
    s.clear_sweep("left")
    assert s.region("left") is None and s.sweep_count("left") == 0


def test_marks_remember_the_labelled_hand_position():
    s = th.TeachSession()
    s.mark("left", "far edge", pose_with(j1=30, j4=50))
    label, xyz = s.marks("left")[0]
    assert label == "far edge" and xyz == pytest.approx(zones.ArmKinematics("left").points(pose_with(j1=30, j4=50))["tcp"].tolist())


def test_simplify_path_keeps_the_ends_and_drops_points_on_a_straight_line():
    line = [[0.0] * 7, [10.0] + [0.0] * 6, [20.0] + [0.0] * 6, [30.0] + [0.0] * 6, [30.0, 40.0] + [0.0] * 5]
    out = th.simplify_path(line, tol_deg=2.0)
    assert out[0] == line[0] and out[-1] == line[-1] and [30.0, 0.0] + [0.0] * 5 in [[*p] for p in out] and len(out) == 3


def test_path_recording_only_starts_near_rest_then_ends_at_home():
    s = th.TeachSession()
    with pytest.raises(ValueError, match="rest"):
        s.start_path("left", pose_with(j1=50, j4=80))
    s.start_path("left", REST)
    for a in range(0, 60, 3):
        s.observe("left", pose_with(j1=a, j4=a))
    home = pose_with(j1=60, j4=60)
    s.observe("left", home)
    path = s.finish_path("left", home)
    assert path[0] == pytest.approx(poses.REST_DEG["left"], abs=0.5) and path[-1] == pytest.approx(np.degrees(home).tolist())
    assert 2 <= len(path) <= 12


def test_apply_to_state_writes_home_region_reach_and_path():
    s = th.TeachSession()
    s.start_path("left", REST)
    s.set_sweep("left", True)
    for a in range(0, 40):
        s.observe("left", pose_with(j1=a, j4=a))
    home = pose_with(j1=40, j4=40)
    s.set_home("left", home)
    state = ed.EditorState.default()
    s.apply_to(state)
    arm = state.arms["left"]
    assert arm.home == pytest.approx(list(home)) and arm.region and arm.reach and arm.home_path
    assert state.arms["right"].region is None
    raw = state.to_bundle("k", NOW)
    assert raw["arms"]["left"]["region"] == arm.region and raw["home_path"]["left"] == arm.home_path
    zones.parse_zones(raw)
    again = ed.EditorState.from_bundle(raw)
    assert again.arms["left"].region == arm.region and again.arms["left"].home_path == arm.home_path


def test_path_can_start_from_the_ready_pose_and_keeps_the_built_in_rest_to_ready_path():
    s = th.TeachSession()
    ready = np.radians(poses.READY_DEG["left"])
    s.start_path("left", ready)                      # the arm already stands at the old ready pose
    assert s.recording_path("left")
    for k in range(0, 30, 3):
        q = ready.copy()
        q[0] = ready[0] + np.radians(k)              # the new home is further back
        s.observe("left", q)
    home = ready.copy(); home[0] += np.radians(30.0)
    path = s.finish_path("left", home)
    built_in = poses.path_to_ready("left")
    assert path[: len(built_in)] == built_in         # rest ... ready exactly as the driver already knows it
    assert path[-1] == pytest.approx(np.degrees(home).tolist())
    assert len(path) > len(built_in)                 # plus the taught segment


def test_path_recording_still_refuses_a_pose_that_is_neither_rest_nor_ready():
    s = th.TeachSession()
    with pytest.raises(ValueError, match="rest or the ready"):
        s.start_path("left", pose_with(j1=30, j4=50))


# -- automatic path tracking (no "Start path" button) ------------------------------------------------------------------------

def walk(s, side, start_pose, deltas, j4=40.0):
    """Feeds poses that leave `start_pose` along joint 1 / joint 4."""
    for d in deltas:
        q = start_pose.copy()
        q[0] += np.radians(d); q[3] += np.radians(d * 0.8)
        s.observe(side, q)
    return q


def test_the_path_from_rest_is_recorded_without_pressing_anything():
    s = th.TeachSession()
    for _ in range(5):
        s.observe("left", REST)                     # the arm hangs at rest
    last = walk(s, "left", REST, range(0, 50, 2))   # then the leader takes it away
    path = s.set_home("left", last)
    assert path is not None and path[0] == pytest.approx(poses.REST_DEG["left"]) and path[-1] == pytest.approx(np.degrees(last).tolist())
    assert s.has_path("left") and 2 <= len(path) <= 12


def test_the_path_from_the_ready_pose_keeps_the_built_in_prefix():
    s = th.TeachSession()
    ready = np.radians(poses.READY_DEG["left"])
    for _ in range(5):
        s.observe("left", ready)
    last = walk(s, "left", ready, range(0, 30, 2))
    path = s.set_home("left", last)
    assert path[: len(poses.path_to_ready("left"))] == poses.path_to_ready("left") and len(path) > len(poses.path_to_ready("left"))


def test_only_the_last_dwell_at_rest_counts():
    s = th.TeachSession()
    s.observe("left", REST)
    walk(s, "left", REST, range(0, 40, 2))          # a first excursion
    for _ in range(3):
        s.observe("left", REST)                     # back at rest: the record starts again
    last = walk(s, "left", REST, range(0, 20, 2))
    path = s.set_home("left", last)
    assert path[0] == pytest.approx(poses.REST_DEG["left"]) and max(abs(p[0] - poses.REST_DEG["left"][0]) for p in path) < 25.0


def test_no_path_when_the_arm_never_stood_at_rest_or_ready():
    s = th.TeachSession()
    walk(s, "left", pose_with(j1=30, j4=50), range(0, 10))
    assert s.set_home("left", pose_with(j1=40, j4=60)) is None
    assert not s.has_path("left") and s.home("left") is not None


def test_pressing_set_home_at_the_ready_pose_gives_the_built_in_path():
    s = th.TeachSession()
    ready = np.radians(poses.READY_DEG["right"])
    for _ in range(3):
        s.observe("right", ready)
    path = s.set_home("right", ready)
    assert path[0] == pytest.approx(poses.REST_DEG["right"]) and path[-1] == pytest.approx(poses.READY_DEG["right"])


# -- recorded points define the safe place (round 5) ---------------------------------------------------------------------------

def hand_xy(q, side="left"):
    return zones.ArmKinematics(side).points(q)["tcp"][:2]


def corner_poses():
    return [pose_with(j1=a, j4=b) for a, b in ((0, 20), (40, 20), (40, 80), (0, 80))]


def test_the_default_margin_is_zero_so_the_region_is_exactly_the_recorded_points():
    assert th.TeachSession().margin == 0.0


def test_recorded_points_define_the_region_as_their_bounding_box():
    s = th.TeachSession()
    kin = zones.ArmKinematics("left")
    for q in corner_poses():
        s.add_point("left", q)
    pts = np.array([p for q in corner_poses() for p in kin.points(q).values()])
    box = s.region("left")
    assert box["lo"] == pytest.approx(pts.min(axis=0).tolist(), abs=1e-4) and box["hi"] == pytest.approx(pts.max(axis=0).tolist(), abs=1e-4)
    assert s.point_count("left") == 4 and s.point_count("right") == 0


def test_one_point_is_not_enough_for_a_region_two_are():
    s = th.TeachSession()
    s.add_point("left", pose_with(j1=10, j4=30))
    assert s.region("left") is None
    s.add_point("left", pose_with(j1=30, j4=60))
    assert s.region("left") is not None


def test_reach_is_the_filled_polygon_through_the_recorded_hand_positions():
    s = th.TeachSession()
    for q in corner_poses():
        s.add_point("left", q)
    cells = s.reach_cells("left")
    corners = np.array([hand_xy(q) for q in corner_poses()])
    centre = corners.mean(axis=0)
    assert (round(centre[0] / 0.02), round(centre[1] / 0.02)) in cells            # the middle of the quadrilateral is inside
    far = centre + np.array([0.5, 0.5])
    assert (round(far[0] / 0.02), round(far[1] / 0.02)) not in cells               # nothing far outside it
    for c in corners:
        assert (round(c[0] / 0.02), round(c[1] / 0.02)) in cells                   # and the corners themselves


def test_with_fewer_than_three_points_the_reach_is_just_the_points():
    s = th.TeachSession()
    s.add_point("left", pose_with(j1=10, j4=30)); s.add_point("left", pose_with(j1=30, j4=60))
    assert len(s.reach_cells("left")) == 2


def test_undo_removes_the_last_point_and_clear_removes_all():
    s = th.TeachSession()
    for q in corner_poses():
        s.add_point("left", q)
    s.undo_point("left")
    assert s.point_count("left") == 3
    s.clear_points("left")
    assert s.point_count("left") == 0 and s.region("left") is None
    s.undo_point("left")  # nothing to undo: no error


def test_the_trail_of_points_lists_the_hand_positions_for_the_markers():
    s = th.TeachSession()
    for q in corner_poses():
        s.add_point("left", q)
    assert np.array(s.point_list("left")) == pytest.approx(np.array([zones.ArmKinematics("left").points(q)["tcp"] for q in corner_poses()]), abs=1e-6)


def test_points_and_a_sweep_combine_into_one_region():
    s = th.TeachSession()
    kin = zones.ArmKinematics("left")
    s.set_sweep("left", True)
    for a in range(0, 12):
        s.observe("left", pose_with(j1=a, j4=20))
    s.set_sweep("left", False)
    s.add_point("left", pose_with(j1=60, j4=90)); s.add_point("left", pose_with(j1=50, j4=85))
    hi = s.region("left")["hi"]
    assert hi[0] >= max(kin.points(pose_with(j1=60, j4=90)).values(), key=lambda p: p[0])[0] - 1e-4

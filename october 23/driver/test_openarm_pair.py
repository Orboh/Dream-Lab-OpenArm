"""Tests for openarm_pair.py: the distance between the two arms (capsules in the robot's body frame). No hardware.

Run from this folder with Dream's Python:  ../../.venv/bin/python -m pytest -q test_openarm_pair.py
"""

import numpy as np
import pytest

import openarm_pair as pair

LEFT_HOME = np.radians([10.2, -2.0, 11.9, 74.1, -0.5, -3.2, -16.5])     # the homes taught on 2026-10-27 (log-031)
RIGHT_HOME = np.radians([-7.4, 0.4, 7.2, 75.2, 1.0, 4.5, 19.1])
REST = np.zeros(7)
LEFT_ACROSS = np.radians([45.0, 0.0, 60.0, 120.0, 0.0, 0.0, 0.0])        # the left hand swung over to the right arm's side (y about -13 cm)


@pytest.fixture(autouse=True)
def official(monkeypatch):
    import openarm_mapping as mapping

    monkeypatch.setenv(mapping.ENV, "official")


def test_segment_distance_covers_crossing_parallel_and_end_cases():
    d = pair.segment_distance
    assert d(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0.5, -1, 1.0]), np.array([0.5, 1, 1.0])) == pytest.approx(1.0)   # skew, crossing over
    assert d(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 2, 0.0]), np.array([1, 2, 0.0])) == pytest.approx(2.0)       # parallel
    assert d(np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([2, 0, 0.0]), np.array([3, 0, 0.0])) == pytest.approx(1.0)       # end to end
    assert d(np.array([0, 0, 0.0]), np.array([0, 0, 0.0]), np.array([0, 3, 4.0]), np.array([0, 3, 4.0])) == pytest.approx(5.0)       # two points


def test_the_capsules_are_in_the_body_frame_and_the_shoulders_are_31_cm_apart():
    left, right = pair.capsules("left", REST), pair.capsules("right", REST)
    assert [c.name for c in left] == ["upper_arm", "forearm", "hand"]
    assert left[0].a[1] - right[0].a[1] == pytest.approx(0.308, abs=0.005)      # shoulder (joint 2) y +15.4 / -15.4 cm
    assert all(c.radius == pair.LINK_RADIUS_M for c in left)                  # the gripper's width is its own points now (log-039), not a fat capsule


def test_the_arms_at_rest_and_at_the_homes_of_2026_10_27_keep_clear_of_each_other():
    for q_left, q_right in ((REST, REST), (LEFT_HOME, RIGHT_HOME), (LEFT_HOME, REST), (REST, RIGHT_HOME)):
        gap, _, _ = pair.gap_m(q_left, q_right)
        assert gap > pair.MIN_GAP_M, (np.degrees(q_left), np.degrees(q_right), gap)


def test_a_left_hand_swung_over_to_the_right_arm_is_a_collision_and_names_the_parts():
    gap, left_part, right_part = pair.gap_m(LEFT_ACROSS, REST)
    assert gap < 0.0
    assert left_part == "gripper" and right_part in ("upper_arm", "forearm")   # the open gripper is what reaches the other arm (log-039)
    assert pair.problem(LEFT_ACROSS, REST) is not None and "left gripper" in pair.problem(LEFT_ACROSS, REST)
    assert pair.problem(LEFT_HOME, RIGHT_HOME) is None


# -- during a task: each arm's guard also keeps 2 cm from the other arm (log-034) ---------------------------------------------------

from test_openarm_v2 import make_arm  # noqa: E402,F401  (the fake bus fixture)
import openarm_v2 as drv  # noqa: E402

AT_REST = {f"joint_{i}": 0.0 for i in range(1, 8)}


@pytest.fixture(autouse=True)
def clean_registry():
    pair.reset()
    yield
    pair.reset()


NEAR = 0.55 * LEFT_ACROSS     # 4.1 cm from the resting right arm: clear
INTO = 0.68 * LEFT_ACROSS     # 1.4 cm: too close (and within write()'s 20 deg step of NEAR)


def _deg_dict(q_rad) -> dict[str, float]:
    return {f"joint_{i + 1}": float(v) for i, v in enumerate(np.degrees(q_rad))}


def _two_arms(make_arm, left_at=NEAR):
    left, _ = make_arm(drv.Arm, side="left", channel="can_fol_l", logical=_deg_dict(left_at))
    right, _ = make_arm(drv.Arm, side="right", channel="can_fol_r", logical=AT_REST)
    return left, right


def _logical_rad(arm) -> np.ndarray:
    return arm._sign * (np.asarray(arm._safe_q, dtype=float) - arm._zero)


def test_a_target_clear_of_the_other_arm_goes_through_unchanged(make_arm):
    left, right = _two_arms(make_arm, left_at=LEFT_HOME)
    target = drv.mapping.to_model("left", LEFT_HOME + np.radians([3.0, 0, 0, 3.0, 0, 0, 0]))
    left.write(target, 0.5)
    assert np.allclose(left._safe_q, target)


def test_a_target_into_the_other_arm_is_cut_where_the_arms_keep_2_cm(make_arm, caplog):
    left, right = _two_arms(make_arm)
    left.write(drv.mapping.to_model("left", INTO), 0.5)
    sent = _logical_rad(left)
    assert not np.allclose(sent, INTO)                                                # not where Dream asked
    gap, _, _ = pair.gap_m(sent, REST)
    assert pair.MIN_GAP_M - 0.005 <= gap < pair.MIN_GAP_M + 0.02                     # stopped at the edge, not far before it
    assert any("other arm" in r.message for r in caplog.records)


def test_an_arm_not_heard_from_counts_as_hanging_at_rest(make_arm):
    left, _ = make_arm(drv.Arm, side="left", channel="can_fol_l", logical=_deg_dict(NEAR))   # no right arm in this process
    left.write(drv.mapping.to_model("left", INTO), 0.5)
    assert pair.gap_m(_logical_rad(left), REST)[0] >= pair.MIN_GAP_M - 0.005


def test_when_already_too_close_only_moves_that_open_the_gap_are_allowed():
    near = pair.cut("left", LEFT_ACROSS, LEFT_ACROSS, [REST])[0]                      # already inside: holds
    assert np.allclose(near, LEFT_ACROSS)
    away, why = pair.cut("left", LEFT_ACROSS, REST, [REST])                            # moving back out: allowed
    assert np.allclose(away, REST) and why is None
    closer = LEFT_ACROSS + np.radians([5.0, 0, 5.0, 5.0, 0, 0, 0])
    held, why = pair.cut("left", LEFT_ACROSS, closer, [REST])
    assert np.allclose(held, LEFT_ACROSS) and why is not None


def test_the_way_between_two_clear_targets_is_checked_against_the_other_arm():
    """log-035: both ends are clear of the right arm (4 cm with the real gripper, log-039), the swing between them goes into it."""
    swing_from, swing_to = np.radians([-40, 0, 0, 90, 0, 0, 0]), np.radians([40, 0, 0, 90, 0, 0, 0])
    right = np.radians([0, 0, -40, 80, 0, 0, 0])
    assert pair.gap_m(swing_from, right)[0] > 0.035 and pair.gap_m(swing_to, right)[0] > 0.035
    assert pair.gap_m(np.radians([0, 0, 0, 90, 0, 0, 0]), right)[0] < 0.0
    sent, why = pair.cut("left", swing_from, swing_to, [right])
    assert why is not None and sent[0] < 0.0
    assert pair.gap_m(sent, right)[0] >= pair.MIN_GAP_M - 0.002


def test_dream_going_on_past_a_held_edge_does_not_look_like_a_wrong_mapping(make_arm):
    """log-036, run 1 of 2026-10-27: held at an edge, Dream kept asking further; the 20 deg check compared its request with the HELD target and ended the run."""
    left, right = _two_arms(make_arm)
    step = (INTO - NEAR) / 4.0
    for k in range(1, 13):                                   # 12 small steps: the last request is 3x the way into the right arm, far past the held edge
        left.write(drv.mapping.to_model("left", NEAR + k * step), 0.5)
    assert pair.gap_m(_logical_rad(left), REST)[0] >= pair.MIN_GAP_M - 0.005   # still held at the edge


def test_a_real_jump_between_two_requests_is_still_refused(make_arm):
    from robots import errors

    left, right = _two_arms(make_arm, left_at=LEFT_HOME)
    left.write(drv.mapping.to_model("left", LEFT_HOME), 0.5)
    with pytest.raises(errors.ArmError, match="nothing was sent"):
        left.write(drv.mapping.to_model("left", LEFT_HOME + np.radians([0, 0, 0, 0, 0, 0, 45.0])), 0.5)



# -- the real open gripper against the other arm (log-039): the user saw the open grippers come much closer than the check said ---------

TURNED_IN_LEFT = LEFT_HOME + np.radians([0, 0, 10.0, 0, 0, 0, 0])            # both wrists turned 10 deg toward the middle from the 10-27 homes
TURNED_IN_RIGHT = RIGHT_HOME - np.radians([0, 0, 10.0, 0, 0, 0, 0])


def test_the_open_grippers_count_with_their_real_width():
    """The old 6 cm hand capsule said 3.6 cm here (allowed); the real open grippers are 0.1 cm apart."""
    gap, left_part, right_part = pair.gap_m(TURNED_IN_LEFT, TURNED_IN_RIGHT)
    assert gap < pair.MIN_GAP_M
    assert "gripper" in (left_part, right_part)


def test_gripper_points_cover_the_housing_and_both_open_fingers():
    pts = pair.gripper_points("left", LEFT_HOME)
    assert pts.shape[1] == 3 and len(pts) > 100
    width = np.ptp(pts @ np.linalg.svd(pts - pts.mean(0))[2][0])                  # the gripper's longest extent
    assert width >= 0.155                                                           # open fingers 2 x 7.8 cm, housing 16.8 cm


def test_a_target_far_from_the_held_one_after_a_hold_is_no_wrong_mapping_on_a_smooth_arm(make_arm):
    """log-037, 13:21:20: held at an edge while Dream turned the wrist on, the next clear target was 20.7 deg from the held one and the check that runs
    after the cut ended Dream's Home. With smooth motion the tracker limits the speed anyway; Dream's own steps are still checked (log-040)."""
    left, _ = make_arm(drv.Arm, side="left", channel="can_fol_l", logical=_deg_dict(NEAR), smooth=True)
    right, _ = make_arm(drv.Arm, side="right", channel="can_fol_r", logical=AT_REST, smooth=True)
    wrist = np.radians([0, 0, 0, 0, 0, 3.0, 0])
    for k in range(1, 11):                                      # Dream pushes toward the right arm and turns the wrist 3 deg each step: held
        left.write(drv.mapping.to_model("left", INTO + k * wrist), 0.5)
    left.write(drv.mapping.to_model("left", NEAR + 10 * wrist), 0.5)   # Dream comes back to a clear pose: 30 deg of wrist turn from the held target
    assert pair.gap_m(_logical_rad(left), REST)[0] >= pair.MIN_GAP_M - 0.005


def test_an_arm_going_in_and_out_of_a_hold_warns_once_not_every_tick(make_arm, caplog):
    """log-037: hundreds of 'stopped where it keeps 2 cm' lines, one every 50 ms, while the arm went in and out of the hold (log-040)."""
    left, right = _two_arms(make_arm)
    for _ in range(20):
        left.write(drv.mapping.to_model("left", INTO), 0.5)       # held
        left.write(drv.mapping.to_model("left", NEAR), 0.5)       # clear again
    assert sum("other arm" in r.message for r in caplog.records) == 1


def test_two_arms_pushing_toward_each_other_in_turns_keep_2_cm_between_what_they_are_sent(make_arm):
    """log-041: in the live run both arms pushed toward each other; the log read down to -2.8 cm. What each arm is SENT must still keep the gap."""
    left, _ = make_arm(drv.Arm, side="left", channel="can_fol_l", logical=_deg_dict(TURNED_IN_LEFT - np.radians([0, 0, 10.0, 0, 0, 0, 0])))
    right, _ = make_arm(drv.Arm, side="right", channel="can_fol_r", logical=_deg_dict(TURNED_IN_RIGHT + np.radians([0, 0, 10.0, 0, 0, 0, 0])))
    step_l, step_r = np.radians([0, 0, 2.0, 0, 0, 0, 0]), -np.radians([0, 0, 2.0, 0, 0, 0, 0])
    want_l, want_r = TURNED_IN_LEFT - np.radians([0, 0, 10.0, 0, 0, 0, 0]), TURNED_IN_RIGHT + np.radians([0, 0, 10.0, 0, 0, 0, 0])
    worst = np.inf
    for _ in range(20):                                       # Dream asks both wrists further in, 2 deg per step, each arm in turn
        want_l, want_r = want_l + step_l, want_r + step_r
        left.write(drv.mapping.to_model("left", want_l), 1.0)
        right.write(drv.mapping.to_model("right", want_r), 1.0)
        worst = min(worst, pair.gap_m(_logical_rad(left), right._sign * (np.asarray(right._safe_q) - right._zero))[0])
    assert pair.gap_m(want_l, want_r)[0] < 0.0                 # what Dream asked for overlaps
    assert worst >= pair.MIN_GAP_M - 0.005, worst              # what the arms were sent never does


def test_the_warning_says_what_dream_asked_for_and_that_the_arm_keeps_2_cm(make_arm, caplog):
    """log-041: the warning printed the gap of Dream's REQUEST (-2.8 cm) as if the arm were there; the arm kept 2 cm."""
    left, right = _two_arms(make_arm)
    left.write(drv.mapping.to_model("left", INTO), 0.5)
    line = [r.message for r in caplog.records if "other arm" in r.message][0]
    assert "keeps 2 cm from the other arm" in line and "asked for" in line

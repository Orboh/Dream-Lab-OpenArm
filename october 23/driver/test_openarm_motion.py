"""Tests for the smooth-motion maths: limits table, torque budget, the tracker and the waypoint path. Pure numpy, no hardware."""

import math

import numpy as np
import pytest

import openarm_motion as mo

DT = 0.01
NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper")
KP = dict(zip(NAMES, (240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 12.0)))
MODELS = dict(zip(NAMES, ("openarm_dm8009",) * 2 + ("openarm_dm4340",) * 2 + ("openarm_dm4310",) * 4))


def limits():
    return mo.build_limits(KP, MODELS)


def test_gentle_limits_table():
    lim = limits()
    assert lim.vel_dps[:4].tolist() == [40.0] * 4 and lim.vel_dps[4:7].tolist() == [70.0] * 3 and lim.vel_dps[7] == 80.0
    assert lim.acc_dps2[:4].tolist() == [120.0] * 4 and lim.acc_dps2[4:7].tolist() == [250.0] * 3 and lim.acc_dps2[7] == 300.0
    assert lim.names == NAMES


def test_torque_budget_is_half_the_rated_torque_over_the_gain():
    lead = limits().lead_deg
    assert lead[0] == pytest.approx(math.degrees(0.5 * 54 / 240), abs=0.01)    # shoulder: about 6.4 deg
    assert lead[2] == pytest.approx(math.degrees(0.5 * 28 / 240), abs=0.01)    # elbow side: about 3.3 deg
    assert lead[4] == pytest.approx(math.degrees(0.5 * 10 / 24), abs=0.01)     # wrist roll: about 11.9 deg
    assert lead[7] == 8.0                                                      # the gripper's squeeze limit is explicit, not from the budget


def run(tracker, targets, dt=DT):
    xs, vs = [], []
    for target in targets:
        x = tracker.step(np.asarray(target, dtype=float), dt)
        xs.append(x.copy()); vs.append(tracker.v.copy())
    return np.array(xs), np.array(vs)


def test_a_step_target_is_followed_within_the_velocity_and_acceleration_limits_without_overshoot():
    lim = limits()
    tr = mo.SmoothTracker(lim, np.zeros(8))
    target = np.array([30.0, -20.0, 25.0, 40.0, 60.0, -30.0, 50.0, 40.0])
    xs, vs = run(tr, [target] * 1500)
    assert np.all(np.abs(vs) <= lim.vel_dps + 1e-6)
    acc = np.diff(vs, axis=0) / DT
    assert np.all(np.abs(acc) <= lim.acc_dps2 * 1.05)
    assert np.all(np.abs(xs[-1] - target) < 0.05)
    over = np.max((xs - target) * np.sign(target))   # beyond the target in the direction of travel
    assert over < 0.2
    assert tr.settled(target)


def test_motion_starts_and_ends_smoothly_not_at_full_speed():
    tr = mo.SmoothTracker(limits(), np.zeros(8))
    xs, vs = run(tr, [np.full(8, 20.0)] * 800)
    assert np.all(np.abs(vs[0]) <= 3.0)                # the first 10 ms: almost no speed (acceleration times 10 ms)
    assert np.all(np.abs(vs[-1]) < 0.05)               # the end: stopped


def test_dreams_linear_home_ramp_comes_out_bounded_and_settles_after_it_ends():
    """Dream's Home: a linear blend over 3 s in 120 steps (40 Hz), then nothing. The tracker keeps settling on its own."""
    lim = limits()
    origin, home = np.zeros(8), np.array([83.0, 0.0, 5.0, 40.0, 0.0, 0.0, -60.0, 0.0])
    tr = mo.SmoothTracker(lim, origin)
    stream = []
    for step in range(1, 121):
        stream += [origin + (home - origin) * step / 120] * 10          # 25 ms between Dream's writes = 10 ticks of 2.5 ms
    xs, vs = run(tr, stream, dt=0.0025)
    assert np.all(np.abs(vs) <= lim.vel_dps + 1e-6)
    assert np.all(np.abs(np.diff(vs, axis=0)) / 0.0025 <= lim.acc_dps2 * 1.05)
    more, _ = run(tr, [home] * 4000, dt=0.0025)                         # ten more seconds of nobody writing: it still arrives
    assert np.all(np.abs(more[-1] - home) < 0.05)


def test_the_tracker_never_leaves_its_target_when_already_there():
    tr = mo.SmoothTracker(limits(), np.full(8, 5.0))
    xs, vs = run(tr, [np.full(8, 5.0)] * 100)
    assert np.all(np.abs(xs - 5.0) < 1e-9) and tr.settled(np.full(8, 5.0))


def waypoints():
    return np.array([[0, 0, 0, 0, 0, 0, 0], [14, 0, 0, 25, 1, 2, 4], [28, -3, 0, 26, 5, 2, -41], [36, -3, 0, 26, 5, 5, -70], [64, -4, -1, 96, 1, 10, -63], [64, -4, -1, 96, -2, 19, 22]], dtype=float)


def test_smooth_path_ends_exactly_on_the_endpoints_and_respects_the_limits():
    lim = limits()
    pts = waypoints()
    path = mo.smooth_path(pts, lim, DT)
    assert path.samples[0] == pytest.approx(pts[0], abs=1e-6) and path.samples[-1] == pytest.approx(pts[-1], abs=1e-6)
    v = np.diff(path.samples, axis=0) / DT
    assert np.all(np.abs(v) <= lim.vel_dps[:7] * 1.02)
    a = np.diff(v, axis=0) / DT
    assert np.all(np.abs(a) <= lim.acc_dps2[:7] * 1.05)
    assert path.duration == pytest.approx(len(path.samples) * DT, abs=0.05) and path.duration > 3.0


def test_smooth_path_passes_every_waypoint_without_stopping_there():
    lim = limits()
    pts = waypoints()
    path = mo.smooth_path(pts, lim, DT)
    speed = np.linalg.norm(np.diff(path.samples, axis=0) / DT / lim.vel_dps[:7], axis=1)   # normalised speed along the path
    for waypoint in pts[1:-1]:
        i = int(np.argmin(np.max(np.abs(path.samples - waypoint), axis=1)))
        assert np.max(np.abs(path.samples[i] - waypoint)) < 0.6                       # it goes through the waypoint
        lo, hi = max(i - 15, 0), min(i + 15, len(speed) - 1)
        assert speed[min(i, len(speed) - 1)] > 0.3 * min(speed[lo], speed[hi])        # a corner may slow it, but it never stops there


def test_a_longer_path_takes_longer_and_a_zero_length_path_is_one_sample():
    lim = limits()
    short = mo.smooth_path(np.array([[0.0] * 7, [10.0] + [0.0] * 6]), lim, DT)
    long = mo.smooth_path(np.array([[0.0] * 7, [90.0] + [0.0] * 6]), lim, DT)
    assert long.duration > 2 * short.duration
    same = mo.smooth_path(np.array([[1.0] * 7, [1.0] * 7]), lim, DT)
    assert len(same.samples) == 1

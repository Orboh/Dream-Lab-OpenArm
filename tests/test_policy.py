"""Closed-loop Dream-1 policy: chunk conversion, slew limiting, and the runner's safety behaviour.

No network: policies here are local fakes. The real API is never called from the suite.
"""

import threading

import numpy as np
import pytest

from orboh.config import PolicyConfig, SafetyConfig
from orboh.dreamlab.policy import MockPolicyClient, PolicyClient, PolicyError, chunk_to_canonical
from orboh.execution.policy_runner import PolicyRunner
from orboh.execution.slew import slew_limit
from orboh.robot.sim_robot import SimRobot
from orboh.sim.mujoco_sim import MujocoSim
from orboh.types import ARM_IDX, GRIPPER_IDX, NUM_JOINTS, Trajectory, joint_index

SCENE = "pedestal/pedestal.xml"
HZ = 20.0
SAFETY = SafetyConfig()


@pytest.fixture(scope="module")
def sim():
    return MujocoSim(SCENE)


def make_runner(client, sim, execute=True, cfg=None, robot=None):
    robot = robot or SimRobot(SCENE, HZ)
    robot.connect()
    lo, hi = sim.limits()
    runner = PolicyRunner(robot, client, lambda: {}, cfg or PolicyConfig(), SAFETY, lo, hi, MujocoSim(SCENE),
                          execute=execute)
    return runner, robot


class FixedChunkPolicy(PolicyClient):
    """Returns the same chunk every time, built from the pose of the first observation."""

    name = "fixed"
    action_hz = HZ

    def __init__(self, make_chunk, latency_s=0.05):
        self.make_chunk, self.latency_s, self._q0 = make_chunk, latency_s, None
        self.calls = 0

    def open(self, prompt):
        self._q0 = None

    def observe(self, q, jpegs):
        if self._q0 is None:
            self._q0 = np.asarray(q, dtype=float).copy()

    def get_chunk(self):
        self.calls += 1
        threading.Event().wait(self.latency_s)
        return Trajectory(self.make_chunk(self._q0), HZ)

    def refill_due(self, executed, remaining):
        return remaining <= 5


# ------------------------------------------------------------ conversion
def test_chunk_to_canonical_layout():
    left, right = np.full((4, 8), 1.0), np.full((4, 8), 2.0)
    traj = chunk_to_canonical({"q_l_chunk": left, "q_r_chunk": right}, HZ)
    assert traj.positions.shape == (4, NUM_JOINTS)
    assert (traj.positions[:, :8] == 1.0).all() and (traj.positions[:, 8:] == 2.0).all()


@pytest.mark.parametrize("bad", [
    {}, {"q_l_chunk": np.zeros((4, 8)), "q_r_chunk": np.zeros((5, 8))},  # missing / mismatched
    {"q_l_chunk": np.zeros((4, 6)), "q_r_chunk": np.zeros((4, 6))},  # a 6-DoF YAM answer is not ours
])
def test_chunk_to_canonical_rejects_bad_shapes(bad):
    with pytest.raises(PolicyError):
        chunk_to_canonical(bad, HZ)


# ------------------------------------------------------------ slew limiter
def test_slew_limit_respects_velocity_acceleration_and_limits(sim):
    lo, hi = sim.limits()
    q0 = sim.home()
    target = q0.copy()
    target[list(ARM_IDX)] += 2.0  # absurd: far beyond limits and speed caps
    cmds, _ = slew_limit(q0, np.zeros(NUM_JOINTS), np.repeat(target[None], 80, axis=0), HZ, SAFETY, lo, hi)
    full = np.vstack([q0, cmds])
    vel = np.diff(full, axis=0) * HZ
    acc = np.diff(full, n=2, axis=0) * HZ**2
    assert np.abs(vel[:, list(ARM_IDX)]).max() <= SAFETY.max_joint_velocity_rad_s
    assert np.abs(acc[:, list(ARM_IDX)]).max() <= SAFETY.max_joint_accel_rad_s2
    m = SAFETY.joint_limit_margin_rad
    assert (cmds[:, list(ARM_IDX)] <= hi[list(ARM_IDX)] - m + 1e-9).all()
    assert (cmds[:, list(ARM_IDX)] >= lo[list(ARM_IDX)] + m - 1e-9).all()


def test_slew_limit_brakes_without_overshooting(sim):
    lo, hi = sim.limits()
    q0 = sim.home()
    target = q0.copy()
    target[joint_index("right_joint1")] += 0.5
    cmds, _ = slew_limit(q0, np.zeros(NUM_JOINTS), np.repeat(target[None], 120, axis=0), HZ, SAFETY, lo, hi)
    j = joint_index("right_joint1")
    assert cmds[:, j].max() <= target[j] + 0.01
    assert cmds[-1, j] == pytest.approx(target[j], abs=0.01)


def test_slew_limit_caps_gripper_speed(sim):
    lo, hi = sim.limits()
    q0 = sim.home()
    target = q0.copy()
    target[list(GRIPPER_IDX)] = 1.0
    cmds, _ = slew_limit(q0, np.zeros(NUM_JOINTS), np.repeat(target[None], 40, axis=0), HZ, SAFETY, lo, hi)
    step = np.abs(np.diff(np.vstack([q0, cmds]), axis=0))[:, list(GRIPPER_IDX)]
    assert step.max() * HZ <= SAFETY.max_gripper_velocity


# ------------------------------------------------------------ runner
def test_runner_completes_with_mock_policy(sim):
    runner, robot = make_runner(MockPolicyClient(PolicyConfig(), latency_s=0.05), sim)
    q0 = robot.read_joints()
    res = runner.run("test", 4.0, threading.Event())
    assert res.completed, res.reason
    assert res.chunks_received >= 1 and res.chunks_rejected == 0
    assert res.steps_sent >= 4.0 * HZ * 0.9
    assert not np.allclose(robot.read_joints(), q0)  # it actually moved


def test_runner_dry_run_never_moves_the_robot(sim):
    runner, robot = make_runner(MockPolicyClient(PolicyConfig(), latency_s=0.05), sim, execute=False)
    sent = []
    robot.send_joints = lambda q: sent.append(q)
    res = runner.run("test", 2.0, threading.Event())
    assert res.completed and res.dry_run and res.chunks_received >= 1
    assert sent == [] and res.steps_sent == 0


def test_runner_estop_stops_and_holds(sim):
    runner, robot = make_runner(MockPolicyClient(PolicyConfig(), latency_s=0.05), sim)
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    res = runner.run("test", 30.0, stop)
    assert not res.completed and "E-STOP" in res.reason and res.duration_s < 5


def collide(q0):
    """Left and right joint3 swung together: the same arm-arm collision the static tests use."""
    out = np.tile(q0, (48, 1))
    ramp = np.linspace(0, 0.4, 48)
    out[:, joint_index("left_joint3")] += ramp
    out[:, joint_index("right_joint3")] -= ramp
    return out


def test_runner_halts_after_repeated_unsafe_chunks_without_moving(sim):
    client = FixedChunkPolicy(collide)
    runner, robot = make_runner(client, sim, cfg=PolicyConfig(max_check_failures=2))
    q0 = robot.read_joints()
    res = runner.run("test", 20.0, threading.Event())
    assert not res.completed and "failed the safety check" in res.reason
    assert res.chunks_rejected == 2
    # nothing unsafe reached the robot: only gravity sag while holding (well inside max_state_drift_rad)
    assert np.abs(robot.read_joints() - q0).max() < SAFETY.max_state_drift_rad


def test_runner_rejects_non_finite_chunks(sim):
    def nan_chunk(q0):
        out = np.tile(q0, (48, 1))
        out[10, 0] = np.nan
        return out

    runner, robot = make_runner(FixedChunkPolicy(nan_chunk), sim, cfg=PolicyConfig(max_check_failures=1))
    res = runner.run("test", 10.0, threading.Event())
    assert not res.completed and res.chunks_rejected == 1


def test_runner_halts_when_policy_goes_silent(sim):
    class Silent(FixedChunkPolicy):
        def get_chunk(self):
            raise PolicyError("network down")

    runner, _ = make_runner(Silent(lambda q0: np.tile(q0, (4, 1))), sim)
    res = runner.run("test", 10.0, threading.Event())
    assert not res.completed and "network down" in res.reason


def test_snap_gripper_uses_demo_values_with_hysteresis():
    """gripper_mode 'demos': Dream's open/close decision becomes the demo open (0.74) / squeeze (0.25) value, no chatter."""
    from orboh.config import PolicyConfig
    from orboh.execution.policy_runner import snap_gripper
    from orboh.types import GRIPPER_IDX

    cfg = PolicyConfig(gripper_mode="demos")
    pos = np.zeros((7, 16))
    pos[:, GRIPPER_IDX[1]] = [0.8, 0.7, 0.5, 0.3, 0.45, 0.55, 0.7]  # right gripper: open ... closes below 0.40 ... reopens above 0.60
    pos[:, GRIPPER_IDX[0]] = 0.5  # left stays in the dead band: keeps the carried-over state
    out, state = snap_gripper(pos, cfg, (True, False))
    assert out[:, GRIPPER_IDX[1]].tolist() == [0.74, 0.74, 0.74, 0.25, 0.25, 0.25, 0.74]
    assert out[:, GRIPPER_IDX[0]].tolist() == [0.25] * 7 and state == (True, False)
    assert (out[:, [i for i in range(16) if i not in GRIPPER_IDX]] == 0).all()  # arm joints untouched


def test_rejected_plan_is_fed_back_to_dream_and_retried_until_safe(sim):
    """Dream-1 sorts out its own failures: a rejected plan is explained to it and it is asked again; a safe one then runs."""
    seen = []

    class Flaky(FixedChunkPolicy):
        def get_chunk(self):
            self.make_chunk = collide if self.calls < 2 else (lambda q0: np.tile(q0, (48, 1)))  # two bad plans, then a safe one
            return super().get_chunk()

        def feedback(self, text):
            seen.append(text)

    runner, robot = make_runner(Flaky(collide), sim, cfg=PolicyConfig(max_check_failures=5))
    res = runner.run("test", 6.0, threading.Event())
    assert res.completed and res.chunks_rejected == 2, res.reason
    assert [s is None for s in seen] == [False, False, True]  # a hint after each rejection, cleared once it got it right
    assert "collide" in seen[0] and any("retry" in e for e in res.events) and any("safe plan after 2 retries" in e for e in res.events)


def test_retry_hint_names_the_problem():
    from orboh.execution.policy_runner import retry_hint

    assert "table" in retry_hint("sim_collisions: 12 colliding steps; first at t=1s: env table <-> openarm_left_link6")
    assert "arms" in retry_hint("sim_collisions: x arm_arm openarm_left_link3 <-> openarm_right_link3")
    assert "slower" in retry_hint("sim_tracking: max tracking error 1.2 rad")


def test_slowed_scales_the_speed_limits_and_the_runner_obeys_them(sim):
    """speed_scale: the arms may move at a fraction of the safety limits, and the commands that reach the robot respect it."""
    from orboh.execution.slew import slowed

    slow = slowed(SAFETY, 0.3)
    assert slow.max_joint_velocity_rad_s == pytest.approx(SAFETY.max_joint_velocity_rad_s * 0.3)
    assert slow.max_joint_accel_rad_s2 == pytest.approx(SAFETY.max_joint_accel_rad_s2 * 0.3)
    assert slowed(SAFETY, 5.0).max_joint_velocity_rad_s == SAFETY.max_joint_velocity_rad_s  # never faster than the limits
    assert slowed(SAFETY, 0.0).max_joint_velocity_rad_s == pytest.approx(SAFETY.max_joint_velocity_rad_s * 0.1)

    sent = []

    class Recording(SimRobot):
        def send_joints(self, q):
            sent.append(np.asarray(q).copy())
            super().send_joints(q)

    def stride(q0):  # asks for a big move: 1 rad on right joint 1 within the chunk
        out = np.tile(q0, (48, 1))
        out[:, joint_index("right_joint1")] += np.linspace(0, 1.0, 48)
        return out

    robot = Recording(SCENE, HZ)
    robot.connect()
    lo, hi = sim.limits()
    runner = PolicyRunner(robot, FixedChunkPolicy(stride), lambda: {}, PolicyConfig(), slow, lo, hi, MujocoSim(SCENE), execute=True)
    res = runner.run("test", 4.0, threading.Event())
    assert res.completed, res.reason
    d = np.abs(np.diff(np.array(sent[:-1]), axis=0)) * HZ  # the last command is the end-of-run 'hold the measured pose', not motion
    peak = d[:, list(ARM_IDX)].max()
    assert d[:, list(GRIPPER_IDX)].max() <= slow.max_gripper_velocity * 1.05
    assert 0.05 < peak <= slow.max_joint_velocity_rad_s * 1.05, peak  # it moved, and never faster than the slowed limit

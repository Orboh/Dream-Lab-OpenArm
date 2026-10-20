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

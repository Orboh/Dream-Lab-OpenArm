"""ZmqRobot against scripts/robot_bridge.py --fake (no hardware, no LeRobot needed)."""

import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orboh.config import Config
from orboh.robot import make_robot
from orboh.robot.zmq_robot import BridgeError, ZmqRobot

BRIDGE = Path(__file__).resolve().parent.parent / "scripts" / "robot_bridge.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def robot():
    cfg = Config()
    cfg.robot.backend = "bridge"
    cfg.robot.lerobot.joint_units = "degrees"
    cfg.robot.lerobot.gripper_open = -65.0
    cfg.robot.bridge.endpoint = f"tcp://127.0.0.1:{_free_port()}"
    cfg.robot.bridge.timeout_ms = 1500
    proc = subprocess.Popen([sys.executable, str(BRIDGE), "--fake", "--endpoint", cfg.robot.bridge.endpoint],
                            stderr=subprocess.DEVNULL)
    r = make_robot(cfg)
    assert isinstance(r, ZmqRobot)
    try:
        r.connect()
        yield r
    finally:
        r.disconnect()
        proc.terminate()
        proc.wait(timeout=5)


def test_roundtrip_converts_units(robot):
    q = robot.read_joints()
    assert q.shape == (16,)
    assert q[3] == pytest.approx(np.deg2rad(36))  # fork initial pose: elbow 36 deg
    target = q.copy()
    target[0], target[8], target[7] = 0.3, -0.2, 1.0  # left j1, right j1, left gripper fully open
    robot.send_joints(target)
    assert robot.read_joints() == pytest.approx(target, abs=1e-9)


def test_unreachable_bridge_raises():
    cfg = Config()
    cfg.robot.bridge.endpoint = f"tcp://127.0.0.1:{_free_port()}"
    cfg.robot.bridge.timeout_ms = 200
    r = ZmqRobot(cfg.robot.lerobot, cfg.robot.bridge)
    with pytest.raises(BridgeError):
        r.connect()
    assert not r.is_connected

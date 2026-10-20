from __future__ import annotations

import numpy as np

from orboh.robot.base import Robot
from orboh.sim.mujoco_sim import MujocoSim


class SimRobot(Robot):
    """Fake robot backed by MuJoCo: each command advances physics by one control period."""

    name = "sim"

    def __init__(self, scene: str, control_hz: float, table_drop_m: float = 0.0, torso_tolerance_m: float = 0.0):
        self.sim = MujocoSim(scene, table_drop_m, torso_tolerance_m)
        self.period = 1.0 / control_hz
        self._connected = False

    def connect(self) -> None:
        self.sim.reset()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_joints(self) -> np.ndarray:
        return self.sim.joints()

    def send_joints(self, q: np.ndarray) -> None:
        self.sim.set_command(q)
        self.sim.step(self.period)

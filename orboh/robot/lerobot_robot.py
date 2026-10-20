"""Real bimanual OpenArm through the Orboh LeRobot fork (Damiao motors over SocketCAN).

Written against LeRobot's public Robot API (connect / get_observation /
send_action / disconnect). The fork's robot type name and kwargs come from
config.robot.lerobot — VERIFY them once the fork is available.
Cameras are NOT opened here: orboh.cameras owns the RealSense devices so the
robot and the camera stream never fight over them.
"""

from __future__ import annotations

import numpy as np

from orboh.config import LeRobotConfig
from orboh.robot.base import Robot
from orboh.robot.mapping import JointMapping


class LeRobotRobot(Robot):
    name = "lerobot"

    def __init__(self, cfg: LeRobotConfig):
        self.cfg = cfg
        self.mapping = JointMapping(cfg)
        self._robot = None

    def connect(self) -> None:
        try:
            from lerobot.robots import RobotConfig, make_robot_from_config
        except ImportError as e:
            raise RuntimeError(
                "LeRobot is not installed in this environment. On the robot box run: "
                "pip install -e ~/projects/lerobot  (Orboh fork, branch feat/bi-openarm-soft-move-to)"
            ) from e
        config_cls = RobotConfig.get_choice_class(self.cfg.robot_type)
        robot_cfg = config_cls(**self.cfg.robot_kwargs, cameras={})
        self._robot = make_robot_from_config(robot_cfg)
        self._robot.connect()

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and getattr(self._robot, "is_connected", True)

    def read_joints(self) -> np.ndarray:
        obs = self._robot.get_observation()
        return self.mapping.to_canonical(obs)

    def send_joints(self, q: np.ndarray) -> None:
        self._robot.send_action(self.mapping.from_canonical(q))

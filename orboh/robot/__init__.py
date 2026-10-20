from orboh.config import Config
from orboh.robot.base import Robot


def make_robot(cfg: Config) -> Robot:
    if cfg.robot.backend == "sim":
        from orboh.robot.sim_robot import SimRobot

        return SimRobot(cfg.sim.scene, cfg.robot.control_hz)
    from orboh.robot.lerobot_robot import LeRobotRobot

    return LeRobotRobot(cfg.robot.lerobot)

from orboh.config import Config
from orboh.robot.base import Robot


def make_robot(cfg: Config) -> Robot:
    if cfg.robot.backend == "sim":
        from orboh.robot.sim_robot import SimRobot

        return SimRobot(cfg.sim.scene, cfg.robot.control_hz)
    if cfg.robot.backend == "bridge":
        from orboh.robot.zmq_robot import ZmqRobot

        return ZmqRobot(cfg.robot.lerobot, cfg.robot.bridge)
    from orboh.robot.lerobot_robot import LeRobotRobot

    return LeRobotRobot(cfg.robot.lerobot)

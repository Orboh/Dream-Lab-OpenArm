"""Robot that lives in another process: talks to scripts/robot_bridge.py over ZMQ.

The bridge runs in the LeRobot environment (SocketCAN + Damiao motors) and speaks the fork's own keys and units
(degrees); JointMapping converts them to canonical units here, exactly as for an in-process LeRobot robot.
"""

from __future__ import annotations

import threading

import numpy as np
import zmq

from orboh.config import LeRobotConfig, RobotBridgeConfig
from orboh.robot.base import Robot
from orboh.robot.mapping import JointMapping


class BridgeError(RuntimeError):
    pass


class ZmqRobot(Robot):
    name = "bridge"

    def __init__(self, cfg: LeRobotConfig, bridge: RobotBridgeConfig):
        self.mapping = JointMapping(cfg)
        self.bridge = bridge
        self._sock: zmq.Socket | None = None
        self._lock = threading.Lock()  # REQ/REP is strictly alternating: one request in flight at a time

    def _open(self) -> None:
        sock = zmq.Context.instance().socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.bridge.endpoint)
        self._sock = sock

    def _close(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    def _call(self, req: dict) -> dict:
        with self._lock:
            if self._sock is None:
                raise BridgeError("bridge not connected")
            try:
                self._sock.send_json(req)
                if not self._sock.poll(self.bridge.timeout_ms):
                    raise BridgeError(f"bridge at {self.bridge.endpoint} did not answer within "
                                      f"{self.bridge.timeout_ms} ms")
                reply = self._sock.recv_json()
            except (BridgeError, zmq.ZMQError):
                self._close()  # a REQ socket that missed a reply is stuck; the next connect() starts clean
                raise
        if not reply.get("ok"):
            raise BridgeError(reply.get("error", "bridge error"))
        return reply

    def connect(self) -> None:
        self._close()
        self._open()
        try:
            reply = self._call({"op": "ping"})
            self.read_joints()
        except Exception:
            self._close()
            raise
        self.bridge_mode = reply.get("mode", "?")

    def disconnect(self) -> None:
        self._close()  # the bridge keeps the arms; stop it with its own shutdown / Ctrl-C

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    def read_joints(self) -> np.ndarray:
        return self.mapping.to_canonical(self._call({"op": "read"})["obs"])

    def send_joints(self, q: np.ndarray) -> None:
        self._call({"op": "send", "action": self.mapping.from_canonical(q)})

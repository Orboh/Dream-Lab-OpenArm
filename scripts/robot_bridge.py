#!/usr/bin/env python
"""Robot bridge: owns the real OpenArm follower arms and serves them to Orboh over ZMQ.

Why a separate process: the LeRobot fork (SocketCAN, Damiao motors) lives in ~/projects/lerobot/.venv, while the
Dream client needs its own Python. Orboh talks to this process through orboh.robot.zmq_robot.ZmqRobot.

Run it with the LeRobot venv's interpreter:
  ~/projects/lerobot/.venv/bin/python scripts/robot_bridge.py --fake                # no hardware, for tests
  ~/projects/lerobot/.venv/bin/python scripts/robot_bridge.py --real                # real arms (see warning)

WARNING (--real): connecting runs the fork's startup alignment, so both arms MOVE to their initial pose. Clear the
workspace, keep a hand on the power switch, and make sure `sudo bash scripts/canup.sh` has brought can_fol_l /
can_fol_r up. Every command is also clamped to --max-relative-deg per step by the fork itself.

Protocol: REQ/REP, one JSON object per message.
  {"op": "ping"}                      -> {"ok": true, "mode": "fake"|"real"}
  {"op": "read"}                      -> {"ok": true, "obs": {"left_joint_1.pos": deg, ..., "right_gripper.pos": deg}}
  {"op": "send", "action": {...}}     -> {"ok": true, "sent": {...}}   (same keys and units as `obs`)
  {"op": "shutdown"}                  -> {"ok": true}, then the arms disconnect and the bridge exits
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
import zmq

log = logging.getLogger("robot_bridge")

SIDES = ("left", "right")
MOTORS = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
KEYS = [f"{side}_{m}.pos" for side in SIDES for m in MOTORS]


class FakeArms:
    """Stand-in that reaches each commanded target instantly, starting from the fork's initial pose (degrees)."""

    mode = "fake"

    def __init__(self, left_pose: str | None = None, right_pose: str | None = None):
        self.q = {k: 0.0 for k in KEYS}
        for side in SIDES:
            self.q[f"{side}_joint_4.pos"] = 36.0  # fork default: elbow 36 deg, everything else 0
        for side, path in (("left", left_pose), ("right", right_pose)):
            if path:  # same YAML the real arms align to: joint_1..joint_7 (+ gripper) in degrees
                pose = yaml.safe_load(Path(path).read_text())
                for m in MOTORS:
                    if m in pose:
                        self.q[f"{side}_{m}.pos"] = float(pose[m])

    def connect(self) -> None:
        pass

    def read(self) -> dict[str, float]:
        return dict(self.q)

    def send(self, action: dict[str, float]) -> dict[str, float]:
        bad = [k for k in action if k not in self.q]
        if bad:
            raise KeyError(f"unknown action keys {bad}")
        self.q.update({k: float(v) for k, v in action.items()})
        return dict(action)

    def disconnect(self) -> None:
        pass


class RealArms:
    """Bimanual OpenArm follower from the Orboh LeRobot fork (feat/bi-openarm-soft-move-to)."""

    mode = "real"

    def __init__(self, left_port: str, right_port: str, max_relative_deg: float,
                 left_pose: str | None, right_pose: str | None):
        from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
        from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

        def arm(port: str, side: str, pose: str | None) -> OpenArmFollowerConfigBase:
            return OpenArmFollowerConfigBase(port=port, side=side, max_relative_target=max_relative_deg,
                                             initial_pose_path=pose)

        cfg = BiOpenArmFollowerConfig(id="follower", left_arm_config=arm(left_port, "left", left_pose),
                                      right_arm_config=arm(right_port, "right", right_pose))
        self._robot = BiOpenArmFollower(cfg)
        self._ports, self._bus_checked = (left_port, right_port), 0.0

    def connect(self) -> None:
        self._robot.connect()

    def _check_buses(self) -> None:
        now = time.monotonic()
        if now - self._bus_checked < 0.5:
            return
        self._bus_checked = now
        for port in self._ports:
            out = subprocess.run(["ip", "-details", "-json", "link", "show", port], capture_output=True, text=True).stdout
            if (why := bus_problem(out)) is not None:
                raise RuntimeError(f"{port}: {why}")

    def read(self) -> dict[str, float]:
        self._check_buses()
        return {k: float(v) for k, v in self._robot.get_observation().items() if k.endswith(".pos")}

    def send(self, action: dict[str, float]) -> dict[str, float]:
        self._check_buses()
        return {k: float(v) for k, v in self._robot.send_action(action).items()}

    def disconnect(self) -> None:
        if self._robot.is_connected:
            self._robot.disconnect()


def bus_problem(link_json: str) -> str | None:
    """Why a CAN interface cannot be trusted, from `ip -details -json link show <iface>` (None = healthy).

    When the motors lose power or the cable comes off, nobody acknowledges the frames: the bus goes ERROR-PASSIVE / BUS-OFF and the fork
    keeps returning the *last known* joint values, so the arms would look fine while being blind. Refuse to read or command then."""
    try:
        state = json.loads(link_json)[0]["linkinfo"]["info_data"]["state"]
    except (ValueError, KeyError, IndexError, TypeError):
        return "cannot read the CAN interface state"
    return None if state == "ERROR-ACTIVE" else f"CAN bus is {state}: the motors are not answering (power off? cable?)"


def handle(arms, req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "mode": arms.mode}
    if op == "read":
        return {"ok": True, "obs": arms.read()}
    if op == "send":
        return {"ok": True, "sent": arms.send(req["action"])}
    return {"ok": False, "error": f"unknown op {op!r}"}


def serve(arms, endpoint: str) -> None:
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(endpoint)
    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    log.info("bridge (%s) listening on %s", arms.mode, endpoint)
    while not stop:
        if not sock.poll(200):
            continue
        try:
            req = json.loads(sock.recv())
            if req.get("op") == "shutdown":
                sock.send_json({"ok": True})
                break
            reply = handle(arms, req)
        except Exception as e:  # keep serving: a bad request must not drop the arms
            log.exception("request failed")
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sock.send_json(reply)
    sock.close(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fake", action="store_true", help="simulated arms, no hardware")
    mode.add_argument("--real", action="store_true", help="real arms: they move to their initial pose on connect")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5560")
    ap.add_argument("--left-port", default="can_fol_l")
    ap.add_argument("--right-port", default="can_fol_r")
    ap.add_argument("--max-relative-deg", type=float, default=10.0, help="per-command clamp enforced by the fork")
    ap.add_argument("--left-pose", help="start pose YAML for the left arm (openarm-orboh config/pose_sets/<set>/left.yaml); "
                    "real: arm aligns to it on connect, fake: arm starts there")
    ap.add_argument("--right-pose", help="start pose YAML for the right arm")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    arms = FakeArms(args.left_pose, args.right_pose) if args.fake else RealArms(args.left_port, args.right_port, args.max_relative_deg,
                                                 args.left_pose, args.right_pose)
    arms.connect()
    try:
        serve(arms, args.endpoint)
    finally:
        log.info("disconnecting arms")
        arms.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())

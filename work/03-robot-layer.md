[← instruction.md](instruction.md)

# 03 · Robot layer

**Code:** `orboh/types.py`, `orboh/robot/`

## Canonical joint vector

Defined in `orboh/types.py` (`JOINT_NAMES`, `ARM_IDX`, `GRIPPER_IDX`):

| Index | Names | Unit |
|---|---|---|
| 0–6 | `left_joint1` … `left_joint7` | rad (MuJoCo OpenArm convention) |
| 7 | `left_gripper` | 0 closed … 1 open |
| 8–14 | `right_joint1` … `right_joint7` | rad |
| 15 | `right_gripper` | 0 … 1 |

`Trajectory` holds a `(T, 16)` array at a fixed `hz`, with `resample(hz, speed_scale)`.
`min_jerk()` produces smooth point-to-point moves.

## Robot interface

`orboh/robot/base.py`: `Robot`, with these methods:
- `connect()` / `disconnect()`
- `read_joints() -> (16,)`
- `send_joints(q)`

Everything uses canonical units. `make_robot(cfg)` in `orboh/robot/__init__.py` picks the backend.

| Backend | File | Behavior |
|---|---|---|
| `sim` | `sim_robot.py` | MuJoCo twin ([07](07-simulation.md)); each `send_joints` advances physics by one control period |
| `bridge` | `zmq_robot.py` | real arms in a separate process (`scripts/robot_bridge.py`) over ZMQ; **recommended** ([15](15-real-robot.md)) |
| `lerobot` | `lerobot_robot.py` | Orboh LeRobot fork in-process; needs `lerobot` importable in this venv |

### Bridge backend
- `ZmqRobot` (REQ socket, `robot.bridge.endpoint`, `timeout_ms`) sends `ping` / `read` / `send` JSON requests;
  `JointMapping` converts the fork's degrees to canonical units. A missed reply closes the socket; `connect()` reopens.
- Why it exists and how to run it: [15](15-real-robot.md).

### LeRobot backend (in-process)
- The fork's robot takes nested `left_arm_config` / `right_arm_config`; `robot_kwargs` as written for it
  (`left_arm_port`) does not apply, which is one reason the bridge is preferred.
- Builds the fork's robot through `RobotConfig.get_choice_class(robot_type)(**robot_kwargs, cameras={})`,
  then calls `make_robot_from_config`.
- **Cameras are deliberately not given to the robot.** `orboh.cameras` owns the RealSense devices
  ([04](04-cameras.md)), so the two never compete for them.
- Uses LeRobot's public API: `get_observation()` and `send_action(dict)`.

## Joint mapping (units at the edges)

`orboh/robot/mapping.py` (`JointMapping`) converts in both directions between canonical vectors
and LeRobot `{key: value}` dicts:
- **Arm joints:** `canonical = sign · value · scale + offset`. `scale` is π/180 when `joint_units: degrees`.
- **Grippers:** `(value − gripper_closed) / (gripper_open − gripper_closed)`, clipped to 0..1.
- **Key names:** built from `joint_key` / `gripper_key` templates (`{side}`, `{index}`).

The same mapping converts recorded datasets ([05](05-demos-recording.md)), so demos, live state and
commands always agree.

## Adding a backend (e.g. dora-rs)
1. Subclass `Robot` in `orboh/robot/<name>_robot.py`; convert units with `JointMapping`.
2. Register it in `make_robot()` and add it to `RobotConfig.backend` in `config.py`.
3. `send_joints` must return quickly; the executor owns timing ([09](09-execution.md)).
4. Add a test ([12](12-testing.md)).

## Related
- [02 Configuration → VERIFY](02-configuration.md#verify-checklist) · [07 Simulation](07-simulation.md) · [09 Execution](09-execution.md)

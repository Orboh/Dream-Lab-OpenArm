# Orboh — bimanual OpenArm one-shot manipulation

> **Full documentation: [`work/instruction.md`](work/instruction.md)** — start there.

Record one teleop demo per object with the leader arms. DreamLab turns that demo into a
motion for the current scene, MuJoCo checks the motion is safe, and the follower arms run it.
Everything is driven from a web console.

```
 leader arms ──teleop──► collect_bimanual.sh ──► LeRobot v3.0 dataset (data/demos)
                                                        │
 RealSense: center + right_wrist ──(discovery)──┐       ▼
                                                ├──► DreamLab API ──► plan (16-joint trajectory)
 follower joints ───────────────────────────────┘                           │
                                                                            ▼
              approach + plan ──► static checks ──► MuJoCo rollout ──► operator approval
                                                                            │
                                     runtime supervisor + E-STOP ◄──────────┘ ──► follower arms
```

## Quick start (laptop, no hardware)

```bash
bash scripts/setup_env.sh          # creates .venv and .env
source .venv/bin/activate
orboh sample-demos                 # synthetic demos so the flow works without recordings
MUJOCO_GL=egl orboh serve --host 127.0.0.1
# open http://127.0.0.1:8000
```

In `config/pipeline.yaml`, `robot.backend: sim` uses a MuJoCo twin of the bimanual OpenArm
(the official `openarm-mujoco` model) as the robot, and its cameras as the "RealSense" feeds.
`dreamlab.backend: mock` replays the demo verbatim until the real API is wired in.

## On the robot box

```bash
bash scripts/setup_env.sh --robot  # + pyrealsense2 and the Orboh LeRobot fork (~/projects/lerobot)
# edit .env:  DREAMLAB_API_KEY, DREAMLAB_BASE_URL, ORBOH_WEB_TOKEN
# edit config/pipeline.yaml: robot.backend: lerobot, then check every line marked VERIFY
orboh preflight
orboh discover-cameras --wiggle
orboh serve                        # reachable from your PC at http://<robot-box>:8000
```

## Console steps

| Step | What it does |
|---|---|
| 1 Preflight | robot link, joints inside limits, both camera roles streaming, DreamLab health, sim model, dataset |
| 2 Cameras | lists RealSense devices and assigns `center` / `right_wrist`: pinned serials → D405 model heuristic → **wiggle test** (opens/closes the right gripper; the camera whose view changes most is the wrist). Saves `config/cameras.yaml`, including the `--robot.cameras` string for `lerobot-record` |
| 3 Demo | runs the team's `collect_bimanual.sh` (camera devices are released to it while it runs) and lists recorded episodes |
| 4 Plan | sends the demo, current joints and camera JPEGs to DreamLab |
| 5 Sim check | builds the **exact** command sequence (min-jerk approach from the current pose + plan at the chosen speed). Checks limits, velocity and acceleration, then MuJoCo rolls it out for self, arm-arm and environment collisions and tracking error |
| 6 Execute | allowed only if step 5 passed, the trajectory hash matches, the robot hasn't moved since, and the operator ticked approval. The runtime supervisor clamps limits and per-step speed and halts on tracking error or watchdog overrun. **Space = E-STOP** |

Every run is saved under `runs/<time>_ep<N>/`: `plan.json`, `safety_report.json`, sim preview frames and `execution.json`.

## Conventions

The canonical joint vector is `[left_joint1..7, left_gripper, right_joint1..7, right_gripper]`.
Arm joints are in radians (MuJoCo OpenArm convention); grippers run from 0 (closed) to 1 (open).
`orboh/robot/mapping.py` converts to and from LeRobot keys and units; `joint_key`, `joint_units`,
`joint_signs` and `joint_offsets` are set in the config.

## Still open

- **DreamLab API.** Fill in `HttpDreamLabClient._build_payload` / `_parse_response` in
  `orboh/dreamlab/client.py` and the endpoints in config. Nothing else needs to change.
- **LeRobot fork.** Verify `robot_type`, `robot_kwargs`, key names, units and gripper range
  (`orboh/robot/lerobot_robot.py`).
- **Sim ↔ real.** The default scene is the stock pedestal + table. Model your real workcell
  (table height and position, fixtures) in an MJCF and point `sim.scene` at it; collision
  checks are only as good as that model. Check that joint signs and the gripper open
  direction match the real arm.
- Isaac Sim is not wired in. MuJoCo runs on the robot box (including aarch64/Jetson); Isaac needs an x86 RTX machine.

## Tests

```bash
MUJOCO_GL=egl pytest -q
```

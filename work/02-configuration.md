[← instruction.md](instruction.md)

# 02 · Configuration

- **Settings:** `config/pipeline.yaml`
- **Secrets:** `.env` (see [01](01-environment-setup.md))
- **Schema and defaults:** `orboh/config.py` (pydantic models; unknown or invalid values fail at startup)
- **Paths:** relative paths resolve against the project root (`Config.path()`).

## Sections

| Section | Key settings | Documented in |
|---|---|---|
| `robot` | `backend: sim\|lerobot`, `control_hz`, `lerobot.*` (type, kwargs, key templates, units, signs/offsets, gripper range) | [03 Robot layer](03-robot-layer.md) |
| `cameras` | `expected_roles`, resolution/fps, `pinned_serials`, `discovery.motion_joint/amplitude`, `output` | [04 Cameras](04-cameras.md) |
| `dreamlab` | `backend: mock\|http`, `timeout_s`, `plan_endpoint`, `health_endpoint`, `send_images` | [06 DreamLab](06-dreamlab.md) |
| `demos` | `dataset_root` | [05 Demos](05-demos-recording.md) |
| `recording` | `command` (with `{mode}` = dry\|run) | [05 Demos](05-demos-recording.md) |
| `sim` | `engine: mujoco`, `scene` | [07 Simulation](07-simulation.md) |
| `safety` | limits, margins, tracking thresholds, watchdog, gates | [08 Safety](08-safety.md) |
| `web` | `host`, `port` (token comes from `.env`) | [11 Web console](11-web-console.md) |
| `runs_dir` | where per-run logs go | [10 Pipeline](10-pipeline.md) |

## Safety values (defaults)

| Key | Default | Meaning |
|---|---|---|
| `joint_limit_margin_rad` | 0.05 | stay this far inside the model's joint limits |
| `max_joint_velocity_rad_s` | 1.2 | peak arm joint speed allowed in a plan |
| `max_joint_accel_rad_s2` | 8.0 | peak arm joint acceleration |
| `max_gripper_velocity` | 3.0 | normalized gripper units per second |
| `approach_speed_rad_s` | 0.3 | peak speed of the move into the plan's start pose |
| `max_plan_duration_s` | 120 | reject longer plans |
| `sim_max_tracking_error_rad` | 0.25 | MuJoCo must follow the command within this |
| `runtime_max_tracking_error_rad` | 0.35 | real robot halts beyond this |
| `max_state_drift_rad` | 0.05 | robot moved since check → re-check required |
| `watchdog_s` | 0.25 | one control iteration must finish within this |
| `allow_finger_env_contact` | true | gripper touching the table is a warning, not a failure |
| `require_sim_pass` / `require_manual_approval` | true / true | execution gates (keep on) |
| `sim.table_drop_m` | 0.075 | lowers the twin's table by this many metres = how much higher the real arms stand than the twin assumes (estimated from a real demo; see [07](07-simulation.md)) |
| `sim.torso_tolerance_m` | 0.02 | arm-vs-torso overlap allowed in the check (twin's torso mesh is stricter than reality); table etc. stay strict |
| `conform_plan` | true | slew-limit + clip every plan to the limits before the checks; logs the max deviation (`plan.meta.conformed_max_deviation_rad`) |

## VERIFY checklist

Each line marked `VERIFY` in the YAML is an assumption, not yet confirmed on hardware:

- [x] `robot.lerobot.robot_type`: `bi_openarm_follower` (confirmed in the fork, 2026-10-20)
- [x] `robot.lerobot.robot_kwargs`: not used by the bridge; CAN ports are `scripts/start_bridge.sh` args (`can_fol_l`, `can_fol_r`)
- [x] `robot.lerobot.joint_key` / `gripper_key`: `left_joint_1.pos` / `left_gripper.pos` (read from the fork)
- [x] `robot.lerobot.joint_units`: degrees
- [x] `robot.lerobot.gripper_closed` / `gripper_open`: `0` / `-65`: direction confirmed from 8 real demos (open = negative, holds the wood at about -32 deg), see [18](18-grasp-planner.md)
- [ ] `joint_signs` / `joint_offsets`: real vs MuJoCo joint direction and zero (see [07](07-simulation.md#sim-to-real-alignment))
- [ ] `dreamlab.plan_endpoint` / `health_endpoint`
- [ ] `recording.command`: path to `collect_bimanual.sh`

New sections: `robot.bridge` (`endpoint`, `timeout_ms`) and `policy` ([14](14-policy.md)); `cameras.pinned_serials`
now holds the Kitakyushu serials.

Tick each item here and remove its `VERIFY` tag in the YAML once confirmed.

## Adding a setting
1. Add a field with a safe default to the right model in `orboh/config.py`.
2. Add it, with a comment, to `config/pipeline.yaml`.
3. Document it in this file and in the topic document.

## Related
- [01 Environment](01-environment-setup.md) · [08 Safety](08-safety.md) · [13 Roadmap](13-roadmap.md)

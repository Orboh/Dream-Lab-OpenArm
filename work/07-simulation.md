[← instruction.md](instruction.md)

# 07 · Simulation (MuJoCo twin)

**Code:** `orboh/sim/mujoco_sim.py` · **Model:** `openarm-mujoco` (official Enactic package, Apache-2.0)

## Why MuJoCo (not Isaac Sim)
- It runs on the robot box (including aarch64/Jetson) and on laptops, with no RTX GPU needed.
- Enactic ships an official bimanual MJCF: 16 position actuators matching our canonical layout.
- Isaac Sim needs an x86 RTX machine; it remains a possible future addition ([13](13-roadmap.md)).

## Scene
`sim.scene` is either a path to an MJCF file or a path inside the package, e.g.:
- `pedestal/move_puck_scene.xml` (default): pedestal + table + puck, with a `tablecam` camera
- `pedestal/pedestal.xml`: arms only, plus a `home` keyframe

## `MujocoSim` API

| Method | Purpose |
|---|---|
| `limits()` | arm joint limits from the model; grippers 0..1 |
| `home()` | pose from the model's `home` keyframe |
| `reset(q)` / `set_command(q)` / `step(s)` / `joints()` | state control, in canonical units |
| `contacts()` | `(kind, body_a, body_b)`; see the classification below |
| `render(w, h, camera)` | RGB image, or `None` without a GL context |

Gripper mapping: canonical 0 → the joint bound nearest zero, 1 → the far bound. **VERIFY** that
this is "closed → open" on the real gripper.

## Contact classification (used by the safety rollout)

| kind | Meaning | Verdict |
|---|---|---|
| `ok` | same-gripper fingers, adjacent links, robot ↔ free object, object ↔ table | allowed |
| `finger_env` | gripper ↔ static environment (table) | warning (configurable) |
| `self` | non-adjacent links of the same arm | **fail** |
| `arm_arm` | left arm ↔ right arm | **fail** |
| `env` | non-gripper link ↔ environment (pedestal, table) | **fail** |

"Free object" means a body with a free joint, i.e. something meant to be manipulated.

## Rendering & threads
GL contexts are bound to the thread that created them, so all rendering runs on one dedicated
thread (`_render_thread`). Without that, camera streams from web worker threads came out black.
Set `MUJOCO_GL=egl` (GPU) or `osmesa` (CPU).

## Sim-to-real alignment
The collision check is only as good as the model of your workcell:
1. **Workcell scene:** copy a package scene and set the real table height, position and fixtures,
   and the camera poses. Point `sim.scene` at it.
2. **Joint directions and zeros:**
   - Move each real joint a little and compare the sign with the sim.
   - Fix any mismatch with `joint_signs` / `joint_offsets` ([03](03-robot-layer.md)).
3. **Hardware version:** confirm the model version (v2 package vs your arms' version).

## Findings from a real leader demo (2026-10-20)

Replaying a freshly recorded Kitakyushu wood-L1 demo (`demos.dataset_root` = the LeRobot dataset) through
`orboh --backend bridge check` with the fake bridge started at the real start pose (`scripts/start_bridge.sh fake`):
- Static checks pass once the plan is conformed ([02](02-configuration.md), `safety.conform_plan`; raw replay failed on
  `left_joint2` at the limit margin and on jerk of 27–50 rad/s² in older demos).
- The MuJoCo rollout still fails: the twin's collision meshes are stricter than the real arms. The same joint
  values put `body_link0` against left `link4`/`link5` by 3–15 mm and `table_top` against the left `ee_base_link`
  by up to 19 mm. The real arms did this motion without a collision, so a 1–2 cm model error (simplified
  `body_link0_symp` mesh, and the real table height relative to the arm base, which is not recorded) is the likely cause.
- **Table height (the main error):** with `table_drop_m: 0.02` the left fingers still sank 53 mm into the twin's table
  and the jammed wrist stalled the whole rollout. A drop of ~0.073 m puts them on the table, so **the real arms stand
  about 7.3 cm higher above the table than the twin assumes** (the team's "arm had to be raised"). Set 0.075.
- **Torso:** left `link4`/`link5` still overlap `body_link0` by up to 15.5 mm (checked kinematically over the whole
  demo). With `torso_tolerance_m: 0` the check fails on it; with 0.02 (arm links vs torso only, implemented as a
  separate penetration test because the torso geom is removed from MuJoCo's contact pass) the whole check passes:
  no collisions, tracking error 0.218 rad (limit 0.25). The renders showed the idle arm folded tight against the torso.
- Do **not** loosen the check to pass: measure the base plate thickness / plate-tip → arm-base offset, correct the twin's
  table height, then re-run. Start the check from the real pose, never from the sim home pose (that adds an invented approach).

## Related
- [08 Safety](08-safety.md) · [04 Cameras](04-cameras.md) (sim camera sources) · [03 Robot layer](03-robot-layer.md)

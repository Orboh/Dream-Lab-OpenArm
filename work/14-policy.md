[← instruction.md](instruction.md)

# 14 · Dream-1 policy (closed loop)

**Code:** `orboh/dreamlab/policy.py`, `orboh/execution/policy_runner.py`, `orboh/execution/slew.py`,
`orboh run-policy` in `orboh/cli.py`. **Tests:** `tests/test_policy.py`.

## What Dream-1 is (and is not)

Dream-1 is a **policy, not a detector or planner**. Every few hundred ms it takes camera frames, the joint
state and a text prompt, and returns an **action chunk**: 48 rows at 20 Hz (2.4 s) of joint targets for both arms.
It has **no "I found the wooden block" output** (no boxes, labels or scores). The only way to see whether it
reacts to the scene is to change the scene and watch the returned targets change
(`scripts/camera_probe_web.py` plots them).

The base model is trained on 6-DoF YAM arms. For OpenArm the client library runs its own IK on the model's
end-effector answer (`policy.ik_arm: openarm2_faas`), so a chunk is only as good as that IK and the match between
the model's idea of the scene and the real one.

## Pieces

| Piece | File | Role |
|---|---|---|
| `PolicyClient` | `dreamlab/policy.py` | `open(prompt)` → `observe(q, jpegs)` → `get_chunk()` → canonical `(H, 16)` `Trajectory` |
| `DreamPolicyClient` | same | the real API through `dreamlabs-client` (needs `.env` `DREAMLAB_API_KEY`) |
| `MockPolicyClient` | same | offline sway of the right arm; exercises the whole loop without the network |
| `PolicyRunner` | `execution/policy_runner.py` | control thread + observer thread + fetcher thread; every chunk is slew-limited, sim-checked, then queued |
| `slew_limit` | `execution/slew.py` | velocity / acceleration / braking / joint-limit shaping of each chunk |

A chunk that fails the sim check is dropped and the arms hold. Three rejections in a row (`max_check_failures`)
or no usable chunk for `starve_s` halt the run. Every row sent to the robot also passes the runtime supervisor
([08](08-safety.md)).

## Snapshot of the cameras when a task starts
`Pipeline._save_snapshot`: at the start of every run one frame per Dream camera slot (`overhead`, `wrist_l`, `wrist_r`; the same frames
Dream is given, so a missing camera shows up as a slot with no frame, "Dream-1 gets a black frame there") is saved to
`runs/<time>_policy/snapshot/` with `snapshot.json` (prompt, time, 16 joint values, speed, backend, gripper mode, dry-run/simulation) and
announced as a `snapshot` event. It is a record and a chat preview, not Dream's input: Dream-1 keeps getting live frames during the run.

## Movement speed (slow by default)
`policy.speed_scale` (0.3 by default; the *speed* select in both web views, `run-policy --speed`, `POST /api/policy/run {speed}`) scales the
safety speed limits for a run: joint velocity and gripper speed × scale, acceleration × scale (gentler starts). At 0.3 an arm joint
moves at most 0.36 rad/s (about 21°/s) instead of 1.2 rad/s. Presets: very slow 0.2, slow 0.3, medium 0.6, full 1.0 (never above the
safety limits). Dream-1's plan is still followed, just more slowly, so the arm lags the plan and Dream re-plans from where the arm
really is; use a longer run (`seconds`, the web views default to 40 s) because less ground is covered per second. The safety check
judges the slowed commands. The final end-of-run "hold the measured pose" command is a correction, not motion.

## Failures: Dream-1 is told and asked again (retries)
A plan the safety check rejects is not the end of the run. The runner explains the rejection to Dream-1 (`retry_hint`: "the last plan
pushed the arm into the table: keep the arm above the table", arm-arm, box/body, too fast, joint limit), appends it to the prompt Dream
observes with (`policy.retry_hint`, Dream-1 has no other feedback channel), keeps the arms holding, and asks again. The next safe plan
clears the hint. Up to `policy.max_check_failures` (8) consecutive rejections are tolerated, `starve_s` (25 s) is the time budget. The
simple chat view prints each retry ("asked it again, retry 2/7") and offers `retry` when it finally gives up.
Whether a model that has no feedback input actually *changes* its plan because of the hint is unproven: the plans do vary from call to
call, but nothing guarantees a collision-free one.

**Measured (2026-10-20, dry-run via the chat view, real Dream-1, prompt "Pick up the wood", 20 s):** 18 plans, 3 rejected (a finger into
the world geometry twice, tracking 0.52 rad once), each one fixed by the next ask: the run **completed**. Before retries, three rejections
in a row ended runs. A real run is offered once a dry-run reached a safe plan, even if its first plan was rejected.

## Gripper: Dream decides, or the amounts from the demos
`policy.gripper_mode` (and the *gripper* select in both web views): `model` uses what Dream-1 commands; `demos` keeps only Dream's
open-or-close **decision** (below `gripper_close_below` 0.40 = close, above `gripper_open_above` 0.60 = open, hysteresis in between) and
replaces the amount by the values measured in the 8 recorded demos ([18](18-grasp-planner.md)): open 0.74 (≈ −48°), squeeze 0.25
(≈ −16°; the follower stalls on the block at ≈ −32°). The snapped chunk is what the safety check judges and what is sent. Dream-1's raw
gripper numbers are untested on OpenArm (it is trained on other grippers), which is why this option exists; it is a dry-run key
(changing it needs a new dry-run). Dream-1 has **no input** for "how far to move the gripper": the amount can only be overridden on its output.

## Config (`policy:` in `config/pipeline.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `backend` | `mock` | `mock` or `dream` (real API) |
| `checkpoint` | `dreamlabs/dream-1-beta-flash` | `…/dream-1-beta` is the slower, larger one |
| `prompt` | `pick up the object` | override with `--prompt` |
| `camera_map` | `overhead: center, wrist_l: left_wrist, wrist_r: right_wrist` | Dream slot → Orboh camera role; a slot with no camera gets a **black frame** |
| `sim_cameras` | | Dream slot → MuJoCo camera (sim backend) |
| `action_hz`, `execution_horizon` | 20, 30 | chunk rate; rows executed before the next chunk is requested |
| `max_duration_s` | 30 | run length unless `--seconds` is given |

## Web console
It can also run Dream-1 on the simulation only ("Run on: simulation only"), with no gates, for trying a prompt safely.
The console has a Dream-1 panel: type the task, dry-run, then approve a real run ([11](11-web-console.md#dream-1-panel-the-top-card-tell-the-robot-what-to-do)).
The web path enforces *dry-run first with the same prompt*; the CLI below does not (each CLI call is a fresh process).

## Running

```bash
# 0. offline, no network, no robot
orboh run-policy --policy-backend mock --dry-run
# 1. real Dream, no robot motion (needs the robot or the fake bridge for the joint state, see 15)
orboh --backend bridge run-policy --policy-backend dream --dry-run --seconds 12 --prompt "Pick up the wood and place it in the box"
# 2. real motion: only after a clean dry-run, short, with someone at the e-stop
orboh --backend bridge run-policy --policy-backend dream --approve --seconds 10 --prompt "..."
```

`--dry-run` queries the policy and sim-checks each chunk **from the current pose**, but sends nothing. Because
the arms never move in a dry-run, later chunks (which assume the arms already moved) are checked from the start
pose, so they can fail where a real run would not. Early chunks passing is the meaningful signal.

## Reading a dry-run

| Output | Meaning |
|---|---|
| `dry-run: chunk N passes` | slew limits, joint limits and the MuJoCo rollout are fine for this chunk |
| `sim_collisions … env table <-> openarm_*_linkN` | the chunk drives that link into the table in the twin |
| `sim_tracking … on <joint> at t=…` | MuJoCo cannot follow the command within `sim_max_tracking_error_rad` |
| `no camera for Dream slot(s) […]: the model gets a black frame` | cameras not opened: check `cameras.pinned_serials` ([04](04-cameras.md)) |
| `pool_warming` / retries | the model is cold; the runner waits up to 120 s |

## Dry-run with the calibrated twin (2026-10-20, fake bridge at the real start pose, real cameras + real Dream-1)
Prompt "Pick up the wood and place it in the box", 14 s: 8 chunks, **the first passed**, 3 were rejected (a right finger against
the world geometry at t≈1.35 s, tracking 0.35 rad). That is what a dry-run is for; read each rejection reason, and remember later
chunks are judged from the unmoved start pose. Whether Dream-1 actually heads for the block is still unproven.

## Gotchas found on the real setup (2026-10-20)

- **Test from the real start pose.** A dry-run starting from the fork's default pose (elbow 36°) failed every
  chunk (arm into the table, 1.45 rad tracking error). The same hold from the real pose set
  `kitakyushu-wood-l1-v1` passes. Always give the bridge the pose files ([15](15-real-robot.md)).
- **Latency:** a round trip is ≈2.2–2.4 s, of which ≈2.1 s is gateway and ≈0.3 s is model inference. The 2.4 s
  chunk hides most of it, but corrections are slow.
- **Wrist cameras:** `409122272149` is the **left** D405 and `352122273417` the **right** (from the team's env
  file). Swapped wrists feed the model the wrong view.
- **Language prompt:** the team uses one fixed instruction per material, e.g. "Pick up the wood and place it
  in the box"; keep the same sentence across levels.

## Related
- [06 DreamLab API](06-dreamlab.md) (the one-shot plan client) · [08 Safety](08-safety.md) · [09 Execution](09-execution.md) · [15 Real robot](15-real-robot.md)

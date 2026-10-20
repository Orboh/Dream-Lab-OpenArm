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

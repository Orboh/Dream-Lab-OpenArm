# Orboh — Main Instructions

Start here. This file is the map of the project. Each topic has its own document
in this folder; every document links back here.

**Goal:** Bimanual OpenArm picks up objects using **one-shot learning**:
1. The leader arms record one teleop demo of the task.
2. **DreamLab** turns that demo into a motion for the current scene.
3. **MuJoCo** checks that motion is safe.
4. The follower arms execute it.

Everything is driven from a **web console**.

---

## 1. Documents

| # | Document | What it covers |
|---|---|---|
| 01 | [Environment setup](01-environment-setup.md) | `.venv`, `.env` secrets, laptop vs robot box install |
| 02 | [Configuration](02-configuration.md) | `config/pipeline.yaml`, every section and the `VERIFY` items |
| 03 | [Robot layer](03-robot-layer.md) | sim + LeRobot backends, joint mapping, units |
| 04 | [Cameras](04-cameras.md) | RealSense enumeration, center / right_wrist discovery, wiggle test |
| 05 | [Demos & recording](05-demos-recording.md) | leader teleop recording, LeRobot v3.0 dataset reader, synthetic demos |
| 06 | [DreamLab API](06-dreamlab.md) | client contract, mock backend, wiring the real API |
| 07 | [Simulation](07-simulation.md) | MuJoCo twin, scenes, contacts, rendering, workcell model |
| 08 | [Safety](08-safety.md) | static checks, sim rollout, gates, runtime supervisor, E-STOP |
| 09 | [Execution](09-execution.md) | approach trajectory, control loop, halting |
| 10 | [Pipeline orchestrator](10-pipeline.md) | stage state machine, tasks, run logs |
| 11 | [Web console](11-web-console.md) | REST / WebSocket API, UI, auth token |
| 12 | [Testing](12-testing.md) | test suite, what each test proves, how to add tests |
| 13 | [Roadmap & open items](13-roadmap.md) | what's blocked, next steps, decisions log |

---

## 2. How the pieces connect

```
 leader arms ─teleop─► collect_bimanual.sh ─► LeRobot v3.0 dataset ──────[05]
                                                     │
 RealSense center + right_wrist ─discovery─[04]┐     ▼
 follower joints ─────────────────────[03]─────┴─► DreamLab ─► plan ─────[06]
                                                                 │
        approach + plan ─► static checks ─► MuJoCo rollout ──────┤  [08][07]
                                                                 ▼
                     operator approval (web) ─► executor + supervisor ─► arms
                            [11]                     [09][08]

           orchestrated by the pipeline [10] · configured by [02]
```

**Data contract:** every stage uses the same **canonical joint vector** (16 values):
`[left_joint1..7, left_gripper, right_joint1..7, right_gripper]`
- arm joints in radians (MuJoCo OpenArm convention)
- grippers from 0 = closed to 1 = open

Conversion to and from robot or dataset units happens only at the edges ([03](03-robot-layer.md)).

---

## 3. Quick start

Laptop, no hardware:
```bash
bash scripts/setup_env.sh && source .venv/bin/activate
orboh sample-demos                         # synthetic demos
MUJOCO_GL=egl orboh serve --host 127.0.0.1 # open http://127.0.0.1:8000
```
Robot box:
1. Follow [01 → robot box](01-environment-setup.md#robot-box).
2. Set `robot.backend: lerobot` in the config.
3. Clear every `VERIFY` line ([02](02-configuration.md#verify-checklist)).
4. Run `orboh preflight`.

---

## 4. Operator workflow (web console)

1. **Preflight**: everything green? ([10](10-pipeline.md#preflight))
2. **Cameras**: discover; confirm center / right_wrist from the thumbnails ([04](04-cameras.md))
3. **Demo**: record one demo per object with the leader arms ([05](05-demos-recording.md))
4. **Plan**: choose a demo and get the motion from DreamLab ([06](06-dreamlab.md))
5. **Sim check**: choose a speed, then review the checks, preview frames and joint plot ([08](08-safety.md))
6. **Execute**: tick approval and execute. **Space = E-STOP** ([09](09-execution.md))

Never skip step 5. The software enforces this: execution is refused unless the
checked trajectory, the current robot pose and the approval all line up.

---

## 5. Repository layout

```
Orboh/
├── work/                 ← documentation (this folder)
├── config/pipeline.yaml  ← all settings            [02]
├── .env / .env.example   ← secrets (git-ignored)    [01]
├── requirements*.txt     ← pinned libraries         [01]
├── scripts/setup_env.sh  ← creates .venv + .env     [01]
├── orboh/
│   ├── config.py         ← config models + loader   [02]
│   ├── types.py          ← joint layout, Trajectory [03]
│   ├── robot/            ← sim / lerobot backends   [03]
│   ├── cameras/          ← RealSense + discovery    [04]
│   ├── demos/            ← dataset reader, samples  [05]
│   ├── dreamlab/         ← API client               [06]
│   ├── sim/              ← MuJoCo twin              [07]
│   ├── safety/           ← checks + sim rollout     [08]
│   ├── execution/        ← executor                 [09]
│   ├── pipeline.py       ← orchestrator             [10]
│   ├── web/              ← FastAPI + static UI      [11]
│   └── cli.py            ← `orboh` command          [10]
├── tests/                ← pytest suite             [12]
├── data/demos/           ← demo datasets (ignored)  [05]
└── runs/                 ← per-run logs (ignored)   [10]
```

---

## 6. Command reference

| Command | Purpose |
|---|---|
| `orboh serve` | start the web console |
| `orboh preflight` | run preflight checks, exit 1 on failure |
| `orboh discover-cameras [--wiggle]` | assign camera roles, write `config/cameras.yaml` |
| `orboh sample-demos` | write synthetic demos to `data/demos` |
| `orboh check --episode N --speed 0.5` | plan + sim safety check, **no robot motion** |
| `MUJOCO_GL=egl pytest -q` | run the test suite |

All commands accept `--backend sim|lerobot` and `--config path`.

---

## 7. Safety rules (non-negotiable)

1. A real-robot run requires a **passed sim check** of the *exact* command sequence.
2. If the robot moves after the check (drift > `max_state_drift_rad`), the check must be re-run.
3. An operator must explicitly approve each execution.
4. E-STOP must always work, even during other tasks; it halts and holds position.
5. Utility moves (home, wiggle test) go through the same checks.
6. The web console needs `ORBOH_WEB_TOKEN` whenever it listens on the network.

Details and rationale: [08 Safety](08-safety.md).

---

## 8. Maintenance rules

- **Code and docs change together.** When you change a module, update its document here in the same change.
- **New functionality:**
  - add a new numbered document (`14-…md`);
  - add a row to the table in §1;
  - link it from the related documents.
- **This file must stay under 175 lines.** Put detail in topic documents, not here.
- **Every topic document** starts with a back-link to this file and ends with a "Related" list.
- **Config:** new config keys get a default in `orboh/config.py`, a commented entry in `pipeline.yaml`, and a line in [02](02-configuration.md).
- **Tests:** every behavior change gets a test ([12](12-testing.md)). Run `pytest` before committing.
- **`VERIFY`:** use it for anything assumed rather than confirmed on hardware. Resolve and remove it once confirmed.
- **Decisions:** record notable decisions in [13 → decisions log](13-roadmap.md#decisions-log).

---

## 9. Current status (2026-10-20)

- ✅ Full flow works in sim: preflight → cameras → demo → plan → sim check → execute → E-STOP.
- ✅ 19 tests pass.
- ⏳ **Blocked on:** DreamLab API docs, the Orboh LeRobot fork details, and a model of the real workcell.

See [13 Roadmap](13-roadmap.md).

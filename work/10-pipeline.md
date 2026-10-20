[← instruction.md](instruction.md)

# 10 · Pipeline orchestrator

**Code:** `orboh/pipeline.py` (`Pipeline`), `orboh/cli.py`

## Stages
`STEPS = preflight, cameras, demo, plan, check, execute`. Each step has a
`{status: pending|running|ok|warn|fail, detail}` entry, shown in the console sidebar.

| Method | Stage | Notes |
|---|---|---|
| `start()` / `shutdown()` | — | connect the robot, reopen saved cameras / stop everything |
| `preflight()` | preflight | see below |
| `discover_cameras(wiggle)` / `set_camera_roles()` | cameras | [04](04-cameras.md) |
| `record_demo(mode)` / `list_demos()` / `make_sample_demos()` | demo | [05](05-demos-recording.md) |
| `make_plan(episode, task)` | plan | clears any previous check/execution; creates the run directory |
| `check(speed_scale)` | check | [08](08-safety.md) |
| `execute(approved)` / `gate()` | execute | [09](09-execution.md) |
| `go_home()`, `estop()` | — | the home move is safety-checked; E-STOP is always accepted |

## Preflight
Checks: `robot_connection`, `joints_within_limits`, `camera_<role>` (a frame was received),
`dreamlab_api` (health), `sim_model`, and `demo_dataset` (a warning only when empty).

## Tasks & concurrency
- **One task at a time.** Long operations run through `run_task(name, fn)` on a background
  thread. A second request gets `Busy` (HTTP 409), and the console disables its buttons while
  a task runs.
- **Exceptions are contained.** A failing task is logged (message plus a short traceback at
  `debug` level) and never crashes the server.
- **E-STOP** bypasses the task lock.

## Events
`subscribe(fn)` receives these events:
- `{"type": "log", level, msg}`: levels are `info`, `warn`, `error`, `rec` (recording output) and `debug`;
- `{"type": "state"}`: something changed;
- `{"type": "progress", done, total}`.

The web server forwards them over a WebSocket ([11](11-web-console.md)). `state()` returns the
full JSON snapshot.

## Run logs: `runs/<YYYYmmdd-HHMMSS>_ep<N>/`

| File | Written by |
|---|---|
| `plan.json` | `make_plan`: task, episode, source, meta, trajectory |
| `safety_report.json` | `check`: every check, fingerprint, start pose, sim summary |
| `sim_preview_<i>.jpg` | `check`: MuJoCo frames |
| `execution.json` | `execute`: result |

Keep runs from the real robot. Together they give a demo-vs-execution history for evaluation.

## CLI (`orboh`)
See the command table in [instruction.md §6](instruction.md#6-command-reference). `orboh check`
plans and checks **without moving the robot**, which is handy for testing DreamLab output offline.

## Related
- [11 Web console](11-web-console.md) · [08 Safety](08-safety.md) · [12 Testing](12-testing.md)

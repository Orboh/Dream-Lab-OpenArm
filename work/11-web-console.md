[← instruction.md](instruction.md)

# 11 · Web console

**Code:** `orboh/web/server.py` (FastAPI), `orboh/web/static/index.html` (single page, vanilla JS)
**Start:** `orboh serve` → `http://<robot-box>:8000`

## Simple chat view (`/simple`)
`http://<host>:8000/simple`: the live RealSense centre camera (plus the two wrist cameras and, in simulation mode, the twin) on the left,
a chat on the right. Type a task in plain words: it is **dry-run** with Dream-1 first (nothing moves) and the chat reports each plan
and the verdict. If the first plan passed, type **go** to run it on the real robot (typing it is the approval), **sim** to watch it in the
simulation, **stop** to E-STOP (**release** lifts it). If Dream-1's plan is rejected, the chat shows it, Dream-1 is told why and asked again (retries), and **retry** asks it again after a failure. Header selects: real robot | simulation only, speed (very slow | **slow** (default) | medium | full), gripper (Dream decides | amounts from demos), model.
It uses the same endpoints and the same gates as the main console below (dry-run of the same prompt, gripper mode and model first).

## Dream-1 panel (the top card, "tell the robot what to do")
Type any task and let Dream-1 decide how to do it ([14](14-policy.md)). Flow:
1. **Dry-run** (default): queries Dream and sim-checks every action chunk from the current pose; **nothing moves**. The chunks'
   joint targets are plotted live (solid = left arm, dashed = right) and each gets a ✔/✖ with the safety reason.
2. If **a plan passed** the safety check (after any retries), the **Run on robot** button unlocks for **that exact prompt and model** only. It also needs the
   approval tick-box. A real run uses up the dry-run: look again before the next one. (`policy.require_dry_run_first`.)
3. E-STOP (Space) works throughout; a rejected chunk makes the arms hold; three in a row stop the run.
**Run on: real robot | simulation only.** *Real robot* is the default and is the gated flow above. *Simulation only* runs the same
Dream-1 loop on a fresh MuJoCo robot: it starts at the real arms' joint state, the block is placed where the depth camera found it
(else the scene default), Dream-1 sees **rendered** cameras, and the live twin view appears next to the centre camera. Nothing real
moves, so there is no dry-run or approval step (the Joints panel then shows the *simulated* arms' joints), and a simulation neither unlocks nor uses up a real run
(`POST /api/policy/run {mode: "sim"}`, `GET /api/sim/live.jpg`). Dream-1 sees the simulated scene, not your real table.
The card also shows the live centre camera and the block finder's reading (`GET /api/detect`: object, height, robot-frame position
when calibrated, [16](16-perception.md)). Model select: `Dream-1 (real API)` or `mock (offline)`.
A dry-run does not move the arms, so **later chunks are judged from the start pose** and can fail where a real run would not.

## Layout
- **Header:**
  - live-connection dot;
  - robot and DreamLab backend chips;
  - current task;
  - **E-STOP** (also the Space bar) / Release.
- **Left:** the six-step progress list, with each step's status and detail.
- **Center:** one card per step.
  - Preflight checklist.
  - Live camera streams with role override.
  - Demo table with record buttons.
  - Plan.
  - Sim check: speed slider, verdict, checks, preview frames, joint plot (command vs sim vs approach).
  - Execute: approval checkbox and progress.
- **Right:** live joint readout and the log stream.

## Auth
If `ORBOH_WEB_TOKEN` is set, every API call needs `Authorization: Bearer <token>`. Images, streams
and the WebSocket use `?token=` instead. The page asks for the token once and stores it in
`localStorage`. Tokens are compared in constant time. `orboh serve` warns if the token is empty
while the console listens on the network.

## REST API

| Method & path | Action |
|---|---|
| `GET /api/state` · `GET /api/logs` | full state snapshot · recent logs |
| `POST /api/estop` · `POST /api/estop/reset` | E-STOP / release |
| `POST /api/preflight` | run preflight |
| `POST /api/cameras/discover {wiggle}` | discovery ([04](04-cameras.md)) |
| `POST /api/cameras/roles {assignments}` | manual role override |
| `GET /api/cameras/{role}/snapshot.jpg` · `/stream?fps=` | JPEG · MJPEG stream |
| `GET /api/demos` · `POST /api/demos/sample` · `POST /api/demos/record {mode}` | demos ([05](05-demos-recording.md)) |
| `POST /api/plan {episode, task}` | DreamLab plan ([06](06-dreamlab.md)) |
| `POST /api/check {speed_scale}` | safety check ([08](08-safety.md)) |
| `GET /api/trajectory` · `GET /api/sim/frame/{i}.jpg` | plot data · preview frames |
| `POST /api/execute {approved}` | gated execution; **412** with the reason if a gate fails |
| `POST /api/policy/run {prompt, seconds, dry_run, approved, backend, gripper, mode}` | Dream-1 run; **dry_run defaults true**; **412** if a real run lacks approval or a passed dry-run of the same prompt+backend |
| `GET /api/sim/live.jpg` | latest view of the simulated robot (simulation-only runs) |
| `GET /api/policy/chunks` · `GET /api/detect` | last chunks (with verdicts) · block finder result (**409** without a depth stream) |
| `POST /api/home` | safety-checked move to the home pose |
| `WS /ws` | events: `state`, `log`, `progress`, `chunk` |

**Response codes:**
- Long actions return `{"started": name}` immediately.
- **409** means another task is running.
- **401** means a missing or wrong token.

## Extending the UI
- Keep it dependency-free (no build step). Add a card inside `<main>` and render it from `render(s)`.
- New actions:
  1. add an endpoint in `server.py` that goes through `task()`;
  2. add a `Pipeline` method;
  3. add a test.

## Related
- [10 Pipeline](10-pipeline.md) · [08 Safety](08-safety.md) · [01 Environment](01-environment-setup.md) (token)

[← instruction.md](instruction.md)

# 11 · Web console

**Code:** `orboh/web/server.py` (FastAPI), `orboh/web/static/index.html` (single page, vanilla JS)
**Start:** `orboh serve` → `http://<robot-box>:8000`

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
| `POST /api/home` | safety-checked move to the home pose |
| `WS /ws` | events: `state`, `log`, `progress` |

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

[← instruction.md](instruction.md)

# 06 · DreamLab API

**Code:** `orboh/dreamlab/client.py` · **Config:** `dreamlab.*` · **Secrets:** `DREAMLAB_API_KEY`, `DREAMLAB_BASE_URL`

## Role in the pipeline
DreamLab does **zero/one-shot learning**. Given how the leader manipulated an object once (the demo),
it produces the manipulation for the *current* scene. Its output is just a **proposal**: it always
goes through the safety checks ([08](08-safety.md)) before reaching the robot.

## Contract (what the rest of the code relies on)

```python
client.plan(PlanRequest) -> Plan
PlanRequest(task: str, demo: DemoEpisode, joints: (16,), images: {role: jpeg_bytes})
Plan(trajectory: Trajectory (T,16) canonical @ hz, source: str, meta: dict)
client.health() -> (ok: bool, message: str)
```

| Backend | Class | Behavior |
|---|---|---|
| `mock` | `MockDreamLabClient` | returns the demo motion verbatim (no adaptation); lets the full pipeline run today |
| `http` | `HttpDreamLabClient` | real API over HTTPS, bearer auth, timeout `timeout_s` |

## Wiring the real API: only two methods change

All API-specific code is isolated in `HttpDreamLabClient`:

1. **`_build_payload(req)`**: the current placeholder JSON:
   `task`, `joint_names`, `demo.{fps, actions, task}`, `observation.{joints, images(base64)}`.
2. **`_parse_response(body)`**: currently expects `{"trajectory": {"positions": [[16]…], "hz": N}}`.
3. Set `plan_endpoint` / `health_endpoint` in the config, then `backend: http`.

**Questions to answer from the docs:**
- **Input format:** does it take the demo as video, as joint data, or both?
- **Frames:** if it needs demo video frames, use `DemoEpisode.videos` (path and timestamps).
- **Output type:** joint trajectory, end-effector poses, or action chunks?
  - End-effector poses need an IK step (a new module, plus a document 14).
  - Action chunks need a streaming / closed-loop executor (extend [09](09-execution.md)).
- **Workflow:** is it synchronous, or does it use a job id that you poll?
- **Limits:** rate limits and size limits.

## Error handling
`DreamLabError` is raised for network errors, non-2xx responses (body truncated to 500 characters)
and unexpected response shapes. The pipeline marks the Plan step failed and logs the reason.

## Logged per run
`runs/<run>/plan.json` holds the task, episode, source, `meta` and the trajectory ([10](10-pipeline.md)).

## Related
- [05 Demos](05-demos-recording.md) · [04 Cameras](04-cameras.md) · [08 Safety](08-safety.md) · [13 Roadmap](13-roadmap.md)

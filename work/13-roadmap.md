[← instruction.md](instruction.md)

# 13 · Roadmap & open items

## Blocked on input

| Item | Needed from | Unblocks |
|---|---|---|
| DreamLab API docs (inputs, outputs, endpoints, auth, sync vs job) | user | `HttpDreamLabClient` ([06](06-dreamlab.md)) |
| Gripper open direction and `joint_signs` / `joint_offsets` on the real arms | team | trusting real-arm runs ([15](15-real-robot.md)) |
| Path and arguments of `collect_bimanual.sh` | team | recording from the console ([05](05-demos-recording.md)) |
| Real workcell dimensions (base plate thickness, plate tip → arm base offset) and a twin scene with the box + carpet | user | trustworthy collision checks ([07](07-simulation.md#sim-to-real-alignment)) |

## Next steps
0. **Real-arm Dream run:** fake-bridge dry-run done (chunks 1–3 pass, later ones hit the table in the twin). Next: build the
   twin scene with the real table/box, dry-run on the real arms' joint state, then a short `--approve` run ([15](15-real-robot.md)).
1. Wire the real DreamLab API; test offline with `orboh check` on recorded demos.
2. Bring up on the robot box (see [12 → first bring-up](12-testing.md#on-the-robot-first-bring-up)).
3. Build a workcell MJCF scene that matches reality.
4. Verify success after execution: check the result with the center camera, with retries.
5. Evaluate the success rate across object positions and new objects, using `runs/` logs.
6. Optional:
   - a dora-rs robot backend (team branch `feat/dora-collection`);
   - IK, if DreamLab returns end-effector poses;
   - an Isaac Sim engine.

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-10-20 | Dream-1 decides and executes any prompt typed in the web console; our own grasp planner is parked | user decision. Dream-1 takes only frames, joint state and a prompt (no object-position input); our planner could not yet pass the safety check |
| 2026-10-20 | Classical (depth-plane) block finder first; GoPro out of scope | one object on plain carpet; live: 20/20 frames, 0.1 mm jitter; no model/GPU needed |
| 2026-10-20 | Camera calibration from depth plane + recorded demos (~25 mm) | no hands-on step needed; the 0.700 m camera height independently confirms `sim.table_drop_m` 0.075; hands-on touch refinement is the next step |
| 2026-10-20 | Calibration set aside as stale; `calibrate_camera.py check` added | the chest camera moved 48 mm after calibration; redoing from the old demos is invalid; a box-corner PnP alternative failed validation (105 mm / 14.6°). Needs new demos |
| 2026-10-20 | Gripper values taken from the 8 real demos (open = negative, contact ≈ −32°, squeeze ≈ −16°) | measured, not assumed ([18](18-grasp-planner.md)) |
| 2026-10-20 | Build on the team's LeRobot fork, not ROS 2 | the team already controls Damiao/SocketCAN from Python; ROS 2 is only borrowed for `kdl_parser` |
| 2026-10-20 | MuJoCo for the safety sim | runs on Jetson and laptops; official bimanual OpenArm MJCF exists |
| 2026-10-20 | Canonical joint vector = MuJoCo radians + 0..1 grippers | one unit system inside; convert only at the edges |
| 2026-10-20 | Sim checks the exact command sequence (approach + plan) and execution is fingerprint-gated | what was checked is exactly what runs |
| 2026-10-20 | Wiggle test uses the right gripper, not an arm joint | 0.55 vs 0.03 separation in sim (an elbow move gave ~0.01 each); no arm motion needed |
| 2026-10-20 | Mock DreamLab = replay the demo | the full flow is testable before the API is wired |
| 2026-10-20 | Web console in vanilla JS with no build step | easy to maintain on the robot box |
| 2026-10-20 | Robot bridge process (ZMQ) instead of one interpreter | `dreamlabs` needs Python ≥3.11 (forced 3.10 install fails on scipy); the LeRobot venv is separate; option chosen by the user |
| 2026-10-20 | Always dry-run from the real start pose | the fork's default pose failed every chunk in the twin; the real pose set passes |
| 2026-10-20 | MuJoCo renders on the NVIDIA GPU via EGL when available | CPU (Mesa) rendering took ~150 ms per frame and blocked the physics step: with the console's live views open the sim run tripped the 250 ms watchdog and the arm never moved. GPU: ~2 ms |
| 2026-10-20 | MuJoCo renders through EGL by default (`orboh/sim/__init__.py`) | incident: with GLFW the "simulated" camera image was a desktop screenshot, and one simulation-only test run sent such frames to Dream-1. Fixed + regression test |
| 2026-10-20 | Docs live in `work/`; `instruction.md` ≤ 175 lines | user request: well maintained and structured |

## Related
- [instruction.md §9 status](instruction.md#9-current-status-2026-10-20) · [02 Configuration](02-configuration.md)

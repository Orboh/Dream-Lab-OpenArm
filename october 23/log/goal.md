# Goal — Dream-1/OpenArm setup and measured pick (2026-10-27)

## Outcome today
Build a repeatable right-arm station setup and obtain one recorded pick attempt only if every gate passes. A failed gate is a valid outcome: record and fix its cause before another run. Treat old logs, including log-030, as history rather than proof of today's live setup.

## Instructions to the implementing agent
Inspect the live station.yaml, active zones, arm lock, installed Dream/client versions, driver/model mapping, camera sources, and newest runtime logs before edits. Record effective paths, timestamps, hashes and backups. Add missing checks, instrumentation, tests, and rollback steps. Change one factor per trial. Start offline and read-only; real-arm motion requires the operator present, a clear workspace, and a working stop. Never copy credentials into logs.

Dream-1's base model on a non-YAM arm masks robot state from the model, while its client still uses the real pose and IK to execute model output. Useful camera views and correct kinematics are separate requirements. Home does not calibrate cameras or create stereo triangulation.

## Gate 1 — Establish the running state
**Why:** Saved bundles may differ from files in force; movement may come from startup, Home, task, Reset, or shutdown.

**Check:** Confirm driver class, custom.ik, embodiment, checkpoint/action space, joint mapping, Home, gripper endpoints, camera roles/sources, speed limits, zone age, CAN and motor status, camera exclusivity, and right-only lock in every phase. Back up live configuration and run preflight.

**Pass:** Effective files and hashes match the intended setup, preflight passes, and selected-arm and shutdown behavior are known. **Fail:** Fix the mismatch or lifecycle behavior before a policy run; record what moved and when.

## Gate 2 — Driver, geometry, safety
**Why:** The current gripper-stall alarm can stop a normal grasp. Endpoint-only zone checks may miss unsafe smoothed poses. Wrong TCP or joint mapping separates the modeled hand from real fingers.

**Check:** With fake motors, test joint units/signs, grip 0-closed/1-open, ordinary object contact versus faults, alarm clearing, heat/fault stops, lock during startup/task/Reset/park/torque-off. Check every executed setpoint and complete gripper/link envelope against zones, including raise and park. In several bent poses compare measured fingertips and orientation against FK, with numeric tolerances set before testing. Compare Dream's three-second Home report with actual settling time.

**Pass:** No unexplained pose error; contact does not become a fatal alarm; real faults stop safely; paths clear hazards; Home timing matches measurement. **Fail:** Fix driver, geometry, trajectory check or timing offline and repeat. Do not weaken a safety boundary to force a pick.

## Gate 3 — Cameras and Home
**Why:** The saved head view is low, the right-wrist block is near the edge, and the locked-left view shows hardware. The base model relies heavily on images on this custom robot.

**Home rule:** Choose a repeatable collision-free start pose from which the selected arm reaches the task and returns safely. The overhead camera should show the table, object, target area, and preferably the **active gripper** at Home. **Both arms do not need to appear in the top camera.** Do not sacrifice clearance, reach, or active wrist visibility to frame both. Park/lock the unused arm safely and make its camera input intentional; verify what the chosen model receives.

**Check:** Capture actual model-resolution frames at Home and near grasp with timestamps, frame rate, and staleness. Ensure object and active fingertips remain distinguishable through approach. Change camera aim or Home only after previewing full paths and updating the bundle. Seeing common table features in two cameras does not by itself provide calibrated 3D coordinates.

**Pass:** Timely images show the task clearly; Home and paths remain safe. **Fail:** Fix view, lighting, occlusion, role/source or stale frames, then recapture before policy trials.

## Gate 4 — Table and active zones
**Why:** The saved keep-out box extends 5 cm above the measured table, plus Dream IK clearance, potentially preventing a small tabletop grasp. Physical touch checks found pose-dependent error.

**Check:** Re-measure separated tabletop/reference points, near edge, object height, TCP-to-fingertip underside, and wrist orientation. Inspect the full arm and gripper over proposed grasp in top and side views. Verify freshness and Home/raise/park paths. Back up and apply only a verified bundle, then confirm live station and guard use it. Keep the left path blocked until its recorded clearance issue is resolved.

**Pass:** The object is reachable within verified clearances and full-body paths are clear. **Fail:** Correct measurement, model, or zone design. Never simply lower the table box or disable the guard.

## Gate 5 — Dry run, then one supervised trial
**Why:** A dry run separates model choice and IK from physical execution; a live trial needs measured feedback.

**Check:** Dry-run the exact scene and prompt, saving model input frames, requested tool poses, IK targets, guard verdicts and both-arm requests. If coherent, run one short right-arm task with the operator at the stop. Synchronize frames, requested pose, IK joint target, driver setpoint, measured joints/hand/grip, clamps, timestamps and object result. Stop on unexpected motion/contact or stale data. Change one factor between trials.

**Pass:** The pick succeeds repeatably under one verified setup, or a failure is classified with evidence as camera/perception, model, IK/geometry, station limit, driver, or physical grasp. **Fail:** Missing trace means improve instrumentation; a classified failure returns to its owning gate. NO_CAPACITY and timeouts are service failures, not evidence about spatial ability.

## Logging and finish line
Create the next numbered log per gate or attempt; keep old logs intact and update log.md with one-line outcomes. Record date/time, objective, effective file hashes/versions, operator actions/commands, measurements, images/plots, pass/fail, cause, change, rollback, and next gate. Record cancelled runs too. Finish today with a verified station snapshot, reproducible start/stop instructions, known risks, and either a measured pick result or the exact failed gate and evidence needed to resume.

# Goal — Dream-1/OpenArm setup and measured pick (2026-10-27)

## Outcome today
Build a repeatable bimanual station setup and obtain one recorded two-arm task attempt only if every gate passes. A failed gate is a valid outcome: record and fix its cause before another run. Treat old logs, including log-030, as history rather than proof of today's live setup.

## Instructions to the implementing agent
Inspect the live station.yaml, active zones, arm lock, installed Dream/client versions, driver/model mapping, camera sources, and newest runtime logs before edits. Treat log-031 and the 2026-10-27 zone bundle as the latest saved evidence, while verifying their live application. Record effective paths, timestamps, hashes and backups. Add missing checks, instrumentation, tests, and rollback steps. Change one factor per trial. Start offline and read-only; real-arm motion requires the operator present, a clear workspace, and a working stop. Never copy credentials into logs.

Dream-1's base model on a non-YAM arm masks robot state from the model, while its client still uses the real pose and IK to execute model output. Useful camera views and correct kinematics are separate requirements. Home does not calibrate cameras or create stereo triangulation.

## Gate 1 — Establish the running state
**Why:** Saved bundles may differ from files in force; movement may come from startup, Home, task, Reset, or shutdown.

**Check:** Confirm driver class, custom.ik, embodiment, checkpoint/action space, joint mapping, Home, gripper endpoints, camera roles/sources, speed limits, zone age, CAN and motor status, camera exclusivity, and both-active selection in every phase, including the actual sequential Go-to-home and park schedule while the other arm holds position. Back up live configuration and run preflight.

**Pass:** Effective files and hashes match the intended setup, preflight passes, and both-arm startup, task, Reset and shutdown behavior are known. **Fail:** Fix the mismatch or lifecycle behavior before a policy run; record what moved and when.

## Gate 2 — Driver, geometry, safety
**Why:** The current gripper-stall alarm can stop a normal grasp. Endpoint-only zone checks may miss unsafe smoothed poses. Wrong TCP or joint mapping separates the modeled hand from real fingers.

**Check:** With fake motors, test joint units/signs, grip 0-closed/1-open, ordinary object contact versus faults, alarm clearing, heat/fault stops, both-arm behavior during startup/task/Reset/park/torque-off, including simultaneous moves and safe failure if either arm faults. Check every executed setpoint and both complete gripper/link envelopes against zones and each other, including the actual sequential raise/park schedule and simultaneous policy commands. In several bent poses compare measured fingertips and orientation against FK, with numeric tolerances set before testing. Compare Dream's three-second Home report with actual settling time.

**Pass:** No unexplained pose error; contact does not become a fatal alarm; real faults stop safely; paths clear hazards; Home timing matches measurement. **Fail:** Fix driver, geometry, trajectory check or timing offline and repeat. Do not weaken a safety boundary to force a pick.

## Gate 3 — Cameras and Home
**Why:** The older camera study was poor, but the head camera and both Home poses moved today; the current combined three-camera view is unmeasured. The base model relies heavily on images on this custom robot.

**Home rule:** Choose a repeatable collision-free start pose from which both arms reach the task and return safely. The overhead camera should show the table, object, target area, and preferably **both active grippers** at Home. During coordinated action, both grippers and the object should remain visible in at least useful overhead/wrist views. Do not sacrifice collision clearance or reach merely to frame both at Home. Confirm that each wrist camera sees its own working area and verify what the selected model actually receives.

**Check:** Capture actual model-resolution frames at Home and near grasp with timestamps, frame rate, and staleness. Ensure object and both sets of fingertips remain distinguishable through the coordinated approach. Change camera aim or Home only after previewing full paths and updating the bundle. Seeing common table features in two cameras does not by itself provide calibrated 3D coordinates.

**Pass:** Timely images show the task and both active arms clearly enough; combined Home, park and task paths remain safe. **Fail:** Fix view, lighting, occlusion, role/source or stale frames, then recapture before policy trials.

## Gate 4 — Table and active zones
**Why:** The saved keep-out box extends 5 cm above the measured table, plus Dream IK clearance, potentially preventing a small tabletop grasp. Physical touch checks found pose-dependent error.

**Check:** Re-measure separated tabletop/reference points, near edge, object height, TCP-to-fingertip underside, and wrist orientation. Inspect the full arm and gripper over proposed grasp in top and side views. The latest saved bundle says `verify: unverified` and has no reference points; establish a physical verification method and tolerances, then verify freshness and Home/raise/park paths. Back up and apply only a verified bundle, then confirm live station and guard use it. Use the 2026-10-27 re-taught left and right paths and verify their current application. Check paired swept-volume clearance for the actual order: first arm held at Home while the second raises, then each arm parks while the other holds. Individual raise tests do not establish this.

**Pass:** Both grippers can reach the object within measured clearance, and each full-body path and their interaction are clear. **Fail:** Correct measurement, model, or zone design. Never simply lower the table box or disable the guard.

## Gate 5 — Dry run, then one supervised trial
**Why:** A dry run separates model choice and IK from physical execution; a live trial needs measured feedback.

**Check:** Dry-run the exact two-arm scene and prompt from a documented simultaneous Home state, saving model input frames, requested tool poses, IK targets, guard verdicts and both-arm requests and predicted inter-arm clearances. Only after a supervised combined Go-to-home/Park test passes, run one short bimanual task with the operator at the stop. Synchronize frames, requested pose, IK joint target, driver setpoint, measured joints/hand/grip, clamps, timestamps and object result. Stop on unexpected motion/contact or stale data. Change one factor between trials.

**Pass:** The coordinated task succeeds repeatably under one verified setup, or a failure is classified with evidence as camera/perception, model, IK/geometry, station limit, driver, or physical grasp. **Fail:** Missing trace means improve instrumentation; a classified failure returns to its owning gate. NO_CAPACITY and timeouts are service failures, not evidence about spatial ability.

## Logging and finish line
Create the next numbered log per gate or attempt; keep old logs intact and update log.md with one-line outcomes. Record date/time, objective, effective file hashes/versions, operator actions/commands, measurements, images/plots, pass/fail, cause, change, rollback, and next gate. Record cancelled runs too. Finish today with a verified station snapshot, reproducible start/stop instructions, known risks, and either a measured pick result or the exact failed gate and evidence needed to resume.


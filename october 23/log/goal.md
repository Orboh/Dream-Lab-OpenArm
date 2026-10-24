# Goal — Diagnose and stabilize Dream-1 on OpenArm

## Context and intended outcome
Continue logs 009–014 and the existing driver/ and workspace/ code. The robot already has taught Home poses, paths from rest, three camera feeds, a point-recording editor, zone bundles, and a one-arm lock. Keep these components and verify what they actually do.

The goal is to answer two observed questions with evidence: why both arms move when one is requested, and why actions seem poorly oriented to the table. A successful pipeline must distinguish Dream-1's requested motion from station homing, driver motion, bad robot coordinates, restrictive workspace settings, and poor camera observations.

## Instructions to the implementing agent
Inspect the current live station.yaml, active zones.yaml, lock file, camera configuration, code, and recent logs before editing. Treat older logs as history. Expand the stages below into exact implementation steps, measurements, tests, acceptance thresholds, operator actions, and rollback. Add any missing process needed to make daily use repeatable. Record timestamps and versions of the files actually applied. Use offline and read-only checks first. Any real-arm motion requires the user present with a working stop mechanism. Do not proceed past a failed stage.

Dream Labs publicly describes image views plus robot state, but does not publish a required three-camera calibration procedure. Do not assume that common image content means stereo triangulation, or that the taught Home pose calibrates cameras to the robot.

## Logging requirements

Create a new numbered log for each stage, following the existing log-001.md to log-014.md style; never rewrite old observations as if they were new. In every stage log, record the date, goal, exact files/configuration in force, commands or operator actions, measurements, pass/fail result, unresolved questions, and next step. Include links or paths to captured frames, plots, bundles, test output, and backups without copying credentials. Record failed checks and aborted trials as carefully as successful ones. Update log.md after each stage with the new log and a one-line result. If a stage is repeated, add a new dated log or addendum showing what changed and why.

## Stage 1 — Establish the true running state
**Why:** A saved editor bundle can differ from the station and zones in force. Movement may occur at station startup, during a task, on Reset, or during shutdown. Without a common timeline, these causes look alike.

**Process:** Record versions or hashes of the live station, active zones, latest saved bundle, driver, robot model, camera roles/serials, and lock setting. Capture timestamps for camera frames, joint readings, Dream targets, driver commands, and measured motor positions. Label the phase of each movement.

**Pass:** One session can be reconstructed and each movement assigned to its source and phase. Continue to view and geometry checks.

**Fail:** Missing or inconsistent files, timestamps, or telemetry mean no model conclusion is valid yet. Fix instrumentation and identify the actual configuration before moving the robot.

## Stage 2 — Check what the cameras and robot model agree on
**Why:** Home puts the cameras in a useful pose, but does not tell the robot where image pixels lie in 3D. The earlier straight-arm check could not fully resolve the wrist mapping; an inaccurate 3D gripper position makes all recorded workspace points suspect.

**Process:** At each taught Home, save synchronized head and wrist frames. Check that shared table landmarks and the grippers appear where expected; measure delivered frame rate and missing or stale frames. In several different bent arm poses, compare the real gripper tip and orientation with both the editor's 3D model and Dream's forward-kinematics result. Check joint signs/zeros, tool point, base-frame axes, and units. If useful, calibrate camera-to-robot transforms with a measured target as an independent diagnostic; do not assume Dream-1 consumes those transforms.

**Pass:** The three feeds are timely and useful, and the modeled gripper agrees with the real one within an explicit measured tolerance over the work poses. Proceed to table capture.

**Fail:** A camera fault calls for fixing the feed, mount, or timing. A pose-dependent 3D mismatch calls for correcting kinematics or calibration before any new safe area is trusted.

## Stage 3 — Capture the real table and allowed area
**Why:** The inspected zones-20261023-1802.yaml is unverified: its recorded reach is behind a placeholder table. The robot can see a table yet be constrained to coordinates that never reach it. The editor's “Record this point” defines safe poses; it is not automatically a measurement of the tabletop.

**Process:** Physically measure at least three separated tabletop points, including the near edge, and use extra points to check the fitted surface. Capture each arm's safe hand points above that table and a path from rest to Home. Review table, Home, reach, region, keep-out zones, and complete arm links in top and side views. Prevent sparse points or a filled polygon from silently treating unvisited unsafe space as safe. Add a check that each reach area intersects the measured tabletop and that Home and the complete raise/park paths clear hazards.

**Pass:** The displayed table matches the physical one, each arm has a usable area over it, and all intended poses and paths pass collision checks. Save a verified bundle.

**Fail:** No intersection means the points, table placement, or coordinate frame are wrong; recapture. A collision or uncertain area must be corrected before applying zones.

## Stage 4 — Apply the verified configuration
**Why:** Editing and saving do not change the running station by themselves. A correct bundle cannot help if another station or stale zones file is active.

**Process:** Verify physical reference points; build the bundle; back up and apply it. Compare every applied Home, reach, region, keep-out zone, driver, speed setting, and camera role with the chosen bundle. Run preflight and a read-only dry run. Keep a known-good rollback copy.

**Pass:** The live station and active zones match the verified bundle and preflight passes. Only then begin supervised motion checks.

**Fail:** Mismatched, stale, or rejected settings mean stop; restore the backup or recapture and reapply. Do not interpret a Dream-1 run made with unknown settings.

## Stage 5 — Make one-arm selection a hard control rule
**Why:** Dream-1 is a bimanual controller and may send targets for both arms despite a one-arm prompt. The current driver checks the lock on action writes but can raise an arm before that check. Task prompts cannot substitute for a physical lock.

**Process:** Define the nonselected arm's behavior at startup, action writes, Reset, stop, and motor-off. Enforce the lock before automatic raising and every commanded move. Handle a raised locked arm safely at shutdown; neither an unexpected park nor an unsupported torque-off is acceptable. Test both selections with fake motors, then with short supervised checks. Log Dream targets, station outputs, driver commands, and measured motion for both arms.

**Pass:** The selected arm behaves as commanded; the other follows the agreed hold/shutdown behavior in every phase. Any remaining two-arm request is visible as a model output rather than unexplained motor motion.

**Fail:** If the driver moves an arm without a matching permitted command, fix the lifecycle or lock code. If Dream requests both but the lock works, investigate prompting or model behavior separately.

## Stage 6 — Evaluate Dream-1
**Why:** Only after camera, geometry, configuration, and lock checks pass can a task failure reasonably be attributed to Dream-1.

**Process:** Run short, bounded one-arm tasks with clear objects and targets visible in the saved frames. Change one factor at a time. Log the prompt, synchronized images and robot state, requested targets, executed commands, and result. Classify failures as perception/model, camera quality, coordinate model, station constraints, driver/lock, or physical execution.

**Pass:** Tasks succeed repeatably, or failures have a clear category and evidence for the next change.

**Fail:** Unclassified failures indicate missing observations or a failed earlier gate. Return to the relevant stage; do not change several settings at once.

## Done
The real and modeled gripper agree in bent poses; measured table and allowed reach overlap; verified files are the files in force; one-arm behavior is deterministic; and any remaining Dream-1 shortcoming is supported by synchronized evidence.

# Goal — Make Dream-1 and the OpenArm workspace agree

## Goal
Build a repeatable pipeline that makes the three camera views, the physical table, the robot's 3D model, the saved workspace, and one-arm control agree. Use it to separate Dream-1 model errors from camera, calibration, station, and driver errors.

## Existing work
Continue logs 009–014 and the files under driver/ and workspace/. Home poses, rest-to-Home paths, a four-card teaching editor, zone bundles, preflight checks, and a one-arm lock already exist. Keep and validate that work rather than rebuilding it. Home is a useful camera-view pose; it is not camera calibration. Point recording uses follower joint readings and forward kinematics, not image triangulation.

## Findings to investigate
- In the inspected zones-20261023-1802.yaml, reach is behind the placeholder table and verification is incomplete. Determine whether this bundle or another one is actually applied.
- The driver checks the one-arm lock on Dream action writes, but its automatic raise can occur before that check. Distinguish startup, task, reset, and shutdown movement.
- Straight-arm checks did not settle joint/wrist mapping in bent poses. The head camera previously delivered fewer frames than configured.
- The live station.yaml, active zones file, and named plan were unavailable during this review. Inspect their present state; do not infer it from old logs.

## Instructions to the implementing agent
Create a staged implementation plan from the current files. For each stage, state the exact code/config changes, measurements, pass/fail criteria, tests, operator actions, and rollback. Add any missing step needed for a robust daily pipeline. Preserve a dated log of assumptions, evidence, failures, and applied file versions. Use read-only/offline checks first. Do not move real arms without the user present and controlling the stop mechanism. Do not advance after a failed gate.

## Stage 1 — Capture a reproducible baseline
Record the versions and safe hashes of the live station, active zones, saved bundle, driver, model, camera roles/serials, and lock setting. Record whether the station is stopped. Identify which process owns each camera and CAN link. Add a single session identifier and timestamps shared by camera frames, joint readings, Dream targets, driver commands, and measured motor motion. Avoid logging credentials.

Pass when one session can be reconstructed without guessing which files or camera frames were used.

## Stage 2 — Validate views and coordinate frames
At both taught Home poses, save synchronized head, left-wrist, and right-wrist frames. Mark common tabletop landmarks and when each gripper enters view. Measure actual frame rate and dropped/stale frames. Visual overlap alone does not prove calibrated stereo or 3D triangulation; document whether any depth or camera extrinsics are really used.

With the follower in several distinct bent poses, compare the real gripper tip and orientation with the 3D viewer and Dream's forward-kinematics result. Check both arms, joint signs/zeros, tool point, base-frame direction, and units. Set a measured position/orientation tolerance before testing; correct the mapping if it fails.

Pass when the displayed gripper follows the real one through the intended work poses.

## Stage 3 — Teach the physical table and safe work area
Use at least three separated physical tabletop contacts, including the near edge, to establish the table plane and location; check more points for error. Confirm the table representation agrees with the real surface in top and side views. Then record each arm's safe hand positions above the real table, plus hazards and the path from rest to Home. Preserve the distinction between table points and safe-area points in the editor.

Add automatic checks: each arm's reach must intersect the measured tabletop; Home must have room to move; the complete raise/park path must clear hazards; allowed regions must contain the intended arm links without admitting unintended collisions. Sparse points or a convex hull must not silently mark unvisited unsafe space as safe. Show any overlap between the two arms' work areas.

Pass when the saved picture and coordinates match a physical check and the bundle is verified.

## Stage 4 — Apply and verify the station
Build, validate, and apply the verified bundle with a dated backup. Compare the live station.yaml and active zones.yaml to the selected bundle; report the exact home, reach, region, keep-out, driver, speed settings, and active-arm lock. Run preflight and a read-only dry run. If any file differs or the zones are stale, stop and reapply or recapture. Make rollback explicit.

Pass when the running station uses the intended verified geometry and settings.

## Stage 5 — Make one-arm behavior deterministic
Define what “right arm only” means at startup, action writes, reset, stop, and motor-off. Check the lock before automatic raise and every commanded move. Handle a raised locked arm safely during shutdown: neither an unexpected park nor an unsupported torque-off is acceptable. Test both selections and a change while running, first with fake motors and then in a supervised bounded check.

Log, for both arms, what Dream requested, what the station passed, what the driver sent, and what the motors did. A prompt alone must not be treated as a physical one-arm lock.

Pass when the nonselected arm follows the agreed hold/shutdown behavior throughout the lifecycle.

## Stage 6 — Evaluate Dream-1
Use short, bounded one-arm tasks with a clear object and target visible in the saved views. Record prompt, frames, robot state, requested and executed actions, and outcome. Classify each failure as perception/model, camera quality, coordinate model, station constraints, driver/lock, or physical execution. Change one factor at a time and keep failed trials in the log.

## Done
The real and modeled gripper agree in bent poses; measured table and allowed reach overlap; verified files are the files in force; single-arm behavior is deterministic; and remaining Dream-1 shortcomings are demonstrated with synchronized evidence.

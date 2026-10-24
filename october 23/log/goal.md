# Goal — Align Dream-1 with the OpenArm workspace

## Context
Continue logs 009–014. Home and the rest-to-Home path are already taught. Point recording uses follower joint readings and the robot model; it does not triangulate camera images. Saving a bundle does not prove the station is using it.

## Goal
Make camera views, physical table, robot coordinates, allowed area, and one-arm control agree before judging Dream-1.

## Plan
1. At each Home, save head and wrist frames. Confirm shared table landmarks and gripper visibility; retain the taught path from rest.
2. At several bent poses, compare the real gripper tip with its 3D model position. Resolve joint zero/sign, tool-point, or frame errors before recording workspace points.
3. Measure separated points on the real tabletop, including its near edge. Record safe hand points above it for each arm. Review table, Home, reach, region, and hazards in top and side views. Reject bundles whose reach misses the table or whose Home/path clashes.
4. Verify reference points, build and apply the bundle. Confirm live station.yaml and zones.yaml match it; log the applied versions and preflight result.
5. Enforce the one-arm lock during startup/raise, action writes, reset, and park. Compare Dream's requested targets, driver commands, and measured motion for both arms.
6. Run short, bounded one-arm tasks only after those checks. Log the prompt, three frames, requested/executed actions, result, and failure category.

## Done when
The model gripper matches the real one in bent poses; reach intersects the measured table; applied files match the verified bundle; a locked arm stays still through the full lifecycle; and any remaining error can be assigned to perception/model, geometry, configuration, or driver.

## Current evidence
The inspected zones-20261023-1802.yaml is unverified and places reach behind a placeholder table. The driver checks the lock on action writes but may raise an arm before that check.

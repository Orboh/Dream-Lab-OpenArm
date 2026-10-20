[← instruction.md](instruction.md)

# 17 · Camera → robot calibration

**Code:** `orboh/perception/calibration.py`, `scripts/calibrate_camera.py`, `Pipeline.find_block`. **Tests:** `tests/test_calibration.py`.
**Output:** `config/camera_extrinsics.yaml` (git-ignored, per machine): `T_base_cam` (4×4), intrinsics `K`, method, error figures.

The block finder ([16](16-perception.md)) gives a position in the **camera frame**. The planner needs it in the **robot base frame**
(= the MuJoCo twin's world frame, z up). This transform is `T_base_cam`.

## Current method: depth plane + recorded demos (no hands-on step)

`.venv/bin/python scripts/calibrate_camera.py from-demos [--save]` (chest camera free; reads a live depth frame once):

1. **Camera height and tilt from depth.** Fit the table plane in the ROI of one live frame: gives the table normal and the camera's height above the table.
   Combined with the twin's table height this fixes camera z, pitch and roll.
2. **Position and heading from the demos.** For each pick episode of a recorded dataset:
   - pixel = the pale blob in the first chest-camera frame (colour; recorded video has no depth);
   - the arm's grasp point = forward kinematics of the follower's measured joints at the moment its gripper stalls on the block
     (≈ 0.49 canonical, −32°; [18](18-grasp-planner.md)), finger-mesh midpoint slid along the gripper axis by one fitted distance (37 mm).
   The pixels are intersected with the table plane (raised half a block), then a rigid fit (`kabsch`) to the grasp xy gives x, y and the turn about the vertical.
3. Report fit rms and **leave-one-out** error in mm; `--save` writes the file.

### Result (2026-10-20, 8 Kitakyushu wood-L1 episodes)

| Quantity | Value |
|---|---|
| Camera position in the base frame | (0.13, 0.04, 0.700) m |
| Camera height above the table (depth) | 0.375 m; twin table top 0.325 m → z = 0.700 m = the "700 mm above the plate" in the scene file |
| Fit rms / leave-one-out | 23.6 mm / mean 27 mm, max 44 mm |
| Live cube in the base frame | (0.315, 0.007, 0.349) m: z = table + half the block, x/y inside the area the demos grasped |

The 0.700 m agreement is an **independent confirmation of `sim.table_drop_m: 0.075`**.

### How good is it? (honest limit)
About **25 mm**. The floor is the demos themselves: the operator does not grasp the block at its exact centre, and the grasp height varied
±3 cm (the pieces differ and the cylinder is 98 mm tall). The gripper has roughly ±11 mm of clearance around a 47 mm cube
([18](18-grasp-planner.md)), so this calibration is good enough to plan, to dry-run and to approach, **not to close the gripper blind**.
Do not trust a real grasp on it without a refinement.

## Refinement (next, hands-on)
Touch-point method (`kabsch` already supports it): with torque off, put the fingertip on 4+ known spots on the carpet while
the camera sees the fingertip in depth; the fingertip's base-frame position comes from forward kinematics. Target: < 8 mm. A second
refinement for the last centimetres is closing the loop with the wrist D405 camera.

## When to redo it
Whenever the chest camera or its mount is touched (the transform is only valid for that pose), or the table height changes.
`Pipeline.find_block` warns when no extrinsics file exists and then reports camera-frame positions only.

## New-rig procedure
1. Pin the serial ([04](04-cameras.md)), set `cameras.depth_roles`, and check `sim.table_drop_m` ([07](07-simulation.md)).
2. Record ≥ 6 pick demos ([05](05-demos-recording.md)) with the object visible in the chest camera's first frame.
3. `scripts/calibrate_camera.py from-demos --dataset <root> --save`; read the leave-one-out figure before trusting it.

## Related
- [16 Perception](16-perception.md) · [18 Grasp planner](18-grasp-planner.md) · [07 Simulation](07-simulation.md) · [15 Real robot](15-real-robot.md)

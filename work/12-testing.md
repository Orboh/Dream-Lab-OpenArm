[← instruction.md](instruction.md)

# 12 · Testing

**Run:** `MUJOCO_GL=egl pytest -q` (about 11 s; no hardware needed) · **Code:** `tests/test_pipeline.py`

## What the suite proves

| Area | Tests | Proves |
|---|---|---|
| Mapping | `test_mapping_round_trip_degrees_and_signs`, `test_mapping_reports_missing_keys` | unit, sign, offset and gripper conversion is lossless; bad keys fail loudly |
| Trajectory | `test_resample_slows_motion` | `speed_scale` really slows the motion |
| Safety | `test_safe_hold_passes`, `test_joint_limit_violation_blocks_before_sim`, `test_fast_motion_fails_velocity`, `test_arm_arm_collision_detected_in_sim` | each layer catches its failure type; a real arm-arm collision (both `joint3` ±0.4 rad) is detected |
| Execution | `test_executor_completes_and_tracks`, `test_executor_estop_halts_immediately`, `test_supervisor_clamps_step_size` | the executor reaches its target, E-STOP sends nothing more, an unchecked 1 rad jump is capped |
| Cameras | `test_roles_from_wiggle_scores`, `test_ambiguous_wiggle_needs_confirmation`, `test_model_heuristic_and_pinned`, `test_missing_camera_reported`, `test_cameras_yaml_round_trip`, `test_changed_fraction` | role logic, including confidence rules and file round-trip |
| Dataset | `test_sample_dataset_round_trip` | the LeRobot v3.0 writer and reader agree, and units survive |
| Web | `test_web_requires_token`, `test_web_full_flow_and_gates` | auth; the gates refuse an unchecked run (412) and an unapproved run (412); E-STOP aborts a live run |

## Manual browser check
Before a release:
1. Run `orboh serve`.
2. Click through all six steps.
3. Press Space during execution.
4. Confirm the browser console shows no errors.

This found the black-camera threading bug ([07](07-simulation.md#rendering--threads)).

## On the robot (first bring-up)
1. Wedge the arms clear, keep the hardware E-STOP in hand, and set `speed` to 0.2×.
2. Run `orboh preflight` and fix everything until it's green.
3. Check joint signs one joint at a time ([07 → alignment](07-simulation.md#sim-to-real-alignment)).
4. Run the wiggle test and confirm the camera roles.
5. Execute **Move to home pose**, then a mock-replay of a real recorded demo at 0.2×.

## Writing tests
- Use `pedestal/pedestal.xml` for fast sim tests, and `SimRobot` for anything that moves.
- Every bug fix gets a test that would have caught it.
- Never call real hardware or the real DreamLab API from the test suite.

## Related
- [08 Safety](08-safety.md) · [11 Web console](11-web-console.md) · [instruction.md §8](instruction.md#8-maintenance-rules)

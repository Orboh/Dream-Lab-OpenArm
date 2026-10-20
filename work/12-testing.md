[← instruction.md](instruction.md)

# 12 · Testing

**Run:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl pytest -q` (about 11 s; no hardware needed) · **Code:** `tests/test_pipeline.py`

## What the suite proves

| Area | Tests | Proves |
|---|---|---|
| Mapping | `test_mapping_round_trip_degrees_and_signs`, `test_mapping_reports_missing_keys` | unit, sign, offset and gripper conversion is lossless; bad keys fail loudly |
| Trajectory | `test_resample_slows_motion` | `speed_scale` really slows the motion |
| Safety | `test_safe_hold_passes`, `test_joint_limit_violation_blocks_before_sim`, `test_fast_motion_fails_velocity`, `test_arm_arm_collision_detected_in_sim` | each layer catches its failure type; a real arm-arm collision (both `joint3` ±0.4 rad) is detected |
| Execution | `test_executor_completes_and_tracks`, `test_executor_estop_halts_immediately`, `test_supervisor_clamps_step_size` | the executor reaches its target, E-STOP sends nothing more, an unchecked 1 rad jump is capped |
| Cameras | `test_roles_from_wiggle_scores`, `test_ambiguous_wiggle_needs_confirmation`, `test_model_heuristic_and_pinned`, `test_missing_camera_reported`, `test_cameras_yaml_round_trip`, `test_changed_fraction` | role logic, including confidence rules and file round-trip |
| Dataset | `test_sample_dataset_round_trip` | the LeRobot v3.0 writer and reader agree, and units survive |
| Policy | `test_chunk_to_canonical_*`, `test_slew_limit_*`, `test_runner_*` (`tests/test_policy.py`) | chunk layout and shape checks; slew limits respect velocity, acceleration, joint limits and brake without overshoot; the runner completes, never moves in dry-run, stops on E-STOP, halts on repeated unsafe, non-finite or silent chunks ([14](14-policy.md)) |
| Plan conform | `test_plan_conform_removes_jerk_and_limit_overshoot` | a jerky demo on a limit is smoothed and clipped before the checks |
| Perception | `test_plane_fit_*`, `test_finds_cube_*`, `test_position_follows_the_object`, `test_empty_table_*`, `test_too_small_blob_*`, `test_missing_depth_pixels_*` (`tests/test_perception.py`) | the block finder recovers height / size / centre on a synthetic RGB-D scene, follows the object, finds nothing on an empty table, rejects tiny blobs, and survives missing-depth holes (a live-camera regression) |
| Calibration | `test_kabsch_*`, `test_pnp_*`, `test_pixel_to_*`, `test_extrinsics_round_trip`, `test_colour_block_pixel_*`, `test_attach_base_positions_*` (`tests/test_calibration.py`) | the rigid fit and PnP recover a known camera pose, degenerate input is refused, the file round-trips, the colour blob ignores specks |
| Speed | `test_slowed_scales_the_speed_limits_and_the_runner_obeys_them` (`tests/test_policy.py`) | `slowed()` scales the limits (never above 1, floor 0.1) and the commands that reach the robot never exceed the slowed velocity / gripper caps |
| Retries | `test_rejected_plan_is_fed_back_to_dream_and_retried_until_safe`, `test_retry_hint_names_the_problem` (`tests/test_policy.py`) | a rejected plan is explained to the policy and re-requested; a safe one then runs and the hint is cleared |
| Gripper mode | `test_snap_gripper_uses_demo_values_with_hysteresis` (`tests/test_policy.py`), `test_web_gripper_mode_is_part_of_the_dry_run_and_simple_page_is_served` | Dream's decision becomes the demo open/close value without chatter; the gripper mode is part of the dry-run gate; `/simple` is served |
| Web policy | `test_web_policy_dry_run_then_gated_real_run` (also: the camera snapshot is served and saved with the prompt and 16 joints) | dry-run moves nothing and unlocks only the same prompt; a real run needs approval; the dry-run is used up |
| Robot bridge | `test_roundtrip_converts_units`, `test_unreachable_bridge_raises` (`tests/test_zmq_robot.py`) | `ZmqRobot` ↔ `robot_bridge.py --fake` round trip converts degrees ↔ canonical; a dead bridge raises instead of hanging |
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

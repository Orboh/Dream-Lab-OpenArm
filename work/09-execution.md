[← instruction.md](instruction.md)

# 09 · Execution

**Code:** `orboh/execution/executor.py`

## 1. Build the command sequence: `build_execution_trajectory(current_q, plan, cfg, speed_scale)`
1. Resample the plan with `speed_scale` (0.1–1.0; the console default is 0.5×).
2. Prepend a **min-jerk approach** from the current pose to the plan's first waypoint:
   - duration = `max(0.5 s, 1.875 · max_arm_distance / approach_speed_rad_s)`;
   - the 1.875 factor keeps the min-jerk *peak* speed at `approach_speed_rad_s`.
3. Return `(trajectory, approach_steps)`. The console plots the approach in grey.

This array is what the safety check fingerprints and simulates ([08](08-safety.md)),
and what the executor streams. They are the same object.

## 2. Stream it: `Executor.run(traj, stop_event) -> ExecutionResult`
On each tick (period = 1/hz, with drift-free scheduling):
1. Stop if `stop_event` is set (E-STOP).
2. Read the measured joints and check tracking error against the previous command.
3. Clamp the target to the limits, and cap the arm step size.
4. `robot.send_joints(cmd)`.
5. Check the watchdog. Report progress every 5 steps (shown as the console's progress bar).

`ExecutionResult`: `completed`, `steps_sent`, `total_steps`, `reason`, `max_tracking_error_rad`.
It is saved to `runs/<run>/execution.json`.

On any halt, `_halt()` commands the measured pose, so the arms hold where they are. If that
command itself fails, the error is appended to `reason`.

## Utility moves
`Pipeline._safe_move_to(q)` covers **Move to home pose** and the camera wiggle test. It builds
a short trajectory with the same function, runs the same safety check, then the same executor.
If the check fails, the move is refused.

## Extending
- **Closed-loop / action chunks:** if DreamLab returns chunks to run while the next is requested,
  add a streaming executor here. Every chunk must still pass layers 1–2 and the supervisor.
- **Gripper force limits:** add them to the supervisor once the fork exposes current or torque.

## Related
- [08 Safety](08-safety.md) · [03 Robot layer](03-robot-layer.md) · [10 Pipeline](10-pipeline.md)

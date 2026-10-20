[← instruction.md](instruction.md)

# 08 · Safety

**Code:** `orboh/safety/checks.py`, `orboh/safety/sim_check.py`, `orboh/safety/__init__.py`,
`Pipeline.gate`, `orboh/execution/executor.py` · **Config:** `safety.*` ([02](02-configuration.md#safety-values-defaults))

Safety works in four layers. Each layer assumes the one before it may have missed something.

## Layer 1 · Static checks (`static_checks`)
These are pure math on the trajectory, so they're instant:

| Check | Fails when |
|---|---|
| `finite_values` | any NaN/inf |
| `duration` | > `max_plan_duration_s` or < 2 steps |
| `joint_limits` | outside model limits minus `joint_limit_margin_rad` |
| `joint_velocity` | peak arm speed > `max_joint_velocity_rad_s` |
| `gripper_velocity` | > `max_gripper_velocity` |
| `joint_acceleration` | > `max_joint_accel_rad_s2` |

If any of these fails, the sim rollout is skipped (`sim_rollout: skipped`).

## Layer 2 · MuJoCo rollout (`simulate`)
The command sequence is replayed step by step in a **separate** twin (`Pipeline.check_sim`, not
the sim robot):

| Check | Fails when |
|---|---|
| `sim_collisions` | any `self`, `arm_arm` or `env` contact ([07](07-simulation.md#contact-classification-used-by-the-safety-rollout)) |
| `sim_tracking` | simulated joints lag the command by > `sim_max_tracking_error_rad` (infeasible motion) |
| `finger_environment_contact` | *warning only*: the gripper touches the table |

The report includes the first collision (time, bodies), the top colliding pairs, 6 preview frames
and the simulated joint trajectory, which the console plots against the command.

## Layer 3 · Execution gates (`Pipeline.gate`)
Execution is refused (HTTP 412) unless **all** of these hold:
1. A plan exists and was checked, and `report.passed` is true (`require_sim_pass`).
2. The **fingerprint** (a hash of the exact command array) matches what was checked.
3. The robot hasn't drifted more than `max_state_drift_rad` since the check.
4. The operator approved the run (`require_manual_approval`).

**Key design point:** the sim checks the *exact* command sequence that will run. That sequence
is the min-jerk approach from the current pose plus the plan at the chosen speed, built once
by `build_execution_trajectory` ([09](09-execution.md)). Nothing is re-planned after the check.

## Layer 4 · Runtime supervisor (inside `Executor.run`)
On every control tick, the supervisor:
- clamps the command to the joint limits;
- caps each arm-joint step at `max_joint_velocity_rad_s / hz × 1.5`;
- **halts** if the measured joints differ from the last command by more than `runtime_max_tracking_error_rad`;
- **halts** if one iteration takes longer than `watchdog_s`.

Halting sends the *measured* pose as the new target, so the arms hold where they are.

## E-STOP
- Triggered by the header button, the Space bar, `POST /api/estop`, or `Pipeline.estop()`.
- Accepted even while another task is running.
- Effects:
  - sets `stop_event`;
  - the executor halts within one tick;
  - teleop recording is terminated;
  - new executions are disabled until the operator clicks "Release E-STOP".
- **This is a software stop. Keep the hardware E-STOP within reach at all times.**

## Changing limits
Tighten freely. Loosen only after testing on the real arms, and record the change in the
[decisions log](13-roadmap.md#decisions-log).

## Related
- [07 Simulation](07-simulation.md) · [09 Execution](09-execution.md) · [11 Web console](11-web-console.md) · [12 Testing](12-testing.md)

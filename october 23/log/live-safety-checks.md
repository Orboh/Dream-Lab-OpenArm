# Live View safety checks for the Dream-1 driver (2026-10-27)

Scope: we design the **driver's safety points**; Dream-1 does the task. Each check below gives a Live View prompt that pushes Dream-1 toward one condition, and the station-log line that shows whether the driver handled it. Dream-1 is zero-shot here: a prompt encourages a motion but does not guarantee it, so each check is judged by the log, not by the prompt alone. Claude watches `~/.dreamlabs/station.log` live and reports each result.

## Before every session
1. Teleop stopped (esc), website stopped (it holds the cameras), both arms hanging at rest.
2. `workspace.py preflight`: every line PASS. Known today: the right raise/park path is 0.2 cm short of the strict rule (log-036); it ran only as a user-approved exception.
3. The station loads the driver at start: start it again after any driver change (`dreamcli robot up`).
4. Operator at the emergency stop, table clear except the test object.
5. **Rule until the drop is fixed (log-036 §5):** before pressing **Reset** or **Stop** while the arm is far from its home (low over the table, reaching out), **support the arm by hand**. Far from its path, the driver switches it off without parking and it drops.

## The checks (right arm only, left locked, unless noted)

| # | Safety point | Live View prompt | Pass: the station log shows | Fail |
|---|---|---|---|---|
| 1 | **Table, fingers (rule C):** fingertips stop at the table box (5 cm above the real table) | `touch the table with the right gripper` | `arm right: stopped at the edge of the allowed space (right grip:finger_… 'table')`, and the run **continues** | the fingers go lower; or the run ends |
| 2 | **Table, hand housing:** the 16.8 cm wide housing keeps 2 cm from the table box | `lay the right hand flat on the table` | `stopped at the edge … (right grip:hand:… 'table')` | the housing goes lower |
| 3 | **Held at an edge is not a "wrong mapping"** (the bug that ended run 1, fixed in log-036) | same as 1; let it run 10 s after the edge line | the edge line, **no** `A wrong angle mapping looks like this` | that error ends the run |
| 4 | **The way to a target** (log-035): a move between two safe poses that would pass through the table is cut | `sweep the right hand quickly from the left side of the table to the right side, close to the table` | an edge line ending in `(on the way to the target)` | none, while the hand clearly dips toward the table |
| 5 | **Arm against arm** (log-034), with the left arm locked and hanging | `move the right hand over to the left side of the robot, next to the left arm` | `arm right: stopped where it keeps 2 cm from the other arm (…)` | the hands or arms touch |
| 6 | **Gripper holding an object** (log-032): a grasp no longer ends the run | put the block **on a raised stand** (its top above the table box: about 8 cm above the table), then `pick up the wooden block with the right gripper and hold it` | `arm right gripper holds an object (…): lighter grip, 4 deg lead`, the run continues, no alarm | `gripper is blocked or squeezing` alarm; the run parks |
| 7 | **Locked arm stays still** | any prompt | `arm left: Dream asks: … this arm is LOCKED, so it holds still` and the left arm does not move | the left arm moves |
| 8 | **Speed cap 0.6 m/s** (Dream's default) | any prompt with a fast motion, e.g. `move the right hand up quickly` | `/status`: `max_ee_speed_m_s 0.6`; clamps counted, no error | the arm is slowed to 0.15 (old setting) |
| 9 | **Normal stop from home parks** | after a run, with the arm back at home: press Stop / `dreamcli robot down` | `parking at rest along the path`, then switched off at rest | it switches off without parking |

## Not tested live (offline tests only, by design)
- Motor fault and over-temperature stops (would need a broken or hot motor).
- A real wrong angle mapping (would send a 20 deg+ jump on purpose).
- Two arms moving at once into each other (needs the left path passing the preflight first).
- The drop on Reset/Stop far from the path (log-036 §5): not to be provoked; fix first.

## After each check
Claude writes the result in the next numbered log: the prompt, the time, the log lines, pass or fail, and anything unexpected.

# Daily safety area — runbook

Defines each day where the arms may work and where they must never go, and makes the station and the driver enforce it.
Everything is derived from one file per capture, `zones/zones-YYYYMMDD-HHMM.yaml` (positions in each arm's base frame, metres).

Run from this folder with the project's Python (the station must be stopped: `dreamcli robot down`; nothing here switches a motor on):

    cd "october 23/workspace"
    PY="env -u PYTHONPATH -u AMENT_PREFIX_PATH ../../.venv/bin/python"

## Teach mode on four cards (the leader moves the follower; this tool only listens)  <- the main workflow
Terminal 1 (your proven teleop, unchanged): `cd ~/projects/openarm-orboh && bash scripts/run_teleop_bimanual.sh`  (arms hanging at the start; Ctrl-C is the emergency stop)
Terminal 2: `$PY workspace.py design --observe --cameras` and open http://localhost:8080. `--observe` only LISTENS to the follower's CAN frames: it sends nothing.
The top strip shows a light per arm ("receiving" / "waiting") and the first clash. The side panel has four tabs, one card each:
1. **1 Position** — live joint angles (J1..J7, degrees) and hand position (x, y, z in cm) of the left and right arm while you move the leader.
2. **2 Home** — move the leader until the camera sees the work space (tick a camera below), press **Set LEFT home here** / **Set RIGHT home here**. The way from rest (or from the old ready pose) is recorded automatically.
   **Preview** buttons (left, right, both) play the recorded way from rest to home on the 3D robot first, with an orange line for the hand's path; nothing real moves.
   Each card says "home set, path from rest recorded" or warns "NO path": then move the leader from the rest or ready pose and press the button again.
3. **3 Danger area** — *Allowed area*: pick left / right / both, move the leader to a farthest safe point (down to just above the table), press **Record this point**; repeat for every corner and edge. **Undo last point** and **Clear all points** fix mistakes.
   The safe place is made from your points: the allowed box (green) is the box around every arm point at those poses (2 points are enough), and the work area is the polygon through the hand positions (3 or more points). Points show as numbered orange balls in the 3D view.
   *Table*: four numbers in cm (height above the hanging hand, front edge, depth, width). *Danger boxes*: **Add a danger box**, set its size in cm, drag it in the 3D view, **Delete this box**.
4. **4 Save** — a checklist (home set, path, allowed area for each arm, table placed, no clash) with what to do for a missing item, then **Save bundle + layout picture** and **Export station keys**.
The **Advanced** folder keeps the rarely used controls (margin, work-area rectangle, marks, manual path, "show robot at", operator).
Then `verify`, `build`, `apply` as below.

## The output (fixed format)
Every capture gives ONE bundle, `zones/zones-YYYYMMDD-HHMM.yaml`, plus `zones/layout-YYYYMMDD-HHMM.png` (top and side view). The bundle always states:
- `home`: where each arm starts and ends a run (7 joint angles in radians, per arm);
- `home_path` (optional): per arm, the poses in degrees from rest to home, as you moved the arm (the driver follows it);
- per arm `region` (the allowed box `lo`/`hi`, `[x, y, z]`, from the sweep), `reach` (where the hand may be, seen from above, `[x, y]`) and `keep_out` (boxes the arm must stay out of: `center` and `size`, `[x, y, z]`), in that arm's base frame
  (= the robot body frame shifted by (0, +0.031, 0.698) m for the left arm and (0, -0.031, 0.698) m for the right; x forward, y left, z up; metres).
`apply` writes exactly `home`, `region`, `reach`, `keep_out` into `~/.dreamlabs/station.yaml` for `follower_l` / `follower_r` (backup first). The editor's **Export station keys** writes the same keys to
`zones/station_keys.yaml` to paste by hand.

## The visual editor (easiest way to make the bundle)
    $PY workspace.py design --live follower --cameras      # open http://localhost:8080
The real OpenArm model (URDF meshes) with the table (blue), danger boxes (red) and the margin Dream keeps (amber). Drag the handles or type numbers; choose "show robot at" home / rest / ready / live;
the status line names every clash with the rest, ready and home poses (and the live arm). **Save bundle + layout picture** writes the bundle and the PNG; **Export station keys** writes the keys file.
`--new` starts from the defaults (table top 30 cm above the hanging hand), otherwise from the newest bundle. `--detail collision` loads lighter meshes.

## The daily order
1. `$PY workspace.py capture --source follower` — arms limp, you move them by hand. In the terminal type:
   `arm left` / `arm right` (who records), `paint on` ... `paint off` (move the hand over the usable table area),
   `table` (touch the table top with the gripper tip; at least 3 points, include the near edge), `hazard NAME` (touch each corner of a danger zone: torso, camera mount, post),
   `ref` twice (two easy-to-find points on the table for tomorrow's check), `undo`, `save`.
2. `$PY workspace.py view --live follower` — open http://localhost:8080. The skeleton follows the limp arm; it turns red inside a danger zone. Check the picture matches the room.
3. `$PY workspace.py verify` — touch the stored reference points again (the daily 2-minute check). Fails if anything is more than 1.5 cm off: the table or the robot moved, capture again.
4. `$PY workspace.py build` — checks that the rest pose, the ready pose and the path between them are clear of every zone, and prints what will be written.
5. `$PY workspace.py apply` — backs up `~/.dreamlabs/station.yaml`, writes `reach` and `keep_out` per arm, checks Dream accepts it, and makes the zones current for the driver's guard.
6. `dreamcli robot up`, then Live View as usual.

Leader arms: `align` (leader and follower in the same pose, both limp) then `capture --source leader --base zones/zones-....yaml` paints the work area and marks hazards
with the leaders on top of a follower capture. The table and reference points always come from the follower.

## One arm only
`cd ../driver && $PY arms.py right` (or `left`, `both`, `status`). The other arm keeps holding its pose whatever Dream-1 asks. Instant, no restart.

## What enforces it
- Dream-1 plans inside `reach` and 2 cm away from `keep_out` (soft: clipped targets and a solver cost).
- The driver (`driver/openarm_guard.py`) refuses any target whose arm would enter a `keep_out` zone, and refuses all motion when there are no zones or they are older than
  24 h (`OPENARM_ZONES_MAX_AGE_H`). A refusal stops the run and the arms hold. `OPENARM_ZONES_REQUIRED=0` turns the guard off for bench work only.
- `region` (the swept box) is enforced by the driver guard for every target the station sends, and by Dream's IK (clipped targets). The rest pose and the raise/park path are exempt: they hang below the table and are driver moves.

## Files
`workspace.py` commands · `capture.py` interactive session · `zonebuild.py` zones maths and station writer · `viewer.py` browser view · `zones/` captures (`zones/demo/` is invented, never apply it).
Tests: `$PY -m pytest -q` here and in `../driver`.

## Gentle motion and motor protection (the driver, `driver/openarm_v2.py` + `openarm_motion.py`)
- Every move is smooth: the raise to home and the park follow your recorded path as one spline with a soft start and stop (no stops at waypoints), and every command Dream sends, including its own 3-second linear Home move, goes through a tracker that
  limits speed and acceleration ("Gentle": big joints 40 deg/s and 120 deg/s squared, wrist 70 and 250, gripper 80 and 300) and keeps settling after Dream stops writing. A control thread at 100 Hz owns the motors and keeps sending.
- Motor protection: a target may never lead the real angle by more than half a motor's rated torque allows (about 6.4 deg shoulder, 3.3 deg elbow side, 9-12 deg wrist, 8 deg gripper), targets stay 3 deg inside the joint limits, the gains ramp up over 0.5 s at start,
  a motor over 60 C is logged and over 70 C makes the arm hold and `alarm()` stops the run, a motor fault does the same, and a gripper that squeezes against something for 2 s is relaxed and reported. The gripper's gain is 12 (it was 25 before the 2026-09-21 over-temperature).
- Before every session: `$PY workspace.py preflight` (it also names any program that has the cameras open: close the editor started with `--cameras` before the station) (or `$PY workspace.py start`, which runs it and then `dreamcli robot up` only if everything passes). `apply` also writes `ramp_s 3`, `ramp_max_rad_s 0.3`, `max_ee_speed_m_s 0.15` and the `Arm` driver (`--no-tuning` skips that).
- Edge of the allowed space: a target past the allowed region or into a keep-out zone is no longer an error that ends the run. The driver sends the furthest allowed point on the way and the arm holds at the edge (one log line: "stopped at the edge of the allowed space"). Stale or missing zones, or an arm that is already outside, still stop the run.
- One arm only: Dream sends both arms on every tick, so say which one may move BEFORE the station starts: `python driver/arms.py right` (or `left`; `both` to undo; `status` shows it; it also works while running). The other arm then holds still, is not raised at start, and the log shows what Dream asked of it ("Dream asks ... this arm is LOCKED, so it holds still"). A Reset still goes through Dream's Home for both arms, but a locked arm just holds.
- Home is the camera-view pose, not a limit: the editor gives it 5 cm of room on every side inside the allowed area, and warns on the Save card when the left and right allowed areas overlap (then the hands can meet in the middle).
- `workspace.py camera-pose [--arm left|right|both] [--hold SECONDS]` raises the arm(s) to that pose for a look at the camera views, then parks and switches them off. The arms MOVE: station stopped, space clear, hand on the emergency stop. Same checks as `start`.

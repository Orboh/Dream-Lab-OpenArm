# Daily safety area — runbook

Defines each day where the arms may work and where they must never go, and makes the station and the driver enforce it.
Everything is derived from one file per capture, `zones/zones-YYYYMMDD-HHMM.yaml` (positions in each arm's base frame, metres).

Run from this folder with the project's Python (the station must be stopped: `dreamcli robot down`; nothing here switches a motor on):

    cd "october 23/workspace"
    PY="env -u PYTHONPATH -u AMENT_PREFIX_PATH ../../.venv/bin/python"

## Teach mode (the leader moves the follower; this tool only listens)  <- the main workflow
Terminal 1 (your proven teleop, unchanged): `cd ~/projects/openarm-orboh && bash scripts/run_teleop_bimanual.sh`  (arms hanging at the start; Ctrl-C is the emergency stop)
Terminal 2: `$PY workspace.py design --observe --cameras` and open http://localhost:8080. `--observe` only LISTENS to the follower's CAN frames (like your dmlog logger): it sends nothing.
On the page, per arm:
1. With the arm at rest (hanging) press **Start path**, move the leader until the cameras (Cameras panel: tick the box) see the work space, press **Set HOME = follower now**.
   The way you moved the arm from rest to there is kept as the home path: the driver raises the arm along it at station start and parks it back along it on stop, so it never cuts across the table.
2. Tick **Record extremes** and move the leader to the farthest points the follower may safely reach (down to just above the table, left, right, far). Untick, press **Apply sweep**.
   The swept points become the allowed REGION (a box with the margin you set, default 1 cm) and the work area; the website shows the trail (blue), the region (green box) and your marks (pink).
3. Set the table and any danger boxes in the same page if you want them, then **Save bundle + layout picture** and **Export station keys**.
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

# Daily safety area — runbook

Defines each day where the arms may work and where they must never go, and makes the station and the driver enforce it.
Everything is derived from one file per capture, `zones/zones-YYYYMMDD-HHMM.yaml` (positions in each arm's base frame, metres).

Run from this folder with the project's Python (the station must be stopped: `dreamcli robot down`; nothing here switches a motor on):

    cd "october 23/workspace"
    PY="env -u PYTHONPATH -u AMENT_PREFIX_PATH ../../.venv/bin/python"

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
- `region` is not used: the rest and ready poses hold the hand below the table-top height, so a region floor would pull the hand up at the start of a run.

## Files
`workspace.py` commands · `capture.py` interactive session · `zonebuild.py` zones maths and station writer · `viewer.py` browser view · `zones/` captures (`zones/demo/` is invented, never apply it).
Tests: `$PY -m pytest -q` here and in `../driver`.

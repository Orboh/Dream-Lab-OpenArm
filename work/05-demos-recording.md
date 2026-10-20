[← instruction.md](instruction.md)

# 05 · Demos & recording (leader teleop)

**Code:** `orboh/demos/dataset.py`, `orboh/demos/sample.py`, `Pipeline.record_demo`

## Recording

The project reuses the team's existing recording flow (Orboh LeRobot fork, `lerobot-record`)
rather than reimplementing it:

```yaml
recording:
  command: "bash ~/projects/openarm-orboh/scripts/collect_bimanual.sh {mode}"   # VERIFY
```

**From the web console:**
- **Record (dry run)** → `{mode}=dry`; **Record demo** → `{mode}=run`.
- The script's output streams into the log panel. E-STOP terminates it.

**Teammate's operator routine:**
1. Line up.
2. Start recording.
3. Do the task with both leader arms.
4. End (it saves immediately).
5. Press `a` to return to the start position.
6. `esc`.

**Guidance:**
- Record one clean demo per object or task (the "one shot"), plus a few extra demos for evaluation.
- Give each demo a clear task text, e.g. "pick up the wooden block with both arms".
- Check the episode length (the teammate was testing 40 s).

## Dataset format: LeRobot Dataset v3.0

```
data/demos/
  meta/info.json                               fps, features (+ names), path templates
  meta/episodes/chunk-000/file-000.parquet     one row per episode
  data/chunk-000/file-000.parquet              frames (episode_index column)
  videos/<camera key>/chunk-000/file-000.mp4
```

`LeRobotDemoDataset` (`dataset.py`):
- **`list_episodes()`** returns index, task, length and duration.
- **`load(i)`** returns a `DemoEpisode` with:
  - `trajectory`: the leader **actions**, converted to canonical units via `JointMapping` ([03](03-robot-layer.md));
  - `states`: the follower `observation.state`, canonical;
  - `videos`: per-camera file path and timestamp range.

Joint names come from `features.action.names`, so the dataset must have been recorded with the
same key names as `robot.lerobot.joint_key` / `gripper_key`.

## Synthetic demos (no hardware)

`orboh sample-demos` (or the console button) writes 3 episodes in v3.0 format to
`demos.dataset_root`. The motion is a gentle bimanual reach-down → close grippers → lift → home.
The data is generated in canonical units and written in robot units, so the full conversion path
is exercised. `info.json` carries `"orboh_synthetic": true`. The command refuses to overwrite an
existing dataset.

## Related
- [04 Cameras](04-cameras.md) (device hand-over) · [06 DreamLab](06-dreamlab.md) (demo is the one-shot input) · [02 Configuration](02-configuration.md)

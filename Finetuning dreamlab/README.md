# Finetuning dreamlab: fine-tuning Dream-1 on our OpenArm (wood block task)

The whole path from teleop recordings to a Dream-1 checkpoint the Live View can run, for one task: **"Pick up the wood and place it in the box"** (the wood block).
Logs of this work: [`log/log.md`](log/log.md). Start with [`log/log-001.md`](log/log-001.md) (what Dream needs, what we have, the plan).

## Folder
| Path | What it holds |
|---|---|
| `pipeline/` | The scripts, run in this order (`pipeline.sh` runs each stage) |
| `pipeline/config.py` | Every setting: the task, 15 fps, 1280x720, camera serials, Dream's body `koutaueda/openarm2-faas` (its config verbatim) |
| `pipeline/link_sources.py` | Links the earlier wood recordings into `dataset/source/` and writes `dataset/source/SOURCES.md` (where each one is) |
| `pipeline/make_start_pose.py` | Writes `pipeline/start_pose/{left,right}.yaml`: start at the station's home along the driver's taught path |
| `pipeline/record.sh` | Records new episodes with `openarm-orboh`'s recorder at the station's camera settings into `dataset/raw/` |
| `pipeline/convert.py` | Our LeRobot v3.0 recordings → Dream's LeRobot v2.1 sessions in the body's layout, into `dataset/dream/` |
| `pipeline/dream_steps.py` | Dream's client: validate, upload, status, fine-tune, job status; ids logged to `log/jobs.jsonl` |
| `pipeline/test_pipeline.py` | Tests (no hardware, no Dream service) |
| `dataset/source/` | Links to the earlier wood recordings (reference only, see `SOURCES.md`) |
| `dataset/raw/` | The new recordings (LeRobot v3.0, as recorded) |
| `dataset/dream/` | The converted sessions that are uploaded |
| `log/` | This work's logs (`log.md` index, `log-NNN.md`), and `jobs.jsonl` |

The dataset folders are kept out of git (`.gitignore`); `SOURCES.md` is kept.

## The pipeline
All commands from `Finetuning dreamlab/pipeline/`. Python is the project's `.venv` (it has Dream's client and `dream_ik`, the same as the station's).

1. **Prepare** (no motor moves): `bash pipeline.sh prepare`. This links the earlier recordings and writes the start pose from `~/.dreamlabs/station.yaml` and `~/.dreamlabs/zones.yaml`. Run it again whenever the station's home changes.
2. **Record** (the arms move: your own terminal, at the e-stop; the station must be down):
   - New session: `bash pipeline.sh record`, or `NUM_EPISODES=20 bash pipeline.sh record`.
   - Add to a session: `RESUME=<folder in dataset/raw> bash pipeline.sh record`.
   - Stop with esc.
3. **Convert:** `bash pipeline.sh convert`, for every folder in `dataset/raw`, or name the folders.
   - The joints go through the driver's `official` mapping to Dream's radians.
   - The grippers become 0–1 using the station's gripper ends.
   - The hand poses come from Dream's own FK.
   - The chest camera becomes the head camera.
   - Each session's `meta/source.json` names its source recording and any episode shorter than 1.6 s.
4. **Validate** (Dream's local checks, nothing sent): `bash pipeline.sh validate`.
5. **Upload** (sends the data to Dream): `bash pipeline.sh upload <name> --yes`. Keep the `ds_…` id (also in `log/jobs.jsonl`). Then run `bash pipeline.sh status <ds_id>` until it shows `processed`.
6. **Fine-tune:** `bash pipeline.sh finetune koutaueda/<name> <suffix> --yes`. This uses `use_eef=True` (the body's action carries hand rotations) and `lr_schedule="cosine"`. Then run `bash pipeline.sh job <job_id>` until `fine_tuned_model` is set. A job takes about 4 to 5.5 hours with 3 cameras (Dream's docs).
7. **Run it:** choose `dream-1-ft:<uid8>-<suffix>` in the Live View, and use the prompt in exactly the words of `config.TASK`. The driver's guard stays on, with the user at the e-stop.

Tests: `bash pipeline.sh test`.

## Rules the pipeline keeps (from Dream's code and docs)
| Rule | Where |
|---|---|
| LeRobot v2.1: `meta/info.json`, `episodes.jsonl`, `tasks.jsonl`; `data/chunk-NNN/episode_NNNNNN.parquet`; `videos/chunk-NNN/<key>/episode_NNNNNN.mp4` | `shared/datasets/validation.py` |
| fps 30, 15, 10 or 5; every episode at least 1.6 s (25 frames at 15 fps) | `content_checks.USABLE_FPS`, `min_episode_frames()` |
| H.264, HEVC, AV1 or VP9 in MP4 (we write H.264); camera `names` = height, width, channels | `video_formats.py`, `check_camera_axes()` |
| At most 3 cameras; one fixed head camera (`head` → `observation.images.head_left`) | `embodiments.MAX_CAMERAS`, the body's `head_camera` |
| The body's spans: state 36 values, action 34 (hand pose, rot6 = the first two columns of the rotation, grip 0 closed to 1 open) | `koutaueda/openarm2-faas`, `fine_tuning.create()` docstring |
| Episode count: Dream's docs give **none**. The only figure is on dreamlabs.ai: "A few hundred demonstrations adapt it to yours." | log-001 |

"""Settings shared by every step of the fine-tuning pipeline: where things are, the task, the recording settings and Dream's body.

Nothing here talks to a motor or to Dream's services.
"""

import os
import pathlib

## This folder (`Finetuning dreamlab/`) and its parts.
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
SOURCE_DIR = DATASET / "source"      # links to the earlier wood recordings (reference only, not uploaded)
RAW_DIR = DATASET / "raw"            # the new recordings, as `lerobot-record` writes them (LeRobot v3.0)
DREAM_DIR = DATASET / "dream"        # the converted sessions Dream takes (LeRobot v2.1, Dream's body layout)
JOBS_FILE = ROOT / "log" / "jobs.jsonl"   # every upload and fine-tune job id, one JSON object per line

## Where the earlier recordings live (LeRobot's cache) and the recorder's run folders.
LEROBOT_CACHE = pathlib.Path("~/.cache/huggingface/lerobot/Orboh").expanduser()
COLLECTIONS = pathlib.Path("~/collections").expanduser()
## Only these recordings are linked into `dataset/source`: the wood block task.
SOURCE_PATTERN = "*wood*"

## The driver folder: its joint mapping is the one Dream's model reads at run time.
DRIVER_DIR = ROOT.parent / "october 23" / "driver"
## The mapping the station runs with (`OPENARM_MODEL_MAPPING=official`, log-018).
MAPPING = "official"
## The station file: gripper ends per arm and the body Dream runs.
STATION_FILE = pathlib.Path(os.environ.get("DREAM_STATION_FILE", "~/.dreamlabs/station.yaml")).expanduser()

## The task: the wood block. The prompt is the text Dream trains on; use the same words in the Live View.
TASK = "Pick up the wood and place it in the box"
TASK_SLUG = "dream_wood_block"

## Recording settings: the station's cameras (1280x720) at 15 fps (user decision, 2026-10-27).
FPS = 15
CAMERA_W, CAMERA_H = 1280, 720
## The cameras by serial: the same physical cameras as `station.yaml`.
CAMERA_SERIALS = {"chest": "344422070898", "wrist_right": "352122273417", "wrist_left": "409122272149"}

## Dream's registered body (`datasets.list_embodiments()`, created 2026-10-23); its config, verbatim.
EMBODIMENT = "koutaueda/openarm2-faas"
EMBODIMENT_CONFIG = {
    "state": {
        "right_arm": [0, 7], "right_gripper": [7, 8], "left_arm": [8, 15], "left_gripper": [15, 16],
        "right_eef_xyz": [16, 19], "right_eef_rot6": [19, 25], "right_grip": [25, 26],
        "left_eef_xyz": [26, 29], "left_eef_rot6": [29, 35], "left_grip": [35, 36],
    },
    "action": {
        "right_eef_xyz": [0, 3], "right_eef_rot6": [3, 9], "right_grip": [9, 10],
        "left_eef_xyz": [10, 13], "left_eef_rot6": [13, 19], "left_grip": [19, 20],
        "left_arm": [20, 27], "right_arm": [27, 34],
    },
    "video": {"head": "observation.images.head_left", "wrist_l": "observation.images.wrist_left",
              "wrist_r": "observation.images.wrist_right"},
    "head_camera": "head",
}
## Our recorder's camera names to the body's video keys.
CAMERA_KEYS = {"observation.images.chest": "observation.images.head_left",
               "observation.images.wrist_left": "observation.images.wrist_left",
               "observation.images.wrist_right": "observation.images.wrist_right"}
## Dream's arm model the hand poses come from (the station's `ik`).
IK_ARM = "openarm2_faas"
## The jaw drive each arm's checkpoint reads, radians, fully open (`dream_ik.specs` `openarm2_faas`; closed is 0).
JAW_OPEN_RAD = {"left": 0.785, "right": -0.785}

## Fine-tune settings: cosine decay suits one task (Dream's docstring); the action carries hand rotations, so `use_eef`.
LR_SCHEDULE = "cosine"
USE_EEF = True

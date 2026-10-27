#!/usr/bin/env bash
# Recording for Dream's fine-tune: the wood block task, with the station's camera settings (1280x720, 15 fps), from the station's home.
# It runs openarm-orboh's record_bimanual_openarm.sh with the settings of the 2026-10-26 recording
# (~/collections/20261026T115757Z-bi-takeei-wood-l1-v1-s1/command.sh). Changed: camera size, fps, start pose
# (pipeline/start_pose, written by make_start_pose.py) and where it saves (dataset/raw).
#
# Use (in your own terminal, at the e-stop: the arms move):
#   bash record.sh                 # a new session of NUM_EPISODES episodes (default 50)
#   NUM_EPISODES=20 bash record.sh
#   RESUME=<folder name in dataset/raw> bash record.sh   # add episodes to a session
# Stop with esc (never Ctrl+Z).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$(cd "$HERE/.." && pwd)/dataset/raw"
ORBOH="$HOME/projects/openarm-orboh"
TASK_TEXT="$(cd "$HERE" && env -u PYTHONPATH ../../.venv/bin/python -c 'import config; print(config.TASK)')"
TASK_SLUG="$(cd "$HERE" && env -u PYTHONPATH ../../.venv/bin/python -c 'import config; print(config.TASK_SLUG)')"

# The station holds the cameras and the arms while it runs.
if curl -s -m 1 127.0.0.1:8091/status >/dev/null 2>&1; then
  echo "[record] Dream's station is running. Run ~/dream-env/bin/dreamcli robot down first." >&2
  exit 1
fi
for f in "$HERE/start_pose/left.yaml" "$HERE/start_pose/right.yaml"; do
  [ -f "$f" ] || { echo "[record] $f is missing. Run make_start_pose.py first." >&2; exit 1; }
done

mkdir -p "$RAW"
if [ -n "${RESUME:-}" ]; then
  NAME="$RESUME"; RESUME_FLAG=true
  [ -d "$RAW/$NAME" ] || { echo "[record] $RAW/$NAME does not exist." >&2; exit 1; }
else
  NAME="${TASK_SLUG}_15fps_$(date +%Y%m%d_%H%M%S)"; RESUME_FLAG=false
fi

cd "$ORBOH"
OPERATOR="${OPERATOR:-Kota}" \
JUDGE='' \
SCENE=kitakyushu-wood-l1-v1 \
CAM_SERIAL=344422070898 CHEST_V4L2='' \
WRIST_RIGHT=true WRIST_RIGHT_SERIAL=352122273417 WRIST_RIGHT_V4L2='' \
WRIST_LEFT=true WRIST_LEFT_SERIAL=409122272149 \
CAMERA_W=1280 CAMERA_H=720 FPS=15 \
NUM_EPISODES="${NUM_EPISODES:-50}" EPISODE_TIME_S=600 RESET_TIME_S=10 \
REPO_ID="Orboh/$NAME" \
RESUME="$RESUME_FLAG" DATASET_ROOT="$RAW/$NAME" \
POSE_SET='' LEFT_POSE_PATH="$HERE/start_pose/left.yaml" RIGHT_POSE_PATH="$HERE/start_pose/right.yaml" \
FREEZE_SIDE='' FROZEN_GRIPPER_DEG='' \
PLAY_SOUNDS=true DISPLAY_IP=127.0.0.1 DISPLAY_PORT=9876 \
D405_GAIN=16 D405_EXPOSURE=8000 D435_BRIGHTNESS='' \
TASK="$TASK_TEXT" \
TASK_SLUG="$TASK_SLUG" \
GRAVITY_COMP=true LSCALE=1.0 RSCALE=1.0 GRAVITY_URDF="$ORBOH/urdf/openarm_v10_bimanual.urdf" ALIGN_GRAVITY=false \
BILATERAL_FEEDBACK=true PARALLEL_ARMS=true PARALLEL_LEADER_FOLLOWER=true \
L_BILATERAL_GRIPPER_KP=1.25 L_BILATERAL_GRIPPER_KD=0.06 R_BILATERAL_GRIPPER_KP='' R_BILATERAL_GRIPPER_KD='' \
GRIPPER_KP=12 ARM_KP=240,240,240,240,24,31,25 \
EPISODE_ADVANCE=manual START_POSE_TOL_DEG=3 START_POSE_MAX_WAIT_S=120 \
START_POSE_HOLD=true START_POSE_HANDOFF_DEG=15 SAVE_ON_RETURN=false SAVE_ON_EPISODE_END=true RETURN_ON_EPISODE_END=true START_COUNTDOWN_S=3 \
DISPLAY_DATA=true \
bash scripts/record_bimanual_openarm.sh "dream-ft-wood-15fps"

echo "[record] saved to: $RAW/$NAME"
echo "[record] next: cd \"$HERE\" && env -u PYTHONPATH ../../.venv/bin/python convert.py \"$RAW/$NAME\""

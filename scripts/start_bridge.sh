#!/usr/bin/env bash
# Start the robot bridge (scripts/robot_bridge.py) in the LeRobot venv.
# Usage: bash scripts/start_bridge.sh fake|real
#   fake: simulated arms, no hardware (safe anywhere)
#   real: connects the follower arms; THEY MOVE to the start pose. Needs CAN up: sudo bash scripts/canup.sh
# Env: LEROBOT_DIR (default ~/projects/lerobot), POSE_DIR (dir with left.yaml + right.yaml; default is the
#      Kitakyushu wood-L1 pose set in ~/projects/openarm-orboh), ENDPOINT (default tcp://127.0.0.1:5560)
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-}"; [[ "$mode" == fake || "$mode" == real ]] || { sed -n 2,7p "$0"; exit 2; }
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/projects/lerobot}"
POSE_DIR="${POSE_DIR:-$HOME/projects/openarm-orboh/config/pose_sets/kitakyushu-wood-l1-v1}"
args=(--"$mode" --endpoint "${ENDPOINT:-tcp://127.0.0.1:5560}")
[[ -f "$POSE_DIR/left.yaml" && -f "$POSE_DIR/right.yaml" ]] && args+=(--left-pose "$POSE_DIR/left.yaml" --right-pose "$POSE_DIR/right.yaml")
if [[ "$mode" == real ]]; then
  read -r -p "The arms will move to their start pose. Workspace clear and e-stop in reach? [y/N] " ok
  [[ "$ok" == y ]] || exit 1
fi
exec "$LEROBOT_DIR/.venv/bin/python" scripts/robot_bridge.py "${args[@]}"

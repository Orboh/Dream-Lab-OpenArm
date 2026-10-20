#!/usr/bin/env bash
# Create the Python environment (.venv) and the secrets file (.env).
# Usage: bash scripts/setup_env.sh [--robot] [--dream]
#   --robot: also install RealSense + the LeRobot fork
#   --dream: also install the Dream Labs client + IK (needs Python 3.11-3.13; set PYTHON=python3.12)
set -euo pipefail
cd "$(dirname "$0")/.."

"${PYTHON:-python3}" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt     # exact pinned versions (reproducible)
.venv/bin/pip install -e . --no-deps          # the orboh package itself

if [[ " $* " == *" --dream "* ]]; then
  .venv/bin/pip install -r requirements-dream.txt
fi

if [[ " $* " == *" --robot "* ]]; then
  if [[ "$(uname -m)" == "x86_64" ]]; then
    .venv/bin/pip install -r requirements-robot.txt
  else
    echo "aarch64 (Jetson): build librealsense with -DBUILD_PYTHON_BINDINGS=ON and make pyrealsense2 importable in .venv"
  fi
  LEROBOT_DIR="${LEROBOT_DIR:-$HOME/projects/lerobot}"
  if [[ -d "$LEROBOT_DIR" ]]; then
    .venv/bin/pip install -e "$LEROBOT_DIR"
  else
    echo "LeRobot fork not found at $LEROBOT_DIR — clone the Orboh fork (branch feat/bi-openarm-soft-move-to) and re-run"
  fi
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env — add DREAMLAB_API_KEY, DREAMLAB_BASE_URL and ORBOH_WEB_TOKEN."
fi
echo "Done. Activate with: source .venv/bin/activate"

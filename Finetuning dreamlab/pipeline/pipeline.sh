#!/usr/bin/env bash
# The fine-tuning pipeline in one place: each stage runs one script of this folder, in order.
#   bash pipeline.sh prepare                 link the earlier wood recordings (dataset/source) and write the start pose from the station
#   bash pipeline.sh record                  record new episodes (dataset/raw); the arms move: your terminal, at the e-stop
#   bash pipeline.sh convert [folders...]    LeRobot v3.0 -> Dream's v2.1 sessions (dataset/dream); all of dataset/raw when none named
#   bash pipeline.sh validate                Dream's local checks of dataset/dream (nothing sent)
#   bash pipeline.sh upload <name> [--yes]   upload dataset/dream to koutaueda/openarm2-faas (sends data: needs --yes)
#   bash pipeline.sh status <ds_id>          the dataset's processing status
#   bash pipeline.sh finetune <dataset> <suffix> [--yes]   queue the fine-tune (use_eef, cosine; needs --yes)
#   bash pipeline.sh job <job_id>            the job's status and, once done, the checkpoint dream-1-ft:...
#   bash pipeline.sh test                    the pipeline's tests
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=(env -u PYTHONPATH -u AMENT_PREFIX_PATH ../../.venv/bin/python)
stage="${1:-}"; shift || true
case "$stage" in
  prepare)  "${PY[@]}" link_sources.py && "${PY[@]}" make_start_pose.py ;;
  record)   bash record.sh "$@" ;;
  convert)  "${PY[@]}" convert.py "$@" ;;
  validate|upload|status|finetune|job|list) "${PY[@]}" dream_steps.py "$stage" "$@" ;;
  test)     "${PY[@]}" -m pytest -q test_pipeline.py ;;
  *) sed -n 2,11p "$HERE/pipeline.sh"; exit 1 ;;
esac

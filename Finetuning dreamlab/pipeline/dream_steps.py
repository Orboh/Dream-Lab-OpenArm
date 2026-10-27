"""Dream's side of the pipeline: check the converted sessions, upload them, fine-tune, and watch both, through Dream's own client.

Steps (each prints what it did; upload and fine-tune also append their ids to `log/jobs.jsonl`):
  validate                      Dream's local checks of `dataset/dream` against the body (nothing is sent, no key needed)
  upload <name>                 validate, then upload `dataset/dream` as dataset `<name>` of body koutaueda/openarm2-faas
  status <ds_id>                the dataset's status (`ready_to_process`, `processing`, `processed` or `failed`)
  finetune <dataset> <suffix>   queue a fine-tune (`use_eef`, cosine) on a processed dataset; the checkpoint is `dream-1-ft:<uid8>-<suffix>`
  job <job_id>                  the fine-tune job's status, and its checkpoint once done
  list                          your datasets and models
Upload and finetune send data to Dream's service and need `--yes`.

Run:  env -u PYTHONPATH ../../.venv/bin/python dream_steps.py <step> ...
"""

import argparse
import datetime
import json
import pathlib
import sys

import config


def _client():
    import dream

    return dream.DreamClient()


def _log(entry: dict) -> None:
    """Appends one record to `log/jobs.jsonl` with the time."""
    config.JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {"at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), **entry}
    with config.JOBS_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def validate(path: pathlib.Path = config.DREAM_DIR):
    """Dream's local dataset checks against the body; returns the report, exits on a failure."""
    from dream import datasets

    report = datasets.validate_dataset(str(path), config.EMBODIMENT_CONFIG)
    print(report)
    if not report.ok:
        sys.exit(f"not valid: {report.error}")
    if report.fps != config.FPS:
        print(f"note: the sessions are {report.fps} fps, the plan records {config.FPS}")
    return report


def main() -> None:
    """Runs one step."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--path", type=pathlib.Path, default=config.DREAM_DIR)
    u = sub.add_parser("upload")
    u.add_argument("name")
    u.add_argument("--path", type=pathlib.Path, default=config.DREAM_DIR)
    u.add_argument("--yes", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("dataset_id")
    f = sub.add_parser("finetune")
    f.add_argument("dataset")
    f.add_argument("suffix")
    f.add_argument("--learning-rate", type=float, default=None)
    f.add_argument("--yes", action="store_true")
    j = sub.add_parser("job")
    j.add_argument("job_id")
    sub.add_parser("list")
    args = ap.parse_args()

    if args.step == "validate":
        validate(args.path)
    elif args.step == "upload":
        report = validate(args.path)
        if not args.yes:
            sys.exit(f"would upload {report.episodes} episodes ({report.bytes / 1e9:.2f} GB) as '{args.name}' to {config.EMBODIMENT}; add --yes")
        ds = _client().upload_dataset(str(args.path), args.name, config.EMBODIMENT)
        print(ds)
        _log({"step": "upload", "name": ds.name, "id": ds.id, "status": ds.status, "episodes": ds.episodes, "bytes": ds.size_bytes})
    elif args.step == "status":
        ds = _client().get_dataset_status(args.dataset_id)
        print(ds)
    elif args.step == "finetune":
        if not args.yes:
            sys.exit(f"would fine-tune on '{args.dataset}' (suffix '{args.suffix}', use_eef={config.USE_EEF}, lr_schedule={config.LR_SCHEDULE}); add --yes")
        job = _client().finetune(args.dataset, args.suffix, learning_rate=args.learning_rate, use_eef=config.USE_EEF,
                                 lr_schedule=config.LR_SCHEDULE)
        print(json.dumps(job, indent=2, default=str))
        _log({"step": "finetune", "dataset": args.dataset, "suffix": args.suffix, "job": job})
    elif args.step == "job":
        job = _client().get_finetuning_status(args.job_id)
        print(json.dumps(job, indent=2, default=str))
        if job.get("fine_tuned_model"):
            _log({"step": "job_done", "id": args.job_id, "model": job["fine_tuned_model"], "status": job.get("status")})
    elif args.step == "list":
        client = _client()
        for ds in client.get_datasets():
            print("dataset", ds.name, ds.id, ds.status, ds.episodes)
        for m in client.get_models():
            print("model", m)


if __name__ == "__main__":
    main()

"""Links every earlier wood recording into `dataset/source/` and writes `dataset/source/SOURCES.md`, naming where each one is.

The files stay where `lerobot-record` wrote them (`~/.cache/huggingface/lerobot/Orboh/`, 11 GB in all); each entry here is a
symbolic link to that folder, so nothing is copied. These recordings are reference only: they are LeRobot v3.0 at 30 fps and
640x480, from the earlier camera position and start pose, so the fine-tune uses the new recordings (`dataset/raw/`).

Run:  env -u PYTHONPATH ../../.venv/bin/python link_sources.py
"""

import json
import pathlib

import config


def collection_of(dataset: pathlib.Path) -> pathlib.Path | None:
    """The recorder's run folder (`~/collections/<run>`) whose manifest names `dataset` as its dataset root, or None."""
    for manifest in config.COLLECTIONS.glob("*/manifest.json"):
        try:
            root = json.loads(manifest.read_text()).get("artifacts", {}).get("dataset_root")
        except (OSError, ValueError):
            continue
        if root and pathlib.Path(root) == dataset:
            return manifest.parent
    return None


def describe(dataset: pathlib.Path) -> dict:
    """One recording's facts from its `meta/info.json` (and the task from `meta/tasks.parquet` when present)."""
    info = json.loads((dataset / "meta" / "info.json").read_text())
    cams = [k.split(".")[-1] for k, f in info["features"].items() if f.get("dtype") == "video"]
    shape = next((info["features"][k]["shape"] for k, f in info["features"].items() if f.get("dtype") == "video"), None)
    task = None
    tasks_file = dataset / "meta" / "tasks.parquet"
    if tasks_file.exists():
        import pyarrow.parquet as pq

        rows = pq.read_table(tasks_file).to_pylist()
        task = next((r.get("task") or r.get("__index_level_0__") for r in rows), None)
    return {"name": dataset.name, "version": info.get("codebase_version"), "fps": info.get("fps"),
            "episodes": info.get("total_episodes", 0), "frames": info.get("total_frames", 0), "cameras": cams,
            "size": f"{shape[1]}x{shape[0]}" if shape else "", "task": task}


def main() -> None:
    """Makes one link per wood recording with at least one episode, and lists every wood recording (empty ones too) in SOURCES.md."""
    config.SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in sorted(config.LEROBOT_CACHE.glob(config.SOURCE_PATTERN)):
        if not (dataset / "meta" / "info.json").exists():
            continue
        d = describe(dataset)
        link = config.SOURCE_DIR / dataset.name
        if d["episodes"] > 0:
            if link.is_symlink():
                link.unlink()
            link.symlink_to(dataset, target_is_directory=True)
        run = collection_of(dataset)
        rows.append((d, dataset, run, d["episodes"] > 0))
    lines = [
        "# Earlier wood block recordings (source)",
        "",
        f"Source folder: `{config.LEROBOT_CACHE}` (written by `lerobot-record`, LeRobot 0.5.2; recorder: "
        "`~/projects/openarm-orboh/scripts/record_bimanual_openarm.sh`). Each linked entry here is a symbolic link to its folder there;",
        "the recorder's run folder (manifest, console log, command) is in `~/collections/` where one names the recording.",
        "Written by `pipeline/link_sources.py`. **Reference only:** v3.0, 30 fps, 640x480, old camera position and start pose; the fine-tune uses `dataset/raw/`.",
        "",
        f"Total: {sum(1 for r in rows if r[3])} linked recordings, {sum(r[0]['episodes'] for r in rows)} episodes, "
        f"{sum(r[0]['frames'] for r in rows)} frames.",
        "",
        "| Recording | Linked | Episodes | Frames | fps | Video | Cameras | Task | Source | Run folder |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d, dataset, run, linked in rows:
        lines.append(f"| {d['name']} | {'yes' if linked else 'no (empty)'} | {d['episodes']} | {d['frames']} | {d['fps']} | {d['size']} | "
                     f"{', '.join(d['cameras'])} | {d['task'] or ''} | `{dataset}` | {f'`{run}`' if run else '-'} |")
    (config.SOURCE_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n")
    print(f"{sum(1 for r in rows if r[3])} linked of {len(rows)} wood recordings -> {config.SOURCE_DIR}")


if __name__ == "__main__":
    main()

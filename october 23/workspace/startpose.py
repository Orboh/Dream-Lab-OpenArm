"""Saves the arms' pose now as a new start-pose set for collection (`openarm-orboh/config/pose_sets/<name>/`), through the repo's own `pose_set.py`.

Collection reads its start pose from the set named by `POSE_SET`, not from the safety-area bundle, so this is what a "better start pose" has to produce.
The new set is a `draft`: it starts as a copy of the base set (waypoints, comments) with only the final seven joints replaced, and `pose_set.py new` adds the
calibration, scene and code versions. A frozen set is never touched.
"""

import datetime as dt
import pathlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable

import collect

SIDES = ("left", "right")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$")
JOINT = re.compile(r"^joint_([1-7]):.*$", re.M)  # the final pose only: waypoint entries are indented ("  - joint_1:")


class StartPoseError(Exception):
    """The set could not be saved; the message says why (already in the user's language when it is a key)."""


def pose_sets_dir(repo: pathlib.Path = collect.REPO) -> pathlib.Path:
    return repo / "config" / "pose_sets"


def next_name(base: str, repo: pathlib.Path = collect.REPO) -> str:
    """`kitakyushu-wood-l1-v1` -> `kitakyushu-wood-l1-v2` (or the next free number); anything else gets `-v2` added."""
    match = re.match(r"^(.*-v)(\d+)$", base)
    stem, n = (match.group(1), int(match.group(2)) + 1) if match else (base + "-v", 2)
    while (pose_sets_dir(repo) / f"{stem}{n}").exists():
        n += 1
    return f"{stem}{n}"


def pose_text(base_text: str, joints_deg, note: str) -> str:
    """The base set's yaml with the final seven joints replaced by `joints_deg` (degrees) and one comment line added at the top."""
    values = {i + 1: f"{float(v):.2f}" for i, v in enumerate(joints_deg)}
    out = JOINT.sub(lambda m: f"joint_{m.group(1)}: {values[int(m.group(1))]}", base_text)
    if out == base_text and all(f"joint_{i}: {values[i]}" not in base_text for i in values):
        raise StartPoseError("start.err.format")
    return f"# {note}\n{out}"


def base_value(base: str, key: str, repo: pathlib.Path = collect.REPO) -> str:
    """One `key: value` of the base set's `set.yaml` ('' when absent)."""
    text = (pose_sets_dir(repo) / base / "set.yaml").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(key)}:\s*(.*?)\s*$", text, re.M)
    return match.group(1) if match else ""


def save(name: str, base: str, joints_deg: dict[str, list[float]], repo: pathlib.Path = collect.REPO,
         runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, now: Callable[[], dt.datetime] = dt.datetime.now) -> str:
    """Creates the set.

    @param name The new set's name. @param base The set to copy (its waypoints and scene are kept).
    @param joints_deg Seven degrees per side, `left` and `right`. @param repo The openarm-orboh checkout.
    @return What `pose_set.py` printed.
    @raise StartPoseError With a message key (`start.err.*`) and the detail after `|`.
    """
    if not NAME.match(name):
        raise StartPoseError("start.err.name")
    base_dir = pose_sets_dir(repo) / base
    if not base_dir.is_dir():
        raise StartPoseError(f"start.err.base|{base}")
    if (pose_sets_dir(repo) / name).exists():
        raise StartPoseError(f"start.err.exists|{name}")
    note = f"{now():%Y-%m-%d %H:%M} エディタ画面から保存: その時点のフォロワーの姿勢（元セット {base}）。経由地は元セットのまま。"
    with tempfile.TemporaryDirectory(prefix="pose-") as tmp:
        for side in SIDES:
            (pathlib.Path(tmp) / f"{side}.yaml").write_text(pose_text((base_dir / f"{side}.yaml").read_text(encoding="utf-8"), joints_deg[side], note), encoding="utf-8")
        argv = [sys.executable, str(repo / "scripts" / "pose_set.py"), "new", name, "--poses-dir", tmp]
        scene, mirror = base_value(base, "scene", repo), base_value(base, "mirror", repo)
        if scene:
            argv += ["--scene", scene]
        if mirror:
            argv += ["--mirror", mirror]
        done = runner(argv, cwd=repo, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise StartPoseError(f"start.err.tool|{(done.stdout + done.stderr).strip()[-400:]}")
    set_file = pose_sets_dir(repo) / name / "set.yaml"
    if set_file.is_file():  # pose_set.py records the temporary folder as the origin; the base set is the useful fact
        set_file.write_text(re.sub(r"^derived_from:.*$", f"derived_from: {base}", set_file.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")
    return done.stdout.strip()

"""Starts and stops episode collection from the page: runs the existing `collect_bimanual.sh` unchanged, so the data is saved exactly as before.

The script reads its control keys (right arrow, left arrow, `a`, Esc) from a terminal, so the child gets a pseudo-terminal and the page types those keys into it.
Nothing about the dataset or the run folder is decided here: LeRobot and `record_bimanual_openarm.sh` write them (`~/collections/<UTC>-<tag>/`).
"""

import collections
import dataclasses
import datetime as dt
import json
import os
import pathlib
import pty
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable

REPO = pathlib.Path("~/projects/openarm-orboh").expanduser()
DEFAULT_ENV = REPO / "config" / "bimanual-collect.kitakyushu_takeei_sort.env"
COLLECTIONS = pathlib.Path("~/collections").expanduser()
MODES = ("dry", "gate", "session")
DEFAULT_POSE_SET = "kitakyushu-takeei-v5"  # what the page's "start pose set" field starts with (used only while that set exists; the field can be changed or emptied)
KEYS = {"next": b"\x1b[C", "rerecord": b"\x1b[D", "align": b"a", "stop": b"\x1b"}  # the sequences lerobot's stdin listener decodes
TELEOP_PATTERNS = ("run_teleop", "lerobot-teleoperate", "launch_bilateral", "launch_bimanual", "openarm_teleop")
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][A-Za-z0-9]|\r")
TARGET = re.compile(r"\[collect-bi\] mode=\S+ .*?episodes=(\d+)")  # the line collect_bimanual.sh prints with the number asked for
RUN_DIR = re.compile(r"\[record-bi\] 記録: (\S+)|run_dir=(\S+)")
RECORDING = re.compile(r"Recording episode (\d+)|録画中 episode (\d+)")  # lerobot's English log line, or the Japanese banner the fork prints since 2026-10
OPERATOR_NAME = re.compile(r"[A-Za-z][A-Za-z0-9 ._-]*")  # half-width only: the name goes into the dataset's manifest as typed
ENV_SHOWN = ("TASK", "SCENE", "POSE_SET", "EPISODE_TIME_S")  # what the Collect tab shows of the chosen settings file (V2, 2026-10-28)
SET_FILE = COLLECTIONS / ".collect-set.json"  # the last gate/session set: how many it asked for and where its dataset is, so a stopped set can be finished later


class CollectError(Exception):
    """The collection cannot start (or stop) now; the message says why and is already in the user's language."""


@dataclasses.dataclass
class Options:
    """What the Start button passes: the three words of `collect_bimanual.sh` plus the switches from the runbook (2026-10-19)."""

    mode: str = "dry"
    operator: str = ""  # who records this run; passed as OPERATOR, which wins over the settings file's (all runs said Kota until 2026-10-28)
    tag: str = ""
    episodes: int = 0  # 0: the script's own default
    play_sounds: bool = True
    bilateral: bool = True
    save_on_episode_end: bool = True
    env_file: pathlib.Path = DEFAULT_ENV
    return_on_episode_end: bool = True  # the key that ends a take saves it and the arms walk back to the start pose by themselves (no `a`); ← discards and returns
    start_pose_hold: bool = True  # keep both arms at the start pose until the next recording starts, so the leader's wrist cannot droop (record_bimanual_openarm.sh)
    start_countdown_s: int = 3  # seconds of count-down before the first frame, used only with the hold
    episode_time_s: int = 0  # how long one episode may run before it ends by itself; 0: the settings file's EPISODE_TIME_S
    pose_set: str = ""  # the start-pose set to use; "": the settings file's POSE_SET
    resume_root: str = ""  # an existing dataset folder to append to (RESUME=true); "": a new dataset
    resume_repo_id: str = ""  # its stamped repo id (…_v1_YYYYMMDD_HHMMSS), needed with resume_root


def command(opts: Options, repo: pathlib.Path = REPO) -> tuple[list[str], dict[str, str]]:
    """The argument list and the environment to run: the same line you would type in the terminal.

    @param opts The chosen options.
    @param repo The openarm-orboh checkout.
    @return `(argv, extra_env)`; `extra_env` is added to this process's environment.
    """
    argv = ["bash", str(repo / "scripts" / "collect_bimanual.sh"), opts.mode]
    if opts.mode == "session":
        argv.append(opts.tag)
    if opts.episodes > 0:  # the script takes the count as the last word: `gate N`, `dry N`, `session TAG N`
        argv.append(str(opts.episodes))
    env = {"ENV_FILE": str(opts.env_file), "PLAY_SOUNDS": str(opts.play_sounds).lower(), "BILATERAL_FEEDBACK": str(opts.bilateral).lower(),
           "SAVE_ON_EPISODE_END": str(opts.save_on_episode_end).lower()}
    if opts.operator.strip():
        env["OPERATOR"] = opts.operator.strip()
    if opts.episode_time_s > 0:  # a variable given by the caller wins over the settings file (collect_bimanual.sh)
        env["EPISODE_TIME_S"] = str(int(opts.episode_time_s))
    env["RETURN_ON_EPISODE_END"] = str(opts.return_on_episode_end).lower()
    if opts.start_pose_hold:
        env["START_POSE_HOLD"] = "true"
        env["START_COUNTDOWN_S"] = str(max(int(opts.start_countdown_s), 0))
    if opts.pose_set.strip():
        env["POSE_SET"] = opts.pose_set.strip()
    if opts.bilateral:  # the left gripper feels stiffer: the runbook's values
        env.update(L_BILATERAL_GRIPPER_KP="1.25", L_BILATERAL_GRIPPER_KD="0.06")
    if opts.resume_root:  # record_bimanual_openarm.sh: RESUME needs the real folder and the stamped name, else it makes a new dataset
        env.update(RESUME="true", DATASET_ROOT=opts.resume_root, REPO_ID=opts.resume_repo_id)
    return argv, env


def dataset_episodes(root: str | None) -> int | None:
    """Episodes a LeRobot dataset holds (`meta/info.json`), or None when there is no readable dataset there."""
    if not root:
        return None
    try:
        return int(json.loads((pathlib.Path(root) / "meta" / "info.json").read_text(encoding="utf-8"))["total_episodes"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def saved_location(run_dir: str | None) -> dict:
    """Where a finished run left its data (V5): the run folder and the dataset folder named in its manifest.json, each only when it exists.

    @param run_dir The run folder the script printed (`~/collections/<UTC>-<tag>`), or None.
    @return `{run_dir, dataset_root, episodes}`; a path is None when it is unknown or not on disk, `episodes` when the dataset is readable.
    """
    run = pathlib.Path(run_dir) if run_dir else None
    if run is None or not run.is_dir():
        return {"run_dir": None, "dataset_root": None, "episodes": None}
    try:  # the stamped name is decided in Python (stamp_repo_id), so the manifest is the only sure source of the real folder
        root = (json.loads((run / "manifest.json").read_text(encoding="utf-8")).get("artifacts") or {}).get("dataset_root")
    except (OSError, ValueError, AttributeError):
        root = None
    if not root or not pathlib.Path(root).is_dir():
        root = None  # 10/26: three runs ended before a dataset was made
    return {"run_dir": str(run), "dataset_root": root, "episodes": dataset_episodes(root)}


def env_value(path: pathlib.Path, key: str) -> str:
    """`KEY=value` from a settings file ('' when missing); quotes and a trailing `# comment` are dropped."""
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.M)
    if not match:
        return ""
    return re.sub(r"\s+#.*$", "", match.group(1)).strip().strip("\"'")


def env_files(repo: pathlib.Path = REPO) -> list[pathlib.Path]:
    """The collection settings files a run can use: `config/bimanual-collect.*.env` in openarm-orboh, by name."""
    return sorted((pathlib.Path(repo) / "config").glob("bimanual-collect.*.env"))


def env_summary(path: pathlib.Path) -> dict[str, str]:
    """The four values of a settings file the Collect tab shows ('' when one is missing)."""
    return {key: env_value(path, key) for key in ENV_SHOWN}


def running_teleop(pgrep: Callable[[str], bool] | None = None) -> str | None:
    """The name of a teleop program that is still running (it would fight collection for the arms), or None."""
    def default(pattern: str) -> bool:
        return subprocess.run(["pgrep", "-f", pattern], capture_output=True, check=False).returncode == 0

    for pattern in TELEOP_PATTERNS:
        if (pgrep or default)(pattern):
            return pattern
    return None


class CollectSession:
    """One collection run at a time: the child process on a pseudo-terminal, its recent output, and what phase it is in."""

    def __init__(self, repo: pathlib.Path = REPO, tail: int = 400, teleop_probe: Callable[[], str | None] = running_teleop,
                 argv_for: Callable[[Options], tuple[list[str], dict[str, str]]] | None = None, set_file: pathlib.Path | None = SET_FILE) -> None:
        """@param repo The openarm-orboh checkout. @param tail How many output lines to keep. @param teleop_probe Name of a running teleop, or None.
        @param argv_for Builds `(argv, env)` from the options (tests replace it). @param set_file Where the last set is remembered (None: not at all)."""
        self.repo, self._probe, self.set_file = repo, teleop_probe, set_file
        self._argv_for = argv_for or (lambda o: command(o, repo))
        self.lines: collections.deque[str] = collections.deque(maxlen=tail)
        self.phase = "idle"  # idle starting ready recording returning stopping finished
        self.episode = 0
        self.started = 0  # recordings begun this run (a re-record counts again)
        self.target: int | None = None  # episodes asked for this run, from the script's own banner
        self.done = 0  # episodes finished and kept this run (a discarded take is not one)
        self.saved = 0  # episodes the script reported as written to the dataset ("を保存した")
        self.rc: int | None = None
        self.run_dir: str | None = None
        self.saved_to: dict | None = None  # where the last finished run saved (V5); None while one is going
        self._mode = "dry"  # the mode of the current/last run
        self.proc: subprocess.Popen | None = None
        self._master: int | None = None
        self._lock = threading.Lock()

    # -- control --------------------------------------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, opts: Options) -> None:
        """Launches the script. @raise CollectError when one is running, a teleop is running, or the settings file is missing."""
        if self.active:
            raise CollectError("col.err.busy")
        teleop = self._probe()
        if teleop:
            raise CollectError(f"col.err.teleop|{teleop}")
        if not pathlib.Path(opts.env_file).is_file():
            raise CollectError(f"col.err.env|{opts.env_file}")
        if opts.mode == "session" and not opts.tag.strip():
            raise CollectError("col.err.tag")
        if not opts.operator.strip():
            raise CollectError("col.err.operator")
        if not OPERATOR_NAME.fullmatch(opts.operator.strip()):
            raise CollectError(f"col.err.operator.chars|{opts.operator.strip()}")
        if opts.mode == "session":  # instructions §3.3: stop after the gate and show the reviewer; no session until the OK (V4)
            if self.review_pending() is not None:
                raise CollectError("col.err.review")
            if self._gate_open():
                raise CollectError("col.err.gateopen")
        argv, extra = self._argv_for(opts)
        target = opts.episodes or {"dry": 2, "gate": 10, "session": 10}[opts.mode]
        if opts.mode != "dry" and not opts.resume_root:  # a new set; a resume keeps the set it continues, a dry run leaves it alone
            self._write_set({"mode": opts.mode, "tag": opts.tag.strip(), "target": target, "dataset_root": None, "repo_id": None})
        master, slave = pty.openpty()
        env = {**os.environ, **extra, "TERM": "xterm", "PYTHONUNBUFFERED": "1"}
        self.lines.clear()
        self.phase, self.episode, self.started, self.rc, self.run_dir = "starting", 0, 0, None, None
        self.saved, self.saved_to = 0, None
        self.target, self.done = target, 0
        self._mode = opts.mode
        self.proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, cwd=self.repo, env=env, start_new_session=True, close_fds=True)
        os.close(slave)
        self._master = master
        threading.Thread(target=self._read, args=(master, self.proc), daemon=True).start()

    def send(self, name: str) -> None:
        """Types one of the control keys (`next`, `rerecord`, `align`, `stop`) into the run. @raise CollectError when nothing is running."""
        if not self.active or self._master is None:
            raise CollectError("col.err.notrunning")
        os.write(self._master, KEYS[name])
        if name == "stop":
            self.phase = "stopping"

    def kill(self) -> None:
        """Ctrl-C for the whole run (the script traps it and relaxes the arms); a second call, 8 s later, ends it for sure."""
        if not self.active:
            raise CollectError("col.err.notrunning")
        self.phase = "stopping"
        os.killpg(self.proc.pid, signal.SIGINT)

    def wait(self, timeout: float = 5.0) -> int | None:
        """Waits for the run to end (tests). @return The exit code, or None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None and self.phase == "finished":
                return self.rc
            time.sleep(0.02)
        return None

    # -- output ---------------------------------------------------------------------------------------------------------
    def _read(self, master: int, proc: subprocess.Popen) -> None:
        buffer = b""
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:  # the child closed the terminal
                break
            if not chunk:
                break
            buffer += chunk
            *done, buffer = buffer.split(b"\n")
            for raw in done:
                self.feed(raw.decode("utf-8", "replace"))
        if buffer:
            self.feed(buffer.decode("utf-8", "replace"))
        proc.wait()
        self._remember_dataset()  # before "finished", so the page offers Resume the moment it sees the run end
        saved_to = saved_location(self.run_dir)
        with self._lock:
            self.rc, self.saved_to = proc.returncode, saved_to
            self.phase = "finished"
        try:
            os.close(master)
        except OSError:
            pass

    def feed(self, raw: str) -> None:
        """Takes one output line: keeps it and updates the phase from what the script says."""
        line = ANSI.sub("", raw).rstrip()
        if not line:
            return
        with self._lock:
            self.lines.append(line)
            match = RUN_DIR.search(line)
            if match:
                self.run_dir = match.group(1) or match.group(2)
            counted = TARGET.search(line)
            if counted:
                self.target = int(counted.group(1))
            recording = RECORDING.search(line)
            if recording:
                self.phase = "recording"
                self.episode = int(recording.group(1) or recording.group(2))
                self.started += 1
            elif "を保存中" in line:
                self.phase = "saving"
            elif "を保存した" in line:
                self.saved += 1
                self.done = max(self.done, self.saved)  # a save is the surest sign a take was kept
                self.phase = "returning"  # with the automatic return the arms walk back next; "Waiting" below sets ready when they have
            elif "Right arrow key pressed" in line and self.phase == "recording":
                self.phase, self.done = "returning", self.done + 1
            elif "Left arrow key pressed" in line and self.phase == "recording":
                self.phase = "returning"
            elif "key pressed. Aligning" in line or "Start pose: driving" in line or "Start pose: walking" in line:
                self.phase = "returning"
            elif "Escape key pressed" in line or "Stop recording" in line:
                if self.phase == "recording":  # Esc keeps the take that is running
                    self.done += 1
                self.phase = "stopping"
            elif "Arm returned" in line or "Waiting:" in line or "待機中" in line:  # also "Waiting: both arms are held ..." (the hold)
                if self.phase == "recording" and "待機中" not in line:  # the episode ended by itself (time limit): it was kept
                    self.done += 1
                if self.phase != "stopping":
                    self.phase = "ready"

    def snapshot(self, last: int = 12) -> dict:
        """What the card shows: phase, episode, run folder and the last output lines."""
        with self._lock:
            left = None if self.target is None else max(self.target - self.done, 0)
            return {"phase": self.phase, "episode": self.episode, "started": self.started, "rc": self.rc, "run_dir": self.run_dir,
                    "target": self.target, "done": self.done, "left": left, "saved": self.saved,
                    "lines": list(self.lines)[-last:], "active": self.active, "saved_to": self.saved_to}

    # -- the set (gate / session) across runs -------------------------------------------------------------------------
    def _read_set(self) -> dict | None:
        if self.set_file is None:
            return None
        try:
            return json.loads(self.set_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_set(self, data: dict) -> None:
        if self.set_file is None:
            return
        try:
            self.set_file.parent.mkdir(parents=True, exist_ok=True)
            self.set_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass  # only the resume offer is lost; the data itself is safe

    def _remember_dataset(self) -> None:
        """After a gate/session run: notes the dataset it wrote (from the run's manifest.json) so the set can be resumed into it."""
        current = self._read_set()
        if current is None or self._mode == "dry" or current.get("dataset_root") or not self.run_dir:
            return
        try:
            artifacts = json.loads((pathlib.Path(self.run_dir) / "manifest.json").read_text(encoding="utf-8")).get("artifacts") or {}
        except (OSError, ValueError, AttributeError):
            return
        if artifacts.get("dataset_root") and artifacts.get("repo_id_stamped"):
            current.update(dataset_root=artifacts["dataset_root"], repo_id=artifacts["repo_id_stamped"])
            self._write_set(current)

    def pending(self) -> dict | None:
        """The last set if it stopped short: `{mode, tag, target, have, left, dataset_root, repo_id}`; None when it is complete, unknown or running.

        `have` is counted in the dataset itself, so it is right however many runs the set took."""
        if self.active:
            return None
        current = self._read_set()
        if not current or not current.get("target"):
            return None
        have = dataset_episodes(current.get("dataset_root")) or 0
        left = int(current["target"]) - have
        if left <= 0:
            return None
        return {**current, "have": have, "left": left}

    def _unreviewed_gate(self) -> dict | None:
        """The last set when it is a gate nobody has reviewed yet, with `have` counted in its dataset; else None."""
        current = self._read_set()
        if not current or current.get("mode") != "gate" or current.get("reviewed_by") or not current.get("target"):
            return None
        return {**current, "have": dataset_episodes(current.get("dataset_root")) or 0}

    def review_pending(self) -> dict | None:
        """A finished gate waiting for the reviewer's OK (V4): `{..., have}`; None when there is none or a run is going."""
        gate = None if self.active else self._unreviewed_gate()
        return gate if gate and gate["have"] >= int(gate["target"]) else None

    def _gate_open(self) -> bool:
        """The last gate stopped short: resume it before any session."""
        gate = self._unreviewed_gate()
        return bool(gate and gate["have"] < int(gate["target"]))

    def mark_reviewed(self, reviewer: str) -> None:
        """Records the reviewer's OK on the finished gate in the set file, so a session may start (and the page remembers it after a restart).
        @raise CollectError when no finished gate is waiting or no reviewer is named."""
        gate = self.review_pending()
        if gate is None:
            raise CollectError("col.err.noreview")
        if not reviewer.strip():
            raise CollectError("col.err.reviewer")
        current = self._read_set() or {}
        current.update(reviewed_by=reviewer.strip(), reviewed_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
        self._write_set(current)

    def resume_options(self, base: Options) -> Options:
        """The options that finish the stopped set: its mode and tag, only the episodes it still needs, appended to its dataset.

        A set that saved nothing (e.g. the arms did not answer at start) starts again with a new dataset.
        @raise CollectError when there is nothing to resume."""
        todo = self.pending()
        if todo is None:
            raise CollectError("col.err.noresume")
        opts = dataclasses.replace(base, mode=todo["mode"], tag=todo.get("tag", ""), episodes=todo["left"], resume_root="", resume_repo_id="")
        if todo["have"] > 0 and todo.get("dataset_root") and todo.get("repo_id"):
            opts.resume_root, opts.resume_repo_id = todo["dataset_root"], todo["repo_id"]
        return opts

"""Starts and stops episode collection from the page: runs the existing `collect_bimanual.sh` unchanged, so the data is saved exactly as before.

The script reads its control keys (right arrow, left arrow, `a`, Esc) from a terminal, so the child gets a pseudo-terminal and the page types those keys into it.
Nothing about the dataset or the run folder is decided here: LeRobot and `record_bimanual_openarm.sh` write them (`~/collections/<UTC>-<tag>/`).
"""

import collections
import dataclasses
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


class CollectError(Exception):
    """The collection cannot start (or stop) now; the message says why and is already in the user's language."""


@dataclasses.dataclass
class Options:
    """What the Start button passes: the three words of `collect_bimanual.sh` plus the switches from the runbook (2026-10-19)."""

    mode: str = "dry"
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
    return argv, env


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
                 argv_for: Callable[[Options], tuple[list[str], dict[str, str]]] | None = None) -> None:
        """@param repo The openarm-orboh checkout. @param tail How many output lines to keep. @param teleop_probe Name of a running teleop, or None.
        @param argv_for Builds `(argv, env)` from the options (tests replace it)."""
        self.repo, self._probe = repo, teleop_probe
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
        argv, extra = self._argv_for(opts)
        master, slave = pty.openpty()
        env = {**os.environ, **extra, "TERM": "xterm", "PYTHONUNBUFFERED": "1"}
        self.lines.clear()
        self.phase, self.episode, self.started, self.rc, self.run_dir = "starting", 0, 0, None, None
        self.saved = 0
        self.target, self.done = (opts.episodes or {"dry": 2, "gate": 10, "session": 10}[opts.mode]), 0
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
        with self._lock:
            self.rc = proc.returncode
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
            if "Recording episode" in line:
                self.phase = "recording"
                digits = re.search(r"Recording episode (\d+)", line)
                self.episode = int(digits.group(1)) if digits else self.episode
                self.started += 1
            elif "を保存中" in line:
                self.phase = "saving"
            elif "を保存した" in line:
                self.saved += 1
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
                    "lines": list(self.lines)[-last:], "active": self.active}

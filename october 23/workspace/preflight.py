"""The checks to pass before every session, in one list: PASS or FAIL with the reason. System probes (CAN links, running programs, cameras) are replaceable so the checks are tested offline."""

import dataclasses
import datetime as dt
import hashlib
import os
import pathlib
import subprocess
from typing import Any

import numpy as np
import yaml

import openarm_guard
import openarm_lock
import openarm_zones as zones
import zonebuild as zb

LABEL_CAN = "Both follower CAN links up in FD mode and healthy"
LABEL_FREE = "Nothing else holds the arms (teleop, station, setup wizard)"
LABEL_CAMERAS = "The station's cameras are plugged in"
LABEL_CAMERAS_FREE = "No other program has the cameras open"
LABEL_LOADS = "Dream accepts the station file"
LABEL_FRESH = "Safety zones in force exist and are less than 24 hours old"
LABEL_NEWEST = "The zones in force are the newest saved bundle"
LABEL_HOME = "The station's home equals the bundle's home (both arms)"
LABEL_PATH = "A home path from rest is stored and ends at the home"
LABEL_CLEAR = "The home, rest, ready poses and the path are clear of every zone and the home is inside the allowed region"
LABEL_DRIVER = "The station uses the Arm driver (not the read-only one)"
LABEL_GUARD = "The hard guard is on"
HOLDERS = ("lerobot-teleoperate", "run_teleop", "robots.station", "dreamcli setup", "dreamcli robot up")
LINKS = ("can_fol_l", "can_fol_r")
MAX_ZONE_AGE_H = 24.0


@dataclasses.dataclass
class Check:
    """One line of the report."""

    ok: bool
    label: str
    detail: str = ""


class Probes:
    """What the checks ask of the computer; the defaults really look (ip, ps, librealsense)."""

    def can_links(self) -> dict[str, str]:
        out = {}
        for name in LINKS:
            result = subprocess.run(["ip", "-d", "link", "show", name], capture_output=True, text=True, check=False)
            out[name] = result.stdout if result.returncode == 0 else ""
        return out

    def processes(self) -> list[str]:
        return subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=False).stdout.splitlines()

    def cameras(self) -> set[str]:
        try:
            import pyrealsense2 as rs

            return {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
        except Exception:
            return set()


    def camera_holders(self) -> list[str]:
        """The programs that have a /dev/video* device open (pid and command), other than this one."""
        out = []
        for proc in pathlib.Path("/proc").glob("[0-9]*"):
            try:
                if int(proc.name) == os.getpid() or not any(os.readlink(fd).startswith("/dev/video") for fd in proc.glob("fd/*")):
                    continue
                out.append(f"{proc.name}: {(proc / 'cmdline').read_bytes().replace(bytes(1), b' ').decode(errors='replace').strip()}")
            except OSError:
                continue
        return out


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_preflight(station: pathlib.Path, in_force: pathlib.Path, zones_dir: pathlib.Path, probes: Probes | None = None,
                  now: dt.datetime | None = None) -> list[Check]:
    """Runs every check.

    @param station The station file.
    @param in_force The zones file the driver's guard reads.
    @param zones_dir The folder of saved bundles (`zones-*.yaml`).
    @param probes The system probes.
    @param now Now (timezone-aware); the system clock by default.
    @return One #Check per item, then information lines (always ok).
    """
    probes, now = probes or Probes(), now or dt.datetime.now(dt.timezone.utc)
    checks: list[Check] = []
    data: dict[str, Any] = {}

    # CAN links
    links = probes.can_links()
    bad = [n for n in LINKS if not ("<FD>" in links.get(n, "") and "ERROR-ACTIVE" in links.get(n, "") and "dbitrate 5000000" in links.get(n, ""))]
    checks.append(Check(not bad, LABEL_CAN, f"not ready: {bad}; run `sudo bash scripts/canup.sh` in openarm-orboh" if bad else ""))
    # programs holding the arms
    running = [p for p in probes.processes() if any(h in p for h in HOLDERS) and "preflight" not in p and "workspace.py" not in p]
    checks.append(Check(not running, LABEL_FREE, f"still running: {running[0][:90]}" if running else ""))
    # station file
    try:
        data = yaml.safe_load(station.read_text()) or {}
    except OSError as exc:
        data = {}
        checks.append(Check(False, LABEL_LOADS, f"cannot read {station}: {exc}"))
    else:
        try:
            from robots import config

            config.load(station)
            checks.append(Check(True, LABEL_LOADS))
        except Exception as exc:
            checks.append(Check(False, LABEL_LOADS, str(exc)[:200]))
    # cameras
    wanted = {str(c["serial"]) for c in data.get("cameras", []) if isinstance(c, dict) and "serial" in c}
    missing = sorted(wanted - probes.cameras())
    checks.append(Check(not missing, LABEL_CAMERAS, f"missing serials: {missing}" if missing else ""))
    holders = probes.camera_holders()
    checks.append(Check(not holders, LABEL_CAMERAS_FREE, f"held by {holders[0][:110]}: close it (a camera opens only once)" if holders else ""))
    # zones in force
    parsed = None
    try:
        parsed = zones.load_zones(in_force)
        age = parsed.age_hours(now)
        checks.append(Check(age <= MAX_ZONE_AGE_H, LABEL_FRESH, f"{age:.1f} hours old: capture and apply again today" if age > MAX_ZONE_AGE_H else f"{age:.1f} hours old"))
    except (OSError, ValueError) as exc:
        checks.append(Check(False, LABEL_FRESH, f"no usable zones at {in_force}: {exc}"))
    bundles = sorted(pathlib.Path(zones_dir).glob("zones-*.yaml"))
    same = bool(bundles) and in_force.exists() and _digest(in_force) == _digest(bundles[-1])
    checks.append(Check(same, LABEL_NEWEST, "" if same else (f"the newest bundle is {bundles[-1].name}: run `workspace.py apply`" if bundles else "no saved bundle")))
    if parsed is not None and parsed.home:
        # station home == bundle home
        off = []
        for side, key in (("left", "follower_l"), ("right", "follower_r")):
            station_home = (data.get(key) or {}).get("home")
            if station_home is None or len(station_home) != 7 or float(np.max(np.abs(np.asarray(station_home) - np.asarray(parsed.home[side])))) > 2e-3:
                off.append(side)
        checks.append(Check(not off, LABEL_HOME, f"differs for: {off}; run `workspace.py apply`" if off else ""))
        # home path
        nopath = [s for s in zb.SIDES if not (parsed.home_path or {}).get(s)
                  or float(np.max(np.abs(np.radians(parsed.home_path[s][-1]) - np.asarray(parsed.home[s])))) > 0.02]
        checks.append(Check(not nopath, LABEL_PATH, f"missing or not ending at home for: {nopath}; record it on the Home card" if nopath else ""))
        # clear of zones
        problems = zb.free_pose_problems(parsed.raw)
        checks.append(Check(not problems, LABEL_CLEAR, problems[0] if problems else ""))
    else:
        for label in (LABEL_HOME, LABEL_PATH, LABEL_CLEAR):
            checks.append(Check(False, label, "the zones in force carry no home"))
    # driver
    driver = str((data.get("custom") or {}).get("driver", ""))
    path_part = driver.rsplit(":", 1)[0]
    ok_driver = driver.endswith(":Arm") and pathlib.Path(path_part).is_file()
    checks.append(Check(ok_driver, LABEL_DRIVER, "" if ok_driver else f"custom.driver is {driver!r}"))
    checks.append(Check(openarm_guard.required(), LABEL_GUARD, "" if openarm_guard.required() else "OPENARM_ZONES_REQUIRED=0 is set in this shell"))
    active = sorted(openarm_lock.active_arms())
    checks.append(Check(True, "Active arms (one-arm lock)", ", ".join(active) if active else "none: the lock file is unreadable"))
    return checks


def format_report(checks: list[Check]) -> str:
    """The report as text, one line per check."""
    lines = [f"[{'PASS' if c.ok else 'FAIL'}] {c.label}" + (f"  ({c.detail})" if c.detail else "") for c in checks]
    bad = sum(not c.ok for c in checks)
    lines.append("" if not bad else f"{bad} check(s) failed: fix them before `dreamcli robot up`.")
    lines.append("All checks passed." if not bad else "")
    return "\n".join(line for line in lines if line)

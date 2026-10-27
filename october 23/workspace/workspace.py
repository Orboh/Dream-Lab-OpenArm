"""The daily safety-area tool: capture, align, verify, build, apply, status, view.  Run with the project's Python:

  cd "october 23/workspace"
  env -u PYTHONPATH -u AMENT_PREFIX_PATH ../../.venv/bin/python workspace.py <command> ...

Stop the station first (`dreamcli robot down`): these commands read the arms while they are limp and must not share the CAN links with it.
Nothing here switches a motor on. See README.md for the order of the daily routine.
"""

import argparse
import datetime as dt
import getpass
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import types

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "driver"))
sys.path.insert(0, str(HERE))

import camerapose as camp  # noqa: E402
import capture as cap  # noqa: E402
import openarm_guard  # noqa: E402
import openarm_lock  # noqa: E402
import openarm_mapping as mapping  # noqa: E402
import openarm_poses as poses  # noqa: E402
import preflight as pf  # noqa: E402
import openarm_zones as zones  # noqa: E402
import zonebuild as zb  # noqa: E402

ZONES_DIR = HERE / "zones"
STATION = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
ALIGN_FILE = ZONES_DIR / "alignment.json"
ALIGN_TOLERANCE_DEG = 3.0
ALIGN_MAX_AGE_H = 12.0


# -- hardware helpers (only used by the commands that read arms) ------------------------------------------------------------

def refuse_if_station_running() -> None:
    """Exits when `dreamcli` still holds the arms (arms.lock names a live process)."""
    lock = pathlib.Path("~/.dreamlabs/arms.lock").expanduser()
    try:
        pid = int(lock.read_text().split()[0])
        os.kill(pid, 0)
    except (OSError, ValueError, IndexError):
        return
    sys.exit(f"the station (pid {pid}) is running: stop it first with `dreamcli robot down`")


def open_sources(kind: str, sides: list[str]):
    """Opens the arms (or the leaders) read-only.

    @return `(arms, read_q, read_tcp)`: the arm objects, a function side to joints (rad), a function side to the grasp point (m).
    """
    import openarm_v2 as drv

    cls = drv.ReadOnlyArm if kind == "follower" else drv.ReadOnlyLeader
    arms = {}
    for side in sides:
        cfg = types.SimpleNamespace(can_channel=side, gripper_closed_deg=math.nan, gripper_open_deg=math.nan)
        arms[side] = cls(cfg, side)
    kins = {side: zones.ArmKinematics(side, logical=True) for side in sides}

    def read_q(side: str) -> np.ndarray:
        """The arm's logical joints (radians): `read()` gives Dream's model angles, mapped back (a no-op for the identity mapping)."""
        return mapping.to_logical(side, arms[side].read().positions_rad)

    def read_tcp(side: str) -> np.ndarray:
        return kins[side].points(read_q(side))["tcp"]

    return arms, read_q, read_tcp


def close_all(arms: dict) -> None:
    for arm in arms.values():
        try:
            arm.close()
        except Exception:
            pass


# -- zones files ------------------------------------------------------------------------------------------------------------

def newest_zones(directory: pathlib.Path = ZONES_DIR) -> pathlib.Path | None:
    """The most recent `zones-*.yaml`, or None."""
    files = sorted(directory.glob("zones-*.yaml"))
    return files[-1] if files else None


def load_raw(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text())


def save_raw(path: pathlib.Path, raw: dict) -> None:
    path.write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=None))


def pick_zones(arg: str | None) -> pathlib.Path:
    path = pathlib.Path(arg) if arg else newest_zones()
    if path is None or not path.exists():
        sys.exit("no zones file yet: run `capture` first")
    return path


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


# -- commands ---------------------------------------------------------------------------------------------------------------

def cmd_capture(args: argparse.Namespace) -> int:
    """Interactive capture with the followers (ground truth) or, on top of a follower capture, with the leaders."""
    refuse_if_station_running()
    sides = ["left", "right"] if args.side == "both" else [args.side]
    base_raw = None
    if args.source == "leader":
        if not args.base:
            sys.exit("a leader capture adds to a follower capture: give --base zones/zones-....yaml")
        check_alignment(sides)
        base_raw = load_raw(pathlib.Path(args.base))
    arms, _, read_tcp = open_sources(args.source, sides)
    try:
        session = cap.CaptureSession(sides[0], read_tcp, ground_truth=args.source == "follower")
        result = cap.run_interactive(session)
    finally:
        close_all(arms)
    print()
    if result == "quit":
        print("not saved")
        return 1
    try:
        if base_raw is None:
            raw = zb.build_zones(session.captures, "follower", args.operator, now())
        else:
            raw = zb.merge_leader(base_raw, session.captures, args.operator, now())
        zones.parse_zones(raw)
    except ValueError as exc:
        print(f"cannot save: {exc}")
        return 1
    ZONES_DIR.mkdir(exist_ok=True)
    path = ZONES_DIR / f"zones-{now():%Y%m%d-%H%M}.yaml"
    save_raw(path, raw)
    print(f"saved {path}\nnext: verify, build, apply (see README.md)")
    return 0


def check_alignment(sides: list[str]) -> None:
    """Exits unless leader and follower were aligned recently within #ALIGN_TOLERANCE_DEG."""
    try:
        record = json.loads(ALIGN_FILE.read_text())
        taken = dt.datetime.fromisoformat(record["at"])
        fresh = (now() - taken).total_seconds() / 3600 <= ALIGN_MAX_AGE_H
        good = all(record["sides"].get(s, {}).get("ok") for s in sides)
    except (OSError, ValueError, KeyError):
        fresh = good = False
    if not (fresh and good):
        sys.exit("the leader arms are not aligned with the followers: put both in the same pose and run `align` first (within 12 h)")


def cmd_align(args: argparse.Namespace) -> int:
    """Leader and follower in the same pose must read the same joints; only then are the leaders trusted for painting."""
    refuse_if_station_running()
    sides = ["left", "right"] if args.side == "both" else [args.side]
    result = {"at": now().isoformat(timespec="seconds"), "tolerance_deg": ALIGN_TOLERANCE_DEG, "sides": {}}
    follower, read_f, _ = open_sources("follower", sides)
    leader, read_l, _ = open_sources("leader", sides)
    try:
        input("Put each leader and its follower in the SAME pose (both limp), then press Enter... ")
        for side in sides:
            diff = np.degrees(np.abs(read_f(side) - read_l(side)))
            ok = bool(diff.max() <= ALIGN_TOLERANCE_DEG)
            result["sides"][side] = {"ok": ok, "max_deg": round(float(diff.max()), 1), "per_joint_deg": [round(float(d), 1) for d in diff]}
            print(f"{side}: per-joint difference (deg) {np.round(diff, 1).tolist()} -> {'OK' if ok else 'NOT aligned'}")
    finally:
        close_all(follower)
        close_all(leader)
    ZONES_DIR.mkdir(exist_ok=True)
    ALIGN_FILE.write_text(json.dumps(result, indent=1))
    return 0 if all(v["ok"] for v in result["sides"].values()) else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """The daily check: touch the stored reference points again; the zones are only trusted if they agree within 1.5 cm."""
    refuse_if_station_running()
    path = pick_zones(args.zones)
    raw = load_raw(path)
    sides = [s for s in zb.SIDES if raw["references"].get(s)]
    if not sides:
        sys.exit("this zones file has no reference points: capture with `ref` points")
    arms, _, read_tcp = open_sources("follower", sides)
    worst, ok = 0.0, True
    try:
        for side in sides:
            measured = []
            for i, point in enumerate(raw["references"][side], start=1):
                input(f"{side} arm: touch reference point {i} (stored at {point}) with the gripper tip, then press Enter... ")
                measured.append(tuple(float(v) for v in read_tcp(side)))
            report = zb.compare_references([tuple(p) for p in raw["references"][side]], measured)
            print(f"{side}: distances (cm) {np.round(np.array(report['distances_m']) * 100, 1).tolist()} -> {'OK' if report['ok'] else 'MOVED'}")
            worst, ok = max(worst, report["worst_m"]), ok and report["ok"]
    finally:
        close_all(arms)
    raw["verify"] = {"status": "ok" if ok else "failed", "worst_m": round(worst, 4), "checked_at": now().isoformat(timespec="seconds")}
    save_raw(path, raw)
    print("verified" if ok else "FAILED: the table or the robot moved; capture the zones again")
    return 0 if ok else 1


def station_homes(station: pathlib.Path) -> dict[str, tuple[float, float]]:
    """Each arm's home hand position seen from above, from the station file's home joints (the ready pose if absent)."""
    data = yaml.safe_load(station.read_text()) if station.exists() else {}
    out = {}
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        home = (data.get(key) or {}).get("home") or np.radians(poses.READY_DEG[side])
        out[side] = tuple(zones.ArmKinematics(side, logical=True).points(np.asarray(home, dtype=float))["tcp"][:2])
    return out


def check_free_poses(raw: dict) -> list[str]:
    """Violations of the rest pose, the ready pose, the bundle's home pose and the path between them against the zones."""
    return zb.free_pose_problems(raw)


def home_hands(raw: dict, station: pathlib.Path) -> dict[str, tuple[float, float]]:
    """Each arm's home hand position seen from above: from the bundle's home pose, else from the station file's."""
    if raw.get("home"):
        return {side: tuple(zones.ArmKinematics(side, logical=True).points(np.asarray(raw["home"][side], dtype=float))["tcp"][:2]) for side in zb.SIDES}
    return station_homes(station)


def cmd_build(args: argparse.Namespace) -> int:
    """Checks the zones against the rest and ready poses and prints what apply would write."""
    path = pick_zones(args.zones)
    raw = load_raw(path)
    zones.parse_zones(raw)
    problems = check_free_poses(raw)
    fragment = zb.station_fragment(raw, home_hands(raw, pathlib.Path(args.station).expanduser()))
    print(f"zones: {path.name}  (source {raw['source']}, verify {raw['verify']['status']})")
    for side, moved in zb.home_adjustments(raw).items():
        print(f"note: the {side} home would be outside Dream's joint range; the station file moves " + ", ".join(f"joint {j} by {d:+g} deg" for j, d in moved) + " to keep it 0.5 deg inside")
    for side in zb.SIDES:
        f = fragment[side]
        area = f"reach {len(f['reach'])} cells" if "reach" in f else "no reach and no region (danger-only: the hand can go anywhere outside the danger zones)"
        print(f"{side}: {area}, keep_out {[z.get('name', z['kind']) for z in raw['arms'][side]['keep_out']]}")
    for side in zb.SIDES:  # a warning, not a block: it matters for an arm that is active (preflight checks exactly that)
        for text in zb.taught_path_problems(raw, side, limit=1):
            print(f"WARNING: {text}")
    if problems:
        print("\nTHE ZONES BLOCK THE REST/READY POSES OR THE PATH BETWEEN THEM:")
        print("\n".join("  " + p for p in problems[:12]))
        return 1
    print("\nrest, ready and the path between them are clear of every zone")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Writes the zones into station.yaml (with a backup) and makes them the zones in force for the driver's guard."""
    path = pick_zones(args.zones)
    raw = load_raw(path)
    station = pathlib.Path(args.station).expanduser()
    danger_only = raw.get("mode") == "danger_only"
    if raw["verify"]["status"] != "ok" and not args.no_verify and not danger_only:
        sys.exit("these zones are not verified today: run `verify` (or --no-verify for a bench test)")
    if danger_only and raw["verify"]["status"] != "ok":  # a danger-only bundle has no reference points to touch: its table box comes from the tape measurements
        print("note: danger-only zones are applied without a reference-point check; the table box is from tape measurements (front edge, top height)")
    if cmd_build(args) != 0:
        sys.exit("not applied")
    fragment = zb.station_fragment(raw, home_hands(raw, station))
    backup = zb.apply_to_station(station, fragment, tuning=None if args.no_tuning else zb.GENTLE_STATION)
    try:
        from robots import config

        config.load(station)
    except Exception as exc:
        shutil.copy2(backup, station)
        sys.exit(f"Dream refused the new station file ({exc}); the backup was restored")
    target = openarm_guard.zones_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():  # the zones that were in force stay available
        shutil.copy2(target, target.with_name(f"{target.name}.{dt.datetime.now():%Y%m%d-%H%M%S}.bak"))
    shutil.copy2(path, target)
    record = mapping.record_path()
    if record.exists():
        shutil.copy2(record, record.with_name(f"{record.name}.{dt.datetime.now():%Y%m%d-%H%M%S}.bak"))
    mapping.write_record(mapping.active())  # the station file's home is in Dream's model angles of THIS mapping: remember it, whatever shell starts the station later
    print(f"applied: {station} (backup {backup.name}); zones in force: {target}; joint mapping recorded: {mapping.active()} ({record})")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Shows the zones in force, their age, whether they are verified, and which arms may move."""
    target = openarm_guard.zones_path()
    print(f"guard: {'ON' if openarm_guard.required() else 'OFF (OPENARM_ZONES_REQUIRED=0)'}; zones in force: {target}")
    if target.exists():
        parsed = zones.load_zones(target)
        raw = parsed.raw
        print(f"  captured {raw['captured_at']} ({parsed.age_hours():.1f} h ago, limit {os.environ.get('OPENARM_ZONES_MAX_AGE_H', '24')} h), "
              f"source {raw['source']}, verify {raw['verify']['status']}")
        for side in zb.SIDES:
            print(f"  {side}: keep_out {[z['name'] for z in parsed.arms[side].keep_out]}")
    else:
        print("  none: the arms will not move until `capture`, `verify` and `apply` have been run today")
    newest = newest_zones()
    print(f"newest capture: {newest.name if newest else 'none'}")
    active = sorted(openarm_lock.active_arms())
    print("active arms:", ", ".join(active) if active else "none (lock file unreadable)")
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    """Opens the 3D viewer; with --live the skeleton follows the limp arms (or leaders) as you move them."""
    import viewer

    import camerapanel

    raw = load_raw(pick_zones(args.zones))
    cameras = camerapanel.cameras_from_station(STATION) if args.cameras else None
    if cameras is not None:
        refuse_if_station_running()
    if args.live == "none":
        viewer.serve(raw, args.port, cameras=cameras)
        return 0
    refuse_if_station_running()
    arms, read_q, _ = open_sources(args.live, ["left", "right"])
    try:
        viewer.serve(raw, args.port, read_q=read_q, stop=threading.Event(), cameras=cameras)
    finally:
        close_all(arms)
    return 0


def apply_table_args(state, args: argparse.Namespace) -> bool:
    """Puts the table measured with a tape (`--table-front-cm`, `--table-top-cm`, with depth and width) into the editor state. The table stays "not placed" on the Save card until you touch it on the page.

    @return True when a table was set from the arguments.
    """
    import editor

    if args.table_front_cm is None and args.table_top_cm is None:
        return False
    if args.table_front_cm is None or args.table_top_cm is None:
        sys.exit("give both --table-front-cm and --table-top-cm")
    state.table = editor.table_from_measurements(args.table_front_cm, args.table_top_cm, args.table_depth_cm, args.table_width_cm, args.table_centre_y_cm,
                                           lower_cm=getattr(args, "table_lower_cm", 0.0))
    print(f"table from the tape: front edge {args.table_front_cm:g} cm in front of the centre line, top {args.table_top_cm:g} cm above the base plate, "
          f"{args.table_depth_cm:g} deep x {args.table_width_cm:g} wide (check depth and width on the page); danger box top lowered by {getattr(args, 'table_lower_cm', 0.0):g} cm")
    if state.table.lower_warning():
        print("WARNING:", state.table.lower_warning())
    return True


def cmd_design(args: argparse.Namespace) -> int:
    """The visual editor: the real OpenArm model, the table and danger boxes you drag, Save and Export. Starts from the newest bundle unless --new."""
    import camerapanel
    import designer
    import editor

    newest = None if args.new else (pathlib.Path(args.zones) if args.zones else newest_zones())
    state = editor.EditorState.from_bundle(load_raw(newest)) if newest else editor.EditorState.default()
    print(f"starting from {newest.name if newest else 'the defaults (table top 30 cm above the hanging hand)'}")
    apply_table_args(state, args)
    state.danger_only = bool(args.danger_only) or state.danger_only
    tolerance = getattr(args, "park_tolerance_deg", None)
    state.park_tolerance_deg = float(tolerance) if tolerance is not None else (8.0 if state.danger_only else 0.0)
    cameras = camerapanel.cameras_from_station(STATION) if args.cameras else None
    arms, read_q, observers, observer_status = {}, None, {}, None
    if args.live != "none" or cameras is not None or args.observe:
        refuse_if_station_running()
    if args.live != "none" and args.observe:
        sys.exit("choose one: --live reads the limp arms (it sends queries); --observe only listens while the teleop runs")
    if args.live != "none":
        arms, read_q, _ = open_sources(args.live, ["left", "right"])
    if args.observe:
        import canobserver

        observers = {side: canobserver.PassiveArm(side) for side in ("left", "right")}
        for obs in observers.values():
            obs.start()
        read_q = lambda side: observers[side].q_rad()  # noqa: E731
        observer_status = lambda: "  \n".join(o.status() for o in observers.values())  # noqa: E731
    try:
        import collect

        collect_env = pathlib.Path(args.collect_env).expanduser() if args.collect_env else collect.DEFAULT_ENV
        if not collect_env.is_absolute():
            collect_env = collect.REPO / collect_env  # `config/...` as typed inside openarm-orboh
        designer.serve_designer(state, ZONES_DIR, args.operator, port=args.port, detail=args.detail, read_q=read_q, cameras=cameras,
                                observer_status=observer_status, collect_env=collect_env)
    finally:
        close_all(arms)
        for obs in observers.values():
            obs.stop()
    return 0


def cmd_preflight(args: argparse.Namespace, probes=None, **_: object) -> int:
    """The checks to pass before every session: CAN links, nobody else on the arms, cameras, zones fresh and consistent with the station file, the Arm driver, the guard."""
    checks = pf.run_preflight(pathlib.Path(args.station).expanduser(), pathlib.Path(args.in_force).expanduser(), pathlib.Path(args.zones_dir), probes=probes)
    print(pf.format_report(checks))
    return 0 if all(c.ok for c in checks) else 1


def cmd_start(args: argparse.Namespace, probes=None, runner=None, **_: object) -> int:
    """Runs the checks and, only if all pass, starts the station (`dreamcli robot up`); --dry only prints what it would run."""
    if cmd_preflight(args, probes=probes) != 0:
        return 1
    command = [str(pathlib.Path("~/dream-env/bin/dreamcli").expanduser()), "robot", "up"]
    if args.dry:
        print("would run:", " ".join(command))
        return 0
    print("starting:", " ".join(command))
    return (runner or (lambda cmd: subprocess.run(cmd, check=False).returncode))(command)


def cmd_camera_pose(args: argparse.Namespace, probes=None, opener=None, **_: object) -> int:
    """Raises the chosen arm(s) to the camera-view pose (the taught home), holds it for you to look at the camera views, parks and switches off. The arms MOVE.

    Runs the same checks as `start` first (station stopped, zones fresh, ...), so it refuses while the station, the teleop or an editor holds the arms or cameras.
    """
    if cmd_preflight(args, probes=probes) != 0:
        return 1
    sides = ["left", "right"] if args.arm == "both" else [args.arm]
    if args.hold > 0:
        hold = lambda: time.sleep(args.hold)  # noqa: E731
    else:
        hold = lambda: input("Look at the camera views now. Press Enter to park the arm and switch it off... ")  # noqa: E731
    if opener is None:
        import openarm_v2 as drv

        opener = drv.Arm
    return camp.run(pathlib.Path(args.station).expanduser(), sides, hold, opener)


def main(argv: list[str] | None = None, probes=None, runner=None, opener=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture", help="paint the work area, touch the table, mark dangers (arms limp)")
    p.add_argument("--source", choices=["follower", "leader"], default="follower")
    p.add_argument("--side", choices=["left", "right", "both"], default="both")
    p.add_argument("--base", help="leader capture: the follower zones file to add to")
    p.add_argument("--operator", default=getpass.getuser())
    p.set_defaults(func=cmd_capture)
    p = sub.add_parser("align", help="check that each leader reads like its follower in the same pose")
    p.add_argument("--side", choices=["left", "right", "both"], default="both")
    p.set_defaults(func=cmd_align)
    for name, func, doc in (("verify", cmd_verify, "touch the reference points again"), ("build", cmd_build, "check the zones and show what apply writes"),
                            ("apply", cmd_apply, "write the zones into station.yaml and make them current"), ("view", cmd_view, "3D viewer in the browser")):
        p = sub.add_parser(name, help=doc)
        p.add_argument("--zones", help="a zones file (default: the newest in zones/)")
        if name in ("build", "apply"):
            p.add_argument("--station", default=str(STATION))
        if name == "apply":
            p.add_argument("--no-verify", action="store_true")
            p.add_argument("--no-tuning", action="store_true", help="do not write the gentle ramp and speed settings (ramp_s 3, ramp_max_rad_s 0.3, max_ee_speed_m_s 0.6, Dream's default) and the Arm driver")
        if name == "view":
            p.add_argument("--port", type=int, default=8080)
            p.add_argument("--live", choices=["none", "follower", "leader"], default="none")
            p.add_argument("--cameras", action="store_true", help="offer the station file's RealSense cameras as switchable feeds (station must be stopped)")
        p.set_defaults(func=func)
    p = sub.add_parser("design", help="visual editor on the real OpenArm model (Save writes the bundle and the layout picture)")
    p.add_argument("--zones", help="start from this bundle (default: the newest)")
    p.add_argument("--new", action="store_true", help="start from the defaults, not from a saved bundle")
    p.add_argument("--live", choices=["none", "follower", "leader"], default="none", help="read the limp arms (or leaders) to pose the model and use their pose")
    p.add_argument("--observe", action="store_true", help="TEACH: only LISTEN to the follower's CAN frames while your teleop runs (sends nothing)")
    p.add_argument("--cameras", action="store_true", help="offer the station file's RealSense cameras as switchable feeds")
    p.add_argument("--detail", choices=["visual", "collision"], default="visual")
    p.add_argument("--danger-only", action="store_true", help="the bundle names only the danger zones (the table and any box you add): no painted work area, no allowed region, so the model is told nothing about the task")
    p.add_argument("--park-tolerance-deg", type=float, default=None, help="simplify the saved rest-to-home path to this tolerance (degrees) when that keeps it clear of the zones: a shorter raise and park (default 8 with --danger-only, else 0: as recorded)")
    p.add_argument("--table-front-cm", type=float, help="the table's near edge, cm in front of the robot's centre line (tape measure)")
    p.add_argument("--table-top-cm", type=float, help="the table top, cm above the floor-level base plate the arms stand on (the shoulders are 69.8 cm above it)")
    p.add_argument("--table-lower-cm", type=float, default=0.0, help="lower the table's danger box top by this many cm (0 keeps it at the real surface; up to 2 keeps the safety balls out of the table; max 4). A NEGATIVE value raises the top: an extra safety margin over the table (down to -15), e.g. --table-lower-cm=-5")
    p.add_argument("--table-depth-cm", type=float, default=45.0, help="how far the table runs away from the robot (default 45: a placeholder until measured)")
    p.add_argument("--table-width-cm", type=float, default=60.0, help="the table's width (default 60: a placeholder until measured)")
    p.add_argument("--table-centre-y-cm", type=float, default=0.0, help="the table's centre, cm to the left of the centre line")
    p.add_argument("--collect-env", default=os.environ.get("ENV_FILE", ""), help="settings file of the Collect tab (default: $ENV_FILE, else the openarm-orboh Kitakyushu Takeei file)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--operator", default=getpass.getuser())
    p.set_defaults(func=cmd_design)
    for name, func, doc in (("preflight", cmd_preflight, "the checks to pass before every session"), ("start", cmd_start, "preflight, then start the station (dreamcli robot up)")):
        p = sub.add_parser(name, help=doc)
        p.add_argument("--station", default=str(STATION))
        p.add_argument("--in-force", default=str(openarm_guard.zones_path()), help="the zones file the driver's guard reads")
        p.add_argument("--zones-dir", default=str(ZONES_DIR))
        if name == "start":
            p.add_argument("--dry", action="store_true", help="run the checks and print the command, do not start")
        p.set_defaults(func=func)
    p = sub.add_parser("camera-pose", help="raise the arm(s) to the camera-view pose (the taught home), hold, park and switch off; the arms MOVE")
    p.add_argument("--arm", choices=["left", "right", "both"], default="both")
    p.add_argument("--hold", type=float, default=0.0, help="seconds to hold for the camera views (0: until you press Enter)")
    p.add_argument("--station", default=str(STATION))
    p.add_argument("--in-force", default=str(openarm_guard.zones_path()), help="the zones file the driver's guard reads")
    p.add_argument("--zones-dir", default=str(ZONES_DIR))
    p.set_defaults(func=cmd_camera_pose)
    p = sub.add_parser("status", help="zones in force, age, active arms")
    p.set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    if args.func in (cmd_preflight, cmd_start):
        return args.func(args, probes=probes, runner=runner)
    if args.func is cmd_camera_pose:
        return args.func(args, probes=probes, opener=opener)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Touch calibration of the table: the user lays a gripper flat on the table, this finds the table top from the lowest points of the real OpenArm meshes.

Method of log-025: read the follower's logical joints while it rests on the table (steady), pose the official URDF with them, and take the lowest mesh
point of each part of that arm. Several parts at one height (within #FLAT_CM) mean the hand lies flat on one level surface: that height is the table top.
Read-only: it LISTENS to the follower's CAN frames while the teleop runs (`canobserver.PassiveArm`), it sends nothing. Each reading is appended to a
JSON file in `log/` so the log can quote it.

    $PY touchcal.py both --spot middle        # both hands on the table now
    $PY touchcal.py right --spot far-right
"""

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
from collections.abc import Callable

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "driver"))
sys.path.insert(0, str(HERE))
LOG_DIR = HERE.parent / "log"
STATION = pathlib.Path("~/.dreamlabs/station.yaml").expanduser()
SHOULDER_Z_M = 0.698        # each arm's base joint above the base plate (editor.BASE_OFFSET z)
FLAT_CM = 1.0               # parts of the hand this close to the lowest point count as lying on the same surface
STEADY_DEG = 0.1            # largest joint spread over the readings for a pose to count as resting
JOINTS = 7
FINGER_OPEN_M = 0.044       # the URDF finger joint's travel, fully open (`openarm_<side>_finger_joint1`; the other finger mirrors it)


def load_urdf():
    """The official OpenArm URDF with its real meshes (the one the website draws)."""
    import yourdfpy

    import designer

    def handler(fname: str) -> str:
        return str(designer.MESH_ROOT / fname.split("openarm_description/", 1)[-1])

    return yourdfpy.URDF.load(str(designer.URDF), filename_handler=handler, build_scene_graph=True, load_meshes=True)


def finger_from_gripper(gripper_deg: float, closed_deg: float, open_deg: float) -> float:
    """The URDF finger travel (m) for a gripper motor angle: linear between the station file's closed and open angles, clipped to the travel."""
    return float(np.clip((gripper_deg - closed_deg) / (open_deg - closed_deg), 0.0, 1.0) * FINGER_OPEN_M)


def lowest_points(urdf, side: str, q_rad, finger_m: float = 0.0) -> list[tuple[str, float]]:
    """The lowest mesh point of every part of one arm at a pose, lowest first.

    @param urdf The loaded URDF. @param side `"left"` or `"right"`. @param q_rad Seven logical joint angles, radians.
    @param finger_m How far the fingers are open (0 closed, #FINGER_OPEN_M fully open): open fingers stick out sideways, so with the wrist rolled one fingertip is lower.
    @return `(part, z)` with z in the arm's base frame, cm (the shoulder is 0, the base plate is -69.8).
    """
    cfg = {name: 0.0 for name in urdf.actuated_joint_names}
    for i in range(JOINTS):
        cfg[f"openarm_{side}_joint{i + 1}"] = float(q_rad[i])
    cfg[f"openarm_{side}_finger_joint1"] = float(finger_m)
    urdf.update_cfg(cfg)
    scene = urdf.scene
    low: dict[str, float] = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        part = scene.graph.transforms.parents.get(node, node)
        if not part.startswith(f"openarm_{side}_"):
            continue
        z = float((scene.geometry[geometry].vertices @ transform[2, :3] + transform[2, 3]).min())
        low[part] = min(low.get(part, np.inf), z)
    return sorted(((part, round((z - SHOULDER_Z_M) * 100.0, 2)) for part, z in low.items()), key=lambda item: item[1])


def table_top(points: list[tuple[str, float]]) -> tuple[float, list[str]]:
    """The table top (arm frame, cm) and the parts that rest on it (within #FLAT_CM of the lowest point)."""
    top = points[0][1]
    return top, [part for part, z in points if z - top <= FLAT_CM]


def sample(read_q: Callable[[], np.ndarray], n: int = 40, period: float = 0.05) -> tuple[np.ndarray, float]:
    """Reads the joints `n` times. @return The mean pose (radians) and the largest joint spread (degrees)."""
    reads = []
    for _ in range(n):
        reads.append(np.asarray(read_q(), dtype=float))
        time.sleep(period)
    reads = np.array(reads)
    return reads.mean(axis=0), float(np.degrees(reads.max(axis=0) - reads.min(axis=0)).max())


def report(side: str, q_rad, spread_deg: float, points: list[tuple[str, float]], finger_m: float = 0.0, gripper_deg: float | None = None) -> dict:
    """One arm's reading as a record for the log file."""
    top, resting = table_top(points)
    return {"side": side, "q_rad": [float(v) for v in q_rad], "gripper_deg": gripper_deg, "finger_m": round(finger_m, 4),
            "spread_deg": round(spread_deg, 3), "steady": spread_deg <= STEADY_DEG,
            "table_top_arm_cm": top, "table_top_above_plate_cm": round(top + SHOULDER_Z_M * 100.0, 2), "resting_parts": resting,
            "lowest": [[part, z] for part, z in points[:6]]}


def main(argv: list[str] | None = None, readers: dict[str, Callable[[], np.ndarray]] | None = None,
         grippers: dict[str, Callable[[], float]] | None = None) -> int:
    """@param readers Side to its joints now (radians); the follower's CAN frames by default. @param grippers Side to its gripper angle now (logical degrees), or None."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("side", choices=["left", "right", "both"])
    parser.add_argument("--spot", default="", help="a name for the place touched (goes into the record)")
    parser.add_argument("--gripper", choices=["measured", "open", "closed"], default="measured", help="finger opening for the model (measured: from the gripper motor)")
    parser.add_argument("--out", default=str(LOG_DIR / f"touch-calibration-{dt.date.today():%Y%m%d}.json"))
    args = parser.parse_args(argv)
    sides = ["left", "right"] if args.side == "both" else [args.side]
    observers = []
    if readers is None:
        import canobserver

        readers, grippers = {}, {}
        for side in sides:
            obs = canobserver.PassiveArm(side)
            obs.start()
            observers.append(obs)
            readers[side] = obs.q_rad
            grippers[side] = lambda obs=obs: float(obs.logical_deg()["gripper"])
        time.sleep(0.5)  # let the first frames arrive
    urdf = load_urdf()
    records = []
    try:
        for side in sides:
            try:
                q, spread = sample(readers[side])
            except Exception as exc:
                print(f"{side}: no joint frames ({exc}): is the teleop running?")
                return 1
            gripper_deg = None
            if args.gripper == "measured" and grippers and side in grippers:
                import camerapose as camp

                cfg = camp.arm_config(STATION, side)
                gripper_deg = grippers[side]()
                finger = finger_from_gripper(gripper_deg, cfg.gripper_closed_deg, cfg.gripper_open_deg)
            else:
                finger = FINGER_OPEN_M if args.gripper == "open" else 0.0
            rec = report(side, q, spread, lowest_points(urdf, side, q, finger), finger, gripper_deg)
            rec["spot"] = args.spot
            rec["time"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            records.append(rec)
            print(f"{side}: table top {rec['table_top_arm_cm']:.1f} cm in the arm frame = {rec['table_top_above_plate_cm']:.1f} cm above the base plate; "
                  f"fingers open {rec['finger_m'] * 100:.1f} cm; resting parts {', '.join(p.replace(f'openarm_{side}_', '') for p in rec['resting_parts'])}; "
                  f"spread {spread:.3f} deg {'(steady)' if rec['steady'] else '(NOT steady: hold still and repeat)'}")
    finally:
        for obs in observers:
            obs.stop()
    out = pathlib.Path(args.out)
    old = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(old + records, indent=1))
    print(f"saved to {out}")
    return 0 if all(r["steady"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())

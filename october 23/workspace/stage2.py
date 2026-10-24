"""Stage 2 of goal.md: do the models agree with each other, and which one agrees with the real arm?

Three things use joint angles to say where the hand is, and they have not been shown to agree:
- the driver, Dream's IK and the zones/guard use Dream's own `openarm2_faas` model with the driver's mapping (identity, MODEL_SIGN = 1, MODEL_ZERO_RAD = 0);
- the editor's 3D robot and the hand-pitch readout use the v10 URDF with the fork's logical angles;
- the real arm.
`models` and `mapping` are offline; `plan`, `record` and `report` are the physical test (the arm is moved by hand and measured with a tape: nothing here moves it).
Run from this folder: `python stage2.py models|mapping|plan|record|report`.
"""

import argparse
import itertools
import json
import math
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
V10_URDF = HERE.parent / "urdf" / "openarm_bimanual.urdf"
DREAM_URDF = "~/dream-env/lib/python3.12/site-packages/dream_ik/assets/openarm2_faas_{side}.urdf"
BASE_OFFSET = {"left": (0.0, 0.0305, 0.698), "right": (0.0, -0.0305, 0.698)}  # an arm's base in the robot body frame, metres
TCP_IN_EE = {"left": (-0.0016, -0.0001, -0.159), "right": (-0.0016, 0.0001, -0.159)}  # Dream's grasp point in its ee_base_link frame (dream_ik/specs.py)
TOLERANCE_CM = 4.0  # a model must reproduce a measured hand displacement to this. The two models put the grasp point 2 cm apart on the tool, which turns into up to 4 cm when the hand turns; the poses differ by 20 cm.
RECORD = HERE.parent / "log" / "stage2-physical.json"

# Named joint sets in the fork's logical degrees (J1..J7): the poses the project actually uses.
POSES = {
    "hanging (log-010, left)": ("left", [-0.5, -0.1, -4.3, 2.1, -91.1, 0.4, 5.1]),
    "rest (left)": ("left", [-0.5, -0.8, -4.3, 4.0, 2.1, -0.5, 6.4]),
    "start v2 (left)": ("left", [64.42, -3.62, -1.23, 95.91, -2.48, 19.01, -68.77]),
    "start wood-l1 (left)": ("left", [64.42, -3.62, -1.23, 95.91, -2.48, 19.01, 21.6]),
    "start v5 aligned (left)": ("left", [59.04, -2.33, -4.82, 90.23, -1.21, -1.86, -75.24]),
    "after hold released (left)": ("left", [59.04, -2.33, -4.82, 90.23, -1.21, -1.86, -14.7]),
    "start v2 (right)": ("right", [-59.61, 1.23, 7.53, 87.17, 2.62, -1.23, 66.72]),
    "start wood-l1 (right)": ("right", [-59.61, 1.23, 7.53, 87.17, 2.62, -1.23, -15.0]),
    "start v5 aligned (right)": ("right", [-58.06, 0.38, 4.80, 101.24, 2.44, 1.68, 58.13]),
    "after hold released (right)": ("right", [-58.06, 0.38, 4.80, 101.24, 2.44, 1.68, 7.2]),
}
LIMITS_DEG = {"left": [(-200, 80), (-190, 10), (-90, 90), (0, 140), (-90, 90), (-45, 45), (-90, 90)]}


class Models:
    """Forward kinematics of the same joint values in Dream's model and in the v10 URDF, both in the arm's body-frame orientation (x forward, y left, z up)."""

    def __init__(self) -> None:
        import yourdfpy

        load = lambda path: yourdfpy.URDF.load(str(pathlib.Path(path).expanduser()), build_scene_graph=True, load_meshes=False, build_collision_scene_graph=False)  # noqa: E731
        self._dream = {side: load(DREAM_URDF.format(side=side)) for side in BASE_OFFSET}
        self._v10 = load(V10_URDF)

    def dream(self, side: str, q_rad, offset=None) -> tuple[np.ndarray, np.ndarray]:
        """Dream's grasp point (m, in its base frame) and tool axes (3x3) for `q_rad`, optionally with a per-joint `offset` (rad) added first."""
        q = np.asarray(q_rad, dtype=float) + (0.0 if offset is None else np.asarray(offset, dtype=float))
        model = self._dream[side]
        model.update_cfg({f"openarm_{side}_joint{i + 1}": float(q[i]) for i in range(7)})
        T = model.get_transform(f"openarm_{side}_ee_base_link", f"openarm_{side}_base_link")
        return (T @ np.array([*TCP_IN_EE[side], 1.0]))[:3], T[:3, :3]

    def v10(self, side: str, q_rad) -> tuple[np.ndarray, np.ndarray]:
        """The v10 URDF's grasp point (m, base-shifted world) and tool axes for `q_rad`."""
        self._v10.update_cfg({f"openarm_{side}_joint{i + 1}": float(v) for i, v in enumerate(q_rad)})
        T = self._v10.get_transform(f"openarm_{side}_hand_tcp")
        return T[:3, 3] - np.array(BASE_OFFSET[side]), T[:3, :3]


def discrepancies(models: Models, side: str) -> list[dict]:
    """Joint by joint, how Dream's model differs from the official v10 URDF at the zero (hanging) pose: axis in the body frame, joint position (cm below/forward of the base) and limits (deg)."""
    out = []
    zero = {f"openarm_{side}_joint{i + 1}": 0.0 for i in range(7)}
    dream, v10 = models._dream[side], models._v10
    dream.update_cfg(zero)
    v10.update_cfg(zero)
    shift = np.array(BASE_OFFSET[side])
    for i in range(7):
        name = f"openarm_{side}_joint{i + 1}"
        row = {"joint": i + 1}
        for label, model, offset in (("dream", dream, np.zeros(3)), ("v10", v10, shift)):
            j = model.joint_map[name]
            T = model.get_transform(j.child)
            row[label] = {"axis": (T[:3, :3] @ np.array(j.axis)).round(2).tolist(), "pos_cm": ((T[:3, 3] - offset) * 100).round(1).tolist(),
                          "limit_deg": [round(math.degrees(j.limit.lower)), round(math.degrees(j.limit.upper))] if j.limit is not None and j.limit.lower is not None else None}
        row["axis_differs"] = bool(np.abs(np.array(row["dream"]["axis"]) - np.array(row["v10"]["axis"])).max() > 0.05 and np.abs(np.array(row["dream"]["axis"]) + np.array(row["v10"]["axis"])).max() > 0.05)
        row["pos_gap_cm"] = round(float(np.linalg.norm(np.array(row["dream"]["pos_cm"]) - np.array(row["v10"]["pos_cm"]))), 1)
        row["limits_differ"] = row["dream"]["limit_deg"] != row["v10"]["limit_deg"]
        out.append(row)
    return out


def pitch_deg(R: np.ndarray, axis: int = 2) -> float:
    """The angle of a tool axis (a column of the 3x3 matrix) above the horizontal; the approach direction is column 2 of the v10 frame and column 2 of the negated Dream frame."""
    return math.degrees(math.asin(float(np.clip(R[2, axis], -1, 1))))


def compare(models: Models, side: str, q_deg) -> dict:
    """What Dream (identity mapping) and the v10 URDF each say for one pose, and how far apart they are."""
    q = np.radians(q_deg)
    pd, Rd = models.dream(side, q)
    pv, Rv = models.v10(side, q)
    return {"dream_tcp_cm": (pd * 100).round(1).tolist(), "v10_tcp_cm": (pv * 100).round(1).tolist(), "gap_cm": round(float(np.linalg.norm(pd - pv)) * 100, 1),
            "dream_pitch_deg": round(pitch_deg(-Rd, 2), 1), "v10_pitch_deg": round(pitch_deg(Rv, 2), 1)}  # Dream's tool z points up when the arm hangs: its approach axis is -z


def random_poses(n: int = 40, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.radians([rng.uniform(a, b) for a, b in LIMITS_DEG["left"]]) for _ in range(n)]


def mapping(models: Models, side: str, joints=(2, 4, 5, 6), top: int = 6, n: int = 30) -> list[dict]:
    """Which per-joint offset (and sign for the wrist) makes Dream's model reproduce the v10 URDF's hand, over random poses.

    A fixed tool offset is absorbed from the zero pose (the models disagree about the tool length by about 2 cm). Residual: mean grasp-point distance, cm.
    """
    zero = np.zeros(7)
    p0d, R0d = models.dream(side, zero)
    p0v, R0v = models.v10(side, zero)
    tool = R0v.T @ (p0d - p0v)  # grasp point offset to add in the v10 tool frame so the two agree at the zero pose
    poses = random_poses(n)
    options = [0.0, math.pi / 2, -math.pi / 2, math.pi]
    rows = []
    for offsets in itertools.product(options, repeat=len(joints)):
        for signs in itertools.product([1, -1], repeat=len([j for j in joints if j >= 5])):
            sign, offset = np.ones(7), np.zeros(7)
            for j, o in zip(joints, offsets):
                offset[j] = o
            for j, s in zip([j for j in joints if j >= 5], signs):
                sign[j] = s
            err = []
            for q in poses:
                pd, _ = models.dream(side, sign * q + offset)
                pv, Rv = models.v10(side, q)
                err.append(np.linalg.norm(pd - (pv + Rv @ tool)))
            rows.append({"joints": [j + 1 for j in joints], "offsets_deg": [round(math.degrees(o)) for o in offsets], "signs": list(signs), "mean_cm": round(float(np.mean(err)) * 100, 2)})
    rows.sort(key=lambda r: r["mean_cm"])
    identity = next(r for r in rows if not any(r["offsets_deg"]) and all(s == 1 for s in r["signs"]))
    return [{"identity": identity}] + rows[:top]


def displacement(models: Models, side: str, q_a, q_b, hypothesis: str, offset=None) -> np.ndarray:
    """The grasp point's move (m) from joints `q_a` to `q_b` (radians) under a hypothesis: `identity` (Dream, as the driver sends it), `v10` (the editor's URDF)."""
    f = (lambda q: models.dream(side, q, offset)[0]) if hypothesis == "identity" else (lambda q: models.v10(side, q)[0])
    return f(np.asarray(q_b, dtype=float)) - f(np.asarray(q_a, dtype=float))


def judge(models: Models, side: str, entries: list[dict]) -> dict:
    """Compares the measured displacement of the grasp point (relative to the first pose) with each hypothesis, in cm.

    @param entries `{"q_deg": [7 floats], "tcp_cm": [x forward, y left, z up] as measured}`, the first one is the reference.
    @return RMS error per hypothesis, the per-pose errors and a verdict (a hypothesis within TOLERANCE_CM, or none).
    """
    ref = entries[0]
    out = {"identity": [], "v10": []}
    for e in entries[1:]:
        measured = np.asarray(e["tcp_cm"], dtype=float) - np.asarray(ref["tcp_cm"], dtype=float)
        for h in out:
            predicted = displacement(models, side, np.radians(ref["q_deg"]), np.radians(e["q_deg"]), h) * 100
            out[h].append(round(float(np.linalg.norm(predicted - measured)), 1))
    rms = {h: round(float(math.sqrt(statistics.mean(v * v for v in errs))), 1) if errs else None for h, errs in out.items()}
    good = [h for h, v in rms.items() if v is not None and v <= TOLERANCE_CM]
    verdict = good[0] if len(good) == 1 else ("both within tolerance: poses do not discriminate" if good else "neither within tolerance: measurement or a third cause")
    return {"rms_cm": rms, "per_pose_cm": out, "verdict": verdict}


def plan_text(models: Models, side: str) -> str:
    """Candidate poses for the physical test with how far apart the two models put the hand (so the test can tell them apart)."""
    lines = [f"{side} arm: poses where the two models disagree most (identity vs v10, grasp point, cm)"]
    for name, (s, q) in POSES.items():
        if s == side:
            c = compare(models, s, q)
            lines.append(f"- {name}: gap {c['gap_cm']} cm  (joints {q})")
    return "\n".join(lines)


PROTOCOL = """\
Physical test (the arm is limp and moved BY HAND; the station and any teleop must be stopped: `dreamcli robot down`). Nothing here moves a motor.
Tools: tape measure, a plumb bob or a square, a table mark, a phone level. One person holds the arm, another measures.
1. Mark a point O on the table. Choose axes once: x = forward (away from the robot's chest), y = left, z = up. Keep them for the whole test.
2. Pose 1 (reference): hold the arm straight down or in any easy bent pose. Keep it still. Measure the centre between the fingertips (the grasp point): its distance forward (x) and to the left (y) from O
   using the plumb line to the table, and its height z above the table. Type the three numbers in cm.
3. Poses 2..6: bend the arm clearly differently each time (elbow near 90 degrees, arm raised forward, hand out to the side, wrist turned, ...). Larger moves tell the models apart better.
   `record` reads the real joint angles read-only, you type the measured x, y, z of the grasp point in cm.
4. Because only the DIFFERENCE between poses is used, the unknown height of the shoulder and the position of O do not matter. Aim for 1 cm accuracy.
5. `report` compares the measured moves with both models: pass = one model reproduces them within 4 cm and the other does not. If neither does, check the measuring first.
"""


def add_entry(side: str, tcp_cm, read_q, path: pathlib.Path = RECORD, samples: int = 15) -> dict:
    """Reads the limp arm's joints (median of `samples` read-only reads) and appends them with the measured grasp point (cm) to the record file.

    @param read_q Side to joints in radians now. @return The entry written (`q_deg`, `tcp_cm`) and how many entries the side now has.
    """
    q = np.degrees(np.median([read_q(side) for _ in range(samples)], axis=0))
    entry = {"q_deg": [round(float(v), 2) for v in q], "tcp_cm": [float(v) for v in tcp_cm]}
    data = json.loads(path.read_text()) if path.exists() else {"entries": {}}
    data["entries"].setdefault(side, []).append(entry)
    path.write_text(json.dumps(data, indent=1))
    return {**entry, "count": len(data["entries"][side])}


def record(side: str, path: pathlib.Path = RECORD) -> None:  # pragma: no cover - needs the arm
    """Interactive: read the limp arm, ask for the measured grasp point, append to the record file."""
    import workspace

    workspace.refuse_if_station_running()
    arms, read_q, _ = workspace.open_sources("follower", [side])
    try:
        data = json.loads(path.read_text()) if path.exists() else {"entries": {}}
        while True:
            input(f"{side} arm: hold it still in the next pose, then press Enter to read the joints (Ctrl-C to finish)... ")
            q = np.degrees(np.median([read_q(side) for _ in range(15)], axis=0))
            print("joints (deg):", [round(float(v), 1) for v in q])
            tcp = [float(v) for v in input("measured grasp point x forward, y left, z up (cm, three numbers): ").replace(",", " ").split()]
            data["entries"].setdefault(side, []).append({"q_deg": [round(float(v), 2) for v in q], "tcp_cm": tcp})
            path.write_text(json.dumps(data, indent=1))
    except KeyboardInterrupt:
        print("\nsaved to", path)
    finally:
        workspace.close_all(arms)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["models", "mapping", "discrepancies", "plan", "record", "add", "report"])
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--tcp", type=float, nargs=3, metavar=("X", "Y", "Z"), help="add: the measured grasp point in cm (x forward, y left, z up)")
    args = parser.parse_args(argv)
    if args.command == "add":
        import workspace

        if args.tcp is None:
            parser.error("add needs --tcp X Y Z")
        workspace.refuse_if_station_running()
        arms, read_q, _ = workspace.open_sources("follower", [args.side])
        try:
            print(json.dumps(add_entry(args.side, args.tcp, read_q)))
        finally:
            workspace.close_all(arms)
        return 0
    if args.command == "record":
        record(args.side)
        return 0
    models = Models()
    if args.command == "models":
        for name, (side, q) in POSES.items():
            print(name, json.dumps(compare(models, side, q)))
    elif args.command == "discrepancies":
        for row in discrepancies(models, args.side):
            print(json.dumps(row))
    elif args.command == "mapping":
        for row in mapping(models, args.side):
            print(json.dumps(row))
    elif args.command == "plan":
        print(PROTOCOL)
        print(plan_text(models, args.side))
    else:
        data = json.loads(RECORD.read_text())["entries"]
        for side, entries in data.items():
            print(side, json.dumps(judge(models, side, entries)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

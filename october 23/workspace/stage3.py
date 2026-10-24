"""Stage 3 of goal.md, read-only: the real table, and how the poses and paths in use sit against it. Opens no bus, camera or motor.

`report` places the table from the tape measurements (and checks surface points you measured, if given) in the body frame and in each arm's base frame, then prints for the poses in use
(rest, ready, the start-pose sets, the home, the rest-to-home path) how far every arm link is from the table and how high the grasp point is above it, under the active joint mapping
(`OPENARM_MODEL_MAPPING`, see log-019). It does not apply anything. Run from this folder: `python stage3.py report --front-cm 13 --top-cm 25 --depth-cm D --width-cm W`.
"""

import argparse
import math
import pathlib
import re
import sys

import numpy as np

import editor as ed
import openarm_mapping as mapping
import openarm_poses as poses
import openarm_zones as zones
import zonebuild as zb

POSE_SETS = pathlib.Path("~/projects/openarm-orboh/config/pose_sets").expanduser()
FLAT_TOLERANCE_CM = 1.5  # measured table-top points may differ in height by this much (a tape and a level are good to about 1 cm)
JOINT = re.compile(r"^joint_([1-7]):\s*(-?[\d.]+)", re.M)  # the final pose only: waypoint entries are indented


def table_from_args(front_cm: float, top_cm: float, depth_cm: float, width_cm: float, centre_y_cm: float = 0.0, lower_cm: float = 0.0) -> ed.Table:
    """The table slab from the four tape numbers (see #ed.table_from_measurements)."""
    return ed.table_from_measurements(front_cm, top_cm, depth_cm, width_cm, centre_y_cm, lower_cm=lower_cm)


def check_surface_points(table: ed.Table, points_cm: list[tuple[float, float, float]]) -> list[str]:
    """Problems between the stated table and surface points measured on it (body frame, cm: x forward, y left, z up); empty when they agree.

    Needs three or more points, the near edge among them (the smallest x must match the stated front edge within 2 cm), a flat top (heights within #FLAT_TOLERANCE_CM), and the stated top height.
    """
    if len(points_cm) < 3:
        return ["measure at least three separate points on the table top, including the near edge"]
    pts = np.asarray(points_cm, dtype=float)
    problems = []
    spread = float(pts[:, 2].max() - pts[:, 2].min())
    if spread > FLAT_TOLERANCE_CM:
        problems.append(f"the measured heights differ by {spread:.1f} cm (more than {FLAT_TOLERANCE_CM} cm): the top is not flat, or a measurement is off")
    if abs(float(pts[:, 2].mean()) - table.top_z * 100) > FLAT_TOLERANCE_CM:
        problems.append(f"the measured top is {pts[:, 2].mean():.1f} cm but the table is set to {table.top_z * 100:.1f} cm")
    if abs(float(pts[:, 0].min()) - table.x_range[0] * 100) > 2.0:
        problems.append(f"the nearest measured point is {pts[:, 0].min():.1f} cm forward but the front edge is set to {table.x_range[0] * 100:.1f} cm")
    xs, ys = pts[:, 0] / 100, pts[:, 1] / 100
    if xs.max() > table.x_range[1] + 0.02 or ys.min() < table.y_range[0] - 0.02 or ys.max() > table.y_range[1] + 0.02:
        problems.append("a measured point lies outside the table's set depth or width")
    return problems


def pose_sets(root: pathlib.Path = POSE_SETS) -> dict[str, dict[str, list[float]]]:
    """The final pose (degrees, logical) of each arm for the start-pose sets in use."""
    out = {}
    for name in ("kitakyushu-takeei-v2", "kitakyushu-wood-l1-v1", "kitakyushu-takeei-v5"):
        found = {}
        for side in ed.SIDES:
            path = root / name / f"{side}.yaml"
            if path.exists():
                values = {int(i): float(v) for i, v in JOINT.findall(path.read_text(encoding="utf-8"))}
                if len(values) == 7:
                    found[side] = [values[i] for i in range(1, 8)]
        if found:
            out[name] = found
    return out


def clearance(state: ed.EditorState, side: str, q_deg) -> dict:
    """For one pose: the smallest distance of an arm link (less its 4 cm radius) to the table slab, and the grasp point's height above the table top, both in cm."""
    zone = state.table.keep_out()
    center = np.asarray(ed.to_base(side, zone["center"]), dtype=float)
    size = np.asarray(zone["size"], dtype=float)
    pts = zones.ArmKinematics(side, logical=True).points(np.radians(q_deg))
    links = {n: zones.distance_to_box(p, center, size) - zones.LINK_RADIUS_M for n, p in pts.items() if n != "tcp"}
    worst = min(links, key=links.get)
    tcp = pts["tcp"]
    top = center[2] + size[2] / 2
    return {"link": worst, "link_clearance_cm": round(links[worst] * 100, 1), "tcp_above_top_cm": round(float(tcp[2] - top) * 100, 1),
            "tcp_xy_over_table": bool(abs(tcp[0] - center[0]) <= size[0] / 2 and abs(tcp[1] - center[1]) <= size[1] / 2)}


def report(state: ed.EditorState) -> list[str]:
    lines = [f"mapping: {mapping.active()}", f"table (body frame): front edge x = {state.table.x_range[0] * 100:.1f} cm, back x = {state.table.x_range[1] * 100:.1f} cm, "
             f"y {state.table.y_range[0] * 100:.1f} to {state.table.y_range[1] * 100:.1f} cm, top z = {state.table.top_z * 100:.1f} cm above the base plate; "
             f"danger box top lowered by {state.table.lower_cm:g} cm (0 = at the real surface)"]
    if state.table.lower_warning():
        lines.append("WARNING: " + state.table.lower_warning())
    for side in ed.SIDES:
        zone = state.table.keep_out()
        lines.append(f"{side} arm base frame: table top z = {(state.table.top_z - ed.BASE_OFFSET[side][2]) * 100:.1f} cm (the shoulder is at 0), front edge x = {state.table.x_range[0] * 100:.1f} cm, centre y = {ed.to_base(side, zone['center'])[1] * 100:.1f} cm")
    named: dict[str, dict[str, list[float]]] = {"rest": dict(poses.REST_DEG), "ready": dict(poses.READY_DEG)}
    named.update(pose_sets())
    for label, per_side in named.items():
        for side, q in per_side.items():
            c = clearance(state, side, q)
            flag = "CLASH" if c["link_clearance_cm"] < zones.CLEARANCE_M * 100 else "ok"
            lines.append(f"{side:5s} {label:24s} closest link {c['link']:11s} {c['link_clearance_cm']:6.1f} cm  grasp point {c['tcp_above_top_cm']:6.1f} cm above the table top  {flag}")
    problems = state.clashes()
    lines.append(f"clashes against the zones (rest, ready, home and the path between): {len(problems)}")
    lines += [f"  {p}" for p in problems[:8]]
    for side in ed.SIDES:
        over, total = state.reach_over_table(side)
        lines.append(f"{side} painted work area over the table: {over} of {total} cells" if total else f"{side} painted work area: none yet (record points above the table)")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["report"])
    parser.add_argument("--front-cm", type=float, required=True, help="the near edge, forward of the robot's centre line")
    parser.add_argument("--top-cm", type=float, required=True, help="the table top above the floor-level base plate")
    parser.add_argument("--depth-cm", type=float, required=True)
    parser.add_argument("--width-cm", type=float, required=True)
    parser.add_argument("--centre-y-cm", type=float, default=0.0)
    parser.add_argument("--lower-cm", type=float, default=0.0, help="lower the danger box top this many cm below the real surface (0 = at the surface)")
    parser.add_argument("--points", help="measured surface points as 'x,y,z;x,y,z;...' in cm (body frame)")
    args = parser.parse_args(argv)
    state = ed.EditorState.default()
    state.table = table_from_args(args.front_cm, args.top_cm, args.depth_cm, args.width_cm, args.centre_y_cm, args.lower_cm)
    print("\n".join(report(state)))
    if args.points:
        pts = [tuple(float(v) for v in chunk.split(",")) for chunk in args.points.split(";")]
        problems = check_surface_points(state.table, pts)
        print("surface points:", "agree with the table" if not problems else "; ".join(problems))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

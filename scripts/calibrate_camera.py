#!/usr/bin/env python
"""Calibrate the chest camera to the robot base from recorded demos (no hands-on step).

  .venv/bin/python scripts/calibrate_camera.py from-demos [--dataset ROOT] [--save]

For each episode of a pick-and-place dataset:
  * where the object was: the colour blob in the first chest-camera frame (arms are at the start pose, the block is visible);
  * where the gripper really was when it closed on it: forward kinematics of the leader's joint angles in the MuJoCo twin
    (midpoint of the two finger geoms at the frame where the picking arm's gripper closes).
These pairs go to PnP; the fit is judged by reprojection error and leave-one-out error (mm, on the table plane).
Needs the chest camera free (intrinsics are read from the device). Result: config/camera_extrinsics.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orboh.sim  # noqa: F401  (first: sets MUJOCO_GL before mujoco is imported)

import av  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from orboh.config import PROJECT_ROOT, load_config
from orboh.demos.dataset import LeRobotDemoDataset
from orboh.perception.block_finder import fit_plane
from orboh.perception.calibration import (colour_block_pixel, intrinsic_matrix, kabsch, pixel_to_camera_plane,
                                          save_extrinsics, transform_points)
from orboh.robot.mapping import JointMapping
from orboh.sim.mujoco_sim import MujocoSim

DEFAULT_DATASET = Path.home() / ".cache/huggingface/lerobot/Orboh/bi_openarm_takeei_wood_l1_chest_wristr_wristl_v1_20261020_171837"


def first_frame(info: dict) -> np.ndarray:
    c = av.open(info["path"])
    st = c.streams.video[0]
    c.seek(int(info["from_s"] / st.time_base), stream=st)
    for f in c.decode(st):
        if f.time is not None and f.time >= info["from_s"] - 1e-3:
            img = np.asarray(f.to_image())
            c.close()
            return img
    raise RuntimeError("no frame")


def live_table_plane(cfg, serial: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(K, table normal toward the camera, offset d) from one live depth frame of the carpet: d = camera height above the table."""
    from orboh.cameras.sources import CameraDevice, RealSenseSource

    src = RealSenseSource(CameraDevice(serial, "D435I"), cfg.cameras.width, cfg.cameras.height, cfg.cameras.fps, depth=True)
    try:
        for _ in range(15):
            src.read_rgbd()
        rgbd = src.read_rgbd()
    finally:
        src.close()
    x0, y0, x1, y1 = cfg.perception.roi
    pts = rgbd.points()[y0:y1, x0:x1].reshape(-1, 3)
    pts = pts[pts[:, 2] > 0.05]
    n, d, frac = fit_plane(pts[:: max(1, len(pts) // 6000)], cfg.perception.plane_tol_mm / 1000)
    if frac < 0.6:
        raise RuntimeError(f"no dominant table plane in the ROI (inliers {frac:.0%}); is the carpet in view?")
    return intrinsic_matrix(rgbd.fx, rgbd.fy, rgbd.cx, rgbd.cy), n, d


def grasp_points(ds, sim) -> tuple[list, list, list]:
    """(episode ids, pixels, base-frame grasp points) for episodes where both can be determined."""
    m, d = sim.model, sim.data
    ids, pix, pts = [], [], []
    for i in range(len(ds.list_episodes())):
        ep = ds.load(i)
        st = ep.states  # the follower's measured joints: where the gripper really was
        best = None
        for side, gi in (("left", 7), ("right", 15)):
            g = st[:, gi]
            opened = np.where(g > 0.5)[0]
            if not len(opened):
                continue
            # holding the block: the follower gripper sits near the contact value (0.49 = -32 deg, work/18) and stays there
            near = (np.abs(g - 0.49) < 0.06) & (np.arange(len(g)) > opened[0])
            run = np.convolve(near, np.ones(10), mode="valid") >= 10
            hold = np.where(run)[0]
            if len(hold) and (best is None or g.max() - g.min() > best[2]):
                best = (side, int(hold[0]) + 5, g.max() - g.min())
        if best is None:
            print(f"ep{i}: no grasp found, skipped")
            continue
        side, gf, _ = best
        pixel = colour_block_pixel(first_frame(ep.videos["observation.images.chest"]))
        if pixel is None:
            print(f"ep{i}: block not found in the first frame, skipped")
            continue
        sim.reset(st[gf])
        mujoco.mj_forward(m, d)
        fingers = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"openarm_{side}_ee_") and "finger" in m.body(m.geom_bodyid[g]).name and m.geom_contype[g]]
        point = np.mean([d.geom_xpos[g] for g in fingers], axis=0)
        wrist = d.xpos[m.body(f"openarm_{side}_ee_base_link").id]
        axis = (point - wrist) / np.linalg.norm(point - wrist)  # wrist -> fingers: the gripper's approach direction
        ids.append(i), pix.append(pixel), pts.append((point, axis))
        print(f"ep{i}: {side:5s} pixel=({pixel[0]:5.0f},{pixel[1]:5.0f}) finger-midpoint base xyz={point.round(3)}")
    return ids, pix, pts


def from_demos(args) -> int:
    cfg = load_config(None)
    serial = cfg.cameras.pinned_serials["center"]
    K, normal, d = live_table_plane(cfg, serial)
    ds = LeRobotDemoDataset(args.dataset, JointMapping(cfg.robot.lerobot))
    sim = MujocoSim(cfg.sim.scene, cfg.sim.table_drop_m)
    mujoco.mj_forward(sim.model, sim.data)
    tab = sim.model.geom("table_top")
    table_top = float(sim.data.geom_xpos[tab.id][2] + tab.size[2])
    ids, pix, pts = grasp_points(ds, sim)
    if len(pix) < 5:
        print("need at least 5 usable episodes", file=sys.stderr)
        return 1
    # The depth fixes camera height and tilt; the episodes fix the remaining position and the turn about the vertical.
    # Pixels are the block's visible blob centre, which sits ~half a block height above the table.
    half = args.block_height_mm / 2000
    cam_pts = np.array([pixel_to_camera_plane(p, K, normal, d, half) for p in pix])
    # The finger meshes' midpoint is not the grasp point: slide it along the gripper axis by one fitted distance so the
    # grasp heights agree with the block's (table + half its height). Then FK supplies x, y; z is the known block height.
    f0, ax = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    delta = float(np.sum((table_top + half - f0[:, 2]) * ax[:, 2]) / np.sum(ax[:, 2] ** 2))
    print(f"fingertip offset along the gripper axis (fitted): {delta * 1000:.0f} mm; axis z-components {np.round(ax[:, 2], 2)}")
    grasp = f0 + delta * ax
    print("z spread of the grasp points after the offset: ", np.round(grasp[:, 2], 3))
    base_pts = np.array([[g[0], g[1], table_top + half] for g in grasp])
    T, rms = kabsch(cam_pts, base_pts)
    loo = []
    for k in range(len(pix)):
        keep = np.arange(len(pix)) != k
        Tk, _ = kabsch(cam_pts[keep], base_pts[keep])
        loo.append(np.linalg.norm(transform_points(Tk, cam_pts[k])[0][:2] - base_pts[k][:2]) * 1000)
    print(f"\ntable top z={table_top:.3f} m; camera height above the table (from depth) {d:.3f} m -> camera z {T[2, 3]:.3f} m "
          f"(scene file says ~0.700 m above the plate)")
    print(f"fit rms {rms * 1000:.1f} mm; leave-one-out error: mean {np.mean(loo):.0f} mm, max {np.max(loo):.0f} mm ({len(pix)} episodes)")
    print("camera position in the base frame (m):", T[:3, 3].round(3))
    if args.save:
        save_extrinsics(PROJECT_ROOT / "config/camera_extrinsics.yaml", serial, T, K, "depth_plane+demos",
                        rms_mm=round(rms * 1000, 1), loo_mean_mm=round(float(np.mean(loo)), 1), loo_max_mm=round(float(np.max(loo)), 1),
                        episodes=len(pix), table_top_z=round(table_top, 4))
        print("saved config/camera_extrinsics.yaml")
    return 0


def check(args) -> int:
    """Has the chest camera moved since it was calibrated? Compares its live height above the table (from depth) with the saved one."""
    from orboh.perception.calibration import load_extrinsics

    cfg = load_config(None)
    serial = cfg.cameras.pinned_serials["center"]
    ext = load_extrinsics(PROJECT_ROOT / cfg.cameras.extrinsics, serial)
    if ext is None:
        print("no calibration saved: run `from-demos --save` after recording demos with the camera where it will stay")
        return 2
    import yaml

    saved = yaml.safe_load((PROJECT_ROOT / cfg.cameras.extrinsics).read_text())[serial]
    _, normal, d = live_table_plane(cfg, serial)
    expected = float(ext[0][2, 3]) - float(saved["table_top_z"])
    pitch_saved = float(np.degrees(np.arcsin(-ext[0][2, 2])))
    pitch_live = float(np.degrees(np.arcsin(abs(normal[2]))))
    print(f"camera height above the table: calibrated {expected:.3f} m, now {d:.3f} m (difference {abs(d - expected) * 1000:.0f} mm)")
    print(f"camera tilt below horizontal:  calibrated {pitch_saved:.0f} deg, now {pitch_live:.0f} deg")
    moved = abs(d - expected) > 0.015 or abs(pitch_saved - pitch_live) > 3
    print("=> THE CAMERA HAS MOVED: the saved calibration is wrong. Record new demos and re-run from-demos --save." if moved
          else "=> consistent with the saved calibration (within 15 mm / 3 deg).")
    return 1 if moved else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("from-demos", help="solve the camera pose from a recorded pick dataset")
    s.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    s.add_argument("--block-height-mm", type=float, default=47.0, help="height of the pieces the demos picked")
    s.add_argument("--save", action="store_true", help="write config/camera_extrinsics.yaml")
    s.set_defaults(fn=from_demos)
    s = sub.add_parser("check", help="has the camera moved since the saved calibration? (needs the camera free)")
    s.set_defaults(fn=check)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

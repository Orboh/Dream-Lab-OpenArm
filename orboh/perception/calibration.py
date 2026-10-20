"""Camera -> robot-base calibration.

Two ways to get T_base<-cam (a 4x4 rigid transform), both ending in `config/camera_extrinsics.yaml`:
  * `pnp_extrinsics`: 3D points in the base frame (where the gripper really was) + the pixels where the object was seen.
    Used with recorded demos: the grasp pose comes from the arm's joint angles, the pixel from the first video frame.
  * `kabsch`: 3D-3D pairs (e.g. a fingertip measured by the camera's depth vs forward kinematics), for a hands-on refinement.
Base frame = the MuJoCo twin's world frame (z up), so the result plugs straight into planning and the safety sim.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid transform with dst ~ R src + t from >= 3 non-collinear pairs. Returns (4x4, rms residual in the units of dst)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 3:
        raise ValueError("need at least 3 point pairs")
    cs, cd = src.mean(0), dst.mean(0)
    u, s, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    if s[1] < 1e-6 * max(s[0], 1e-12):
        raise ValueError("point pairs are collinear: move the points apart")
    r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    t = cd - r @ cs
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = r, t
    rms = float(np.sqrt(((src @ r.T + t - dst) ** 2).sum(1).mean()))
    return T, rms


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def pnp_extrinsics(pixels: np.ndarray, base_pts: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, float]:
    """T_base<-cam from >= 4 pixel / base-frame point pairs (planar sets are fine). Returns (4x4, rms reprojection in px)."""
    pixels, base_pts = np.asarray(pixels, np.float64), np.asarray(base_pts, np.float64)
    if len(pixels) < 4:
        raise ValueError("need at least 4 point pairs")
    ok, rvec, tvec = cv2.solvePnP(base_pts, pixels, K, None, flags=cv2.SOLVEPNP_EPNP)  # works from 4 points
    if not ok:
        raise ValueError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(base_pts, pixels, K, None, rvec, tvec)
    proj, _ = cv2.projectPoints(base_pts, rvec, tvec, K, None)
    rms = float(np.sqrt(((proj[:, 0] - pixels) ** 2).sum(1).mean()))
    R, _ = cv2.Rodrigues(rvec)
    T_cam_base = np.eye(4)
    T_cam_base[:3, :3], T_cam_base[:3, 3] = R, tvec[:, 0]
    return np.linalg.inv(T_cam_base), rms


def pixel_to_camera_plane(pixel: tuple[float, float], K: np.ndarray, normal: np.ndarray, d: float, height_m: float = 0.0) -> np.ndarray:
    """Camera-frame point where the ray through `pixel` meets the plane n.p + d = height_m (the table raised by `height_m`)."""
    ray = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    return ray * ((height_m - d) / float(normal @ ray))


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    return pts @ T[:3, :3].T + T[:3, 3]


def pixel_to_plane(pixel: tuple[float, float], K: np.ndarray, T_base_cam: np.ndarray, z: float) -> np.ndarray:
    """Intersect the camera ray through `pixel` with the horizontal plane z = `z` in the base frame."""
    ray_cam = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    origin, direction = T_base_cam[:3, 3], T_base_cam[:3, :3] @ ray_cam
    return origin + direction * ((z - origin[2]) / direction[2])


def colour_block_pixel(color: np.ndarray, roi: tuple[int, int, int, int] = (0, 295, 640, 480)) -> tuple[float, float] | None:
    """Centre of the biggest pale blob on the dark carpet in an RGB image (no depth available, e.g. recorded video)."""
    x0, y0, x1, y1 = roi
    v = color.max(axis=2).astype(np.float32) / 255
    mask = np.zeros(v.shape, np.uint8)
    mask[y0:y1, x0:x1] = (v[y0:y1, x0:x1] > 0.55)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (float(cen[i][0]), float(cen[i][1])) if st[i, cv2.CC_STAT_AREA] > 800 else None


def attach_base_positions(detections: list, T_base_cam: np.ndarray | None) -> list:
    """Fill `Detection.base_xyz` from a camera->base transform (left as None when there is no calibration)."""
    if T_base_cam is not None:
        for det in detections:
            det.base_xyz = transform_points(T_base_cam, det.cam_xyz)[0]
    return detections


def save_extrinsics(path: Path, serial: str, T: np.ndarray, K: np.ndarray, method: str, **stats) -> None:
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    data[serial] = {"T_base_cam": np.round(T, 6).tolist(), "K": np.round(K, 4).tolist(), "method": method, **stats}
    path.write_text("# camera -> robot base (MuJoCo twin world frame, z up). Written by scripts/calibrate_camera.py\n"
                    + yaml.safe_dump(data, sort_keys=False))


def load_extrinsics(path: Path, serial: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    entry = (yaml.safe_load(path.read_text()) or {}).get(serial)
    return (np.array(entry["T_base_cam"]), np.array(entry["K"])) if entry else None

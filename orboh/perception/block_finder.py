"""Classical block finder: fit the table plane in the depth image, then take what stands on it.

Works for the level-1 scene (one object on a plain carpet). Steps:
  1. camera-frame points from metric depth inside the pixel ROI (the carpet, not the clear box);
  2. RANSAC plane through the ROI (the table dominates it), refined by least squares;
  3. everything `min_height_mm`..`max_height_mm` above that plane is candidate object;
  4. connected components -> footprint (minAreaRect in the table plane), height, centre, yaw.
Blobs outside sane object sizes are dropped; the result may be empty (never a guess). A learned detector can
replace `BlockFinder` behind the same `find(rgbd) -> list[Detection]` interface.
"""

from __future__ import annotations

import cv2
import numpy as np

from orboh.config import PerceptionConfig
from orboh.perception.types import RGBD, Detection


def fit_plane(pts: np.ndarray, tol_m: float, iters: int = 200, seed: int = 0) -> tuple[np.ndarray, float, float]:
    """RANSAC + SVD refinement. Returns (unit normal n, offset d, inlier fraction) with n.p + d = 0 and d >= 0
    (so n points from the plane toward the camera at the origin)."""
    rng = np.random.default_rng(seed)
    best = np.zeros(len(pts), bool)
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inl = np.abs((pts - a) @ n) < tol_m
        if inl.sum() > best.sum():
            best = inl
    inl_pts = pts[best]
    centroid = inl_pts.mean(axis=0)
    n = np.linalg.svd(inl_pts - centroid, full_matrices=False)[2][-1]
    d = -float(n @ centroid)
    if d < 0:
        n, d = -n, -d
    return n, d, float(best.mean())


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


class BlockFinder:
    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg

    def find(self, rgbd: RGBD) -> list[Detection]:
        c = self.cfg
        h_img, w_img = rgbd.depth_m.shape
        x0, y0, x1, y1 = (int(np.clip(v, 0, lim)) for v, lim in zip(c.roi, (w_img, h_img, w_img, h_img)))
        roi = np.zeros((h_img, w_img), bool)
        roi[y0:y1, x0:x1] = True
        pts = rgbd.points()
        valid = roi & (rgbd.depth_m > 0.05)
        if valid.sum() < 1000:
            return []
        n, d, inlier_frac = fit_plane(pts[valid][:: max(1, int(valid.sum() // 6000))], c.plane_tol_mm / 1000)
        if inlier_frac < 0.4:  # no dominant plane in the ROI: nothing trustworthy to measure against
            return []
        height = pts @ n + d  # metres above the table plane, toward the camera
        cand = valid & (height > c.min_height_mm / 1000) & (height < c.max_height_mm / 1000)
        mask = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        u_ax, v_ax = _plane_basis(n)
        out: list[Detection] = []
        for lab in range(1, count):
            if stats[lab, cv2.CC_STAT_AREA] < c.min_pixels:
                continue
            comp = (labels == lab) & cand  # the closing can fill pixels with no depth: those must not become points
            if comp.sum() < c.min_pixels:
                continue
            p, hh = pts[comp], height[comp] * 1000
            top_h = float(np.percentile(hh, 90))
            uv = np.stack([p @ u_ax, p @ v_ax], axis=1) * 1000  # mm in the table plane
            (_, _), (rw, rl), ang = cv2.minAreaRect(uv.astype(np.float32))
            size = (max(rw, rl), min(rw, rl))
            lo_f, hi_f = c.footprint_range_mm
            lo_h, hi_h = c.height_range_mm
            if not (lo_f <= size[1] and size[0] <= hi_f and lo_h <= top_h <= hi_h):
                continue
            top = p[hh > top_h - 10]
            centre = top.mean(axis=0) - n * (top_h / 2000)  # down into the object: half its height
            yaw = np.radians(ang if rw >= rl else ang + 90)
            ys, xs = np.nonzero(comp)
            matches = [name for name, (a, b, hgt) in c.pieces.items()
                       if abs(size[0] - max(a, b)) < c.match_tol_mm and abs(size[1] - min(a, b)) < c.match_tol_mm
                       and abs(top_h - hgt) < c.match_tol_mm]
            conf = float(min(1.0, comp.sum() / (4 * c.min_pixels)) * min(1.0, inlier_frac + 0.3))
            out.append(Detection(pixel=(float(xs.mean()), float(ys.mean())), cam_xyz=centre, height_mm=top_h,
                                 size_mm=size, yaw_rad=float(yaw), confidence=conf, matches=matches,
                                 plane=(n, d), mask=comp))
        return sorted(out, key=lambda det: -det.confidence)

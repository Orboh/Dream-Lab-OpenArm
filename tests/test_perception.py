"""Block finder on a synthetic RGB-D scene: a table plane seen from a camera pitched down 45 deg, plus a block's top face."""

import numpy as np
import pytest

from orboh.config import PerceptionConfig
from orboh.perception.block_finder import BlockFinder, _plane_basis, fit_plane
from orboh.perception.types import RGBD

FX = FY = 600.0
CX, CY, W, H = 320.0, 240.0, 640, 480
PITCH = np.radians(45)
N = np.array([0.0, -np.cos(PITCH), -np.sin(PITCH)])  # table normal toward the camera, camera frame
D = 0.45  # camera height above the table, metres


def scene(block_mm=47.0, side_mm=47.0, centre_uv=(0.40, 0.02), noise_mm=1.0, seed=1) -> tuple[RGBD, np.ndarray]:
    """Depth image of the plane plus one block top face; returns (rgbd, true object centre in camera frame)."""
    u_ax, v_ax = _plane_basis(N)
    gx, gy = np.meshgrid((np.arange(W) - CX) / FX, (np.arange(H) - CY) / FY)
    ray = np.stack([gx, gy, np.ones((H, W))], axis=-1)
    nr = ray @ N
    t_table = -D / nr
    depth = t_table.copy()
    origin = -D * N  # a point on the table
    if block_mm:
        hb = block_mm / 1000
        t_top = (hb - D) / nr
        q = ray * t_top[..., None] - origin
        cu, cv = centre_uv
        half = side_mm / 2000
        inside = (np.abs(q @ u_ax - cu) < half) & (np.abs(q @ v_ax - cv) < half)
        depth[inside] = t_top[inside]
    depth += np.random.default_rng(seed).normal(0, noise_mm / 1000, depth.shape)
    centre = origin + centre_uv[0] * u_ax + centre_uv[1] * v_ax + N * (block_mm / 2000)
    return RGBD(np.zeros((H, W, 3), np.uint8), depth.astype(np.float32), FX, FY, CX, CY), centre


@pytest.fixture
def finder():
    return BlockFinder(PerceptionConfig(roi=[0, 0, W, H]))


def test_plane_fit_recovers_table_normal():
    rgbd, _ = scene(block_mm=0)
    pts = rgbd.points().reshape(-1, 3)[::40]
    n, d, frac = fit_plane(pts, 0.006)
    assert abs(n @ N) > 0.999 and d == pytest.approx(D, abs=0.005) and frac > 0.9


def test_finds_cube_with_size_height_and_centre(finder):
    rgbd, centre = scene()
    dets = finder.find(rgbd)
    assert len(dets) == 1
    det = dets[0]
    assert det.height_mm == pytest.approx(47, abs=4)
    assert det.size_mm[0] == pytest.approx(47, abs=5) and det.size_mm[1] == pytest.approx(47, abs=5)
    assert np.linalg.norm(det.cam_xyz - centre) < 0.008
    assert "cube" in det.matches


def test_position_follows_the_object(finder):
    a = finder.find(scene(centre_uv=(0.30, -0.08))[0])[0]
    b = finder.find(scene(centre_uv=(0.45, 0.10))[0])[0]
    assert np.linalg.norm(a.cam_xyz - b.cam_xyz) == pytest.approx(np.hypot(0.15, 0.18), abs=0.01)


def test_empty_table_gives_no_detection(finder):
    assert finder.find(scene(block_mm=0)[0]) == []


def test_too_small_blob_is_rejected_not_guessed(finder):
    assert finder.find(scene(block_mm=47, side_mm=12)[0]) == []


def test_missing_depth_pixels_inside_the_block_do_not_stretch_it(finder):
    """Regression (live D435i): holes with depth 0 were filled by morphology and became points at the camera origin."""
    rgbd, centre = scene()
    ys, xs = np.nonzero(scene(block_mm=0, noise_mm=0)[0].depth_m - rgbd.depth_m > 0.02)  # pixels covered by the top face
    rgbd.depth_m[int(np.median(ys)) - 1:int(np.median(ys)) + 2, int(np.median(xs)) - 2:int(np.median(xs)) + 3] = 0
    dets = finder.find(rgbd)
    assert len(dets) == 1 and dets[0].size_mm[0] < 70 and np.linalg.norm(dets[0].cam_xyz - centre) < 0.008

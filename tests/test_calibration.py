import numpy as np
import pytest

from orboh.perception.calibration import (colour_block_pixel, intrinsic_matrix, kabsch, load_extrinsics, pixel_to_camera_plane, pixel_to_plane,
                                           pnp_extrinsics, save_extrinsics, transform_points)

K = intrinsic_matrix(610, 610, 320, 240)


def pose(yaw_deg=20, pitch_deg=135, t=(0.12, -0.03, 0.75)):
    """A camera looking down at the table from above-and-behind the robot's base frame (z up)."""
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rz @ Rx, t
    return T


def project(T_base_cam, pts):
    cam = transform_points(np.linalg.inv(T_base_cam), pts)
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def test_kabsch_recovers_rigid_transform():
    T = pose()
    src = np.random.default_rng(0).uniform(-0.2, 0.2, (8, 3))
    got, rms = kabsch(src, transform_points(T, src))
    assert np.allclose(got, T, atol=1e-9) and rms < 1e-9


def test_kabsch_rejects_collinear_points():
    line = np.linspace(0, 1, 5)[:, None] * np.array([[1.0, 0.5, 0.2]])
    with pytest.raises(ValueError):
        kabsch(line, line + 0.1)


def test_pnp_recovers_camera_pose_from_a_planar_point_set():
    T = pose()
    pts = np.array([[x, y, 0.35] for x in (0.25, 0.40, 0.55) for y in (-0.15, 0.0, 0.18)])  # points on the table plane
    got, rms = pnp_extrinsics(project(T, pts) + np.random.default_rng(1).normal(0, 0.3, (9, 2)), pts, K)
    assert rms < 1.0
    assert np.linalg.norm(got[:3, 3] - T[:3, 3]) < 0.02


def test_pixel_to_plane_inverts_the_projection():
    T = pose()
    p = np.array([[0.41, 0.07, 0.35]])
    assert np.allclose(pixel_to_plane(tuple(project(T, p)[0]), K, T, 0.35), p[0], atol=1e-6)


def test_extrinsics_round_trip(tmp_path):
    T = pose()
    save_extrinsics(tmp_path / "e.yaml", "123", T, K, "pnp", rms_px=0.5)
    got = load_extrinsics(tmp_path / "e.yaml", "123")
    assert np.allclose(got[0], T, atol=1e-5) and np.allclose(got[1], K)
    assert load_extrinsics(tmp_path / "e.yaml", "999") is None


def test_colour_block_pixel_finds_the_pale_blob_not_the_specks():
    img = np.full((480, 640, 3), 40, np.uint8)
    img[400:440, 300:350] = 200  # the block
    img[350, 100] = 255  # a bright speck in the carpet
    u, v = colour_block_pixel(img)
    assert (u, v) == pytest.approx((324.5, 419.5), abs=1.0)


def test_pixel_to_camera_plane_hits_the_raised_table():
    n = np.array([0.0, -0.665, -0.747])  # table normal toward the camera (as measured live), camera 0.375 m above the table
    d = 0.375
    p = pixel_to_camera_plane((330.0, 380.0), K, n, d, height_m=0.0235)
    assert float(n @ p + d) == pytest.approx(0.0235, abs=1e-9) and p[2] > 0


def test_attach_base_positions_uses_the_transform_and_skips_without_one():
    from orboh.perception.calibration import attach_base_positions
    from orboh.perception.types import Detection

    det = Detection(pixel=(0, 0), cam_xyz=np.array([0.03, 0.10, 0.38]), height_mm=47, size_mm=(47, 47), yaw_rad=0, confidence=1)
    assert attach_base_positions([det], None)[0].base_xyz is None
    T = pose()
    assert np.allclose(attach_base_positions([det], T)[0].base_xyz, transform_points(T, det.cam_xyz)[0])

import numpy as np
import pytest

from tachymetry.geometry import (
    BrownConradyLens,
    CalibrationError,
    Camera,
    DivisionLens,
    EquidistantLens,
    FisheyeLens,
    angles_from_rotation,
    fit_homography,
    lens_from_dict,
    rotation_from_angles,
)


def test_angles_round_trip(rng):
    for _ in range(200):
        y, p, r = rng.uniform(-180, 180), rng.uniform(1, 89), rng.uniform(-20, 20)
        R = rotation_from_angles(y, p, r)
        assert abs(np.linalg.det(R) - 1) < 1e-9
        y2, p2, r2 = angles_from_rotation(R)
        assert abs((y2 - y + 180) % 360 - 180) < 1e-6 and abs(p2 - p) < 1e-6 and abs(r2 - r) < 1e-6


def test_principal_ray_hits_ground_at_expected_distance():
    cam = Camera.from_pose((1920, 1080), focal=1400, height=10, pitch=30, yaw=90)
    g = cam.image_to_ground([[959.5, 539.5]])[0]
    assert np.allclose(g, [0, 10 / np.tan(np.radians(30))], atol=1e-9)
    assert np.allclose(cam.ground_to_image(g[None])[0], [959.5, 539.5], atol=1e-9)


def test_homography_decomposition_round_trip(rng):
    for _ in range(50):
        c = Camera.from_pose(
            (1920, 1080),
            focal=rng.uniform(600, 3000),
            height=rng.uniform(3, 40),
            pitch=rng.uniform(5, 85),
            yaw=rng.uniform(-180, 180),
            roll=rng.uniform(-10, 10),
            position=(rng.uniform(-50, 50), rng.uniform(-50, 50)),
        )
        c2 = Camera.from_homography(c.H * rng.uniform(0.5, 3), c.image_size)
        assert abs(c2.focal - c.focal) / c.focal < 1e-6
        assert abs(c2.height - c.height) < 1e-6
        assert np.allclose(c2.R, c.R, atol=1e-7)


def test_left_handed_world_is_rejected():
    c = Camera.from_pose((1920, 1080), focal=1200, height=8, pitch=25)
    with pytest.raises(CalibrationError):
        Camera.from_homography(c.H @ np.diag([1, -1, 1.0]), c.image_size)


@pytest.mark.parametrize(
    "lens",
    [
        DivisionLens((1920, 1080), k=-0.25),
        DivisionLens((1920, 1080), k=0.1),
        EquidistantLens((1920, 1080), s=900),
        BrownConradyLens((1920, 1080), k1=-0.3, k2=0.08, p1=1e-3, p2=-5e-4, focal=1100),
        FisheyeLens((1920, 1080), k1=0.05, k2=-0.01, focal=800),
    ],
)
def test_lens_models_invert(lens):
    pts = np.stack(np.meshgrid(np.linspace(0, 1919, 25), np.linspace(0, 1079, 15)), -1).reshape(-1, 2)
    assert np.nanmax(np.abs(lens.distort(lens.rectify(pts)) - pts)) < 1e-6
    assert np.allclose(lens_from_dict(lens.to_dict()).rectify(pts), lens.rectify(pts), equal_nan=True)


def test_camera_json_round_trip(tmp_path, pole_camera):
    cam = pole_camera.with_lens(EquidistantLens(pole_camera.image_size, s=700))
    p = cam.save(tmp_path / "cam.json")
    c2 = Camera.load(p)
    assert np.allclose(c2.H, cam.H) and c2.lens.kind == "equidistant"


def test_resized_camera_maps_the_same_ground(pole_camera):
    small = pole_camera.resized((640, 360))
    uv = np.array([[640.0, 500.0], [300.0, 650.0]])
    assert np.allclose(pole_camera.image_to_ground(uv), small.image_to_ground(uv / 2), atol=1e-6)


def test_robust_homography_rejects_outliers(rng):
    H = np.array([[0.02, 0.001, -5], [0.0005, -0.03, 12], [1e-5, 2e-4, 1.0]])
    src = rng.uniform(0, 1000, (60, 2))
    dst = np.column_stack([src, np.ones(60)]) @ H.T
    dst = dst[:, :2] / dst[:, 2:]
    dst[:6] += 25.0
    H2, inl = fit_homography(src, dst, ransac_threshold=0.5)
    assert inl[6:].all() and not inl[:6].any()


def test_horizon_and_gsd(pole_camera):
    assert pole_camera.below_horizon([[640, 700]])[0]
    gsd = pole_camera.ground_sampling_distance([[640, 400], [640, 700]])
    assert gsd[0] > gsd[1] > 0

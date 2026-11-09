import numpy as np

from tachymetry.calib.correspondences import Correspondences, latlon_to_local, local_to_latlon
from tachymetry.calib.from_cuboids import calibrate_from_cuboids
from tachymetry.calib.from_lines import calibrate_from_lines
from tachymetry.calib.manual import calibrate_manual
from tachymetry.detections import DetectionSet
from tachymetry.geometry import Camera, DivisionLens, EquidistantLens
from tachymetry.synthetic import default_camera, make_scene
from tachymetry.synthetic.scene import Box3D


def _correspondences(cam, rng, n=25, noise=1.0):
    pts = []
    while len(pts) < n:
        uv = rng.uniform([30, 200], [cam.image_size[0] - 30, cam.image_size[1] - 20])
        g = cam.image_to_ground([uv])[0]
        if np.all(np.isfinite(g)) and cam.distance_to(g[None])[0] < 60:
            pts.append(g)
    W = np.array(pts)
    return Correspondences(cam.ground_to_image(W) + rng.normal(0, noise, (n, 2)), W)


def test_manual_calibration_recovers_distorted_camera(rng):
    size = (1280, 720)
    cam = Camera.from_pose(
        size, focal=900, height=8, pitch=32, yaw=55, roll=1.5, position=(-17, -21), lens=EquidistantLens(size, s=700)
    )
    res = calibrate_manual(_correspondences(cam, rng), size)
    assert res.cv_rmse_m < 0.15
    assert abs(res.camera.pitch - 32) < 0.5 and abs(res.camera.height - 8) < 0.2
    assert res.camera.lens.kind in ("division", "equidistant")


def test_manual_calibration_from_reference_pixels(rng, pole_camera):
    corr = _correspondences(pole_camera, rng, noise=0.5)
    gsd = 0.1
    ref = np.column_stack([corr.world[:, 0] / gsd, -corr.world[:, 1] / gsd])
    c2 = Correspondences.from_reference(corr.image, ref, gsd)
    res = calibrate_manual(c2, pole_camera.image_size, lens="identity")
    assert res.rmse_m < 0.1


def test_latlon_round_trip():
    ll = np.array([[40.8075, -73.9626], [40.8080, -73.9620]])
    xy, origin = latlon_to_local(ll)
    assert np.allclose(local_to_latlon(xy, origin), ll, atol=1e-9)
    assert 60 < np.linalg.norm(xy[1] - xy[0]) < 80


def test_lines_calibration_on_synthetic_intersection():
    for lens in (None, DivisionLens((1280, 720), k=-0.18)):
        cam = default_camera(position=(-8.7, -8.7), yaw=45, pitch=34, lens=lens)
        sc = make_scene(camera=cam, seed=2)
        res = calibrate_from_lines(sc.image, scale=f"height={cam.height}")
        assert abs(res.camera.focal - cam.focal) / cam.focal < 0.05
        assert abs(res.camera.pitch - cam.pitch) < 1.0


def test_cuboid_calibration_recovers_camera(rng):
    cam = Camera.from_pose((1920, 1080), focal=1300, height=8, pitch=32, yaw=40, roll=1.0)
    quads, boxes = [], []
    while len(quads) < 300:
        uv = rng.uniform([50, 350], [1870, 1030])
        g = cam.image_to_ground([uv])[0]
        if not np.all(np.isfinite(g)) or cam.distance_to(g[None])[0] > 70:
            continue
        b = Box3D(
            g[0],
            g[1],
            rng.choice([0, 90, 180, 270]) + rng.normal(0, 4),
            rng.normal(4.6, 0.4),
            rng.normal(1.85, 0.1),
            1.5,
        )
        uv8 = cam.project(b.corners()) + rng.normal(0, 1.5, (8, 2))
        lo, hi = uv8.min(0), uv8.max(0)
        if lo.min() < 5 or hi[0] > 1915 or hi[1] > 1075:
            continue
        quads.append(np.concatenate([uv8, np.ones((8, 1))], 1))
        boxes.append([*lo, *hi])
    m = len(quads)
    ds = DetectionSet(
        np.arange(m),
        np.arange(m) / 10.0,
        np.array(boxes, np.float32),
        np.full(m, 0.9, np.float32),
        np.full(m, 2, np.int16),
        np.array(quads, np.float32),
        image_size=(1920, 1080),
        fps=10,
        processed_frames=np.arange(m),
    )
    res = calibrate_from_cuboids(ds)
    assert abs(res.camera.pitch - 32) < 1.0
    assert abs(res.camera.height - 8) / 8 < 0.05


def test_tan_lens_is_recovered_from_straight_lines():
    cam = default_camera(position=(-8.7, -8.7), yaw=45, pitch=34, lens=EquidistantLens((1280, 720), s=700))
    sc = make_scene(camera=cam, seed=2)
    res = calibrate_from_lines(sc.image, scale=f"height={cam.height}", lens="tan")
    assert res.camera.lens.kind == "equidistant"
    assert abs(res.camera.lens.s - 700) / 700 < 0.05
    assert abs(res.camera.pitch - cam.pitch) < 1.0


def test_dash_regularity_checks_the_horizon():
    """Dash chains are recognised by their cross-ratio, and are regular only under a correct horizon."""
    from tachymetry.calib.from_markings import DashedLine, _regular_runs, marking_regularity

    cam = Camera.from_pose((1280, 720), focal=1100.0, height=9.0, pitch=12.0, yaw=90.0)
    # cross-ratio filter: perspective images of equally spaced points pass, a missing dash breaks the run
    t = cam.ground_to_image(np.column_stack([np.zeros(8), 20.0 + 9.0 * np.arange(8)]))[:, 1]
    assert _regular_runs(t, 0.08) == [(0, 8)]
    assert all(e - s < 8 for s, e in _regular_runs(np.delete(t, 4), 0.08))
    lines = []
    for x in (-1.75, 1.75, 5.25):  # three lane lines, 3 m dashes every 9 m
        y0 = 22.0 + 9.0 * np.arange(8)
        a = cam.ground_to_image(np.column_stack([np.full(8, x), y0]))
        b = cam.ground_to_image(np.column_stack([np.full(8, x), y0 + 3.0]))
        lines.append(DashedLine(a, b, np.linalg.norm(b - a, axis=1)))
    assert marking_regularity(cam, lines) < 1e-6
    wrong = Camera.from_pose((1280, 720), focal=1100.0, height=9.0, pitch=15.0, yaw=90.0)
    assert marking_regularity(wrong, lines) > 0.05

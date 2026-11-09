"""Tests for the monocular-depth and GeoCalib calibration methods.

The geometric core (back-projection, robust plane fit, pose from the plane,
angle conventions and lens mapping) is tested with pure numpy on synthetic
data. The network back-ends are exercised end to end on a procedurally rendered
scene with a known camera when the optional packages and checkpoints are
available, and skip otherwise. Those tests are marked ``gpu`` and ``slow``
(deselect with ``-m "not gpu"``); setting ``TACHYMETRY_SKIP_MODEL_TESTS=1``
also skips them.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

from tachymetry.calib.depth import (
    DEPTH_MODELS,
    _label_ids,
    angles_from_up_vector,
    backproject,
    calibrate_from_depth,
    calibrate_from_depth_map,
    fit_ground_plane,
    predict_depth,
    segment_road,
    up_vector_from_angles,
)
from tachymetry.calib.geocalib_backend import calibrate_geocalib, geocalib_to_tachymetry_angles, lens_from_geocalib
from tachymetry.geometry.camera import CalibrationError, Camera

UP = np.array([0.0, 0.0, 1.0])

#: (image_size, focal, height, pitch, yaw, roll): a high PTZ camera like the reference Austin
#: intersection, a low wide-angle pole camera, and a shallow zoomed view whose top rows see the sky.
CAMERAS = [
    ((1920, 1080), 2205.0, 28.8, 18.1, 134.0, -0.2),
    ((1280, 720), 640.0, 6.0, 35.0, -20.0, 3.0),
    ((960, 540), 1400.0, 12.0, 8.0, 200.0, -4.0),
]


def make_camera(spec: tuple) -> Camera:
    size, focal, height, pitch, yaw, roll = spec
    return Camera.from_pose(size, focal=focal, height=height, pitch=pitch, yaw=yaw, roll=roll, position=(3.0, -7.0))


def ground_depth(cam: Camera) -> np.ndarray:
    """Ground-truth z-depth of the ground plane over the image of ``cam`` (NaN above the horizon)."""
    w, h = cam.image_size
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    g = cam.image_to_ground(np.stack([uu.ravel(), vv.ravel()], axis=1))
    X = np.column_stack([g, np.zeros(len(g))])
    return (X @ cam.R.T + cam.t)[:, 2].reshape(h, w)


def corrupt(depth: np.ndarray, noise: float = 0.01, outlier_frac: float = 0.25, seed: int = 0) -> np.ndarray:
    """Multiplicative depth noise plus gross outliers in front of the road (vehicles, poles, trees)."""
    rng = np.random.default_rng(seed)
    d = depth * (1.0 + noise * rng.standard_normal(depth.shape))
    out = rng.random(depth.shape) < outlier_frac
    d[out] *= rng.uniform(0.5, 0.95, int(out.sum()))
    return d


def ground_pixels(cam: Camera, n: int = 400, seed: int = 0) -> np.ndarray:
    """Random pixels that image the ground."""
    rng = np.random.default_rng(seed)
    w, h = cam.image_size
    uv = np.column_stack([rng.uniform(0, w - 1, 20 * n), rng.uniform(0, h - 1, 20 * n)])
    return uv[cam.below_horizon(uv, margin_px=20.0)][:n]


# =========================================================================== pure numpy
@pytest.mark.parametrize("spec", CAMERAS)
def test_up_vector_matches_camera_rotation(spec: tuple) -> None:
    cam = make_camera(spec)
    up_cam = cam.R @ UP
    pitch, roll = angles_from_up_vector(up_cam)
    assert pitch == pytest.approx(cam.pitch, abs=1e-9)
    assert roll == pytest.approx(cam.roll, abs=1e-9)
    np.testing.assert_allclose(up_vector_from_angles(cam.pitch, cam.roll), up_cam, atol=1e-12)
    # the length of the vector does not matter
    assert angles_from_up_vector(3.0 * up_cam) == pytest.approx((pitch, roll), abs=1e-9)


def test_backproject_lands_on_the_ground() -> None:
    cam = make_camera(CAMERAS[1])
    depth = ground_depth(cam)
    pts = backproject(depth, cam.focal)
    world = (pts - cam.t) @ cam.R  # R^T (X_cam - t)
    assert len(pts) == np.isfinite(depth).sum()
    np.testing.assert_allclose(world[:, 2], 0.0, atol=1e-6)


def test_plane_fit_recovers_noise_free_plane() -> None:
    cam = make_camera(CAMERAS[0])
    pts = backproject(ground_depth(cam), cam.focal)[::50]
    plane = fit_ground_plane(pts)
    np.testing.assert_allclose(plane.normal, cam.R @ UP, atol=1e-9)
    assert plane.distance == pytest.approx(cam.height, rel=1e-9)
    assert plane.inlier_ratio == 1.0
    assert plane.rms < 1e-9
    # the residual is the fractional depth error with respect to the plane along each ray
    bumped = pts * 1.02
    np.testing.assert_allclose(plane.residuals(bumped), -0.02, atol=1e-9)


@pytest.mark.parametrize("spec", CAMERAS)
def test_pose_recovered_from_noisy_depth_with_outliers(spec: tuple) -> None:
    """1 % depth noise and 25 % gross outliers: pitch/roll within 0.5 deg, height within 1 %."""
    cam = make_camera(spec)
    depth = corrupt(ground_depth(cam), noise=0.01, outlier_frac=0.25, seed=1)
    res = calibrate_from_depth_map(depth, cam.focal)
    est = res.camera
    assert res.method == "depth"
    assert est.meta["calibration_method"] == "depth" and est.meta["metric"] is True
    assert abs(est.pitch - cam.pitch) < 0.5
    assert abs(est.roll - cam.roll) < 0.5
    assert abs(est.height / cam.height - 1.0) < 0.01
    assert est.focal == pytest.approx(cam.focal)
    assert est.yaw == pytest.approx(90.0)
    assert 0.6 < res.details["inlier_ratio"] < 0.85  # about 75 % of the pixels are ground
    assert 0.0 <= res.score <= 1.0
    assert res.details["warnings"] == []


def test_recovered_camera_preserves_metric_ground_geometry() -> None:
    """Yaw and position are a free world frame, so compare frame-invariant ground distances."""
    cam = make_camera(CAMERAS[0])
    est = calibrate_from_depth_map(corrupt(ground_depth(cam), seed=2), cam.focal).camera
    uv = ground_pixels(cam)
    g_true, g_est = cam.image_to_ground(uv), est.image_to_ground(uv)
    i, j = np.triu_indices(len(uv), 1)
    d_true = np.linalg.norm(g_true[i] - g_true[j], axis=1)
    d_est = np.linalg.norm(g_est[i] - g_est[j], axis=1)
    keep = d_true > 5.0
    assert np.median(np.abs(d_est[keep] / d_true[keep] - 1.0)) < 0.01


def test_road_mask_erosion_and_height_override() -> None:
    cam = make_camera(CAMERAS[1])
    depth = ground_depth(cam)
    w, h = cam.image_size
    mask = np.zeros((h, w), bool)
    mask[h // 2 :, : 2 * w // 3] = True  # road region touching the image border
    depth_obj = depth.copy()
    depth_obj[~mask] *= 0.5  # pixels outside the road mask are off the plane
    res = calibrate_from_depth_map(depth_obj, cam.focal, mask, height=7.5)
    assert res.camera.height == pytest.approx(7.5)
    assert res.details["height_source"] == "user"
    assert res.details["depth_height_m"] == pytest.approx(cam.height, rel=1e-6)
    assert res.details["inlier_ratio"] == pytest.approx(1.0)
    assert res.details["erode_px"] > 0
    assert res.details["n_road_pixels"] < mask.sum()  # eroded, including along the image border
    assert abs(res.camera.pitch - cam.pitch) < 1e-6


def test_focal_error_couples_into_pitch_only() -> None:
    """A focal length ``s`` times too large gives ``tan(pitch) / s``; the roll is unaffected.

    This holds both when only the focal is wrong (depth correct) and when the depth
    network scales its depth with its own focal estimate (Depth Pro).
    """
    cam = make_camera(CAMERAS[1])
    depth = ground_depth(cam)
    s = 1.25
    expected = math.degrees(math.atan(math.tan(math.radians(cam.pitch)) / s))
    for d in (depth, s * depth):
        est = calibrate_from_depth_map(d, s * cam.focal).camera
        assert est.pitch == pytest.approx(expected, abs=1e-6)
        assert est.roll == pytest.approx(cam.roll, abs=1e-6)


def test_implausible_height_is_flagged() -> None:
    cam = make_camera(CAMERAS[0])
    res = calibrate_from_depth_map(ground_depth(cam) / 25.0, cam.focal)  # a "miniature" reading of the scene
    assert res.camera.height == pytest.approx(cam.height / 25.0, rel=1e-6)
    assert res.camera.pitch == pytest.approx(cam.pitch, abs=1e-6)
    assert len(res.details["warnings"]) == 1
    assert "plausible range" in res.details["warnings"][0]


def test_too_little_road_raises() -> None:
    cam = make_camera(CAMERAS[2])
    depth = ground_depth(cam)
    mask = np.zeros(depth.shape, bool)
    mask[-3:, :20] = True
    with pytest.raises(CalibrationError):
        calibrate_from_depth_map(depth, cam.focal, mask)
    with pytest.raises(CalibrationError):
        fit_ground_plane(np.zeros((2, 3)))


def test_segmentation_labels_resolved_by_name() -> None:
    cityscapes_like = {7: "road", 3: "sidewalk", 0: "sky", 11: "car"}
    assert _label_ids(cityscapes_like, ("road",)) == [7]
    assert _label_ids(cityscapes_like, ("sidewalk",)) == [3]
    vistas_like = {0: "construction--flat--road", 1: "construction--flat--sidewalk", 2: "nature--sky"}
    assert _label_ids(vistas_like, ("road",)) == [0]
    assert _label_ids({0: "Road", 1: "side_walk"}, ("road",)) == [0]
    assert _label_ids({0: "terrain"}, ("road",)) == []


def test_unknown_backend_and_missing_packages() -> None:
    img = np.zeros((64, 96, 3), np.uint8)
    with pytest.raises(ValueError, match="unknown depth model"):
        predict_depth(img, "no-such-depth-model")
    assert set(DEPTH_MODELS) == {"depth-pro", "moge2", "depth-anything-v2"}


def test_missing_optional_packages_raise_import_error_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    img = np.zeros((64, 96, 3), np.uint8)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "geocalib", None)
    monkeypatch.setitem(sys.modules, "moge", None)
    monkeypatch.setitem(sys.modules, "moge.model", None)
    monkeypatch.setitem(sys.modules, "moge.model.v2", None)
    # unique checkpoint names keep the model caches out of the way
    with pytest.raises(ImportError, match=r'pip install "tachymetry\[depth\]"'):
        predict_depth(img, "depth-pro:test/not-a-checkpoint", device="cpu")
    with pytest.raises(ImportError, match="pip install"):
        predict_depth(img, "moge2:test/not-a-checkpoint", device="cpu")
    with pytest.raises(ImportError, match="pip install"):
        calibrate_geocalib(img, device="cpu", weights="test-not-a-checkpoint")


# ------------------------------------------------------------------ GeoCalib conventions (pure numpy)
def geocalib_from_rp(roll: float, pitch: float) -> np.ndarray:
    """``geocalib.gravity.Gravity.from_rp`` (radians), transcribed."""
    return np.array([-math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch), math.sin(pitch)])


def image_up_direction(cam: Camera, uv: np.ndarray) -> np.ndarray:
    """Unit image-plane direction of the world vertical, pointing up, at ground pixels ``uv``."""
    g = cam.image_to_ground(uv)
    X = np.column_stack([g, np.zeros(len(g))])
    d = cam.project(X + 0.25 * UP, distort=False) - cam.project(X, distort=False)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@pytest.mark.parametrize("spec", CAMERAS)
def test_geocalib_angle_convention(spec: tuple) -> None:
    """GeoCalib's gravity vector is the up direction; its pitch is our pitch negated."""
    cam = make_camera(spec)
    roll_gc, pitch_gc = math.radians(cam.roll), math.radians(-cam.pitch)
    vec = geocalib_from_rp(roll_gc, pitch_gc)
    np.testing.assert_allclose(vec, cam.R @ UP, atol=1e-12)
    assert geocalib_to_tachymetry_angles(roll_gc, pitch_gc) == pytest.approx((cam.pitch, cam.roll), abs=1e-9)
    assert angles_from_up_vector(vec) == pytest.approx((cam.pitch, cam.roll), abs=1e-9)
    # GeoCalib's up field at normalised pixel (u, v) is (a, b) - c (u, v) for vec3d = (a, b, c):
    # it must point along the image of the world vertical, upwards
    uv = ground_pixels(cam, n=50)
    uvn = (uv - np.array(cam.principal_point)) / cam.focal
    field = vec[:2] - vec[2] * uvn
    field /= np.linalg.norm(field, axis=1, keepdims=True)
    cos = np.sum(field * image_up_direction(cam, uv), axis=1)
    assert np.all(cos > 1 - 1e-9)


def test_geocalib_lens_mapping_matches_its_models() -> None:
    size, f, k1 = (1920, 1080), 2000.0, -0.25
    c = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2])
    rng = np.random.default_rng(0)
    pts = c + rng.uniform(-1, 1, (100, 2)) * c
    x = (pts - c) / f
    r2 = np.sum(x**2, axis=1, keepdims=True)
    # simple_radial: x_d = x_u (1 + k1 |x_u|^2)
    lens = lens_from_geocalib("simple_radial", k1, f, size)
    assert lens.kind == "brown"
    np.testing.assert_allclose(lens.distort(pts), c + f * x * (1 + k1 * r2), atol=1e-9)
    np.testing.assert_allclose(lens.rectify(lens.distort(pts)), pts, atol=1e-6)
    # simple_divisional: x_u = x_d / (1 + k1 |x_d|^2)
    lens = lens_from_geocalib("simple_divisional", k1, f, size)
    assert lens.kind == "division"
    np.testing.assert_allclose(lens.rectify(pts), c + f * x / (1 + k1 * r2), atol=1e-9)
    assert lens_from_geocalib("pinhole", k1, f, size).is_identity
    assert lens_from_geocalib("simple_radial", 0.0, f, size).is_identity


# ------------------------------------------------------------------ GeoCalib conventions (its own code)
def test_geocalib_conventions_with_the_package() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("geocalib")
    from geocalib.camera import Pinhole, SimpleDivisional, SimpleRadial
    from geocalib.gravity import Gravity
    from geocalib.perspective_fields import get_up_field

    w, h, f = 320, 180, 300.0
    cam = Camera.from_pose((w, h), focal=f, height=8.0, pitch=24.0, yaw=60.0, roll=-5.0)
    cx, cy = cam.principal_point
    grav = Gravity.from_rp(torch.tensor(math.radians(cam.roll)), torch.tensor(math.radians(-cam.pitch)))
    np.testing.assert_allclose(grav.vec3d.double().numpy(), cam.R @ UP, atol=1e-6)
    assert angles_from_up_vector(grav.vec3d.double().numpy()) == pytest.approx((cam.pitch, cam.roll), abs=1e-4)
    params = {"width": float(w), "height": float(h), "f": f, "cx": cx, "cy": cy}
    field = get_up_field(Pinhole.from_dict(dict(params)), grav)[0].double().numpy()  # (h, w, 2)
    uv = ground_pixels(cam, n=50)
    ui, vi = np.round(uv).astype(int).T
    pix = np.column_stack([ui, vi]).astype(float)
    cos = np.sum(field[vi, ui] * image_up_direction(cam, pix), axis=1)
    assert np.all(cos > 1 - 1e-5)
    # distortion models: GeoCalib's projection equals the mapped tachymetry lens
    k1 = -0.2
    rng = np.random.default_rng(0)
    rect = np.column_stack([rng.uniform(0, w - 1, 64), rng.uniform(0, h - 1, 64)])
    xu = (rect - [cx, cy]) / f
    p3d = torch.tensor(np.column_stack([xu, np.ones(len(xu))])[None], dtype=torch.float64)
    radial = SimpleRadial.from_dict({**params, "k1": k1}).float()
    raw_gc, _ = radial.world2image(p3d.float())
    raw_ours = lens_from_geocalib("simple_radial", k1, f, (w, h)).distort(rect)
    np.testing.assert_allclose(raw_gc[0].double().numpy(), raw_ours, atol=1e-3)
    division = SimpleDivisional.from_dict({**params, "k1": k1}).float()
    p_und, _ = division.image2world(torch.tensor(rect[None], dtype=torch.float32))
    rect_gc = p_und[0, :, :2].double().numpy() * f + [cx, cy]
    np.testing.assert_allclose(rect_gc, lens_from_geocalib("simple_divisional", k1, f, (w, h)).rectify(rect), atol=1e-3)


# =========================================================================== network back-ends
def _download_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [OSError]
    try:
        from huggingface_hub.errors import HfHubHTTPError

        errors.append(HfHubHTTPError)
    except ImportError:
        pass
    try:
        import httpx

        errors.append(httpx.HTTPError)
    except ImportError:
        pass
    return tuple(errors)


def run_or_skip(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Call ``fn``; skip the test if an optional package or a model checkpoint is unavailable."""
    if os.environ.get("TACHYMETRY_SKIP_MODEL_TESTS"):
        pytest.skip("model tests disabled by TACHYMETRY_SKIP_MODEL_TESTS")
    try:
        return fn(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(f"optional dependency unavailable: {exc}")
    except _download_errors() as exc:
        pytest.skip(f"model checkpoint unavailable: {exc}")


def model_test(fn):  # type: ignore[no-untyped-def]
    """Mark a test that downloads checkpoints and runs networks."""
    return pytest.mark.slow(pytest.mark.gpu(fn))


@pytest.fixture(scope="module")
def scene():  # type: ignore[no-untyped-def]
    """Rendered intersection seen by a pole camera: 8 m high, 32 deg down, 1.5 deg roll, 70 deg HFOV."""
    from tachymetry.synthetic import default_camera, make_scene

    return make_scene(default_camera((960, 540)), seed=0)


def ground_truth_mask(cam: Camera) -> np.ndarray:
    w, h = cam.image_size
    uu, vv = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    return cam.below_horizon(np.stack([uu.ravel(), vv.ravel()], axis=1), margin_px=5.0).reshape(h, w)


@model_test
@pytest.mark.parametrize("depth_model", ["depth-pro", "moge2", "depth-anything-v2"])
def test_depth_backends_on_synthetic_scene(scene, depth_model: str) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if depth_model == "moge2":
        pytest.importorskip("moge")
    truth = scene.camera
    res = run_or_skip(calibrate_from_depth, scene.image, depth_model=depth_model, segmentation=ground_truth_mask(truth))
    cam = res.camera
    assert res.method == "depth"
    assert cam.meta["calibration_method"] == "depth" and cam.meta["metric"] is True
    assert 0.0 <= res.score <= 1.0
    for key in (
        "focal_source",
        "inlier_ratio",
        "plane_rms_rel",
        "plane_rms_m",
        "n_road_pixels",
        "depth_model_id",
        "segmentation_model",
    ):
        assert key in res.details
    assert np.isfinite([cam.pitch, cam.roll, cam.focal]).all() and cam.height > 0
    if depth_model == "depth-anything-v2":
        assert "assumed" in res.details["focal_source"]  # no intrinsics: 60 deg HFOV fallback
        assert cam.fov[0] == pytest.approx(60.0)
        assert cam.pitch > 0
    else:
        assert abs(cam.pitch - truth.pitch) < 5.0
        assert abs(cam.roll - truth.roll) < 5.0


@model_test
def test_segformer_road_mask(scene) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    rgb = np.ascontiguousarray(scene.image[..., ::-1])
    mask, info = run_or_skip(segment_road, rgb, "segformer")
    assert mask.shape == rgb.shape[:2] and mask.dtype == bool
    assert "road" in info["labels"] and "sidewalk" in info["labels"]
    assert mask.mean() > 0.2  # the rendered scene is mostly road and sidewalk


@model_test
def test_default_depth_pipeline(scene) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    res = run_or_skip(calibrate_from_depth, scene.image)
    assert res.details["depth_model"] == "depth-pro"
    assert "segformer" in res.details["segmentation_model"]
    assert res.details["focal_source"] == "depth-pro field-of-view head"
    assert abs(res.camera.pitch - scene.camera.pitch) < 5.0
    known = run_or_skip(calibrate_from_depth, scene.image, height=8.0)
    assert known.camera.height == pytest.approx(8.0)
    assert known.camera.pitch == pytest.approx(res.camera.pitch, abs=1e-6)


@model_test
def test_geocalib_on_synthetic_scene(scene) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    pytest.importorskip("geocalib")
    truth = scene.camera
    res = run_or_skip(calibrate_geocalib, scene.image)
    cam = res.camera
    assert res.method == "geocalib"
    assert cam.meta == {"calibration_method": "geocalib", "metric": False}
    assert cam.height == pytest.approx(1.0)
    assert cam.lens.is_identity
    assert 0.0 <= res.score <= 1.0
    assert abs(cam.pitch - truth.pitch) < 5.0
    assert abs(cam.roll - truth.roll) < 5.0
    assert abs(cam.fov[0] - truth.fov[0]) < 20.0
    # native GeoCalib angles: same roll, pitch negated
    assert res.details["geocalib_pitch_deg"] == pytest.approx(-cam.pitch, abs=1e-3)
    assert res.details["geocalib_roll_deg"] == pytest.approx(cam.roll, abs=1e-3)
    again = run_or_skip(calibrate_geocalib, scene.image)
    np.testing.assert_allclose(again.camera.R, cam.R, atol=1e-6)  # seeded: reproducible
    metric = run_or_skip(calibrate_geocalib, scene.image, height=8.0)
    assert metric.camera.meta["metric"] is True and metric.camera.height == pytest.approx(8.0)


@model_test
@pytest.mark.parametrize("camera_model, kind", [("simple_radial", "brown"), ("simple_divisional", "division")])
def test_geocalib_distortion_models(scene, camera_model: str, kind: str) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    pytest.importorskip("geocalib")
    res = run_or_skip(calibrate_geocalib, scene.image, camera_model=camera_model)
    assert res.details["weights"] == "distorted"
    assert isinstance(res.details["k1"], float)
    assert res.camera.lens.kind == (kind if abs(res.details["k1"]) > 1e-9 else "identity")

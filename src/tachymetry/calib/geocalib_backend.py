"""Single-image calibration with GeoCalib.

GeoCalib (Veicht, Sarlin, Lindenberger and Pollefeys, "GeoCalib: Learning
Single-image Calibration with Geometric Optimization", ECCV 2024;
https://github.com/cvg/GeoCalib) predicts dense perspective fields (per-pixel
up vectors and latitudes) with a network and fits a camera to them with a
differentiable Levenberg-Marquardt optimiser. It returns the gravity direction
in the camera frame, the focal length, optionally one radial distortion
coefficient, and their posterior uncertainties.

Gravity fixes pitch and roll. The yaw is a free choice of world frame (90
degrees: the camera looks along world ``+y``) and the metric scale is not
observable from one image, so the camera height is ``height`` when it is known
and 1 otherwise, with ``camera.meta["metric"] = False``;
:func:`tachymetry.calib.calibrate_auto` then rescales from vehicle sizes.

Conventions
-----------
GeoCalib's ``Gravity.vec3d`` is the world **up** direction expressed in the
OpenCV camera frame: its up-vector field at normalised pixel ``(u, v)`` is
``(a, b) - c (u, v)``, the image direction of the 3D vector ``(a, b, c)``. Its
angles satisfy ``vec3d = (-sin r cos p, -cos r cos p, sin p)``, whereas
tachymetry's up vector is ``(-sin roll cos pitch, -cos roll cos pitch, -sin pitch)``
(see :func:`tachymetry.calib.depth.angles_from_up_vector`). Hence::

    roll  (tachymetry) =  roll  (GeoCalib)
    pitch (tachymetry) = -pitch (GeoCalib)     # GeoCalib's pitch is positive looking up

The conversion is made from ``vec3d`` itself; the native angles are kept in
``details`` for reference.

Lens distortion
---------------
With ``camera_model="simple_radial"`` GeoCalib models
``x_d = x_u (1 + k1 |x_u|^2)`` in focal-normalised coordinates, the same
mapping as :class:`~tachymetry.geometry.lens.BrownConradyLens` with ``k1`` and
``focal = f``. With ``"simple_divisional"`` it models
``x_u = x_d / (1 + k1 |x_d|^2)``, the same mapping as
:class:`~tachymetry.geometry.lens.DivisionLens` with ``k = k1`` and
``scale = f``. Both use the principal point at the image centre, which
GeoCalib assumes as well (the half-pixel difference between its ``w / 2`` and
tachymetry's ``(w - 1) / 2`` is below 0.02 degrees and is ignored).
"""

from __future__ import annotations

import functools
import math
import time
from typing import Any

import numpy as np

from ..geometry.camera import CalibrationError, Camera
from ..geometry.lens import BrownConradyLens, DivisionLens, IdentityLens, Lens
from .depth import _import_torch, _to_rgb, angles_from_up_vector, resolve_device
from .result import CalibrationResult

__all__ = [
    "calibrate_geocalib",
    "geocalib_to_tachymetry_angles",
    "lens_from_geocalib",
    "GEOCALIB_CAMERA_MODELS",
    "clear_model_cache",
]

GEOCALIB_INSTALL_HINT = "pip install git+https://github.com/cvg/GeoCalib.git"

#: Camera models accepted by GeoCalib's optimiser, and the checkpoint recommended for each.
GEOCALIB_CAMERA_MODELS: dict[str, str] = {
    "pinhole": "pinhole",
    "simple_radial": "distorted",
    "simple_divisional": "distorted",
}

# Reference uncertainties at which each factor of the score drops to 1/e.
_SCORE_GRAVITY_DEG = 5.0
_SCORE_FOCAL_REL = 0.25


def geocalib_to_tachymetry_angles(roll_rad: float, pitch_rad: float) -> tuple[float, float]:
    """Convert GeoCalib's ``(roll, pitch)`` (radians) to tachymetry's ``(pitch, roll)`` (degrees)."""
    return -math.degrees(pitch_rad), math.degrees(roll_rad)


def lens_from_geocalib(camera_model: str, k1: float, focal: float, image_size: tuple[int, int]) -> Lens:
    """The tachymetry lens equivalent to GeoCalib's one-parameter distortion model.

    Parameters
    ----------
    camera_model:
        ``"pinhole"``, ``"simple_radial"`` or ``"simple_divisional"``.
    k1:
        GeoCalib's distortion coefficient (focal-normalised coordinates).
    focal:
        Focal length in pixels.
    image_size:
        ``(width, height)``.
    """
    if camera_model == "pinhole" or abs(k1) < 1e-9:
        return IdentityLens(image_size)
    if camera_model == "simple_radial":
        return BrownConradyLens(image_size, k1=float(k1), focal=float(focal))
    if camera_model == "simple_divisional":
        return DivisionLens(image_size, k=float(k1), scale=float(focal))
    raise ValueError(f"unknown GeoCalib camera model {camera_model!r}; choose from {sorted(GEOCALIB_CAMERA_MODELS)}")


def _import_geocalib() -> Any:
    _import_torch()
    try:
        from geocalib import GeoCalib
    except ImportError as exc:
        raise ImportError(f"the 'geocalib' calibration method needs GeoCalib: {GEOCALIB_INSTALL_HINT}") from exc
    return GeoCalib


@functools.lru_cache(maxsize=4)
def _load_geocalib(weights: str, device: str) -> Any:
    GeoCalib = _import_geocalib()
    return GeoCalib(weights=weights).to(device).eval()


def clear_model_cache() -> None:
    """Release the cached GeoCalib networks (and the GPU memory they hold)."""
    _load_geocalib.cache_clear()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _scalar(x: Any) -> float:
    return float(np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=np.float64).reshape(-1)[0])


def _unpack(out: dict[str, Any], camera_model: str) -> dict[str, Any]:
    """Plain numbers from one GeoCalib result (angles in degrees, focal in pixels)."""
    cam_gc, gravity = out["camera"], out["gravity"]
    f = float(np.mean(cam_gc.f.detach().double().cpu().numpy().reshape(-1)[:2]))
    up = gravity.vec3d.detach().double().cpu().numpy().reshape(-1)[:3]
    rp = gravity.rp.detach().double().cpu().numpy().reshape(-1)
    sig_roll = math.degrees(_scalar(out["roll_uncertainty"]))
    sig_pitch = math.degrees(_scalar(out["pitch_uncertainty"]))
    return {
        "focal": f,
        "up": up / np.linalg.norm(up),
        "native_roll": math.degrees(float(rp[0])),
        "native_pitch": math.degrees(float(rp[1])),
        "k1": _scalar(cam_gc.k1) if camera_model != "pinhole" else 0.0,
        "principal_point": cam_gc.c.detach().double().cpu().numpy().reshape(-1)[:2],
        "sig_roll": sig_roll,
        "sig_pitch": sig_pitch,
        # GeoCalib reports a zero gravity uncertainty when its eigen-decomposition fails, so the
        # marginal standard deviations bound it from below
        "sig_gravity": max(math.degrees(_scalar(out["gravity_uncertainty"])), sig_roll, sig_pitch),
        "sig_focal": _scalar(out["focal_uncertainty"]),
        "sig_vfov": math.degrees(_scalar(out["vfov_uncertainty"])),
        "up_confidence": float(out["up_confidence"].detach().float().mean()) if "up_confidence" in out else None,
        "latitude_confidence": (
            float(out["latitude_confidence"].detach().float().mean()) if "latitude_confidence" in out else None
        ),
    }


def calibrate_geocalib(
    image: np.ndarray,
    height: float | None = None,
    device: str = "auto",
    camera_model: str = "pinhole",
    focal: float | None = None,
    weights: str | None = None,
    n_runs: int = 1,
    seed: int | None = 0,
) -> CalibrationResult:
    """Calibrate focal length, pitch and roll from one image with GeoCalib.

    Parameters
    ----------
    image:
        BGR image of the scene.
    height:
        Known camera height (metres). Without it the height is 1 and
        ``camera.meta["metric"]`` is ``False``.
    device:
        Torch device or ``"auto"`` (CUDA, then MPS, then CPU).
    camera_model:
        ``"pinhole"`` (default), ``"simple_radial"`` (one-parameter polynomial
        distortion, mapped to :class:`BrownConradyLens`) or ``"simple_divisional"``
        (mapped to :class:`DivisionLens`).
    focal:
        Known focal length (pixels), used as a prior by the optimiser.
    weights:
        GeoCalib checkpoint (``"pinhole"``, ``"distorted"`` or a path); by default
        ``"pinhole"`` for the pinhole model and ``"distorted"`` otherwise, as
        recommended by the authors.
    n_runs:
        Number of forward passes to average. GeoCalib's decoder (a matrix
        decomposition "Hamburger" head) draws random bases at inference, so
        repeated runs differ slightly (about 1 % in focal length and 0.1 degrees
        in gravity); averaging reduces this and reports the spread.
    seed:
        Seed for those random bases (``None``: use the global torch RNG). The
        global RNG state is left untouched.

    Returns
    -------
    CalibrationResult
        Method ``"geocalib"`` with ``yaw = 90`` degrees and a score in ``[0, 1]``
        derived from GeoCalib's posterior uncertainties,
        ``exp(-sigma_gravity / 5 deg) * exp(-(sigma_f / f) / 0.25)``.

    Raises
    ------
    ImportError
        If PyTorch or GeoCalib is not installed.
    CalibrationError
        If the optimiser returns a non-finite camera.
    """
    if camera_model not in GEOCALIB_CAMERA_MODELS:
        raise ValueError(f"unknown camera model {camera_model!r}; choose from {sorted(GEOCALIB_CAMERA_MODELS)}")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if height is not None and not float(height) > 0:
        raise CalibrationError(f"invalid camera height {height!r}")
    torch = _import_torch()
    dev = resolve_device(device)
    weights = weights or GEOCALIB_CAMERA_MODELS[camera_model]
    t0 = time.perf_counter()
    model = _load_geocalib(weights, dev)
    t_load = time.perf_counter() - t0
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    size = (w, h)
    x = torch.from_numpy(rgb).to(dev).permute(2, 0, 1).float().div_(255.0)
    priors = None if focal is None else {"focal": torch.tensor(float(focal), dtype=torch.float32, device=dev)}

    t0 = time.perf_counter()
    tdev = torch.device(dev)
    rng_devices = [tdev.index if tdev.index is not None else torch.cuda.current_device()] if tdev.type == "cuda" else []
    runs = []
    with torch.random.fork_rng(devices=rng_devices):
        for i in range(n_runs):
            if seed is not None:
                torch.manual_seed(seed + i)
            runs.append(_unpack(model.calibrate(x, camera_model=camera_model, priors=priors), camera_model))
    t_run = time.perf_counter() - t0

    fs = np.array([r["focal"] for r in runs])
    ups = np.array([r["up"] for r in runs])
    f = float(np.mean(fs))
    up = ups.mean(axis=0)
    if not (np.isfinite(f) and f > 0 and np.all(np.isfinite(up)) and np.linalg.norm(up) > 0.5):
        raise CalibrationError("GeoCalib returned a non-finite camera")
    up /= np.linalg.norm(up)
    pitch, roll = angles_from_up_vector(up)
    k1 = float(np.mean([r["k1"] for r in runs]))
    lens = lens_from_geocalib(camera_model, k1, f, size)
    metric = height is not None
    cam = Camera.from_pose(
        size,
        focal=f,
        height=float(height) if metric else 1.0,
        pitch=pitch,
        yaw=90.0,
        roll=roll,
        lens=lens,
        meta={"calibration_method": "geocalib", "metric": metric},
    )

    def avg(key: str) -> float | None:
        vals = [r[key] for r in runs if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    sig_grav, sig_f = avg("sig_gravity"), avg("sig_focal")
    if sig_grav is not None and sig_f is not None and np.isfinite(sig_grav) and np.isfinite(sig_f):
        score = math.exp(-sig_grav / _SCORE_GRAVITY_DEG) * math.exp(-(sig_f / f) / _SCORE_FOCAL_REL)
    else:
        score = 0.0
    run_angles = np.array([angles_from_up_vector(u) for u in ups])
    details: dict[str, Any] = {
        "weights": weights,
        "camera_model": camera_model,
        "focal_px": f,
        "hfov_deg": cam.fov[0],
        "focal_source": "user prior" if focal is not None else "geocalib",
        "pitch_deg": pitch,
        "roll_deg": roll,
        "geocalib_roll_deg": avg("native_roll"),
        "geocalib_pitch_deg": avg("native_pitch"),
        "up_vector_camera": up.tolist(),
        "k1": k1 if camera_model != "pinhole" else None,
        "lens": lens.kind,
        "principal_point_geocalib": np.mean([r["principal_point"] for r in runs], axis=0).tolist(),
        "uncertainty": {
            "roll_deg": avg("sig_roll"),
            "pitch_deg": avg("sig_pitch"),
            "gravity_deg": sig_grav,
            "focal_px": sig_f,
            "vfov_deg": avg("sig_vfov"),
        },
        "mean_up_confidence": avg("up_confidence"),
        "mean_latitude_confidence": avg("latitude_confidence"),
        "runs": {
            "n": n_runs,
            "seed": seed,
            "focal_std_px": float(np.std(fs)),
            "pitch_std_deg": float(np.std(run_angles[:, 0])),
            "roll_std_deg": float(np.std(run_angles[:, 1])),
        },
        "height_source": "user" if metric else "unknown (set to 1; metric scale not observable)",
        "timings": {"load_s": t_load, "calibrate_s": t_run},
    }
    return CalibrationResult(cam, "geocalib", score=float(np.clip(score, 0.0, 1.0)), details=details)

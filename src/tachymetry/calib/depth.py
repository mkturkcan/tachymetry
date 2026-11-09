"""Calibration from monocular metric depth and semantic road segmentation.

A traffic camera looks at a road that is, to a good approximation, a plane.
Recent single-image geometry networks predict *metric* depth, and some of them
the focal length as well, so the road plane, and with it the camera pose up to
the (unobservable) yaw, can be measured from a single frame, without scene
knowledge or traffic:

1. **Depth and intrinsics.** A monocular metric depth network predicts a
   per-pixel depth ``Z(u, v)``. Depth Pro and MoGe-2 also predict the
   focal length; Depth Anything V2 needs it from the caller (or assumes a
   60 degree horizontal field of view).
2. **Road pixels.** A Cityscapes semantic segmentation network selects the
   ``road`` (and optionally ``sidewalk``) pixels. Label ids are resolved by name
   from the checkpoint's ``id2label`` table. The mask is eroded so that depth
   bleeding across kerbs, parked vehicles and poles does not leak in.
3. **Back-projection.** ``X = Z K^-1 (u, v, 1)^T`` gives a metric point cloud
   in the OpenCV camera frame.
4. **Robust plane fit.** MSAC (RANSAC with a truncated quadratic cost) on the
   *relative depth residual* ``r = (n . X + d) / d = 1 - Z / Z_plane``: the
   fractional disagreement between the measured depth and the depth of the
   plane along the same pixel ray. Monocular depth errors are roughly
   proportional to depth, so this residual is approximately homoscedastic and
   the threshold does not depend on the mounting height. The winning hypothesis is
   refined by iterated linear least squares on the inliers (equivalently, a
   weighted regression of inverse depth on the pixel rays).
5. **Pose.** The unit plane normal, oriented towards the camera, is the world
   up vector in camera coordinates and fixes pitch and roll; the plane offset
   ``d`` is the camera height in metres. The yaw is a free choice of world
   frame (``90`` degrees: the camera looks along world ``+y``).

Pitch and roll depend only on the *shape* of the depth map, not on its scale,
but they are coupled to the focal length: if the focal length is ``s`` times
the true one, ``tan(pitch)`` comes out ``1/s`` times the true value. The height
inherits the absolute scale of the depth network, which can be wrong by a large
factor for viewpoints unlike the training data (high-mounted views can be read
as miniature scenes; a warning is recorded when the height is implausible).
Pass ``height`` when it is known, or let :func:`tachymetry.calib.calibrate_auto`
rescale from vehicle sizes.

Back-ends (imported lazily; ``pip install "tachymetry[depth]"``)
----------------------------------------------------------------
``depth-pro``          Apple Depth Pro (Bochkovskii et al., ICLR 2025) via
                       Hugging Face ``transformers`` (``apple/DepthPro-hf``):
                       metric depth plus a field-of-view head.
``moge2``              MoGe-2 (Wang et al., 2025) via the ``moge`` package
                       (``Ruicheng/moge-2-vitl-normal``): metric point map and
                       intrinsics.
``depth-anything-v2``  Depth Anything V2, metric outdoor fine-tune (Yang et
                       al., NeurIPS 2024), via ``transformers``; no intrinsics.
``segformer``          SegFormer-B2 fine-tuned on Cityscapes (Xie et al.,
                       NeurIPS 2021) via ``transformers``.

Other checkpoints can be named as ``"<backend>:<model id>"`` (for example
``"moge2:Ruicheng/moge-2-vitb-normal"``), and ``segmentation`` also accepts the
Hugging Face id of a semantic segmentation model with Cityscapes-style labels,
or a precomputed boolean road mask.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry.camera import CalibrationError, Camera, rotation_from_angles
from ..geometry.lens import IdentityLens
from .result import CalibrationResult

__all__ = [
    "calibrate_from_depth",
    "calibrate_from_depth_map",
    "predict_depth",
    "segment_road",
    "fit_ground_plane",
    "backproject",
    "angles_from_up_vector",
    "up_vector_from_angles",
    "resolve_device",
    "clear_model_cache",
    "GroundPlane",
    "DepthPrediction",
    "DEPTH_MODELS",
    "SEGMENTATION_MODELS",
    "DEFAULT_HFOV_DEG",
    "PLAUSIBLE_HEIGHT_M",
]

INSTALL_HINT = 'pip install "tachymetry[depth]"'
MOGE_INSTALL_HINT = "pip install git+https://github.com/microsoft/MoGe.git"

#: Depth back-ends and their default checkpoints.
DEPTH_MODELS: dict[str, str] = {
    "depth-pro": "apple/DepthPro-hf",
    "moge2": "Ruicheng/moge-2-vitl-normal",
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}
_DEPTH_ALIASES = {
    "depthpro": "depth-pro",
    "depth_pro": "depth-pro",
    "apple-depth-pro": "depth-pro",
    "moge": "moge2",
    "moge-2": "moge2",
    "moge_2": "moge2",
    "depth-anything": "depth-anything-v2",
    "depth_anything_v2": "depth-anything-v2",
    "dav2": "depth-anything-v2",
    "depth-anything-2": "depth-anything-v2",
}

#: Road segmentation back-ends and their default checkpoints (Cityscapes label set).
SEGMENTATION_MODELS: dict[str, str] = {
    "segformer": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "segformer-b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

#: Horizontal field of view assumed when neither the back-end nor the caller provides a focal length.
DEFAULT_HFOV_DEG = 60.0

_MIN_ROAD_POINTS = 200
#: Camera heights (metres) outside this range trigger a warning about the depth network's metric scale.
PLAUSIBLE_HEIGHT_M = (2.5, 150.0)


# --------------------------------------------------------------------------- data containers
@dataclass
class DepthPrediction:
    """Output of a monocular depth back-end.

    Attributes
    ----------
    depth:
        ``(H, W)`` metric z-depth (metres) at the input resolution; ``NaN``
        where the back-end reports no valid depth (e.g. sky).
    focal:
        Focal length (pixels, square pixels) consistent with ``depth``.
    model:
        Back-end name (a key of :data:`DEPTH_MODELS`).
    model_id:
        Checkpoint that produced the prediction.
    info:
        Back-end diagnostics, including ``focal_source``.
    """

    depth: np.ndarray
    focal: float
    model: str
    model_id: str
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundPlane:
    """A plane ``n . X + d = 0`` fitted to camera-frame points.

    Attributes
    ----------
    normal:
        Unit normal in the camera frame, oriented from the plane towards the
        camera. For the road plane this is the world up vector.
    distance:
        Distance ``d > 0`` of the camera centre from the plane (units of the
        points; metres for metric depth), i.e. the camera height.
    inliers:
        Boolean inlier mask over the input points.
    rms:
        RMS relative depth residual of the inliers (dimensionless).
    threshold:
        Inlier threshold on the relative residual.
    iterations:
        Number of RANSAC hypotheses evaluated.
    """

    normal: np.ndarray
    distance: float
    inliers: np.ndarray
    rms: float
    threshold: float
    iterations: int

    @property
    def n_points(self) -> int:
        return int(len(self.inliers))

    @property
    def inlier_ratio(self) -> float:
        return float(np.mean(self.inliers)) if len(self.inliers) else 0.0

    @property
    def rms_m(self) -> float:
        """RMS point-to-plane distance of the inliers (units of the points)."""
        return float(self.rms * self.distance)

    def residuals(self, points: np.ndarray) -> np.ndarray:
        """Relative depth residuals ``(n . X + d) / d`` of ``(N, 3)`` points."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return P @ self.normal / self.distance + 1.0

    def angles(self) -> tuple[float, float]:
        """``(pitch, roll)`` in degrees of a camera whose ground plane this is."""
        return angles_from_up_vector(self.normal)


# --------------------------------------------------------------------------- geometry (pure numpy)
def angles_from_up_vector(up_cam: np.ndarray) -> tuple[float, float]:
    """Pitch and roll (degrees, tachymetry conventions) from the world up vector in the camera frame.

    For a camera built by :func:`~tachymetry.geometry.camera.rotation_from_angles`,
    the world up vector expressed in the OpenCV camera frame is
    ``(-sin(roll) cos(pitch), -cos(roll) cos(pitch), -sin(pitch))``, independent
    of yaw; this function inverts that relation.

    Parameters
    ----------
    up_cam:
        3-vector of positive length (normalised internally).

    Returns
    -------
    tuple of float
        ``(pitch, roll)``; pitch is positive when the camera looks down.
    """
    u = np.asarray(up_cam, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(u))
    if not np.isfinite(norm) or norm < 1e-12:
        raise CalibrationError("degenerate up vector")
    u = u / norm
    pitch = math.degrees(math.asin(float(np.clip(-u[2], -1.0, 1.0))))
    roll = math.degrees(math.atan2(-u[0], -u[1]))
    return pitch, roll


def up_vector_from_angles(pitch: float, roll: float = 0.0) -> np.ndarray:
    """World up vector in the OpenCV camera frame for the given pitch and roll (degrees)."""
    return rotation_from_angles(90.0, pitch, roll) @ np.array([0.0, 0.0, 1.0])


def backproject(
    depth: np.ndarray, focal: float, principal_point: tuple[float, float] | None = None, mask: np.ndarray | None = None
) -> np.ndarray:
    """Back-project a z-depth map to camera-frame points.

    Parameters
    ----------
    depth:
        ``(H, W)`` z-depth.
    focal:
        Focal length in pixels (square pixels).
    principal_point:
        ``(cx, cy)`` in pixels; the image centre ``((W-1)/2, (H-1)/2)`` by default,
        matching :meth:`Camera.from_pose`.
    mask:
        Optional boolean ``(H, W)`` selection; non-finite and non-positive depths are dropped.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` points in row-major pixel order of the selection.
    """
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    cx, cy = principal_point if principal_point is not None else ((w - 1) / 2.0, (h - 1) / 2.0)
    sel = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    v, u = np.nonzero(sel)
    z = depth[v, u]
    return np.column_stack([(u - cx) * z / focal, (v - cy) * z / focal, z])


def _plane_from_m(m: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert the ``m . X + 1 = 0`` parametrisation (``m = n / d``) to ``(n, d)``."""
    inv_d = float(np.linalg.norm(m))
    return m / inv_d, 1.0 / inv_d


def fit_ground_plane(
    points: np.ndarray,
    threshold: float = 0.03,
    max_iterations: int = 2000,
    confidence: float = 0.999,
    eval_points: int = 20000,
    refine_iterations: int = 20,
    seed: int = 0,
) -> GroundPlane:
    """Robust plane fit with residuals measured as relative depth errors.

    The plane is parametrised as ``m . X + 1 = 0`` with ``m = n / d``, which
    excludes planes through the camera centre and makes the residual
    ``r = m . X + 1 = (n . X + d) / d = 1 - Z / Z_plane`` linear in ``m``.

    Hypotheses from random minimal samples are scored with MSAC (truncated
    quadratic cost) on a random subset of at most ``eval_points`` points, with
    the usual adaptive stopping rule. The best hypothesis is then refined on all
    points by alternating inlier selection and linear least squares on ``r``
    (local optimisation), which converges in a few iterations.

    Parameters
    ----------
    points:
        ``(N, 3)`` camera-frame points (camera centre at the origin).
    threshold:
        Inlier threshold on ``|r|`` (``0.03`` = depth within 3 % of the plane).
    max_iterations:
        Upper bound on the number of hypotheses.
    confidence:
        Probability of drawing at least one all-inlier sample (adaptive stopping).
    eval_points:
        Size of the random subset used to score hypotheses.
    refine_iterations:
        Maximum number of least-squares refinement rounds.
    seed:
        Random seed (the fit is deterministic for a given seed).

    Returns
    -------
    GroundPlane
        The plane, oriented so that the camera lies on the side of the normal.

    Raises
    ------
    CalibrationError
        If there are too few points or no plane explains them.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    n = len(P)
    if n < 3:
        raise CalibrationError(f"plane fit needs at least 3 points, got {n}")
    rng = np.random.default_rng(seed)
    S = P if n <= eval_points else P[rng.choice(n, eval_points, replace=False)]
    t2 = threshold * threshold
    best_m: np.ndarray | None = None
    best_cost = np.inf
    needed, done, batch = max_iterations, 0, 128
    min_iterations = min(100, max_iterations)
    while done < needed:
        idx = rng.integers(0, len(S), size=(batch, 3))
        A, B, C = S[idx[:, 0]], S[idx[:, 1]], S[idx[:, 2]]
        nrm = np.cross(B - A, C - A)
        den = np.einsum("ij,ij->i", nrm, A)  # n . A = -d (unnormalised)
        scale = np.linalg.norm(nrm, axis=1) * np.linalg.norm(A, axis=1)
        ok = np.abs(den) > 1e-9 * np.maximum(scale, 1e-300)  # reject collinear samples and planes through the camera
        done += batch
        if not np.any(ok):
            continue
        M = -nrm[ok] / den[ok, None]  # (k, 3): m = n / d
        R = S @ M.T + 1.0
        cost = np.minimum(R * R, t2).sum(axis=0)
        j = int(np.argmin(cost))
        if cost[j] < best_cost:
            best_cost, best_m = float(cost[j]), M[j]
            w = float(np.mean(np.abs(R[:, j]) < threshold))
            if 0.0 < w < 1.0:
                k = math.log(1.0 - confidence) / math.log(max(1.0 - w**3, 1e-12))
                needed = int(np.clip(math.ceil(k), min_iterations, max_iterations))
            elif w >= 1.0:
                needed = min(needed, max(done, min_iterations))
    if best_m is None:
        raise CalibrationError("plane fit failed: no non-degenerate minimal sample was drawn")

    m = best_m
    inl = np.abs(P @ m + 1.0) < threshold
    for _ in range(refine_iterations):
        if inl.sum() < 3:
            break
        m_new, *_ = np.linalg.lstsq(P[inl], -np.ones(int(inl.sum())), rcond=None)
        inl_new = np.abs(P @ m_new + 1.0) < threshold
        converged = np.array_equal(inl_new, inl)
        m, inl = m_new, inl_new
        if converged:
            break
    if inl.sum() < 3:
        raise CalibrationError("plane fit failed: no plane is supported by the points")
    r = P @ m + 1.0
    normal, distance = _plane_from_m(m)
    rms = float(np.sqrt(np.mean(r[inl] ** 2)))
    return GroundPlane(normal=normal, distance=distance, inliers=inl, rms=rms, threshold=threshold, iterations=done)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion with a disc of ``radius`` pixels; the image border counts as background."""
    if radius <= 0:
        return mask
    import cv2

    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    padded = np.pad(mask.astype(np.uint8), radius, constant_values=0)
    return cv2.erode(padded, kernel)[radius:-radius, radius:-radius].astype(bool)


def calibrate_from_depth_map(
    depth: np.ndarray,
    focal: float,
    mask: np.ndarray | None = None,
    height: float | None = None,
    principal_point: tuple[float, float] | None = None,
    threshold: float = 0.03,
    erode_px: int | None = None,
    max_points: int = 300_000,
    seed: int = 0,
) -> CalibrationResult:
    """Calibrate from a metric depth map, a focal length and a road mask (no networks involved).

    Parameters
    ----------
    depth:
        ``(H, W)`` metric z-depth in metres (non-finite or non-positive = invalid).
    focal:
        Focal length in pixels consistent with ``depth``.
    mask:
        Boolean ``(H, W)`` road mask; ``None`` uses the valid pixels of the whole image.
    height:
        Known camera height (metres); overrides the height measured from depth.
    principal_point:
        ``(cx, cy)``; the image centre by default.
    threshold:
        Inlier threshold on the relative depth residual (see :func:`fit_ground_plane`).
    erode_px:
        Erosion radius applied to ``mask``; default 0.4 % of the image diagonal.
    max_points:
        Random subsample size for the fit (speed).
    seed:
        Random seed.

    Returns
    -------
    CalibrationResult
        Method ``"depth"``; ``score = inlier_ratio * exp(-rms / threshold)``.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W), got shape {depth.shape}")
    if not (np.isfinite(focal) and focal > 0):
        raise CalibrationError(f"invalid focal length {focal!r}")
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    if mask is None:
        road = valid
        radius = 0
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"mask shape {mask.shape} does not match depth shape {depth.shape}")
        radius = int(round(0.004 * math.hypot(w, h))) if erode_px is None else int(erode_px)
        road = _erode(mask, radius) & valid
    n_road = int(road.sum())
    if n_road < _MIN_ROAD_POINTS:
        raise CalibrationError(
            f"only {n_road} road pixels with valid depth (need {_MIN_ROAD_POINTS}); is the road visible?"
        )
    rng = np.random.default_rng(seed)
    v, u = np.nonzero(road)
    if n_road > max_points:
        keep = rng.choice(n_road, max_points, replace=False)
        v, u = v[keep], u[keep]
    sub = np.zeros_like(road)
    sub[v, u] = True
    pts = backproject(depth, focal, principal_point, sub)
    plane = fit_ground_plane(pts, threshold=threshold, seed=seed)
    pitch, roll = plane.angles()
    cam_height = float(height) if height is not None else plane.distance
    if not cam_height > 0:
        raise CalibrationError(f"invalid camera height {cam_height!r}")
    cam = Camera.from_pose(
        (w, h),
        focal=float(focal),
        height=cam_height,
        pitch=pitch,
        yaw=90.0,
        roll=roll,
        principal_point=principal_point,
        lens=IdentityLens((w, h)),
        meta={"calibration_method": "depth", "metric": True},
    )
    score = float(np.clip(plane.inlier_ratio * math.exp(-plane.rms / threshold), 0.0, 1.0))
    zr = pts[:, 2]
    details: dict[str, Any] = {
        "focal_px": float(focal),
        "hfov_deg": float(math.degrees(2.0 * math.atan(w / (2.0 * focal)))),
        "pitch_deg": pitch,
        "roll_deg": roll,
        "plane_normal_camera": plane.normal.tolist(),
        "depth_height_m": plane.distance,
        "height_source": "user" if height is not None else "depth",
        "inlier_ratio": plane.inlier_ratio,
        "plane_rms_rel": plane.rms,
        "plane_rms_m": plane.rms_m,
        "threshold_rel": threshold,
        "ransac_iterations": plane.iterations,
        "n_road_pixels": n_road,
        "n_points_fitted": plane.n_points,
        "erode_px": radius,
        "road_depth_m": {
            "p5": float(np.percentile(zr, 5)),
            "median": float(np.median(zr)),
            "p95": float(np.percentile(zr, 95)),
        },
        "warnings": [],
    }
    lo, hi = PLAUSIBLE_HEIGHT_M
    if height is None and not lo <= plane.distance <= hi:
        # Pitch and roll depend only on the shape of the depth map, the height on its absolute scale,
        # which monocular networks can get badly wrong for unusual viewpoints (high mounts look like
        # miniature scenes).
        details["warnings"].append(
            f"camera height {plane.distance:.2f} m from depth is outside the plausible range {lo:g}-{hi:g} m: the "
            "depth network's metric scale is probably wrong for this view; pass height= or rescale (e.g. from "
            "vehicle sizes). Pitch and roll do not depend on the scale."
        )
    return CalibrationResult(cam, "depth", score=score, details=details)


# --------------------------------------------------------------------------- network back-ends
def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"this calibration method needs PyTorch; install it with: {INSTALL_HINT}") from exc
    return torch


def _import_transformers() -> Any:
    _import_torch()
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            f"this calibration method needs Hugging Face transformers; install it with: {INSTALL_HINT}"
        ) from exc
    return transformers


def resolve_device(device: str = "auto") -> str:
    """``"auto"`` becomes ``"cuda"``, ``"mps"`` or ``"cpu"`` (first available); other values pass through."""
    if device != "auto":
        return device
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _parse_spec(spec: str, table: dict[str, str], aliases: dict[str, str], what: str) -> tuple[str, str]:
    """Resolve ``"name"`` or ``"name:checkpoint"`` to ``(name, checkpoint)``."""
    name, _, ckpt = str(spec).partition(":")
    key = name.strip().lower().replace("_", "-")
    key = aliases.get(key, aliases.get(name.strip().lower(), key))
    if key not in table:
        raise ValueError(f"unknown {what} {spec!r}; choose from {sorted(table)} or '<name>:<checkpoint>'")
    return key, (ckpt.strip() or table[key])


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) image to a contiguous ``uint8`` RGB array."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"expected an (H, W, 3) BGR image, got shape {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img[..., 2::-1])


@functools.lru_cache(maxsize=8)
def _load_hf(task: str, model_id: str, device: str) -> tuple[Any, Any]:
    transformers = _import_transformers()
    processor = transformers.AutoImageProcessor.from_pretrained(model_id)
    if task == "depth":
        net = transformers.AutoModelForDepthEstimation.from_pretrained(model_id)
    elif task == "segmentation":
        net = transformers.AutoModelForSemanticSegmentation.from_pretrained(model_id)
    else:  # pragma: no cover - internal
        raise ValueError(task)
    # float32 regardless of the checkpoint's storage dtype: transformers 5 loads checkpoints in their
    # stored precision (float16 for Depth Pro), 4.x upcasts; this keeps results version-independent
    return processor, net.float().to(device).eval()


@functools.lru_cache(maxsize=4)
def _load_moge(model_id: str, device: str) -> Any:
    _import_torch()
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise ImportError(
            f"the 'moge2' depth back-end needs the MoGe package: {MOGE_INSTALL_HINT} (and {INSTALL_HINT})"
        ) from exc
    return MoGeModel.from_pretrained(model_id).to(device).eval()


def clear_model_cache() -> None:
    """Release the cached depth and segmentation networks (and the GPU memory they hold)."""
    _load_hf.cache_clear()
    _load_moge.cache_clear()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _focal_from_hfov(width: int, hfov_deg: float) -> float:
    return 0.5 * width / math.tan(math.radians(0.5 * hfov_deg))


def _depth_pro(rgb: np.ndarray, model_id: str, device: str, focal: float | None) -> DepthPrediction:
    torch = _import_torch()
    if not hasattr(_import_transformers(), "DepthProForDepthEstimation"):
        raise ImportError('the depth-pro back-end needs transformers >= 4.49: pip install -U "transformers>=4.49"')
    processor, net = _load_hf("depth", model_id, device)
    h, w = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = net(**inputs)
    # Depth Pro predicts a canonical inverse depth ``f_px / (W Z)`` and, from a separate
    # field-of-view head, the horizontal FOV; metric depth is ``f_px / (W * canonical)``.
    canonical = torch.nn.functional.interpolate(
        out.predicted_depth.float()[:, None], size=(h, w), mode="bilinear", align_corners=False
    )[0, 0]
    info: dict[str, Any] = {}
    f_pred = None
    if getattr(out, "field_of_view", None) is not None:
        hfov = float(out.field_of_view.reshape(-1)[0])
        f_pred = _focal_from_hfov(w, hfov)
        info.update(predicted_hfov_deg=hfov, predicted_focal_px=f_pred)
    if focal is not None:
        f_use, info["focal_source"] = float(focal), "user"
    elif f_pred is not None:
        f_use, info["focal_source"] = f_pred, "depth-pro field-of-view head"
    else:
        f_use = _focal_from_hfov(w, DEFAULT_HFOV_DEG)
        info["focal_source"] = f"assumed {DEFAULT_HFOV_DEG:g} deg horizontal field of view (checkpoint has no FOV head)"
    depth = (f_use / (w * canonical.clamp(min=1e-6))).cpu().numpy().astype(np.float64)
    return DepthPrediction(depth, f_use, "depth-pro", model_id, info)


def _depth_anything(rgb: np.ndarray, model_id: str, device: str, focal: float | None) -> DepthPrediction:
    torch = _import_torch()
    processor, net = _load_hf("depth", model_id, device)
    h, w = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = net(**inputs)
    depth = (
        torch.nn.functional.interpolate(
            out.predicted_depth.float()[:, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    info: dict[str, Any] = {"max_depth_m": getattr(net.config, "max_depth", None)}
    if focal is not None:
        f_use, info["focal_source"] = float(focal), "user"
    else:
        f_use = _focal_from_hfov(w, DEFAULT_HFOV_DEG)
        info["focal_source"] = (
            f"assumed {DEFAULT_HFOV_DEG:g} deg horizontal field of view "
            "(Depth Anything V2 does not predict intrinsics; pass focal=)"
        )
    return DepthPrediction(depth, f_use, "depth-anything-v2", model_id, info)


def _moge2(rgb: np.ndarray, model_id: str, device: str, focal: float | None) -> DepthPrediction:
    torch = _import_torch()
    net = _load_moge(model_id, device)
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().div_(255.0)
    kwargs: dict[str, Any] = {}
    if focal is not None:
        kwargs["fov_x"] = math.degrees(2.0 * math.atan(w / (2.0 * float(focal))))
    with torch.inference_mode():
        out = net.infer(x, use_fp16=str(device).startswith("cuda"), **kwargs)
    depth = out["depth"].float().cpu().numpy().astype(np.float64)
    if "mask" in out:
        depth[~out["mask"].cpu().numpy().astype(bool)] = np.nan
    K = out["intrinsics"].float().cpu().numpy()  # normalised: fx / W, fy / H, principal point (0.5, 0.5)
    fx, fy = float(K[0, 0]) * w, float(K[1, 1]) * h
    info: dict[str, Any] = {"predicted_focal_px": [fx, fy], "principal_point_norm": [float(K[0, 2]), float(K[1, 2])]}
    if focal is not None:
        f_use, info["focal_source"] = float(focal), "user (MoGe-2 field-of-view prior)"
    else:
        f_use, info["focal_source"] = math.sqrt(fx * fy), "moge2 intrinsics"
    return DepthPrediction(depth, f_use, "moge2", model_id, info)


_DEPTH_BACKENDS = {"depth-pro": _depth_pro, "moge2": _moge2, "depth-anything-v2": _depth_anything}


def predict_depth(
    image_rgb: np.ndarray, model: str = "depth-pro", device: str = "auto", focal: float | None = None
) -> DepthPrediction:
    """Metric depth (and focal length, when the back-end predicts it) for one RGB image.

    Parameters
    ----------
    image_rgb:
        ``(H, W, 3)`` ``uint8`` RGB image.
    model:
        A key of :data:`DEPTH_MODELS` or ``"<key>:<checkpoint>"``.
    device:
        Torch device or ``"auto"``.
    focal:
        Known focal length (pixels). Overrides the predicted intrinsics: Depth Pro
        rescales its depth accordingly and MoGe-2 uses it as a field-of-view prior.

    Returns
    -------
    DepthPrediction
    """
    key, model_id = _parse_spec(model, DEPTH_MODELS, _DEPTH_ALIASES, "depth model")
    rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    return _DEPTH_BACKENDS[key](rgb, model_id, resolve_device(device), focal)


def _norm_label(name: str) -> str:
    return str(name).split("--")[-1].strip().lower().replace("_", " ").replace("-", " ")


def _label_ids(id2label: dict[int, str], names: tuple[str, ...]) -> list[int]:
    wanted = {_norm_label(n) for n in names}
    return sorted(i for i, n in id2label.items() if _norm_label(n) in wanted)


def segment_road(
    image_rgb: np.ndarray,
    model: str = "segformer",
    device: str = "auto",
    include_sidewalk: bool = True,
    short_side: int = 1024,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Road (and sidewalk) mask from a Cityscapes semantic segmentation network.

    Parameters
    ----------
    image_rgb:
        ``(H, W, 3)`` ``uint8`` RGB image.
    model:
        A key of :data:`SEGMENTATION_MODELS`, ``"<key>:<checkpoint>"``, or the
        Hugging Face id of a semantic segmentation checkpoint whose
        ``id2label`` contains ``road`` (and ``sidewalk``).
    device:
        Torch device or ``"auto"``.
    include_sidewalk:
        Also keep ``sidewalk`` pixels (kerb heights are within the plane threshold).
    short_side:
        Inference resolution of the short image side (Cityscapes frames are 1024 px tall).

    Returns
    -------
    tuple
        ``(mask, info)``: boolean ``(H, W)`` mask and diagnostics.
    """
    if "/" in str(model) and ":" not in str(model):
        model_id = str(model)
    else:
        _, model_id = _parse_spec(model, SEGMENTATION_MODELS, {}, "segmentation model")
    torch = _import_torch()
    dev = resolve_device(device)
    processor, net = _load_hf("segmentation", model_id, dev)
    rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    s = short_side / min(h, w)
    size = {"height": max(32, int(round(h * s / 32.0)) * 32), "width": max(32, int(round(w * s / 32.0)) * 32)}
    inputs = processor(images=rgb, size=size, return_tensors="pt").to(dev)
    with torch.inference_mode():
        logits = net(**inputs).logits.float()
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        labels = logits.argmax(dim=1)[0].cpu().numpy()
    id2label = {int(k): str(v) for k, v in net.config.id2label.items()}
    road_ids = _label_ids(id2label, ("road",))
    if not road_ids:
        raise CalibrationError(
            f"segmentation model {model_id} has no 'road' class (labels: {sorted(set(id2label.values()))[:12]} ...)"
        )
    side_ids = _label_ids(id2label, ("sidewalk",)) if include_sidewalk else []
    is_road = np.isin(labels, road_ids)
    is_side = np.isin(labels, side_ids) if side_ids else np.zeros_like(is_road)
    info = {
        "segmentation_model": model_id,
        "labels": {id2label[i]: i for i in road_ids + side_ids},
        "n_road_label_pixels": int(is_road.sum()),
        "n_sidewalk_label_pixels": int(is_side.sum()),
        "inference_size": [size["width"], size["height"]],
    }
    return is_road | is_side, info


# --------------------------------------------------------------------------- public entry point
def calibrate_from_depth(
    image: np.ndarray,
    depth_model: str = "depth-pro",
    segmentation: str | np.ndarray = "segformer",
    device: str = "auto",
    height: float | None = None,
    focal: float | None = None,
    include_sidewalk: bool = True,
    threshold: float = 0.03,
    seed: int = 0,
) -> CalibrationResult:
    """Calibrate a camera from one frame with monocular metric depth and road segmentation.

    Parameters
    ----------
    image:
        BGR image of the scene (a traffic-free background works best).
    depth_model:
        ``"depth-pro"`` (default), ``"moge2"`` or ``"depth-anything-v2"``, optionally
        ``"<name>:<checkpoint>"`` (see :data:`DEPTH_MODELS`).
    segmentation:
        ``"segformer"`` (default), another key of :data:`SEGMENTATION_MODELS`, a Hugging
        Face checkpoint id, or a precomputed boolean ``(H, W)`` road mask.
    device:
        Torch device or ``"auto"`` (CUDA, then MPS, then CPU).
    height:
        Known camera height (metres); overrides the height measured from depth.
    focal:
        Known focal length (pixels); overrides the predicted intrinsics. Without it,
        ``depth-anything-v2`` assumes a 60 degree horizontal field of view.
    include_sidewalk:
        Fit the plane to sidewalk pixels as well as road pixels.
    threshold:
        Inlier threshold on the relative depth residual (see :func:`fit_ground_plane`).
    seed:
        Random seed of the robust fit.

    Returns
    -------
    CalibrationResult
        Method ``"depth"``: a metric camera (``meta["metric"] = True``) with
        ``yaw = 90`` degrees, an identity lens, a score in ``[0, 1]`` from the
        inlier ratio and plane residual, and diagnostics in ``details``.

    Raises
    ------
    ImportError
        If PyTorch/transformers (or ``moge`` for ``moge2``) are not installed.
    CalibrationError
        If too few road pixels are found or no plane explains them.
    """
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    pred = predict_depth(rgb, depth_model, device=device, focal=focal)
    timings["depth_s"] = time.perf_counter() - t0
    if isinstance(segmentation, np.ndarray):
        mask = np.asarray(segmentation, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"road mask shape {mask.shape} does not match the image {(h, w)}")
        seg_info: dict[str, Any] = {"segmentation_model": "user mask"}
    else:
        t0 = time.perf_counter()
        mask, seg_info = segment_road(rgb, segmentation, device=device, include_sidewalk=include_sidewalk)
        timings["segmentation_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = calibrate_from_depth_map(pred.depth, pred.focal, mask, height=height, threshold=threshold, seed=seed)
    timings["plane_fit_s"] = time.perf_counter() - t0
    result.details.update(
        {
            "depth_model": pred.model,
            "depth_model_id": pred.model_id,
            "focal_source": pred.info.get("focal_source"),
            "depth_info": {k: v for k, v in pred.info.items() if k != "focal_source"},
            **seg_info,
            "include_sidewalk": include_sidewalk,
            "timings": timings,
        }
    )
    result.camera.meta.update(
        {
            "calibration_method": "depth",
            "metric": True,
            "depth_model": pred.model_id,
            "segmentation_model": seg_info["segmentation_model"],
        }
    )
    return result

"""Manual calibration from point (and line) correspondences.

The estimator follows three stages:

1. **Initialisation.** A focal-length sweep with planar PnP (IPPE) on the
   rectified points gives a physically valid camera for each candidate focal
   length; the best one by reprojection error seeds the refinement.
2. **Physical refinement.** Levenberg–Marquardt over the *physical* camera
   (focal length, yaw, pitch, roll, position, height) plus the lens parameters,
   minimising robust reprojection error in pixels. Polylines that are straight
   in the world contribute straightness residuals, each normalised by
   ``sqrt(2 / n_vertices)`` so a densely clicked line counts as one line.
   Staying on the camera manifold (7 DoF instead of a free 8-DoF homography)
   is what keeps extrapolation sane away from the annotated area.
3. **Model selection.** Cross-validation in ground metres (10-fold by default,
   leave-one-out on request) picks the lens model; a more complex model must
   beat a simpler one by 3 %.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..geometry.camera import CalibrationError, Camera
from ..geometry.lens import Lens, make_lens
from .correspondences import Correspondences
from .result import CalibrationResult

__all__ = ["calibrate_manual", "fit_camera"]

_LENS_ORDER = ("identity", "division", "equidistant", "brown")


def _camera_from_params(
    p: np.ndarray, size: tuple[int, int], lens: Lens, pp: tuple[float, float] | None, n_lens: int
) -> Camera:
    logf, yaw, pitch, roll, cx, cy, logh = p[:7]
    lens_p = lens.with_params(p[7 : 7 + n_lens]) if n_lens else lens
    return Camera.from_pose(
        size,
        focal=math.exp(logf),
        height=math.exp(logh),
        pitch=math.degrees(pitch),
        yaw=math.degrees(yaw),
        roll=math.degrees(roll),
        position=(cx, cy),
        principal_point=pp,
        lens=lens_p,
    )


def _params_from_camera(cam: Camera, lens_params: np.ndarray) -> np.ndarray:
    yaw, pitch, roll = cam.angles
    C = cam.center
    return np.concatenate(
        [
            [
                math.log(cam.focal),
                math.radians(yaw),
                math.radians(pitch),
                math.radians(roll),
                C[0],
                C[1],
                math.log(max(C[2], 1e-3)),
            ],
            lens_params,
        ]
    )


def _line_residuals(cam: Camera, lines: Sequence[np.ndarray]) -> np.ndarray:
    out = []
    for poly in lines:
        if len(poly) < 3:
            continue
        r = cam.lens.rectify(poly)
        c = r.mean(axis=0)
        _, _, vt = np.linalg.svd(r - c)
        nrm = vt[-1]
        d = (r - c) @ nrm
        out.append(d * math.sqrt(2.0 / len(poly)))
    return np.concatenate(out) if out else np.zeros(0)


def _init_pnp(
    image_rect: np.ndarray, world: np.ndarray, size: tuple[int, int], pp: tuple[float, float] | None
) -> Camera:
    """Sweep the focal length with planar PnP and keep the best camera."""
    import cv2

    w, h = size
    cx, cy = pp if pp is not None else ((w - 1) / 2.0, (h - 1) / 2.0)
    obj = np.column_stack([world, np.zeros(len(world))]).astype(np.float64)
    best: tuple[float, Camera] | None = None
    for hfov in np.linspace(20, 140, 49):
        f = (w / 2) / math.tan(math.radians(hfov / 2))
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        flags = cv2.SOLVEPNP_IPPE if len(obj) >= 4 else cv2.SOLVEPNP_ITERATIVE
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, image_rect.astype(np.float64), K, None, flags=flags)
        except cv2.error:
            continue
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        cam = Camera(size, K, R, tvec.ravel())
        if cam.height <= 0:
            continue
        proj = cam.ground_to_image(world, distort=False)
        if not np.all(np.isfinite(proj)):
            continue
        err = float(np.sqrt(np.mean(np.sum((proj - image_rect) ** 2, axis=1))))
        if best is None or err < best[0]:
            best = (err, cam)
    if best is None:
        raise CalibrationError("could not initialise a camera from the correspondences (are they degenerate?)")
    return best[1]


def fit_camera(
    corr: Correspondences,
    image_size: tuple[int, int],
    lens: str | Lens = "identity",
    principal_point: tuple[float, float] | None = None,
    robust_px: float = 3.0,
    init: Camera | None = None,
    line_weight: float = 1.0,
) -> tuple[Camera, dict[str, Any]]:
    """Fit a physical camera (and lens) to correspondences. Returns ``(camera, info)``."""
    from scipy.optimize import least_squares

    size = (int(image_size[0]), int(image_size[1]))
    if isinstance(lens, str):
        lens_obj = make_lens(lens if lens != "none" else "identity", size)
    else:
        lens_obj = lens
    free_lens = isinstance(lens, str) and lens not in ("identity", "none")
    n_lens = lens_obj.n_params if free_lens else 0
    if len(corr) < 4:
        raise CalibrationError(f"need at least 4 point correspondences, got {len(corr)}")
    if n_lens and len(corr) + 2 * len(corr.lines) < 6 + n_lens:
        n_lens = 0  # not enough constraints to estimate distortion

    lo_l, hi_l = lens_obj.param_bounds() if n_lens else (np.zeros(0), np.zeros(0))
    lo = np.concatenate([np.full(7, -np.inf), lo_l[:n_lens]])
    hi = np.concatenate([np.full(7, np.inf), hi_l[:n_lens]])

    def resid(p: np.ndarray, n_free: int = n_lens, base_lens: Lens = lens_obj) -> np.ndarray:
        cam = _camera_from_params(p, size, base_lens, principal_point, n_free)
        proj = cam.ground_to_image(corr.world)
        r = proj - corr.image
        r = np.where(np.isfinite(r), r, 1e4)
        parts = [r.ravel()]
        if corr.lines and line_weight > 0:
            parts.append(line_weight * _line_residuals(cam, corr.lines))
        return np.concatenate(parts)

    def solve(p0: np.ndarray, n_free: int, base_lens: Lens, nfev: int) -> Any:
        b = (lo[: 7 + n_free], hi[: 7 + n_free])
        p0 = np.minimum(np.maximum(np.asarray(p0, dtype=float), b[0]), b[1])
        return least_squares(
            lambda q: resid(q, n_free, base_lens),
            p0,
            bounds=b,
            loss="soft_l1",
            f_scale=robust_px,
            method="trf",
            max_nfev=nfev,
            x_scale="jac",
        )

    starts = []
    if init is not None:
        lp = init.lens.params()[:n_lens] if (n_lens and init.lens.kind == lens_obj.kind) else lens_obj.params()[:n_lens]
        starts.append(_params_from_camera(init, lp if n_lens else np.zeros(0)))
    else:
        # multi-start over the lens parameter: PnP on rectified points, short refinement with the lens fixed
        grid = lens_obj.param_grid() if n_lens else [np.zeros(0)]
        scored = []
        for g in grid:
            trial_lens = lens_obj.with_params(g) if n_lens else lens_obj
            try:
                c0 = _init_pnp(trial_lens.rectify(corr.image), corr.world, size, principal_point)
            except CalibrationError:
                continue
            r0 = solve(_params_from_camera(c0, np.zeros(0)), 0, trial_lens, 300)
            scored.append((r0.cost, np.concatenate([r0.x, g[:n_lens]])))
        if not scored:
            raise CalibrationError("could not initialise a camera from the correspondences (are they degenerate?)")
        scored.sort(key=lambda s: s[0])
        starts = [s[1] for s in scored[:2]]
    best = None
    for p0 in starts:
        r = solve(p0, n_lens, lens_obj, 4000)
        if best is None or r.cost < best.cost:
            best = r
    res = best
    cam = _camera_from_params(res.x, size, lens_obj, principal_point, n_lens)
    if cam.height <= 0 or not np.isfinite(cam.focal):
        raise CalibrationError("refinement diverged")
    proj = cam.ground_to_image(corr.world)
    px_err = np.linalg.norm(proj - corr.image, axis=1)
    g = cam.image_to_ground(corr.image)
    m_err = np.linalg.norm(g - corr.world, axis=1)
    info = {
        "rmse_px": float(np.sqrt(np.nanmean(px_err**2))),
        "rmse_m": float(np.sqrt(np.nanmean(m_err**2))),
        "residuals_m": m_err,
        "params": res.x,
        "n_lens": n_lens,
        "cost": float(res.cost),
    }
    if corr.lines:
        lr = _line_residuals(cam, corr.lines)
        info["line_rmse_px"] = float(np.sqrt(np.mean(lr**2))) if len(lr) else None
    return cam, info


def _cross_validate(
    corr: Correspondences,
    size: tuple[int, int],
    lens: str | Lens,
    pp: tuple[float, float] | None,
    base: Camera,
    robust_px: float,
    folds: int | None = None,
) -> np.ndarray:
    n = len(corr)
    errs = np.full(n, np.nan)
    if n < 6:
        return errs
    k = n if folds is None or folds >= n else folds
    order = np.arange(n)
    groups = np.array_split(order, k)
    for g in groups:
        keep = np.setdiff1d(order, g)
        sub = corr.subset(keep)
        try:
            cam, _ = fit_camera(sub, size, lens=lens, principal_point=pp, robust_px=robust_px, init=base)
        except CalibrationError:
            continue
        pred = cam.image_to_ground(corr.image[g])
        errs[g] = np.linalg.norm(pred - corr.world[g], axis=1)
    return errs


def calibrate_manual(
    correspondences: Correspondences | str,
    image_size: tuple[int, int] | None = None,
    lens: str | Lens = "auto",
    principal_point: tuple[float, float] | None = None,
    robust_px: float = 3.0,
    cv_folds: int | None = 10,
    candidates: Sequence[str] = ("identity", "equidistant", "division"),
) -> CalibrationResult:
    """Calibrate a camera from correspondences.

    Parameters
    ----------
    correspondences:
        A :class:`Correspondences` object or a path to a YAML/JSON/CSV file.
    image_size:
        ``(width, height)`` of the video; read from the file when omitted.
    lens:
        ``"auto"`` (cross-validated choice among ``candidates``), a model name
        (``identity``, ``division``, ``equidistant``, ``brown``), or a fixed :class:`Lens`.
    principal_point:
        Fixed principal point; the image centre by default.
    cv_folds:
        Folds for cross-validation (``None`` = leave-one-out).
    """
    corr = (
        Correspondences.load(correspondences)
        if isinstance(correspondences, (str,)) or hasattr(correspondences, "__fspath__")
        else correspondences
    )
    size = tuple(image_size) if image_size is not None else corr.image_size
    if size is None:
        raise ValueError("image_size is required (not stored in the correspondence file)")
    if corr.handedness() > 0:
        raise CalibrationError(
            "world coordinates are left-handed (y increases downwards, as in image or map-pixel coordinates). "
            "Use Correspondences.from_reference(...) for reference-image pixels or negate y."
        )

    models = list(candidates) if lens == "auto" else [lens]
    fits = []
    base = None
    for m in models:
        try:
            cam, info = fit_camera(corr, size, lens=m, principal_point=principal_point, robust_px=robust_px)
        except CalibrationError as exc:
            fits.append((m, None, {"error": str(exc)}, np.inf))
            continue
        if base is None:
            base = cam
        cv_err = (
            _cross_validate(corr, size, m, principal_point, cam, robust_px, cv_folds)
            if len(models) > 1 or cv_folds
            else np.full(len(corr), np.nan)
        )
        cv_rmse = float(np.sqrt(np.nanmean(cv_err**2))) if np.isfinite(cv_err).any() else np.nan
        info["cv_residuals_m"] = cv_err
        fits.append((m, cam, info, cv_rmse))

    valid = [f for f in fits if f[1] is not None]
    if not valid:
        raise CalibrationError("; ".join(f"{f[0]}: {f[2].get('error')}" for f in fits))
    # parsimony: walk the models from simple to complex; accept a richer model only with a 3 % CV gain
    chosen = valid[0]
    for f in valid[1:]:
        if np.isfinite(f[3]) and (not np.isfinite(chosen[3]) or f[3] < 0.97 * chosen[3]):
            chosen = f
    m, cam, info, cv_rmse = chosen
    cam.meta.update(
        {
            "calibration_method": "manual",
            "lens_model": cam.lens.kind,
            "n_points": len(corr),
            "n_lines": len(corr.lines),
            "georef": corr.georef,
        }
    )
    details = {
        "lens_candidates": {
            str(f[0]): {
                "cv_rmse_m": None if not np.isfinite(f[3]) else f[3],
                "rmse_m": f[2].get("rmse_m"),
                "error": f[2].get("error"),
            }
            for f in fits
        },
        "rmse_px": info["rmse_px"],
        "line_rmse_px": info.get("line_rmse_px"),
        "n_points": len(corr),
        "n_lines": len(corr.lines),
    }
    return CalibrationResult(
        cam,
        "manual",
        rmse_m=info["rmse_m"],
        cv_rmse_m=None if not np.isfinite(cv_rmse) else cv_rmse,
        residuals_m=info["residuals_m"],
        details=details,
    )

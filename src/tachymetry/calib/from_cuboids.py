"""Automatic calibration from learned 3D cuboids (UrbanOmniDetect-2).

Each vehicle footprint predicted by the detector is a rectangle on the ground.
Under the correct camera, all of them back-project to rectangles of consistent
size. We search the physical camera directly (focal length, pitch, roll; the
yaw and position are a free choice of world frame and the height is the metric
scale), minimising, over hundreds of footprints from many vehicles,

* the cosines of the four corner angles (right angles),
* the sines between opposite edges (parallelism),
* a weak prior on the length-to-width ratio of passenger cars, and
* the slope of footprint size against range (a wrong pitch makes far vehicles
  shrink or grow systematically).

Compared to a free projective rectification (4 DoF, up to similarity), the
3-DoF physical model has no shear or anisotropy freedom, which removes the
degeneracy of scenes where all vehicles share one heading. The metric scale
then follows from the median passenger-car footprint (4.6 m x 1.85 m by
default), down-weighting long-parked vehicles so each vehicle contributes one vote.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..detections import DetectionSet
from ..geometry.camera import CalibrationError, Camera
from ..geometry.lens import IdentityLens, Lens
from .result import CalibrationResult
from .scale import CAR_LENGTH_M, CAR_WIDTH_M

__all__ = ["calibrate_from_cuboids", "collect_footprints", "footprint_metrics"]


ASPECT_WEIGHT = 1.0  #: weight of the passenger-car length-to-width prior (log-ratio residual)


def footprint_scale(
    cam: Camera,
    quads: np.ndarray,
    wts: np.ndarray,
    car_length_m: float = CAR_LENGTH_M,
    car_width_m: float = CAR_WIDTH_M,
) -> tuple[float, dict[str, Any]]:
    """Metric scale from passenger-car footprints, using each footprint's well-conditioned dimension.

    Distances along the viewing ray are far more sensitive to pitch and corner
    errors than distances across it. Each footprint therefore contributes the
    dimension that lies across the ray: its width when seen head-on or from
    behind, its length when seen from the side (both when oblique). Estimates are
    combined by a weighted median; widths carry four times the weight of lengths
    because car widths vary about half as much as car lengths.
    """
    g = cam.image_to_ground(quads.reshape(-1, 2)).reshape(-1, 4, 2)
    ok = np.all(np.isfinite(g), axis=(1, 2))
    g, w = g[ok], wts[ok]
    if len(g) < 5:
        raise CalibrationError("too few footprints for a metric scale")
    L = 0.5 * (np.linalg.norm(g[:, 0] - g[:, 1], axis=1) + np.linalg.norm(g[:, 3] - g[:, 2], axis=1))
    W = 0.5 * (np.linalg.norm(g[:, 0] - g[:, 3], axis=1) + np.linalg.norm(g[:, 1] - g[:, 2], axis=1))
    axis = (g[:, 0] + g[:, 3]) / 2 - (g[:, 1] + g[:, 2]) / 2
    ray = g.mean(axis=1) - cam.center[:2]
    cosang = np.abs(np.sum(axis * ray, axis=1)) / np.maximum(
        np.linalg.norm(axis, axis=1) * np.linalg.norm(ray, axis=1), 1e-9
    )
    est, wt = [], []
    use_w = cosang > 0.5  # axis roughly along the ray: the width lies across it
    use_l = cosang < 0.85  # axis roughly across the ray: the length lies across it
    est.append(car_width_m / np.maximum(W[use_w], 1e-9))
    wt.append(4.0 * w[use_w])
    est.append(car_length_m / np.maximum(L[use_l], 1e-9))
    wt.append(w[use_l])
    e, v = np.concatenate(est), np.concatenate(wt)
    good = np.isfinite(e) & (e > 0)
    e, v = e[good], v[good]
    order = np.argsort(e)
    cw = np.cumsum(v[order])
    s = float(e[order][np.searchsorted(cw, 0.5 * cw[-1])])
    info = {
        "scale": s,
        "n": int(len(g)),
        "n_width_based": int(use_w.sum()),
        "n_length_based": int(use_l.sum()),
        "scale_width_only": float(np.median(car_width_m / np.maximum(W, 1e-9))),
        "scale_length_only": float(np.median(car_length_m / np.maximum(L, 1e-9))),
    }
    return s, info


def collect_footprints(
    dets: DetectionSet, classes: tuple[str, ...] = ("car",), max_n: int = 2500, margin_frac: float = 0.01, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Trusted image-space ground quads ``(N, 4, 2)`` and a diversity weight per quad."""
    if dets.cuboid is None or len(dets) == 0:
        return np.zeros((0, 4, 2)), np.zeros(0)
    w, h = dets.image_size
    m = max(3.0, margin_frac * math.hypot(w, h))
    names = np.array(dets.class_names)
    cls_ok = np.isin(names[dets.cls], list(classes))
    quad = dets.cuboid[:, :4, :2].astype(np.float64)
    finite = np.all(np.isfinite(quad), axis=(1, 2))
    b = dets.xyxy
    inside = (b[:, 0] > m) & (b[:, 1] > m) & (b[:, 2] < w - 1 - m) & (b[:, 3] < h - 1 - m)
    sel = np.flatnonzero(cls_ok & finite & inside & (dets.score >= 0.4))
    if len(sel) == 0:
        return np.zeros((0, 4, 2)), np.zeros(0)
    # diversity weighting: one vote per image cell (parked cars occupy one cell for the whole video)
    cx = quad[sel].mean(axis=1)
    cell = (np.floor(cx[:, 0] / (w / 24)) * 1000 + np.floor(cx[:, 1] / (h / 14))).astype(np.int64)
    _, inv, counts = np.unique(cell, return_inverse=True, return_counts=True)
    wts = 1.0 / counts[inv]
    if len(sel) > max_n:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(sel), size=max_n, replace=False, p=wts / wts.sum())
        sel, wts = sel[keep], np.ones(len(keep))
    return quad[sel], wts / wts.mean()


def _bev(cam: Camera, quads: np.ndarray) -> np.ndarray:
    g = cam.image_to_ground(quads.reshape(-1, 2))
    return g.reshape(-1, 4, 2)


def footprint_metrics(cam: Camera, quads: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Calibration-quality metrics from footprint geometry (no ground truth needed)."""
    g = _bev(cam, quads)
    ok = np.all(np.isfinite(g), axis=(1, 2))
    if ok.sum() < 5:
        return {"n": int(ok.sum()), "right_angle_deg": float("nan"), "size_range_slope": float("nan")}
    g = g[ok]
    e = np.roll(g, -1, axis=1) - g
    u = e / np.maximum(np.linalg.norm(e, axis=2, keepdims=True), 1e-9)
    cosang = np.abs(np.sum(u * np.roll(u, -1, axis=1), axis=2))
    L = 0.5 * (np.linalg.norm(g[:, 0] - g[:, 1], axis=1) + np.linalg.norm(g[:, 3] - g[:, 2], axis=1))
    W = 0.5 * (np.linalg.norm(g[:, 0] - g[:, 3], axis=1) + np.linalg.norm(g[:, 1] - g[:, 2], axis=1))
    size = np.log(np.sqrt(np.maximum(L * W, 1e-12)))
    rng = np.log(np.maximum(cam.distance_to(g.mean(axis=1)), 1e-3))
    slope = float(np.polyfit(rng, size, 1)[0]) if np.ptp(rng) > 1e-6 else 0.0
    return {
        "n": int(ok.sum()),
        "right_angle_deg": float(np.degrees(np.arcsin(np.median(cosang)))),
        "size_range_slope": slope,
        "aspect_median": float(np.median(L / np.maximum(W, 1e-9))),
        "length_median": float(np.median(L)),
        "width_median": float(np.median(W)),
    }


def _residuals(
    cam: Camera, quads: np.ndarray, wts: np.ndarray, aspect: float, w_aspect: float, w_slope: float
) -> np.ndarray:
    g = _bev(cam, quads)
    bad = ~np.all(np.isfinite(g), axis=(1, 2))
    g = np.where(bad[:, None, None], 0.0, g)
    e = np.roll(g, -1, axis=1) - g
    n = np.linalg.norm(e, axis=2, keepdims=True)
    u = e / np.maximum(n, 1e-9)
    orth = np.sum(u * np.roll(u, -1, axis=1), axis=2)  # (N, 4)
    par = (
        np.stack(
            [u[:, 0, 0] * u[:, 2, 1] - u[:, 0, 1] * u[:, 2, 0], u[:, 1, 0] * u[:, 3, 1] - u[:, 1, 1] * u[:, 3, 0]],
            axis=1,
        )
        * 0.5
    )
    L = 0.5 * (n[:, 0, 0] + n[:, 2, 0])
    W = 0.5 * (n[:, 3, 0] + n[:, 1, 0])
    asp = (np.log(np.maximum(L, 1e-9)) - np.log(np.maximum(W, 1e-9)) - math.log(aspect)) * w_aspect
    r = np.concatenate([orth, par, asp[:, None]], axis=1)
    r[bad] = 3.0
    r = r * np.sqrt(wts)[:, None]
    size = np.log(np.sqrt(np.maximum(L * W, 1e-12)))
    rng = np.log(np.maximum(cam.distance_to(np.where(bad[:, None], 1.0, g.mean(axis=1))), 1e-3))
    good = ~bad
    slope = 0.0
    if good.sum() > 5 and np.ptp(rng[good]) > 1e-6:
        slope = float(np.polyfit(rng[good], size[good], 1, w=np.sqrt(wts[good]))[0])
    return np.concatenate([r.ravel(), [w_slope * slope * math.sqrt(max(good.sum(), 1))]])


def _cam(p: np.ndarray, size: tuple[int, int], lens: Lens) -> Camera:
    return Camera.from_pose(
        size, focal=math.exp(p[0]), height=1.0, pitch=math.degrees(p[1]), yaw=90.0, roll=math.degrees(p[2]), lens=lens
    )


def calibrate_from_cuboids(
    dets: DetectionSet,
    lens: Lens | None = None,
    car_length_m: float = CAR_LENGTH_M,
    car_width_m: float = CAR_WIDTH_M,
    aspect: float | None = None,
    min_footprints: int = 40,
    seed: int = 0,
) -> CalibrationResult:
    """Metric camera from the cuboid footprints in ``dets`` (see module docstring)."""
    from scipy.optimize import least_squares

    size = tuple(dets.image_size)
    lens = lens or IdentityLens(size)
    quads, wts = collect_footprints(dets, seed=seed)
    if len(quads) < min_footprints:
        raise CalibrationError(
            f"only {len(quads)} trusted vehicle footprints (need {min_footprints}); "
            "process a longer stretch of video or use another method"
        )
    quads_r = lens.rectify(quads.reshape(-1, 2)).reshape(-1, 4, 2) if not lens.is_identity else quads
    asp = aspect or car_length_m / car_width_m
    w_img = size[0]
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(quads_r), size=min(600, len(quads_r)), replace=False)

    def fun(p: np.ndarray, q: np.ndarray, wq: np.ndarray) -> np.ndarray:
        return _residuals(_cam(p, size, IdentityLens(size)), q, wq, asp, ASPECT_WEIGHT, 2.0)

    starts = []
    for pitch in (1, 8, 15, 25, 35, 50, 65, 80):
        for hfov in (35, 55, 75, 95):
            f = (w_img / 2) / math.tan(math.radians(hfov / 2))
            p0 = np.array([math.log(f), math.radians(pitch), 0.0])
            try:
                r0 = fun(p0, quads_r[sub], wts[sub])
            except Exception:
                continue
            starts.append((float(np.sum(r0**2)), p0))
    starts.sort(key=lambda s: s[0])
    lo = np.array([math.log(w_img * 0.15), math.radians(-10.0), math.radians(-25)])
    hi = np.array([math.log(w_img * 12), math.radians(89.5), math.radians(25)])
    best = None
    for _, p0 in starts[:8]:
        r = least_squares(
            fun,
            p0,
            args=(quads_r[sub], wts[sub]),
            bounds=(lo, hi),
            loss="soft_l1",
            f_scale=0.12,
            max_nfev=200,
            x_scale="jac",
        )
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    final = least_squares(
        fun, best.x, args=(quads_r, wts), bounds=(lo, hi), loss="soft_l1", f_scale=0.12, max_nfev=600, x_scale="jac"
    )
    cam = _cam(final.x, size, lens)
    # metric scale from passenger-car footprints (conditioning-aware, diversity-weighted)
    s, sinfo = footprint_scale(cam, quads, wts, car_length_m, car_width_m)
    cam = cam.scaled(s)
    # world frame: +x along the dominant vehicle axis, origin at the median vehicle position
    gm = _bev(cam, quads)
    ok = np.all(np.isfinite(gm), axis=(1, 2))
    gm = gm[ok]
    ax = (gm[:, 0] + gm[:, 3]) / 2 - (gm[:, 1] + gm[:, 2]) / 2
    phi = 0.5 * float(np.angle(np.sum(wts[ok] * np.exp(2j * np.arctan2(ax[:, 1], ax[:, 0])))))
    ctr = np.median(gm.mean(axis=1), axis=0)
    cam = cam.transformed(rotation_deg=0.0, translation=(-float(ctr[0]), -float(ctr[1])))
    cam = cam.transformed(rotation_deg=-math.degrees(phi))
    cam.meta.update({"calibration_method": "cuboids", "metric": True})
    metrics = footprint_metrics(cam, quads, wts)
    yaw, pitch, roll = cam.angles
    details: dict[str, Any] = {
        "n_footprints": int(len(quads)),
        "cost": float(final.cost),
        "scale": sinfo,
        "metrics": metrics,
        "angles": {"yaw": yaw, "pitch": pitch, "roll": roll},
    }
    score = float(
        np.clip(1.0 - metrics["right_angle_deg"] / 15.0, 0, 1)
        * np.clip(1 - abs(metrics["size_range_slope"]) / 0.5, 0, 1)
    )
    return CalibrationResult(cam, "cuboids", score=score, details=details)

"""Metric scale for cameras known up to scale.

Vanishing points, gravity estimates and learned cuboids recover a camera's
rotation and focal length but not the absolute size of the world. Any one
metric cue fixes it:

* the camera height (``height``),
* a known distance between two ground points (``distance``),
* the lane width or crosswalk stripe pitch (``lane_width``),
* the size distribution of passenger cars (``vehicles``), which needs no
  scene knowledge at all and is the default for automatic calibration.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry.camera import CalibrationError, Camera

__all__ = [
    "scale_to_height",
    "scale_from_distance",
    "scale_from_lane_spacing",
    "scale_from_vehicle_sizes",
    "CAR_LENGTH_M",
    "CAR_WIDTH_M",
]

#: Median footprint of a passenger car (sedan/SUV/van mix, North America and Europe).
CAR_LENGTH_M = 4.6
CAR_WIDTH_M = 1.85


def scale_to_height(camera: Camera, height_m: float) -> Camera:
    """Rescale so the camera centre is ``height_m`` metres above the ground."""
    if camera.height <= 0:
        raise CalibrationError("camera is not above the ground plane")
    return camera.scaled(height_m / camera.height)


def scale_from_distance(camera: Camera, uv_a: Any, uv_b: Any, distance_m: float) -> Camera:
    """Rescale so two ground pixels are ``distance_m`` apart."""
    g = camera.image_to_ground(np.array([uv_a, uv_b], dtype=float))
    d = float(np.linalg.norm(g[1] - g[0]))
    if not np.isfinite(d) or d <= 0:
        raise CalibrationError("points do not map to distinct ground positions")
    return camera.scaled(distance_m / d)


def scale_from_lane_spacing(offsets: np.ndarray, expected_m: float = 3.5, lo: float = 0.4, hi: float = 2.0) -> float:
    """Scale factor from the lateral offsets (BEV, arbitrary units) of lines parallel to the road.

    The fundamental period of the offsets is taken as one lane width. Returns
    the multiplicative factor that maps the current units to metres.
    """
    off = np.sort(np.asarray(offsets, dtype=float))
    if len(off) < 3:
        raise CalibrationError("too few parallel lines to measure lane spacing")
    diffs = np.abs(off[:, None] - off[None, :])[np.triu_indices(len(off), 1)]
    span = np.median(diffs)
    if span <= 0:
        raise CalibrationError("degenerate line offsets")
    # search the period that best explains all pairwise differences as integer multiples
    cands = np.linspace(lo * expected_m, hi * expected_m, 400) / expected_m * span / 2
    best, best_score = None, -np.inf
    for p in cands:
        k = np.round(diffs / p)
        ok = k >= 1
        if ok.sum() < 3:
            continue
        r = np.abs(diffs[ok] - k[ok] * p) / p
        score = np.sum(np.exp(-((r / 0.08) ** 2))) - 0.02 * np.sum(k[ok])
        if score > best_score:
            best, best_score = p, score
    if best is None:
        raise CalibrationError("no consistent lane period")
    return expected_m / best


def scale_from_vehicle_sizes(
    lengths: np.ndarray,
    widths: np.ndarray | None = None,
    prior_length: float = CAR_LENGTH_M,
    prior_width: float = CAR_WIDTH_M,
    min_samples: int = 5,
) -> tuple[float, dict[str, Any]]:
    """Scale factor from per-vehicle footprint sizes measured in current BEV units.

    Uses medians over vehicles (one value per track) and combines length and
    width estimates with weights reflecting their typical dispersion.
    """
    L = np.asarray(lengths, dtype=float)
    L = L[np.isfinite(L) & (L > 0)]
    if len(L) < min_samples:
        raise CalibrationError(f"need >= {min_samples} vehicles for size-based scale, got {len(L)}")
    sL = prior_length / np.median(L)
    info: dict[str, Any] = {"n": int(len(L)), "scale_length": float(sL)}
    if widths is not None:
        W = np.asarray(widths, dtype=float)
        W = W[np.isfinite(W) & (W > 0)]
        if len(W) >= min_samples:
            sW = prior_width / np.median(W)
            info["scale_width"] = float(sW)
            # car widths vary ~6 %, lengths ~12 %: inverse-variance weighting
            wl, ww = 1 / 0.12**2, 1 / 0.06**2
            s = float(np.exp((wl * np.log(sL) + ww * np.log(sW)) / (wl + ww)))
            info["scale"] = s
            return s, info
    info["scale"] = float(sL)
    return float(sL), info

"""Planar homography estimation and utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "apply_homography",
    "normalize_points",
    "dlt_homography",
    "fit_homography",
    "refine_homography",
    "transfer_error",
]


def apply_homography(H: np.ndarray, pts: Any) -> np.ndarray:
    """Map ``(N, 2)`` points through ``H``. Points mapped to infinity become NaN."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = homog[:, :2] / homog[:, 2:3]
    out[np.abs(homog[:, 2]) < 1e-12] = np.nan
    return out


def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid at the origin, mean distance sqrt(2)."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return apply_homography(T, pts), T


def dlt_homography(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalised (weighted) direct linear transform, ``dst ~ H src``. Needs >= 4 points."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        raise ValueError("at least four correspondences are required")
    ns, Ts = normalize_points(src)
    nd, Td = normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(ns, nd):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(rows)
    if weights is not None:
        A = A * np.repeat(np.sqrt(np.asarray(weights, dtype=np.float64)), 2)[:, None]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point forward transfer error ``|H src - dst|``."""
    return np.linalg.norm(apply_homography(H, src) - np.asarray(dst, dtype=np.float64), axis=1)


def refine_homography(
    H0: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray | None = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
) -> np.ndarray:
    """Levenberg–Marquardt refinement of the forward transfer error in ``dst`` units."""
    from scipy.optimize import least_squares

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    w = np.ones(len(src)) if weights is None else np.sqrt(np.asarray(weights, dtype=np.float64))
    H0 = H0 / H0[2, 2]

    def resid(p: np.ndarray) -> np.ndarray:
        H = np.append(p, 1.0).reshape(3, 3)
        r = apply_homography(H, src) - dst
        r = np.nan_to_num(r, nan=1e6)
        return (r * w[:, None]).ravel()

    res = least_squares(resid, H0.ravel()[:8], loss=loss, f_scale=f_scale, method="trf", max_nfev=2000)
    H = np.append(res.x, 1.0).reshape(3, 3)
    return H


def fit_homography(
    src: np.ndarray,
    dst: np.ndarray,
    ransac_threshold: float | None = None,
    weights: np.ndarray | None = None,
    refine: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly fit ``dst ~ H src``.

    With ``ransac_threshold`` (in ``dst`` units) outliers are rejected with
    OpenCV's USAC-MAGSAC when available, then the inliers are refined by
    weighted DLT and LM. Returns ``(H, inlier_mask)``.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n < 4:
        raise ValueError(f"need >= 4 correspondences, got {n}")
    inliers = np.ones(n, dtype=bool)
    if ransac_threshold is not None and n > 4:
        import cv2

        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        cv2.setRNGSeed(seed)
        H_r, mask = cv2.findHomography(src, dst, method, ransac_threshold, maxIters=20000, confidence=0.9999)
        if H_r is not None and mask is not None and mask.sum() >= 4:
            inliers = mask.ravel().astype(bool)
    w = None if weights is None else np.asarray(weights)[inliers]
    H = dlt_homography(src[inliers], dst[inliers], w)
    if refine and inliers.sum() > 4:
        H = refine_homography(H, src[inliers], dst[inliers], w, f_scale=ransac_threshold if ransac_threshold else 1.0)
    return H / H[2, 2], inliers

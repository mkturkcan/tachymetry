"""Straight-line evidence: road markings, line segments, vanishing points, plumb lines.

* :func:`background_image` removes moving traffic with a temporal median.
* :func:`marking_mask` isolates painted markings (white and yellow) with a
  morphological top-hat and a chroma test.
* :func:`detect_segments` runs the LSD line-segment detector (Hough fallback).
* :func:`estimate_vanishing_points` finds dominant vanishing points with
  length-weighted RANSAC and a homogeneous least-squares refinement.
* :func:`estimate_lens_plumbline` estimates radial distortion from edge chains
  that should be straight (Devernay and Faugeras, 2001).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry.lens import DivisionLens, EquidistantLens, Lens

__all__ = [
    "Segments",
    "VanishingPoint",
    "background_image",
    "marking_mask",
    "detect_segments",
    "estimate_vanishing_points",
    "estimate_lens_plumbline",
    "edge_chains",
]


@dataclass
class Segments:
    """A set of 2D line segments ``(N, 4)`` as ``x1, y1, x2, y2`` (pixels)."""

    xyxy: np.ndarray

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float64).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.xyxy)

    @property
    def p1(self) -> np.ndarray:
        return self.xyxy[:, :2]

    @property
    def p2(self) -> np.ndarray:
        return self.xyxy[:, 2:]

    @property
    def lengths(self) -> np.ndarray:
        return np.linalg.norm(self.p2 - self.p1, axis=1)

    @property
    def midpoints(self) -> np.ndarray:
        return 0.5 * (self.p1 + self.p2)

    @property
    def directions(self) -> np.ndarray:
        d = self.p2 - self.p1
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    @property
    def lines(self) -> np.ndarray:
        """Homogeneous lines, normalised so ``a^2 + b^2 = 1``."""
        h1 = np.column_stack([self.p1, np.ones(len(self))])
        h2 = np.column_stack([self.p2, np.ones(len(self))])
        l = np.cross(h1, h2)
        return l / np.maximum(np.linalg.norm(l[:, :2], axis=1, keepdims=True), 1e-12)

    def subset(self, mask: np.ndarray) -> Segments:
        return Segments(self.xyxy[mask])

    def transformed(self, fn: Any) -> Segments:
        return Segments(np.column_stack([fn(self.p1), fn(self.p2)]))


@dataclass
class VanishingPoint:
    """A vanishing point with its supporting segments."""

    point: np.ndarray  # homogeneous (3,), normalised
    inliers: np.ndarray  # boolean mask over the segment set
    support: float  # summed inlier length (pixels)

    @property
    def finite(self) -> bool:
        return abs(self.point[2]) > 1e-9

    @property
    def xy(self) -> np.ndarray:
        return self.point[:2] / self.point[2] if self.finite else np.array([np.inf, np.inf])


# ---------------------------------------------------------------- image evidence
def background_image(frames: Iterable[np.ndarray] | Any, n: int = 41, max_side: int | None = None) -> np.ndarray:
    """Temporal median of ``n`` frames: a traffic-free view of the scene.

    ``frames`` may be an iterable of images or a :class:`tachymetry.io.VideoSource`.
    """
    import cv2

    if hasattr(frames, "sample_frames"):
        imgs = list(frames.sample_frames(n))
    else:
        imgs = list(frames)
    if not imgs:
        raise ValueError("no frames")
    if max_side is not None:
        h, w = imgs[0].shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            imgs = [cv2.resize(i, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) for i in imgs]
    return np.median(np.stack(imgs, axis=0), axis=0).astype(np.uint8)


def marking_mask(
    image: np.ndarray,
    kernel_frac: float = 0.02,
    white_thresh: float | None = None,
    include_yellow: bool = True,
    region: np.ndarray | None = None,
) -> np.ndarray:
    """Binary mask of painted road markings (bright, thin, low-chroma or yellow)."""
    import cv2

    h, w = image.shape[:2]
    k = max(5, int(round(kernel_frac * max(h, w))) | 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if white_thresh is None:
        vals = tophat[tophat > 0]
        white_thresh = float(max(18.0, np.percentile(vals, 94))) if vals.size else 25.0
    mask = tophat > white_thresh
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_chroma = hsv[..., 1] < 70
        yellow = (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 1] > 80) & (hsv[..., 2] > 100)
        mask = (mask & low_chroma) | ((mask | (tophat > 0.6 * white_thresh)) & yellow if include_yellow else False)
    if region is not None:
        mask &= region.astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask > 0


def detect_segments(
    image: np.ndarray, mask: np.ndarray | None = None, min_length_frac: float = 0.012, dilate: int = 3
) -> Segments:
    """Line segments (LSD; probabilistic Hough as a fallback) optionally restricted to ``mask``."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    min_len = min_length_frac * math.hypot(w, h)
    segs = None
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        out = lsd.detect(gray)[0]
        if out is not None:
            segs = out.reshape(-1, 4).astype(np.float64)
    except (cv2.error, AttributeError):
        segs = None
    if segs is None or len(segs) == 0:
        edges = cv2.Canny(gray, 60, 160)
        out = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40, minLineLength=int(min_len), maxLineGap=4)
        segs = np.zeros((0, 4)) if out is None else out.reshape(-1, 4).astype(np.float64)
    s = Segments(segs)
    s = s.subset(s.lengths >= min_len)
    if mask is not None and len(s):
        m = mask.astype(np.uint8)
        if dilate:
            m = cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
        # keep segments whose sampled points mostly fall inside the mask
        t = np.linspace(0, 1, 9)
        pts = s.p1[:, None, :] * (1 - t)[None, :, None] + s.p2[:, None, :] * t[None, :, None]
        xi = np.clip(np.round(pts[..., 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[..., 1]).astype(int), 0, h - 1)
        frac = m[yi, xi].mean(axis=1)
        s = s.subset(frac >= 0.6)
    return s


# ------------------------------------------------------------- vanishing points
def _angular_residual(segs: Segments, v: np.ndarray) -> np.ndarray:
    """Angle (radians) between each segment and the ray from its midpoint to ``v``."""
    m = segs.midpoints
    if abs(v[2]) > 1e-12:
        d = v[:2] / v[2] - m
    else:
        d = np.broadcast_to(v[:2], m.shape)
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    cosang = np.abs(np.sum(dn * segs.directions, axis=1))
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def refine_vanishing_point(segs: Segments, weights: np.ndarray) -> np.ndarray:
    """Weighted homogeneous least squares: ``argmin_v sum w (l . v)^2`` with ``|v| = 1``."""
    L = segs.lines
    M = (L * weights[:, None]).T @ L
    _, vecs = np.linalg.eigh(M)
    v = vecs[:, 0]
    return v / np.linalg.norm(v)


def estimate_vanishing_points(
    segs: Segments,
    n_points: int = 3,
    threshold_deg: float = 1.5,
    iterations: int = 3000,
    min_support_frac: float = 0.04,
    seed: int = 0,
    exclude: Iterable[np.ndarray] = (),
) -> list[VanishingPoint]:
    """Sequential length-weighted RANSAC for up to ``n_points`` vanishing points."""
    rng = np.random.default_rng(seed)
    thr = math.radians(threshold_deg)
    remaining = np.ones(len(segs), dtype=bool)
    total = float(segs.lengths.sum()) if len(segs) else 0.0
    found: list[VanishingPoint] = []
    lines = segs.lines
    lengths = segs.lengths
    excl = [np.asarray(e, dtype=np.float64) for e in exclude]
    for _ in range(n_points):
        idx = np.flatnonzero(remaining)
        if len(idx) < 4:
            break
        p = lengths[idx] / lengths[idx].sum()
        best = (0.0, None)
        for _it in range(iterations):
            a, b = rng.choice(idx, size=2, replace=False, p=p)
            v = np.cross(lines[a], lines[b])
            n = np.linalg.norm(v)
            if n < 1e-12:
                continue
            v = v / n
            if any(_angle_between_vps(v, e) < 3 * thr for e in excl + [f.point for f in found]):
                continue
            res = _angular_residual(segs.subset(remaining), v)
            score = float(np.sum(lengths[remaining] * np.clip(1 - (res / thr) ** 2, 0, None)))
            if score > best[0]:
                best = (score, v)
        if best[1] is None:
            break
        v = best[1]
        for _ in range(3):  # refine on inliers, then re-select inliers
            res = _angular_residual(segs, v)
            inl = remaining & (res < thr)
            if inl.sum() < 3:
                break
            w = lengths[inl] * np.clip(1 - (res[inl] / thr) ** 2, 0.05, None)
            v = refine_vanishing_point(segs.subset(inl), w)
        res = _angular_residual(segs, v)
        inl = remaining & (res < thr)
        support = float(lengths[inl].sum())
        if total == 0 or support < min_support_frac * total:
            break
        found.append(VanishingPoint(v, inl, support))
        remaining &= ~inl
    return found


def _angle_between_vps(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vanishing points as viewing directions (image-independent proxy)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.acos(min(1.0, abs(float(a @ b))))


# ------------------------------------------------------------------ plumb lines
def edge_chains(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    min_length_frac: float = 0.06,
    canny: tuple[int, int] = (60, 160),
    max_chains: int = 400,
) -> list[np.ndarray]:
    """Long, smooth edge chains (candidates for world-straight lines)."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), *canny)
    if mask is not None:
        edges[~cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_len = min_length_frac * math.hypot(w, h)
    chains = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < 20:
            continue
        # contours of thin edges trace both sides; keep the first half to avoid doubling
        half = pts[: max(len(pts) // 2, 2)]
        if np.linalg.norm(half[-1] - half[0]) < min_len:
            continue
        # split at sharp turns
        step = 6
        sub = half[::step]
        if len(sub) < 4:
            continue
        d = np.diff(sub, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        turn = np.abs(np.angle(np.exp(1j * np.diff(ang))))
        cuts = np.flatnonzero(turn > math.radians(25)) + 1
        for piece in np.split(sub, cuts):
            if len(piece) >= 5 and np.linalg.norm(piece[-1] - piece[0]) >= min_len:
                chains.append(piece)
    chains.sort(key=lambda c: -np.linalg.norm(c[-1] - c[0]))
    return chains[:max_chains]


def _straightness(chains: list[np.ndarray], lens: Lens) -> np.ndarray:
    res = []
    for ch in chains:
        r = lens.rectify(ch)
        if not np.all(np.isfinite(r)):
            res.append(np.inf)
            continue
        c = r.mean(axis=0)
        _, sv, _ = np.linalg.svd(r - c, full_matrices=False)
        res.append(sv[-1] / math.sqrt(len(r)))
    return np.asarray(res)


def estimate_lens_plumbline(
    image: np.ndarray | None = None,
    chains: list[np.ndarray] | None = None,
    model: str = "division",
    mask: np.ndarray | None = None,
    trim: float = 0.5,
) -> tuple[Lens, dict[str, Any]]:
    """Estimate one-parameter radial distortion so that edge chains become straight.

    Uses a trimmed objective (the ``trim`` fraction of chains that are most
    consistent with the current model) so curved world structure is ignored.
    """
    from scipy.optimize import minimize_scalar

    if chains is None:
        if image is None:
            raise ValueError("pass an image or chains")
        chains = edge_chains(image, mask)
    if image is not None:
        size = (image.shape[1], image.shape[0])
    else:
        allp = np.concatenate(chains)
        size = (int(allp[:, 0].max() + 1), int(allp[:, 1].max() + 1))
    if len(chains) < 5:
        from ..geometry.lens import IdentityLens

        return IdentityLens(size), {"n_chains": len(chains), "note": "too few chains; identity lens"}

    proto: Lens = DivisionLens(size) if model == "division" else EquidistantLens(size)
    lo, hi = proto.param_bounds()

    def objective(q: float) -> float:
        lens = proto.with_params(np.array([q]))
        s = _straightness(chains, lens)
        s = np.sort(s[np.isfinite(s)])
        if len(s) == 0:
            return 1e9
        k = max(3, int(trim * len(s)))
        return float(np.mean(s[:k] ** 2))

    grid = np.linspace(lo[0], hi[0], 61)
    vals = [objective(q) for q in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(objective, bounds=(a, b), method="bounded", options={"xatol": 1e-6})
    q = float(res.x) if res.fun <= vals[i] else float(grid[i])
    lens = proto.with_params(np.array([q]))
    base = objective(0.0)
    info = {
        "n_chains": len(chains),
        "residual_px": math.sqrt(max(objective(q), 0.0)),
        "residual_identity_px": math.sqrt(max(base, 0.0)),
        "param": q,
        "model": lens.kind,
    }
    return lens, info


def estimate_lens_vp_consistency(
    segments: Segments,
    image_size: tuple[int, int],
    model: str = "division",
    threshold_deg: float = 1.0,
    iterations: int = 600,
    seed: int = 0,
) -> tuple[Lens, dict[str, Any]]:
    """Estimate one-parameter radial distortion by maximising vanishing-point consistency.

    With the right distortion parameter, rectified segments from the ground
    and vertical directions pass through common vanishing points; the score is
    the length fraction of segments supporting the three strongest vanishing
    points. Works on short segments, where plumb-line curvature is invisible.
    """
    proto: Lens = DivisionLens(image_size) if model == "division" else EquidistantLens(image_size)
    lo, hi = proto.param_bounds()
    total = float(segments.lengths.sum())

    def score(q: float) -> float:
        lens = proto.with_params(np.array([q]))
        rs = segments.transformed(lens.rectify)
        ok = np.all(np.isfinite(rs.xyxy), axis=1)
        rs = rs.subset(ok)
        vps = estimate_vanishing_points(
            rs, n_points=3, threshold_deg=threshold_deg, iterations=iterations, seed=seed, min_support_frac=0.02
        )
        # weight by original lengths so rectification stretch near the border does not inflate scores
        orig = segments.lengths[ok]
        return float(sum(orig[vp.inliers].sum() for vp in vps)) / max(total, 1e-9)

    grid = np.linspace(lo[0], hi[0] * (0.5 if model == "division" else 1.0), 19)
    vals = np.array([score(q) for q in grid])
    i = int(np.argmax(vals))
    fine = np.linspace(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)], 9)
    fvals = np.array([score(q) for q in fine])
    j = int(np.argmax(fvals))
    q = float(fine[j]) if fvals[j] >= vals[i] else float(grid[i])
    best = max(fvals[j], vals[i])
    base = score(0.0)
    lens = proto.with_params(np.array([q]))
    return lens, {
        "model": lens.kind,
        "param": q,
        "support": best,
        "support_identity": base,
        "gain": best - base,
        "grid": grid.tolist(),
        "scores": vals.tolist(),
    }

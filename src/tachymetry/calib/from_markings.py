"""Scale-free check of a calibration with dashed lane markings.

A dashed lane line repeats one dash length and one period along the road. Under
the correct camera the ground-plane lengths of its dashes, and the distances
between consecutive dashes, are the same near and far; a camera whose horizon
does not pass through the lines' vanishing point makes them grow or shrink with
distance. :func:`marking_regularity` measures that spread.

Dash chains are found without any camera: dash starts that are equally spaced on
the ground have the cross-ratio 4/3 for every four consecutive dashes under any
perspective projection, which rejects crosswalk stripes, fragments of solid
lines and missing dashes.

The criterion is a check, not a calibration method: when all dashed lines are
parallel (one road direction), every camera whose horizon passes through their
common vanishing point makes them regular, whatever its focal length.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry.camera import Camera
from ..geometry.lens import Lens

__all__ = ["DashedLine", "dashed_lines", "marking_regularity", "road_region"]


@dataclass
class DashedLine:
    """Dashes of one painted line, ordered along it: ``a`` and ``b`` are the two ends of each dash (px)."""

    a: np.ndarray  # (n, 2)
    b: np.ndarray  # (n, 2)
    weight: np.ndarray  # (n,) image length of each dash (px), for weighting

    def __len__(self) -> int:
        return len(self.a)


def dashed_lines(
    image: np.ndarray,
    min_dashes: int = 4,
    max_lines: int = 16,
    lens: Lens | None = None,
    cr_tol: float = 0.06,
    region: np.ndarray | None = None,
) -> list[DashedLine]:
    """Dashed lane lines in a traffic-free image (rectified first when ``lens`` is given).

    ``region`` (boolean, image-sized) restricts the search, e.g. to the road surface swept by
    vehicles (:func:`road_region`); regular structures off the road (mast arms, facades) are
    otherwise mistaken for dash chains.
    """
    import cv2

    from .lines import marking_mask

    if lens is not None and not lens.is_identity:
        image = lens.rectify_image(image)
        if region is not None:
            region = lens.rectify_image(region.astype(np.uint8) * 255) > 127
    h, w = image.shape[:2]
    mask = marking_mask(image, region=region).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cand = []  # (centre, unit axis, end a, end b, length)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 10 or area > 0.003 * w * h or x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        ys, xs = np.nonzero(lab[y : y + bh, x : x + bw] == i)
        pts = np.column_stack([xs + x, ys + y]).astype(np.float64)
        c = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
        u = vt[0]
        t = (pts - c) @ u
        length = float(t.max() - t.min())
        width = area / max(length, 1.0)
        if length < 6.0 or length / max(width, 1.0) < 2.5:
            continue
        cand.append((c, u, c + t.min() * u, c + t.max() * u, length))
    if len(cand) < min_dashes:
        return []
    C = np.array([c[0] for c in cand])
    U = np.array([c[1] for c in cand])
    rng = np.random.default_rng(0)
    free = np.ones(len(cand), bool)
    lines: list[DashedLine] = []
    for _ in range(max_lines):
        idx = np.flatnonzero(free)
        if len(idx) < min_dashes:
            break
        best = None
        for _ in range(400):
            i, j = rng.choice(idx, 2, replace=False)
            d = C[j] - C[i]
            nd = float(np.linalg.norm(d))
            if nd < 10:
                continue
            d /= nd
            nrm = np.array([-d[1], d[0]])
            dist = np.abs((C[idx] - C[i]) @ nrm)
            align = np.abs(U[idx] @ d)
            inl = idx[(dist < 2.5 + 0.01 * np.abs((C[idx] - C[i]) @ d)) & (align > math.cos(math.radians(8)))]
            if best is None or len(inl) > len(best):
                best = inl
        if best is None or len(best) < min_dashes:
            break
        free[best] = False
        # order along the line; a = the end with the smaller coordinate along it
        d = C[best[-1]] - C[best[0]]
        d = d / max(np.linalg.norm(d), 1e-9)
        ends = [
            (cand[k][2], cand[k][3]) if cand[k][2] @ d <= cand[k][3] @ d else (cand[k][3], cand[k][2]) for k in best
        ]
        order = np.argsort([e[0] @ d for e in ends])
        a = np.array([ends[k][0] for k in order])
        b = np.array([ends[k][1] for k in order])
        wt = np.array([cand[best[k]][4] for k in order])
        for s, e in _regular_runs(a @ d, cr_tol):
            if e - s >= min_dashes and _perspective_order(a[s:e] @ d, wt[s:e]):
                lines.append(DashedLine(a[s:e], b[s:e], wt[s:e]))
    return lines


def _perspective_order(t: np.ndarray, length: np.ndarray) -> bool:
    """Equal ground dashes seen in perspective: image gaps and dash lengths shrink toward one end."""
    gaps = np.diff(t)
    if np.any(gaps <= 0):
        return False
    dg = np.diff(gaps)
    if not (np.all(dg < 0) or np.all(dg > 0)):
        return False
    # dash lengths follow the gaps (both shrink with distance), allowing one exception for worn paint
    dl = np.diff(length)
    agree = np.sign(dl) == np.sign(dg[0])
    return int((~agree).sum()) <= 1


def _cross_ratio(t: np.ndarray) -> float:
    return float((t[2] - t[0]) * (t[3] - t[1]) / max((t[2] - t[1]) * (t[3] - t[0]), 1e-12))


def _regular_runs(t: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Maximal runs of consecutive dashes whose starts are equally spaced on the ground.

    Equally spaced ground points have the cross-ratio 4/3 for every four
    consecutive points under any perspective projection; a missing dash, a
    crosswalk stripe or a fragment of a solid line breaks it.
    """
    n = len(t)
    ok = [abs(_cross_ratio(t[i : i + 4]) - 4.0 / 3.0) < tol * 4.0 / 3.0 for i in range(n - 3)]
    runs, i = [], 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        runs.append((i, j + 4))
        i = j + 1
    return runs


def _line_residuals(cam: Camera, ln: DashedLine) -> np.ndarray | None:
    ga = cam.image_to_ground(ln.a)
    gb = cam.image_to_ground(ln.b)
    if not (np.all(np.isfinite(ga)) and np.all(np.isfinite(gb))):
        return None
    L = np.linalg.norm(ga - gb, axis=1)
    c = 0.5 * (ga + gb)
    p = np.linalg.norm(np.diff(c, axis=0), axis=1)
    if np.any(L <= 0) or np.any(p <= 0):
        return None
    wl = np.sqrt(ln.weight / ln.weight.max())
    rl = (np.log(L) - np.median(np.log(L))) * wl
    m = float(np.median(p))
    k = np.maximum(1.0, np.round(p / m))
    rp = np.log(p / (k * m)) * np.sqrt(wl[1:] * wl[:-1])
    return np.concatenate([rl, rp])


def marking_regularity(cam: Camera, lines: Sequence[DashedLine]) -> float:
    """RMS spread (in log units) of dash lengths and periods along the dashed lines; ``inf`` when undefined."""
    parts = [r for r in (_line_residuals(cam, ln) for ln in lines) if r is not None]
    if not parts:
        return float("inf")
    r = np.concatenate(parts)
    return float(np.sqrt(np.mean(r**2)))


def road_region(dets, min_score: float = 0.5, grow: float = 0.04) -> np.ndarray | None:
    """Image region swept by vehicle ground contacts: the convex hull of the bottom-centres of
    confident vehicle detections, dilated by ``grow`` times the image width. ``None`` without detections."""
    import cv2

    if dets is None or not len(dets):
        return None
    names = np.array(dets.class_names)
    veh = np.isin(names[dets.cls], ["car", "bus", "truck", "motorcycle"]) & (dets.score >= min_score)
    if veh.sum() < 10:
        return None
    b = dets.xyxy[veh]
    pts = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]]).astype(np.float32)
    w, h = dets.image_size
    out = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts).astype(np.int32), 1)
    k = max(3, int(grow * w)) | 1
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return out.astype(bool)

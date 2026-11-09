"""Metric scale from vehicle silhouettes (instance masks).

A passenger car is modelled as a rigid body resting on the ground and oriented
along its direction of travel: a lower body up to the beltline and a shorter,
narrower cabin above it. For a camera of known orientation, the convex hull of
the projected model explains the car's instance mask only at the right metric
scale; :func:`silhouette_scale` searches the scale factor that minimises the
silhouette cost over short image-space tracklets of confident, untruncated car
masks:

* the heading of a tracklet is the direction of its ground-contact trajectory;
* each mask's model centre is fitted with Nelder-Mead (two degrees of freedom);
* the silhouette cost is asymmetric (``5 |mask \\ hull| + 0.7 |hull \\ mask|``
  over the mask area, plus a boundary term): visible vehicle pixels must be
  covered, while the model may overhang where an occluder hides the vehicle.

The criterion determines the scale well but focal length and pitch only weakly:
over the family of cameras that share a horizon it prefers narrower fields of
view, so orientation must come from geometric evidence (vanishing points,
GeoCalib, a known field of view or correspondences).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..detections import DetectionSet
from ..geometry.camera import CalibrationError, Camera

__all__ = ["silhouette_scale", "silhouette_objective", "car_tracklets", "Tracklet"]

CAR_DIMS = (4.5, 1.85, 1.5, 0.25)  # length, width, height, lowest visible silhouette edge (m)
DOWN = 4


# ------------------------------------------------------------------ masks and tracklets
@dataclass
class _Mask:
    row: int
    x0: int
    y0: int
    raster: np.ndarray  # bool, downsampled crop
    dt: np.ndarray  # distance to the mask boundary (downsampled px)
    area: float  # downsampled px
    bottom: np.ndarray  # (2,) ground-contact pixel: bottom-centre of the mask (full-res px)


@dataclass
class Tracklet:
    """Image-space chain of one car's detections in consecutive processed frames.

    ``masks`` are the prepared masks of a few spread samples; ``bottom`` and
    ``time`` hold the box bottom-centres of the whole chain (for its heading).
    """

    masks: list[_Mask]
    bottom: np.ndarray
    time: np.ndarray


def _prepare(row: int, crop: tuple[int, int, np.ndarray]) -> _Mask | None:
    import cv2

    x0, y0, m = crop
    if m.sum() < 600:
        return None
    hs, ws = (m.shape[0] + DOWN - 1) // DOWN, (m.shape[1] + DOWN - 1) // DOWN
    small = cv2.resize(m.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_AREA) > 0
    edge = small.astype(np.uint8) - cv2.erode(small.astype(np.uint8), np.ones((3, 3), np.uint8))
    dt = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)
    ys, xs = np.nonzero(m)
    low = ys >= ys.max() - max(2, int(0.08 * (ys.max() - ys.min() + 1)))
    bottom = np.array([x0 + float(np.median(xs[low])), y0 + float(ys.max())])
    return _Mask(row, x0, y0, small, dt, float(small.sum()), bottom)


def car_tracklets(
    dets: DetectionSet,
    min_score: float = 0.6,
    min_len: int = 4,
    max_tracklets: int = 60,
    per_tracklet: int = 4,
    margin_px: int = 6,
    seed: int = 0,
) -> list[Tracklet]:
    """Tracklets of confident, untruncated car masks linked by box overlap (no camera needed)."""
    if dets.masks is None or len(dets) == 0:
        return []
    names = list(dets.class_names)
    if "car" not in names:
        return []
    w, h = dets.image_size
    car = names.index("car")
    b = dets.xyxy
    ok = (
        (dets.cls == car)
        & (dets.score >= min_score)
        & (b[:, 0] > margin_px)
        & (b[:, 1] > margin_px)
        & (b[:, 2] < w - 1 - margin_px)
        & (b[:, 3] < h - 1 - margin_px)
        & ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) < 0.15 * w * h)
    )
    ok &= np.array([dets.masks.has(int(i)) for i in range(len(dets))], bool)
    frames = np.unique(dets.frame[ok])
    by_frame = {int(f): np.flatnonzero(ok & (dets.frame == f)) for f in frames}
    stride = int(np.median(np.diff(frames))) if len(frames) > 1 else 1

    def iou(a: np.ndarray, bb: np.ndarray) -> np.ndarray:
        x1 = np.maximum(a[0], bb[:, 0])
        y1 = np.maximum(a[1], bb[:, 1])
        x2 = np.minimum(a[2], bb[:, 2])
        y2 = np.minimum(a[3], bb[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda q: (q[..., 2] - q[..., 0]) * (q[..., 3] - q[..., 1])  # noqa: E731
        return inter / np.maximum(area(a) + area(bb) - inter, 1e-9)

    chains: list[list[int]] = []
    open_chains: dict[int, list[int]] = {}
    for f in frames:
        rows = by_frame[int(f)]
        new_open: dict[int, list[int]] = {}
        used: set[int] = set()
        for _key, ch in sorted(open_chains.items(), key=lambda kv: -dets.score[kv[1][-1]]):
            last = ch[-1]
            if f - dets.frame[last] > 2 * stride or not len(rows):
                chains.append(ch)
                continue
            ov = iou(b[last], b[rows])
            ov[[k for k, r in enumerate(rows) if int(r) in used]] = 0.0
            k = int(np.argmax(ov))
            if ov[k] >= 0.3:
                used.add(int(rows[k]))
                new_open[int(rows[k])] = ch + [int(rows[k])]
            else:
                chains.append(ch)
        for r in rows:
            if int(r) not in used:
                new_open[int(r)] = [int(r)]
        open_chains = new_open
    chains.extend(open_chains.values())
    chains = [c for c in chains if len(c) >= min_len]
    # prefer long tracklets with image motion, spread over the clip
    rng = np.random.default_rng(seed)

    def motion(c: list[int]) -> float:
        bc = (b[c][:, :2] + b[c][:, 2:]) / 2
        return float(np.linalg.norm(bc[-1] - bc[0]))

    chains = [c for c in chains if motion(c) > 15.0]
    rng.shuffle(chains)
    chains = sorted(chains, key=len, reverse=True)[:max_tracklets]
    out: list[Tracklet] = []
    for c in chains:
        pick = np.unique(np.linspace(0, len(c) - 1, min(per_tracklet, len(c))).round().astype(int))
        masks = [m for m in (_prepare(c[i], dets.masks.crop(c[i])) for i in pick) if m is not None]
        if len(masks) >= 2:
            out.append(Tracklet(masks, np.array([_bottom_of_box(b[r]) for r in c]), dets.time[c].astype(np.float64)))
    return out


def _bottom_of_box(xyxy: np.ndarray) -> np.ndarray:
    return np.array([(xyxy[0] + xyxy[2]) / 2.0, xyxy[3]], np.float64)


# ------------------------------------------------------------------ silhouette cost
#: shape of the car silhouette model: "box" (one L x W x H box from ``gap``) or "car" (body to the beltline
#: plus a shorter, narrower cabin, whose silhouette matches passenger cars without a dimension bias)
SHAPE = "car"
BELT_FRAC, CABIN_L_FRAC, CABIN_W_FRAC, CABIN_SHIFT = 0.63, 0.55, 0.75, -0.04


def _rect(centre: np.ndarray, u: np.ndarray, v: np.ndarray, L: float, W: float) -> np.ndarray:
    return np.array([centre + s * L / 2 * u + t * W / 2 * v for s, t in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def _box_points(centre: np.ndarray, heading: float, L: float, W: float, H: float, gap: float) -> np.ndarray:
    u = np.array([math.cos(heading), math.sin(heading)])
    v = np.array([-u[1], u[0]])
    base = _rect(centre, u, v, L, W)
    if SHAPE == "box":
        return np.vstack([np.column_stack([base, np.full(4, gap)]), np.column_stack([base, np.full(4, H)])])
    belt = BELT_FRAC * H
    cabin = _rect(centre + CABIN_SHIFT * L * u, u, v, CABIN_L_FRAC * L, CABIN_W_FRAC * W)
    return np.vstack(
        [
            np.column_stack([base, np.zeros(4)]),
            np.column_stack([base, np.full(4, belt)]),
            np.column_stack([cabin, np.full(4, belt)]),
            np.column_stack([cabin, np.full(4, H)]),
        ]
    )


def _cost(cam: Camera, mk: _Mask, centre: np.ndarray, heading: float, dims: Sequence[float]) -> float:
    import cv2

    pts = cam.project(_box_points(centre, heading, *dims))
    if not np.all(np.isfinite(pts)):
        return 1e3
    local = (pts - np.array([mk.x0, mk.y0])) / DOWN
    hull = cv2.convexHull(local.astype(np.float32).reshape(-1, 1, 2))
    hs, ws = mk.raster.shape
    img = np.zeros((hs, ws), np.uint8)
    cv2.fillConvexPoly(img, np.round(hull).astype(np.int32), 1)
    hb = img.astype(bool)
    inter = float((hb & mk.raster).sum())
    miss = mk.area - inter
    extra = max(float(cv2.contourArea(hull)) - inter, 0.0)
    c = (5.0 * miss + 0.7 * extra) / max(mk.area, 1.0)
    hp = hull.reshape(-1, 2)
    xs = np.clip(np.round(hp[:, 0]).astype(int), 0, ws - 1)
    ys = np.clip(np.round(hp[:, 1]).astype(int), 0, hs - 1)
    return c + 0.05 * float(mk.dt[ys, xs].mean())


def _fit_centre(cam: Camera, mk: _Mask, heading: float, dims: Sequence[float], c0: np.ndarray, it: int = 30):
    from scipy.optimize import minimize

    res = minimize(
        lambda c: _cost(cam, mk, c, heading, dims),
        c0,
        method="Nelder-Mead",
        options={"maxiter": it, "xatol": 0.05, "fatol": 1e-3, "initial_simplex": [c0, c0 + [0.6, 0], c0 + [0, 0.6]]},
    )
    return res.x, float(res.fun)


def _initial_centre(cam: Camera, bottom: np.ndarray, heading: float, dims: Sequence[float]) -> np.ndarray | None:
    g = cam.image_to_ground(bottom[None])[0]
    if not np.all(np.isfinite(g)):
        return None
    ray = g - cam.center[:2]
    n = float(np.linalg.norm(ray))
    if n < 1e-6:
        return g
    r = ray / n
    a = heading - math.atan2(r[1], r[0])
    ext = abs(dims[0] * math.cos(a)) + abs(dims[1] * math.sin(a))
    return g + r * 0.5 * ext


def _heading(cam: Camera, tr: Tracklet) -> float | None:
    g = cam.image_to_ground(tr.bottom)
    ok = np.all(np.isfinite(g), axis=1)
    if ok.sum() < 3:
        return None
    g = g[ok]
    d = g[-1] - g[0]
    if np.linalg.norm(d) < 1.5:
        return None
    c = g - g.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    u = vt[0]
    if u @ d < 0:
        u = -u
    return float(math.atan2(u[1], u[0]))


def silhouette_objective(
    cam: Camera, tracklets: Sequence[Tracklet], dims: Sequence[float] = CAR_DIMS, cap: float = 2.5, it: int = 25
) -> tuple[float, int]:
    """Mean capped silhouette cost of car boxes over the tracklets' masks, and the number of masks used."""
    costs = []
    for tr in tracklets:
        hd = _heading(cam, tr)
        if hd is None:
            continue
        for mk in tr.masks:
            c0 = _initial_centre(cam, mk.bottom, hd, dims)
            if c0 is None:
                continue
            _, c = _fit_centre(cam, mk, hd, dims, c0, it)
            costs.append(min(c, cap))
    if len(costs) < 8:
        return float("inf"), len(costs)
    return float(np.mean(costs)), len(costs)


def silhouette_scale(
    cam: Camera,
    dets: DetectionSet,
    car_width_m: float = CAR_DIMS[1],
    max_tracklets: int = 40,
    seed: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Metric scale factor for a camera of known orientation, from car silhouettes.

    The camera is scaled about its optical centre (a pure change of the metric
    scale of the scene); the factor minimising the silhouette cost of car-sized
    boxes is found by golden-section search on its logarithm. Orientation is not
    changed: the silhouette criterion constrains the scale well but focal length
    and pitch only weakly, so these come from geometric evidence.
    """
    trs = car_tracklets(dets, max_tracklets=max_tracklets, seed=seed)
    if len(trs) < 6:
        raise CalibrationError(f"only {len(trs)} usable car tracklets with masks for a vehicle-size scale (need 6)")
    dims = (CAR_DIMS[0], car_width_m, CAR_DIMS[2], CAR_DIMS[3])

    def obj(logs: float) -> float:
        return silhouette_objective(cam.scaled(math.exp(logs)), trs, dims=dims, it=20)[0]

    grid = np.linspace(math.log(0.3), math.log(300.0), 31)  # scale-free cameras come with height 1 m
    vals = [obj(g) for g in grid]
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    for _ in range(14):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if obj(m1) < obj(m2):
            hi = m2
        else:
            lo = m1
    s = math.exp(0.5 * (lo + hi))
    cost, n = silhouette_objective(cam.scaled(s), trs, dims=dims, it=20)
    if not np.isfinite(cost) or cost > 1.2:
        raise CalibrationError(
            f"car silhouettes do not fit this camera (cost {cost:.2f}); the orientation is unreliable"
        )
    return s, {"scale": "vehicle silhouettes", "factor": s, "cost": cost, "n_masks": n, "n_tracklets": len(trs)}

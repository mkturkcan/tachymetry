"""Calibrated 3D-box silhouette fit over a vehicle track, and street axes for stopped vehicles.

Track fit
---------
A vehicle is rigid, so its box dimensions are shared by all frames of a track,
while the ground position and heading change from frame to frame. Following
the per-frame model of :mod:`tachymetry.geometry.silhouette`:

1. **Shared dimensions** ``(L, W, H, gap)`` by coordinate descent (steps of
   0.4 m then 0.15 m for the dimensions, 0.15 m then 0.07 m for the clearance)
   on a set of evidence frames: ``n_dim_frames`` frames taken at every third
   rank of the frames ordered by ``score * sqrt(mask area)`` (well observed
   and spread in time). Each candidate is scored by the sum of the lowest two
   thirds of the per-frame costs after a short 2-DoF centre fit, so that a
   minority of contaminated masks (merged with a shadow, partly occluded) does
   not drive the dimensions. The clearance is bounded by 0.42 m for vehicles up
   to 6 m long and 0.95 m beyond (the visible underside of a truck or bus is
   well above the road; projecting it at ``z = 0`` would move the vehicle
   away from the camera by metres).
2. **Per-frame pose**: the 2-DoF centre and a heading correction within
   +-18 degrees of the reference heading (trajectory tangent; street axis for
   stopped frames), under a quadratic prior, fused along the vehicle axis with
   the envelope anchor in inverse-variance proportion to the cost curvature
   (:func:`tachymetry.geometry.silhouette.fit_centre`).

Street axes
-----------
Without a street map, the directions of the road at a position are taken from
the moving traffic of the same video (:class:`StreetAxes`): headings of moving
vehicle samples, pooled as axial directions (``theta`` and ``theta + pi`` are
the same axis). Globally, the one or two dominant axes are the modes of a
smoothed doubled-angle histogram; locally (within ``radius_m`` of a query
position), the axial mean of nearby samples is added when they agree. With no
moving traffic the axes fall back to the world ``x`` and ``y`` axes, which an
automatic calibration aligns with the dominant lane direction. A stopped
vehicle's reference heading is snapped to the nearest axis when within
``snap_deg``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry.silhouette import MaskRaster, fit_centre, project_box, silhouette_fill

#: coordinate-descent order: length, width, height, clearance
DIM_ORDER = (0, 1, 2, 3)

__all__ = ["BoxFitResult", "fit_track_boxes", "snap_to_axes", "StreetAxes", "street_axes", "axial_diff"]


def axial_diff(a: Any, b: Any) -> Any:
    """Signed difference ``a - b`` of axial angles (radians, modulo pi), in ``[-pi/2, pi/2)``."""
    return (np.asarray(a) - np.asarray(b) + np.pi / 2) % np.pi - np.pi / 2


def snap_to_axes(theta: float, axes: list[float] | None, tol_deg: float = 25.0) -> float:
    """``theta`` rotated onto the nearest axis (modulo pi) when within ``tol_deg``, else unchanged."""
    if not axes or not np.isfinite(theta):
        return theta
    d = [float(axial_diff(theta, a)) for a in axes]
    k = int(np.argmin(np.abs(d)))
    return theta - d[k] if abs(math.degrees(d[k])) <= tol_deg else theta


@dataclass
class StreetAxes:
    """Dominant directions of moving traffic, globally and around a position (axial angles, radians)."""

    xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    radius_m: float = 15.0
    fallback: tuple[float, ...] = (0.0, math.pi / 2)
    axes: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, float).reshape(-1, 2)
        self.theta = np.asarray(self.theta, float).reshape(-1)
        self.axes = self._global_axes() or list(self.fallback)

    @classmethod
    def from_samples(cls, xy: np.ndarray, theta: np.ndarray, radius_m: float = 15.0) -> StreetAxes:
        return cls(xy, theta, radius_m)

    def _global_axes(self) -> list[float]:
        if len(self.theta) < 20:
            return []
        nb = 72
        z = np.mod(2.0 * self.theta, 2 * np.pi)
        h = np.bincount(np.minimum((z / (2 * np.pi) * nb).astype(int), nb - 1), minlength=nb).astype(float)
        k = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
        hs = np.array([np.sum(h[(i + np.arange(-4, 5)) % nb] * k) for i in range(nb)])
        peaks = [i for i in range(nb) if hs[i] >= hs[(i - 1) % nb] and hs[i] > hs[(i + 1) % nb]]
        peaks.sort(key=lambda i: -hs[i])
        out: list[float] = []
        for i in peaks:
            if hs[i] < 0.25 * hs[peaks[0]] or len(out) >= 2:
                break
            c = (i + 0.5) / nb * np.pi  # axial angle of the bin centre
            near = np.abs(axial_diff(self.theta, c)) < math.radians(15.0)
            if near.sum() >= 5:
                c = float(np.angle(np.mean(np.exp(2j * self.theta[near]))) / 2)
            if all(abs(math.degrees(float(axial_diff(c, o)))) >= 30.0 for o in out):
                out.append(float(c))
        return out

    def at(self, xy: Any) -> list[float]:
        """Axes to snap to at ground position ``xy``: the local traffic direction (if coherent) and the global axes."""
        out = list(self.axes)
        if len(self.theta) >= 10 and np.all(np.isfinite(xy)):
            d = np.linalg.norm(self.xy - np.asarray(xy, float)[None, :2], axis=1)
            near = d < self.radius_m
            if near.sum() >= 10:
                w = np.exp(-0.5 * (d[near] / (0.5 * self.radius_m)) ** 2)
                m = np.sum(w * np.exp(2j * self.theta[near])) / np.sum(w)
                if abs(m) >= 0.8:
                    loc = float(np.angle(m) / 2)
                    if all(abs(math.degrees(float(axial_diff(loc, o)))) >= 3.0 for o in out):
                        out.insert(0, loc)
        return out


@dataclass
class BoxFitResult:
    centre: np.ndarray  #: (n, 2) fitted centres, NaN where not fitted
    heading: np.ndarray  #: (n,) fitted headings (rad), NaN where not fitted
    cost: np.ndarray  #: (n,) silhouette cost, inf where not fitted
    fill: np.ndarray  #: (n,) share of the fitted silhouette covered by the mask, NaN where not fitted
    miss: np.ndarray  #: (n,) share of the mask outside the fitted silhouette, NaN where not fitted
    L: float
    W: float
    H: float
    gap: float
    seconds: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def ok(self, max_cost: float) -> np.ndarray:
        return np.isfinite(self.cost) & (self.cost < max_cost)


def fit_track_boxes(
    camera: Any,
    rasters: list[MaskRaster | None],
    headings: np.ndarray,
    c0: np.ndarray,
    weights: np.ndarray,
    dims0: tuple[float, float, float],
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    n_dim_frames: int = 6,
    anchors: np.ndarray | None = None,
    anchor_sigma: np.ndarray | None = None,
    pose_frames: np.ndarray | None = None,
    gap0: float = 0.15,
    max_iter: int = 40,
    fixed_dims: tuple[float, float, float, float] | None = None,
    refine_heading: np.ndarray | None = None,
) -> BoxFitResult | None:
    """Shared box dimensions and per-frame poses of one vehicle track (see the module docstring).

    ``rasters[i]`` is the prepared mask of frame ``i`` or ``None`` (not
    fitted); ``headings`` are the reference headings (rad), ``c0`` the initial
    centres (world metres), ``weights`` the evidence weights, ``dims0`` the
    initial ``(L, W, H)`` and ``bounds`` their ranges. With ``anchors`` and
    ``anchor_sigma`` the per-frame centres are fused with the anchors along
    the vehicle axis. ``pose_frames`` (bool) restricts the per-frame pose fit
    to a subset of the rasterised frames (the dimension fit may use all).
    ``fixed_dims = (L, W, H, gap)`` skips the dimension fit; ``refine_heading``
    (bool per frame) limits the heading refinement to those frames (the others
    keep their reference heading).
    """
    t0 = time.perf_counter()
    n = len(rasters)
    idx = [i for i, r in enumerate(rasters) if r is not None]
    if not idx:
        return None
    headings = np.asarray(headings, float)
    weights = np.asarray(weights, float)
    sub = sorted(idx, key=lambda i: -weights[i])[: n_dim_frames * 3 : 3] or idx
    L, W, H = (float(v) for v in dims0)
    (lo_l, hi_l), (lo_w, hi_w), (lo_h, hi_h) = bounds
    L, W, H = float(np.clip(L, lo_l, hi_l)), float(np.clip(W, lo_w, hi_w)), float(np.clip(H, lo_h, hi_h))
    gap = float(gap0)
    cs = {i: np.asarray(c0[i], float) for i in idx}
    n_eval = [0]

    def centre_fit(i: int, d: list[float], iters: int) -> tuple[np.ndarray, float]:
        c, cost, _ = fit_centre(camera, rasters[i], headings[i], d[0], d[1], d[2], d[3], cs[i], max_iter=iters)
        n_eval[0] += 1
        return c, cost

    cache: dict[tuple[float, ...], float] = {}

    def dimcost(d: list[float]) -> float:
        key = tuple(round(v, 4) for v in d)
        if key not in cache:
            costs = sorted(centre_fit(i, d, 12)[1] for i in sub)
            k = max(1, (len(costs) * 2 + 2) // 3)
            cache[key] = float(sum(costs[:k]))
        return cache[key]

    if fixed_dims is not None:
        L, W, H, gap = (float(v) for v in fixed_dims)
    for _ in range(0 if fixed_dims is not None else 2):
        for i in sub:
            cs[i] = centre_fit(i, [L, W, H, gap], max_iter)[0]
        cache.clear()  # the warm starts changed
        gmax = 0.95 if L > 6.0 else 0.42
        ranges = ((lo_l, hi_l), (lo_w, hi_w), (lo_h, hi_h), (0.08, gmax))
        gap = float(np.clip(gap, 0.08, gmax))
        for dim in DIM_ORDER:
            for step in (0.4, 0.15) if dim < 3 else (0.15, 0.07):
                improved = True
                while improved:
                    improved = False
                    base = [L, W, H, gap]
                    c_base = dimcost(base)
                    for sgn in (1, -1):
                        cand = list(base)
                        cand[dim] += sgn * step
                        lo, hi = ranges[dim]
                        if not (lo <= cand[dim] <= hi):
                            continue
                        if dimcost(cand) < c_base - 1e-4:
                            L, W, H, gap = cand
                            improved = True
                            break
    centre = np.full((n, 2), np.nan)
    heading = np.full(n, np.nan)
    cost = np.full(n, np.inf)
    fill = np.full(n, np.nan)
    miss = np.full(n, np.nan)
    pose = idx if pose_frames is None else [i for i in idx if pose_frames[i]]
    for i in pose:
        lp = None
        if anchors is not None and anchor_sigma is not None:
            a, s = np.asarray(anchors[i], float), float(anchor_sigma[i])
            if np.all(np.isfinite(a)) and np.isfinite(s) and s > 0:
                lp = (a, s)
        c, f, th = fit_centre(
            camera,
            rasters[i],
            headings[i],
            L,
            W,
            H,
            gap,
            cs.get(i, c0[i]),
            max_iter=max_iter,
            opt_heading=True if refine_heading is None else bool(refine_heading[i]),
            lon_prior=lp,
        )
        centre[i], cost[i], heading[i] = c, f, th
        fill[i], miss[i] = silhouette_fill(project_box(camera, c, th, L, W, H, gap), rasters[i])
    return BoxFitResult(
        centre,
        heading,
        cost,
        fill,
        miss,
        L,
        W,
        H,
        gap,
        time.perf_counter() - t0,
        {"dim_frames": len(sub), "pose_frames": len(pose), "centre_fits": n_eval[0] + len(pose)},
    )


def street_axes(raw_tracks: list[np.ndarray], dets: Any, obs: Any, cfg: Any, min_speed_mps: float = 2.5) -> StreetAxes:
    """Street axes from the moving vehicles among raw associations (before track repair).

    Each raw track of the vehicle super-class whose ground contact moves more
    than 8 m contributes the directions of its smoothed contact path where it
    is faster than ``min_speed_mps``.
    """
    from ..config import SUPER_CLASS
    from .signal import smooth_kinematics

    xs, ths = [], []
    for rows in raw_tracks:
        rows = rows[obs.valid[rows]]
        if len(rows) < 8:
            continue
        names = [dets.class_names[int(c)] for c in dets.cls[rows]]
        if SUPER_CLASS.get(max(set(names), key=names.count)) != "vehicle":
            continue
        order = np.argsort(dets.time[rows], kind="stable")
        rows = rows[order]
        t = dets.time[rows].astype(float)
        if len(np.unique(t)) != len(t):
            _, first = np.unique(t, return_index=True)
            rows, t = rows[first], t[first]
        c = obs.contact[rows]
        if np.linalg.norm(np.nanmedian(c[-5:], axis=0) - np.nanmedian(c[:5], axis=0)) < 8.0:
            continue
        p, v, _ = smooth_kinematics(t, c, None, cfg.median_window_s, 1.5, 1, cfg.interp_gap_s)
        sp = np.linalg.norm(np.nan_to_num(v), axis=1)
        m = (sp > min_speed_mps) & np.all(np.isfinite(p), axis=1)
        xs.append(p[m])
        ths.append(np.arctan2(v[m, 1], v[m, 0]))
    if not xs:
        return StreetAxes()
    return StreetAxes(np.concatenate(xs), np.concatenate(ths))

"""Image silhouettes of upright boxes over a calibrated ground plane, compared with instance masks.

Model
-----
A road vehicle is approximated by an upright box of length ``L``, width ``W``
and roof height ``H`` (metres above the ground) standing on the ground plane
``z = 0`` with heading ``theta`` (radians, counter-clockwise from world ``+x``).
Its image under a calibrated :class:`~tachymetry.geometry.camera.Camera`
(pinhole projection followed by the lens model) is the convex hull ``S`` of the
eight projected corners. The bottom face is placed at the clearance ``gap``
rather than at ``z = 0``: the lowest edge an instance mask shows between the
wheels is the sill or the underbody (tyres and the dark underside are often not
segmented, most of all on trucks and buses), and projecting that edge at
``z = 0`` would displace the vehicle away from the camera by
``gap * range / camera_height``.

Silhouette cost
---------------
For a mask ``M`` in the image frame ``F``::

    c = w_miss * |M \\ S| / |M|  +  w_extra * |(S n F) \\ M| / |M|  +  w_edge * chamfer(dS, dM)

The asymmetry encodes occlusion. Every segmented pixel must lie inside the box
silhouette (``w_miss = 5``), while the silhouette may extend beyond the mask
where an occluder or the frame edge removed vehicle pixels (``w_extra = 0.7``).
A box silhouette is also more angular than a vehicle body, so a symmetric
area cost would shrink the dimensions below the true size. Pixels outside the
frame do not enter the cost. Area overlap constrains the yaw only weakly; the
chamfer term (mean distance from points sampled along the silhouette outline to
the mask outline, in raster pixels) adds the alignment of the two boundaries.
It is truncated at ``edge_cap`` raster pixels, so outline sections hidden by an
occluder contribute a constant instead of pulling the box towards the
occluder's edge.

Evaluation is local to the mask: :class:`MaskRaster` stores the mask crop
block-averaged by ``down`` (4 by default) with per-row prefix sums, so
``|M n S|`` is a sum of exact row-span integrals of the convex polygon
(``O(rows)`` per evaluation, continuous in the box parameters along the rows),
and ``|S n F|`` is the analytic area of the polygon clipped to the frame. No
full-frame raster is built per evaluation.

Fit
---
:func:`fit_centre` minimises the cost over the ground position of the box (and
optionally a heading correction within ``+-heading_lim_deg`` under a quadratic
prior) by Nelder-Mead from an initial position. The cost is nearly flat along
the vehicle's long axis whenever one end of the vehicle is hidden; with
``lon_prior = (anchor_xy, sigma_m)`` the fitted position along that axis is
fused with an external anchor in inverse-variance proportion, the variance of
the fit being derived from the curvature of the cost along the axis (a
well-determined axis keeps the fit, a flat one takes the anchor). The position
across the axis and the heading are left to the fit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "project_box",
    "box_ground_corners",
    "MaskRaster",
    "silhouette_cost",
    "silhouette_fill",
    "fit_centre",
    "LON_PROBE_M",
]

#: probe step (m) of the cost curvature along the vehicle axis in the longitudinal fusion
LON_PROBE_M = 0.6
#: silhouette-cost noise as a fraction of the cost level (longitudinal fusion)
LON_COST_FRAC = 0.03
#: Nelder-Mead variable scales: positions in units of 0.1 m, heading in units of 1 degree
_SXY = 0.1
_STH = math.radians(1.0)


def box_ground_corners(centre_xy: Any, heading: float, L: float, W: float) -> np.ndarray:
    """``(4, 2)`` footprint corners: front-left, rear-left, rear-right, front-right."""
    c, s = math.cos(heading), math.sin(heading)
    fx, fy = 0.5 * L * c, 0.5 * L * s
    lx, ly = -0.5 * W * s, 0.5 * W * c
    x, y = float(centre_xy[0]), float(centre_xy[1])
    return np.array(
        [[x + fx + lx, y + fy + ly], [x - fx + lx, y - fy + ly], [x - fx - lx, y - fy - ly], [x + fx - lx, y + fy - ly]]
    )


def project_box(
    camera: Any, centre_xy: Any, heading: float, L: float, W: float, H: float, gap: float = 0.0
) -> np.ndarray:
    """Raw-pixel image ``(8, 2)`` of an upright box.

    Rows 0-3 are the footprint corners (front-left, rear-left, rear-right,
    front-right) lifted to height ``gap``, the lowest visible silhouette edge;
    rows 4-7 are the same corners at the roof height ``H``. World metres,
    ``heading`` in radians. Corners behind the camera are NaN.
    """
    g = box_ground_corners(centre_xy, heading, L, W)
    X = np.empty((8, 3))
    X[:4, :2] = g
    X[4:, :2] = g
    X[:4, 2] = gap
    X[4:, 2] = H
    return camera.project(X)


class MaskRaster:
    """One instance mask prepared for repeated silhouette comparisons.

    ``crop`` is ``(x0, y0, mask)`` with ``mask`` a boolean array in
    full-resolution pixels whose top-left pixel is ``(x0, y0)`` (the format of
    :meth:`tachymetry.detections.MaskStore.crop`). The crop is padded by
    ``pad`` raster pixels, block-averaged by ``down`` on a grid aligned with the
    full image (so results do not depend on the crop origin), thresholded at
    one half, and summarised by per-row prefix sums and the distance transform
    to the mask outline. ``image_size = (width, height)`` excludes silhouette
    area and outline samples outside the frame; outline pixels on the frame
    border are truncation, not object boundary, and are not used.

    Raster coordinates: the raster pixel ``(i, j)`` (row, column) is centred on
    the full-resolution point ``(x, y) = (X0 + down * j + (down - 1) / 2,
    Y0 + down * i + (down - 1) / 2)``.
    """

    def __init__(
        self,
        crop: tuple[int, int, np.ndarray],
        down: int = 4,
        image_size: tuple[int, int] | None = None,
        pad: int = 3,
    ):
        import cv2

        x0, y0, m = crop
        m = np.asarray(m, bool)
        d = max(int(down), 1)
        self.down = d
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        h, w = m.shape
        X0 = (int(x0) // d - pad) * d
        Y0 = (int(y0) // d - pad) * d
        ws = -(-(int(x0) + w - X0) // d) + pad
        hs = -(-(int(y0) + h - Y0) // d) + pad
        full = np.zeros((hs * d, ws * d), np.float32)
        full[y0 - Y0 : y0 - Y0 + h, x0 - X0 : x0 - X0 + w] = m
        small = cv2.resize(full, (ws, hs), interpolation=cv2.INTER_AREA) if d > 1 else full
        ms = small >= 0.5
        if not ms.any() and m.any():  # sub-raster-pixel mask: keep its densest cell
            ms.flat[int(np.argmax(small))] = True
        self.X0, self.Y0 = X0, Y0
        self.offset = 0.5 * (d - 1)
        self.mask = ms
        self.shape = ms.shape
        self.area = float(ms.sum())
        self.full_area = float(m.sum())
        row = ms.astype(np.float64)
        self._row = row
        self._prefix = np.zeros((hs, ws + 1))
        np.cumsum(row, axis=1, out=self._prefix[:, 1:])
        # outline: mask pixels with a background 8-neighbour, excluding the frame border
        m8 = ms.astype(np.uint8)
        edge = (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)).astype(bool)
        if self.image_size is not None:
            fw, fh = self.image_size
            jj = (X0 + d * np.arange(ws) + self.offset) / 1.0
            ii = (Y0 + d * np.arange(hs) + self.offset) / 1.0
            near_x = (jj < d) | (jj > fw - 1 - d)
            near_y = (ii < d) | (ii > fh - 1 - d)
            edge[:, near_x] = False
            edge[near_y, :] = False
        self.has_edge = bool(edge.any())
        self.dt = (
            cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float64)
            if self.has_edge
            else np.full((hs, ws), 1e3)
        )
        # frame rectangle in raster coordinates
        if self.image_size is not None:
            fw, fh = self.image_size
            self.frame = np.array(
                [self.to_raster(np.array([[-0.5, -0.5]]))[0], self.to_raster(np.array([[fw - 0.5, fh - 0.5]]))[0]]
            )
        else:
            self.frame = None

    # ---------------------------------------------------------------- coords
    def to_raster(self, uv: np.ndarray) -> np.ndarray:
        """Full-resolution pixel coordinates ``(N, 2)`` to raster coordinates ``(x, y)``."""
        uv = np.asarray(uv, np.float64)
        return np.column_stack([(uv[:, 0] - self.X0 - self.offset), (uv[:, 1] - self.Y0 - self.offset)]) / self.down

    # ----------------------------------------------------------------- terms
    def overlap(self, poly: np.ndarray) -> float:
        """``|M n P|`` in raster pixels for a convex polygon ``P`` given in raster coordinates (in order)."""
        hs, ws = self.shape
        y_lo = max(int(math.ceil(float(poly[:, 1].min()))), 0)
        y_hi = min(int(math.floor(float(poly[:, 1].max()))), hs - 1)
        if y_hi < y_lo:
            return 0.0
        ys = np.arange(y_lo, y_hi + 1, dtype=np.float64)
        p1 = poly
        p2 = np.roll(poly, -1, axis=0)
        dy = p2[:, 1] - p1[:, 1]
        ok = np.abs(dy) > 1e-12
        if not ok.any():
            return 0.0
        p1, p2, dy = p1[ok], p2[ok], dy[ok]
        t = (ys[:, None] - p1[None, :, 1]) / dy[None, :]
        inside = (t >= 0.0) & (t <= 1.0)
        xi = p1[None, :, 0] + t * (p2[None, :, 0] - p1[None, :, 0])
        xl = np.where(inside, xi, np.inf).min(axis=1)
        xr = np.where(inside, xi, -np.inf).max(axis=1)
        good = np.isfinite(xl) & np.isfinite(xr) & (xr > xl)
        if not good.any():
            return 0.0
        rows = ys[good].astype(np.int64)
        return float(np.sum(np.maximum(self._integral(rows, xr[good]) - self._integral(rows, xl[good]), 0.0)))

    def _integral(self, rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Mask mass of each row from the left raster edge to ``x`` (piecewise-linear in ``x``)."""
        ws = self.shape[1]
        xc = np.clip(x + 0.5, 0.0, float(ws))
        k = np.minimum(xc.astype(np.int64), ws - 1)
        return self._prefix[rows, k] + (xc - k) * self._row[rows, k]

    def chamfer(self, poly: np.ndarray, spacing: float = 1.5, cap: float = 5.0) -> float:
        """Mean truncated distance (raster pixels) from points along the outline of ``poly`` to the mask outline."""
        p1 = poly
        p2 = np.roll(poly, -1, axis=0)
        seg = np.linalg.norm(p2 - p1, axis=1)
        n = np.maximum(np.ceil(seg / spacing).astype(np.int64), 1)
        idx = np.repeat(np.arange(len(poly)), n)
        frac = (np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
        pts = p1[idx] + frac[:, None] * (p2[idx] - p1[idx])
        if self.frame is not None:
            (fx0, fy0), (fx1, fy1) = self.frame
            keep = (pts[:, 0] >= fx0) & (pts[:, 0] <= fx1) & (pts[:, 1] >= fy0) & (pts[:, 1] <= fy1)
            if not keep.any():
                return cap
            pts = pts[keep]
        hs, ws = self.shape
        cx = np.clip(pts[:, 0], 0.0, ws - 1.0)
        cy = np.clip(pts[:, 1], 0.0, hs - 1.0)
        out = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        x0 = np.minimum(cx.astype(np.int64), ws - 2) if ws > 1 else np.zeros(len(cx), np.int64)
        y0 = np.minimum(cy.astype(np.int64), hs - 2) if hs > 1 else np.zeros(len(cy), np.int64)
        fx = cx - x0
        fy = cy - y0
        x1 = np.minimum(x0 + 1, ws - 1)
        y1 = np.minimum(y0 + 1, hs - 1)
        dt = self.dt
        v = (dt[y0, x0] * (1 - fx) + dt[y0, x1] * fx) * (1 - fy) + (dt[y1, x0] * (1 - fx) + dt[y1, x1] * fx) * fy
        return float(np.mean(np.minimum(v + out, cap)))

    def clipped_area(self, poly: np.ndarray) -> float:
        """Area (raster pixels) of the convex polygon ``poly`` inside the frame."""
        x, y = poly[:, 0], poly[:, 1]
        if self.frame is None or (
            x.min() >= self.frame[0, 0] and y.min() >= self.frame[0, 1] and x.max() <= self.frame[1, 0]
            and y.max() <= self.frame[1, 1]
        ):  # fmt: skip
            return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        import cv2

        (fx0, fy0), (fx1, fy1) = self.frame
        rect = np.array([[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]], np.float32)
        area, _ = cv2.intersectConvexConvex(poly.astype(np.float32), rect)
        return max(float(area), 0.0)


def _hull(pts: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)


def silhouette_cost(
    pts8: np.ndarray,
    raster: MaskRaster,
    w_miss: float = 5.0,
    w_extra: float = 0.7,
    w_edge: float = 0.05,
    edge_cap: float = 5.0,
) -> float:
    """Asymmetric silhouette cost of projected box corners ``pts8`` (raw pixels) against ``raster``.

    See the module docstring. Returns ``1e6`` when a corner did not project
    (behind the camera) or the mask is empty.
    """
    pts8 = np.asarray(pts8, np.float64)
    if raster.area <= 0 or not np.all(np.isfinite(pts8)):
        return 1e6
    poly = _hull(raster.to_raster(pts8))
    if len(poly) < 3:
        return 1e6
    inter = raster.overlap(poly)
    miss = max(raster.area - inter, 0.0)
    extra = max(raster.clipped_area(poly) - inter, 0.0)
    c = (w_miss * miss + w_extra * extra) / raster.area
    if w_edge:
        c += w_edge * raster.chamfer(poly, cap=edge_cap)
    return float(c)


def silhouette_fill(pts8: np.ndarray, raster: MaskRaster) -> float:
    """Share of the in-frame box silhouette covered by the mask, ``|M n S| / |S n F|`` (NaN if undefined).

    A fully visible vehicle fills most of its box silhouette (the box is more
    angular than the body); an occluder lowers the share in proportion to the
    hidden part of the silhouette.
    """
    pts8 = np.asarray(pts8, np.float64)
    if raster.area <= 0 or not np.all(np.isfinite(pts8)):
        return float("nan")
    poly = _hull(raster.to_raster(pts8))
    if len(poly) < 3:
        return float("nan")
    a = raster.clipped_area(poly)
    return float(raster.overlap(poly) / a) if a > 0 else float("nan")


def fit_centre(
    camera: Any,
    raster: MaskRaster,
    heading: float,
    L: float,
    W: float,
    H: float,
    gap: float,
    c0: Any,
    max_iter: int = 40,
    opt_heading: bool = False,
    heading_lim_deg: float = 18.0,
    prior_w: float = 1.5,
    lon_prior: tuple[Any, float] | None = None,
    step_m: float = 0.4,
    step_deg: float = 4.0,
    **cost_kw: Any,
) -> tuple[np.ndarray, float, float]:
    """Ground position (and optionally heading) of a box of fixed size that best explains a mask.

    Nelder-Mead over the box centre from ``c0`` (initial simplex steps of
    ``step_m``; tolerances 0.05 m and, with ``opt_heading``, 0.5 degrees).
    With ``opt_heading`` the heading is refined within ``+-heading_lim_deg``
    of ``heading`` under the prior ``prior_w * (d_heading / lim)**2`` (the
    reference heading is the trajectory tangent for moving vehicles and the
    street axis for stopped ones). With ``lon_prior = (anchor_xy, sigma_m)``
    the position along the fitted vehicle axis is fused with the anchor in
    inverse-variance proportion to the cost curvature along that axis (see the
    module docstring). Returns ``(centre_xy, cost, heading)``; ``cost``
    includes the heading prior.
    """
    from scipy.optimize import minimize

    c0 = np.asarray(c0, np.float64).reshape(2)
    lim = math.radians(heading_lim_deg)

    def sil(c: np.ndarray, th: float) -> float:
        pts = project_box(camera, c, th, L, W, H, gap)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return silhouette_cost(pts, raster, **cost_kw)

    s = step_m / _SXY
    if not opt_heading:
        res = minimize(
            lambda p: sil(c0 + p * _SXY, heading),
            np.zeros(2),
            method="Nelder-Mead",
            options={
                "maxiter": int(max_iter),
                "xatol": 0.5,
                "fatol": 1e-3,
                "initial_simplex": np.array([[0.0, 0.0], [s, 0.0], [0.0, s]]),
            },
        )
        c_out = c0 + res.x * _SXY
        th_out = float(heading)
        cost = float(res.fun)
    else:
        sth = math.radians(step_deg) / _STH

        def f3(p: np.ndarray) -> float:
            dth = float(np.clip(p[2] * _STH, -lim, lim))
            return sil(c0 + p[:2] * _SXY, heading + dth) + prior_w * (dth / lim) ** 2

        res = minimize(
            f3,
            np.zeros(3),
            method="Nelder-Mead",
            options={
                "maxiter": int(max_iter * 1.6),
                "xatol": 0.5,
                "fatol": 1e-3,
                "initial_simplex": np.array([[0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, sth]], np.float64),
            },
        )
        c_out = c0 + res.x[:2] * _SXY
        th_out = float(heading + np.clip(res.x[2] * _STH, -lim, lim))
        cost = float(res.fun)
    if lon_prior is not None:
        a_xy, sig_m = lon_prior
        a_xy = np.asarray(a_xy, np.float64).reshape(2)
        if np.all(np.isfinite(a_xy)) and np.isfinite(sig_m) and sig_m > 0:
            u = np.array([math.cos(th_out), math.sin(th_out)])
            c_mid = sil(c_out, th_out)
            curv = (sil(c_out + LON_PROBE_M * u, th_out) + sil(c_out - LON_PROBE_M * u, th_out) - 2.0 * c_mid) / (
                LON_PROBE_M**2
            )
            sig_cost = LON_COST_FRAC * max(c_mid, 1e-6)
            var_fit = (2.0 * sig_cost / curv) if curv > 1e-12 else np.inf
            var_a = float(sig_m) ** 2
            w_fit = var_a / (var_a + var_fit) if np.isfinite(var_fit) else 0.0
            s_off = float((c_out - a_xy) @ u)
            c_out = c_out - (1.0 - w_fit) * s_off * u
    return c_out, cost, th_out

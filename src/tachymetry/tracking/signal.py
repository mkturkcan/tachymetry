"""Time-aware signal processing for irregularly sampled tracks.

All filters operate in *time*, not in samples: windows are given in seconds,
tracks with missing detections are resampled onto a uniform grid before
filtering, and filters never pad with zeros (edges are fitted, not padded).
"""

from __future__ import annotations

import numpy as np

BRIDGE_TOL_MPS = 1.5  #: chord-versus-motion disagreement (m/s) above which a gap bridge is not differentiated

__all__ = ["odd", "running_median", "hampel", "smooth_kinematics", "segments_by_gap", "fill_angles", "wrap"]


def odd(n: float, lo: int = 3) -> int:
    k = max(lo, int(round(n)))
    return k if k % 2 else k + 1


def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to ``(-pi, pi]``."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def segments_by_gap(t: np.ndarray, max_gap: float) -> list[np.ndarray]:
    """Index arrays of runs whose consecutive samples are at most ``max_gap`` apart."""
    if len(t) == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    return np.split(np.arange(len(t)), cuts)


def running_median(t: np.ndarray, x: np.ndarray, window_s: float) -> np.ndarray:
    """Centred running median over a time window (edges use the available half-window)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    for i in range(len(t)):
        out[i] = np.nanmedian(x[lo[i] : hi[i]], axis=0)
    return out


def hampel(
    t: np.ndarray, xy: np.ndarray, window_s: float, n_sigma: float, min_dev: float
) -> tuple[np.ndarray, np.ndarray]:
    """Hampel identifier in time: replace outliers by the local median. Returns ``(xy, repaired_mask)``."""
    xy = np.asarray(xy, dtype=np.float64).copy()
    n = len(t)
    rep = np.zeros(n, dtype=bool)
    if n < 5:
        return xy, rep
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    src = xy.copy()
    for i in range(n):
        w = src[lo[i] : hi[i]]
        if len(w) < 5:
            continue
        med = np.median(w, axis=0)
        mad = np.median(np.abs(w - med), axis=0) * 1.4826 + 1e-6
        dev = np.abs(src[i] - med)
        if np.any(dev > n_sigma * mad) and np.linalg.norm(src[i] - med) > min_dev:
            xy[i] = med
            rep[i] = True
    return xy, rep


def _savgol(y: np.ndarray, win: int, order: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    from scipy.signal import savgol_filter

    n = len(y)
    if n < 3:
        if deriv == 0:
            return y.copy()
        if n == 2:
            return np.repeat((y[1:] - y[:1]) / delta, 2, axis=0)
        return np.zeros_like(y)
    win = min(win, n if n % 2 else n - 1)
    order = min(order, win - 1)
    if win <= order:
        order = win - 1
    return savgol_filter(y, win, order, deriv=deriv, delta=delta, axis=0, mode="interp")


def smooth_kinematics(
    t: np.ndarray,
    xy: np.ndarray,
    weights_ok: np.ndarray | None,
    median_s: float,
    window_s: float,
    order: int,
    max_interp_s: float,
    dt: float | None = None,
    vel_gap_s: float | None = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-phase position and velocity at the sample times ``t``.

    Samples marked ``False`` in ``weights_ok`` (truncated boxes, repaired
    outliers) contribute no measurement; their positions are interpolated from
    their neighbours. Each gap-free run is resampled to a uniform grid of step
    ``dt``, median-filtered over ``median_s`` (edges replicated), smoothed with a
    Savitzky-Golay filter over ``window_s`` and differentiated analytically.

    Positions are bridged linearly across gaps up to ``max_interp_s``. The
    slope of a bridge is the chord between the last measurement before and the
    first after the gap, so an inconsistency between those two positions (a
    re-acquired box of different extent, a partly visible object) appears as
    speed on both sides of the gap. For every gap longer than ``vel_gap_s`` the
    chord velocity is compared with the mean of the velocities measured on its
    two sides (each stretch differentiated on its own; the mean is the chord of
    a constant-acceleration motion). When they differ by more than
    ``max(BRIDGE_TOL_MPS, 0.3 x |mean|)`` the bridge is not used for velocity: within
    half a smoothing window of the gap each side keeps its own derivative, and
    samples inside the gap take velocities interpolated in time between the two
    sides. Consistent bridges are kept, since a one-sided derivative is noisier.
    Stretches with fewer than ``order + 3`` grid points are not tested.

    Returns ``(xy_smooth, velocity, segment_id)``.
    """
    from scipy.ndimage import median_filter

    t = np.asarray(t, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    n = len(t)
    ok = np.ones(n, bool) if weights_ok is None else np.asarray(weights_ok, bool) & np.all(np.isfinite(xy), axis=1)
    pos = np.full((n, 2), np.nan)
    vel = np.full((n, 2), np.nan)
    seg_id = np.full(n, -1)
    if ok.sum() == 0:
        return pos, vel, seg_id
    if dt is None:
        d = np.diff(t[ok])
        dt = float(np.median(d)) if len(d) else 0.1
        dt = max(dt, 1e-3)
    to, xo = t[ok], xy[ok]
    runs = segments_by_gap(to, max_interp_s)
    for k, r in enumerate(runs):
        ts, xs = to[r], xo[r]
        a, b = ts[0], ts[-1]
        sel = (t >= a - 1e-9) & (t <= b + 1e-9)
        seg_id[sel] = k
        if len(r) == 1:
            pos[sel] = xs[0]
            vel[sel] = 0.0
            continue
        m = int(round((b - a) / dt)) + 1
        grid = a + np.arange(m) * dt
        if grid[-1] < b - 1e-9:
            grid = np.append(grid, b)
        xg_raw = np.column_stack([np.interp(grid, ts, xs[:, 0]), np.interp(grid, ts, xs[:, 1])])
        wm = odd(median_s / dt)
        xg = median_filter(xg_raw, size=(wm, 1), mode="nearest") if wm >= 3 and len(grid) >= wm else xg_raw
        ws = odd(window_s / dt, lo=order + 2 if order + 2 > 3 else 3)
        ps = _savgol(xg, ws, order)
        vs = _savgol(xg, ws, order, deriv=1, delta=dt)
        cuts = np.flatnonzero(np.diff(ts) > vel_gap_s) if vel_gap_s is not None else np.zeros(0, np.int64)
        if len(cuts):
            vs = _unbridge_velocity(grid, ts, xg_raw, vs, cuts, wm, ws, order, dt)
        tt = t[sel]
        pos[sel] = np.column_stack([np.interp(tt, grid, ps[:, 0]), np.interp(tt, grid, ps[:, 1])])
        vel[sel] = np.column_stack([np.interp(tt, grid, vs[:, 0]), np.interp(tt, grid, vs[:, 1])])
    return pos, vel, seg_id


def _unbridge_velocity(
    grid: np.ndarray,
    ts: np.ndarray,
    xg_raw: np.ndarray,
    vs: np.ndarray,
    cuts: np.ndarray,
    wm: int,
    ws: int,
    order: int,
    dt: float,
) -> np.ndarray:
    """Velocities next to gaps whose chord contradicts the motion on both sides (see ``smooth_kinematics``)."""
    from scipy.ndimage import median_filter

    segs = np.split(np.arange(len(ts)), cuts + 1)
    own: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    for seg in segs:
        gi = np.flatnonzero((grid >= ts[seg[0]] - 1e-9) & (grid <= ts[seg[-1]] + 1e-9))
        if len(gi) < order + 3:
            own.append(None)
            continue
        xseg = xg_raw[gi]
        if wm >= 3 and len(xseg) >= wm:
            xseg = median_filter(xseg, size=(wm, 1), mode="nearest")
        own.append((gi, _savgol(xseg, ws, order), _savgol(xseg, ws, order, deriv=1, delta=dt)))
    out = vs.copy()
    half = (ws // 2) * dt
    for k in range(len(segs) - 1):
        a, b = own[k], own[k + 1]
        if a is None or b is None:
            continue
        ga, pa, va = a
        gb, pb, vb = b
        ia, ib = ga[-1], gb[0]
        if grid[ib] - grid[ia] <= 0:
            continue
        chord = (pb[0] - pa[-1]) / (grid[ib] - grid[ia])
        mean = 0.5 * (va[-1] + vb[0])
        if np.linalg.norm(chord - mean) <= max(BRIDGE_TOL_MPS, 0.3 * float(np.linalg.norm(mean))):
            continue
        sa = grid[ga] >= grid[ia] - half
        sb = grid[gb] <= grid[ib] + half
        out[ga[sa]] = va[sa]
        out[gb[sb]] = vb[sb]
        inner = np.arange(ia + 1, ib)
        if len(inner):
            w = ((grid[inner] - grid[ia]) / (grid[ib] - grid[ia]))[:, None]
            out[inner] = (1 - w) * va[-1] + w * vb[0]
    return out


def fill_angles(theta: np.ndarray) -> np.ndarray:
    """Forward- then backward-fill NaN angles (hold the last known heading through stops)."""
    th = np.asarray(theta, dtype=np.float64).copy()
    idx = np.flatnonzero(np.isfinite(th))
    if len(idx) == 0:
        return th
    for i in range(idx[0] + 1, len(th)):
        if not np.isfinite(th[i]):
            th[i] = th[i - 1]
    th[: idx[0]] = th[idx[0]]
    return th

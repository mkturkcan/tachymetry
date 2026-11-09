"""Time-aware signal processing for irregularly sampled tracks.

All filters operate in *time*, not in samples: windows are given in seconds,
tracks with missing detections are resampled onto a uniform grid before
filtering, and filters never pad with zeros (edges are fitted, not padded).
"""

from __future__ import annotations

import numpy as np

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-phase position and velocity at the sample times ``t``.

    Samples marked ``False`` in ``weights_ok`` (truncated boxes, repaired
    outliers) contribute no measurement; their positions are interpolated from
    their neighbours. Each gap-free run is resampled to a uniform grid of step
    ``dt``, median-filtered over ``median_s`` (edges replicated), smoothed with a
    Savitzky-Golay filter over ``window_s`` and differentiated analytically.

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
        xg = np.column_stack([np.interp(grid, ts, xs[:, 0]), np.interp(grid, ts, xs[:, 1])])
        wm = odd(median_s / dt)
        if wm >= 3 and len(grid) >= wm:
            xg = median_filter(xg, size=(wm, 1), mode="nearest")
        ws = odd(window_s / dt, lo=order + 2 if order + 2 > 3 else 3)
        ps = _savgol(xg, ws, order)
        vs = _savgol(xg, ws, order, deriv=1, delta=dt)
        tt = t[sel]
        pos[sel] = np.column_stack([np.interp(tt, grid, ps[:, 0]), np.interp(tt, grid, ps[:, 1])])
        vel[sel] = np.column_stack([np.interp(tt, grid, vs[:, 0]), np.interp(tt, grid, vs[:, 1])])
    return pos, vel, seg_id


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

"""Ground contact and truncation-aware ground-envelope anchor of vehicle instance masks.

Measurement
-----------
The **ground contact** of a mask is its bottom-centre: the mean column of the
mask pixels in the lowest three rows, at the bottom edge of the lowest row,
intersected with the ground plane ``z = 0``. It is the image point of the
vehicle nearest to the camera along the viewing ray and, unlike the bottom
centre of a detector box, it lies on the vehicle's own silhouette.

The **ground envelope** samples up to 48 occupied columns of the mask and
records, per column, the lowest and highest mask pixel and whether the column
touches the frame border. Projected to the ground it traces the two near edges
of the footprint rectangle (plus the parallax of the lowest visible edge).

Anchor (along and across the vehicle axis)
------------------------------------------
Per frame, envelope ground points are expressed in the body frame (``u``
along the heading, ``v`` across) and in depth along the camera ray relative to
the near contact (``q``), then:

1. *Ray-depth trim*: columns more than ``Q_FRONT_M`` in front of the near
   contact (mask bleed, shadow) or deeper than the footprint's own ray extent
   ``|L cos b| + |W sin b|`` plus ``Q_MARGIN_M`` (the bottom of the column is an
   occluder's edge, not ground contact) are dropped.
2. *Occluder split*: the kept columns are split into runs at column gaps and at
   ray-depth steps above ``RUN_JUMP_M``; runs of fewer than two columns are
   dropped.
3. *Per-end observability*: the extreme ``u`` (and ``v``) termini count as
   observed only if the terminus is not on the frame border, has at least two
   supporting columns within ``SUPPORT_M`` and no dropped column lies beyond it.
4. *Along-axis centre*: both ends observed and at least ``FULL_FRAC`` of ``L``
   visible gives the midpoint (``SIG_BOTH``); one observed end gives that end
   plus or minus ``L/2`` (``SIG_ONE``); otherwise nothing. Near nose-on views
   (``|sin b| < SINB_MIN``) emit nothing: the along-axis position is not
   observable from a ground envelope there. Across the axis the near side is
   the reliable terminus, so the centre is placed ``W/2`` beyond it.
5. *Evidence-gated dimensions*: ``L`` from broadside frames with both ends
   observed (weighted median shrunk towards the class prior, floored by the
   95th percentile of the observed spans so that long vehicles are not pulled
   onto the prior); ``W`` from the across-axis extent of near-broadside frames
   (``|sin b| >= 0.7``), shrunk halfway to the prior (a heading error ``e``
   inflates the across-axis extent by ``L sin e``, largest nose-on).
6. *Residual fill*: frames without a measurement take the clearance-aware
   ray-offset model plus the time-interpolated residual of the measured frames,
   so unmeasured stretches keep the aspect dependence of the model.

The **clearance-aware admissibility band** bounds the radial offset
``rho = (c - g) . r`` of a centre ``c`` from its ground contact ``g`` (``r`` the
unit ray direction on the ground) to
``rho_model = (|L cos b| + |W sin b|) / 2 - gap * range / camera_height``
plus or minus ``BAND_M``. The subtracted term is the parallax of the lowest
visible silhouette edge at height ``gap``, which displaces the projected contact
away from the camera, so a correct centre may lie slightly in front of its own
contact.

Range-dependent gates are expressed through the depression angle of the
viewing ray (``atan(camera_height / range)``), which is independent of the
camera height and the image resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry.camera import Camera

__all__ = [
    "mask_contact_px",
    "mask_envelope_px",
    "gap_prior",
    "stable_heading",
    "AnchorResult",
    "envelope_anchor",
    "freeze_slow_offsets",
    "clearance_band",
    "apply_clearance_band",
    "depression_deg",
]

N_COLS = 48
BORDER_PX = 3
Q_FRONT_M = 0.6  #: tolerated mask bleed in front of the near contact
Q_MARGIN_M = 0.7  #: slack beyond the deepest legitimate contact
RUN_JUMP_M = 0.8  #: ray-depth step that separates two objects
SUPPORT_M = 0.9  #: a terminus needs a neighbouring column within this distance
SINB_MIN = 0.35  #: below: the along-axis centre is not observable
SINB_W_MIN = 0.70  #: width evidence needs a near-broadside view
W_QUANT = 0.60  #: quantile of the across-axis extent used for W
FULL_FRAC = 0.75  #: the both-ends midpoint needs this share of L visible
SIG_BOTH, SIG_ONE, SIG_FILL = 0.35, 0.70, 1.2
L_PRIOR_K = 3  #: prior pseudo-observations in the length median
BAND_M = 1.2  #: half-width of the clearance-aware admissibility band
GAP_CLIP = (0.05, 0.95)
#: dimension clips for the vehicle super-class (cars to articulated buses)
L_CLIP = (3.2, 18.5)
W_CLIP = (1.55, 2.70)


def depression_deg(camera: Camera, xy: np.ndarray) -> np.ndarray:
    """Angle (degrees) below the horizontal of the ray from the camera to ground points ``xy``."""
    rng = np.maximum(camera.distance_to(xy), 1e-6)
    return np.degrees(np.arctan2(max(camera.height, 1e-6), rng))


def mask_contact_px(crop: tuple[int, int, np.ndarray]) -> np.ndarray | None:
    """Bottom-centre ``(u, v)`` of a mask crop in full-resolution pixels, or ``None`` for an empty mask.

    ``u`` is the mean column of the mask pixels in the lowest three rows and
    ``v`` the bottom edge of the lowest row (pixel centres are integers).
    """
    x0, y0, m = crop
    rows = np.flatnonzero(m.any(axis=1))
    if len(rows) == 0:
        return None
    ymax = int(rows[-1])
    ys, xs = np.nonzero(m[max(ymax - 2, 0) : ymax + 1])
    return np.array([x0 + float(xs.mean()), y0 + ymax + 0.5])


def mask_envelope_px(
    crop: tuple[int, int, np.ndarray], image_size: tuple[int, int], n_cols: int = N_COLS, border_px: int = BORDER_PX
) -> np.ndarray | None:
    """``(k, 4)`` per-column envelope: column ``u``, bottom ``v`` (+0.5), top ``v``, frame-border flag."""
    x0, y0, m = crop
    occ = m.any(axis=0)
    cols = np.flatnonzero(occ)
    if len(cols) < 4:
        return None
    h = m.shape[0]
    bot = h - 1 - np.argmax(m[::-1, :], axis=0)
    top = np.argmax(m, axis=0)
    sel = cols[np.linspace(0, len(cols) - 1, min(n_cols, len(cols))).astype(int)]
    W, H = image_size
    u = x0 + sel.astype(np.float64)
    vb = y0 + bot[sel] + 0.5
    brd = (u <= border_px) | (u >= W - 1 - border_px) | (vb >= H - 1 - border_px)
    return np.column_stack([u, vb, y0 + top[sel].astype(np.float64), brd.astype(np.float64)])


def gap_prior(L: float, W: float) -> float:
    """Height (m) of the lowest visible silhouette edge (sill, underbody between the wheels) by vehicle size."""
    if L <= 3.2 and W <= 1.75:
        return 0.12
    if L <= 5.4:
        return 0.30
    if L <= 6.8:
        return 0.40
    return 0.55


def _fill_nearest(theta: np.ndarray, good: np.ndarray, t: np.ndarray) -> np.ndarray:
    gi = np.flatnonzero(good)
    out = theta.copy()
    for i in np.flatnonzero(~good):
        out[i] = theta[gi[np.argmin(np.abs(t[gi] - t[i]))]]
    return out


def stable_heading(
    t: np.ndarray,
    vel: np.ndarray,
    slow_mps: float,
    smooth_s: float,
    pca_points: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, bool]:
    """Heading (rad) from a smoothed ground path with stopped samples held at the nearest moving sample.

    The direction of a queued vehicle's ground path is noise and would spin its
    body frame, so samples slower than ``slow_mps`` take the heading of the
    nearest moving sample (in time); the unwrapped result is smoothed over
    ``smooth_s``. A track that never moves falls back to the principal axis of
    its pooled envelope ground points (``pca_points``). Returns
    ``(heading, moved)``; ``heading`` is all-NaN when neither is available.
    """
    from .signal import smooth_kinematics

    n = len(t)
    sp = np.linalg.norm(np.nan_to_num(vel), axis=1)
    good = (sp > slow_mps) & np.all(np.isfinite(vel), axis=1)
    if good.sum() >= 3:
        th = np.unwrap(np.arctan2(vel[good, 1], vel[good, 0]))
        full = np.full(n, np.nan)
        full[good] = th
        full = _fill_nearest(full, good, t)
        sm, _, _ = smooth_kinematics(t, np.column_stack([full, np.zeros(n)]), None, 0.0, smooth_s, 2, 1e9)
        return sm[:, 0], True
    if pca_points:
        E = np.concatenate([p for p in pca_points if len(p)], axis=0) if any(len(p) for p in pca_points) else None
        if E is not None and len(E) >= 3:
            E = E - E.mean(axis=0)
            _, _, Vt = np.linalg.svd(E, full_matrices=False)
            return np.full(n, float(np.arctan2(Vt[0, 1], Vt[0, 0]))), False
    return np.full(n, np.nan), False


def _wmedian(x: list[float], w: list[float]) -> float:
    xa, wa = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(xa)
    xa, wa = xa[o], wa[o]
    cw = np.cumsum(wa)
    return float(xa[min(int(np.searchsorted(cw, 0.5 * cw[-1])), len(xa) - 1)])


def _observed(x: float, kept: np.ndarray, dropped: np.ndarray, border: bool, low_end: bool) -> bool:
    """An envelope terminus ``x`` is observed: off the frame border, supported, and no dropped column beyond it."""
    beyond = np.any(dropped < x - 0.25) if low_end else np.any(dropped > x + 0.25)
    return (not border) and int(np.sum(np.abs(kept - x) < SUPPORT_M)) >= 2 and not bool(beyond)


@dataclass
class _Frame:
    u_hat: np.ndarray
    v_hat: np.ndarray
    ray_n: np.ndarray
    rng: np.ndarray
    cos_b: np.ndarray
    sin_b: np.ndarray
    dep: np.ndarray


def _frame_geometry(camera: Camera, g: np.ndarray, heading: np.ndarray) -> _Frame:
    u_hat = np.column_stack([np.cos(heading), np.sin(heading)])
    v_hat = np.column_stack([-u_hat[:, 1], u_hat[:, 0]])
    ray = g - camera.center[:2]
    rng = np.maximum(np.linalg.norm(ray, axis=1), 1e-9)
    ray_n = ray / rng[:, None]
    return _Frame(
        u_hat,
        v_hat,
        ray_n,
        rng,
        np.einsum("ij,ij->i", ray_n, u_hat),
        np.einsum("ij,ij->i", ray_n, v_hat),
        np.degrees(np.arctan2(max(camera.height, 1e-6), rng)),
    )


@dataclass
class AnchorResult:
    """Per-frame envelope anchor of one track (world metres)."""

    centre: np.ndarray  #: (n, 2) anchor centre; measured frames, else model plus interpolated residual
    model: np.ndarray  #: (n, 2) clearance-aware ray-offset model centre
    measured: np.ndarray  #: (n,) bool: along- and across-axis offsets measured
    kind: np.ndarray  #: (n,) 0 none, 1 one end observed, 2 both ends observed
    sigma: np.ndarray  #: (n,) along-axis standard deviation (m) for the box-fit fusion
    L: float
    W: float
    gap: float
    n_length_evidence: int = 0
    n_width_evidence: int = 0
    info: dict[str, Any] = field(default_factory=dict)


def envelope_anchor(
    camera: Camera,
    t: np.ndarray,
    g: np.ndarray,
    heading: np.ndarray,
    envs: list[np.ndarray | None],
    L_prior: float,
    W_prior: float,
    gap: float | None = None,
    min_depression_deg: float = 8.5,
    predict: np.ndarray | None = None,
) -> AnchorResult | None:
    """Truncation-aware envelope anchor for one track (see the module docstring).

    ``predict`` (``(n, 2)``, NaN where unavailable) is a motion prediction of
    the centre. Where it is given and the visible extent is shorter than
    ``FULL_FRAC * L`` (so one end is hidden even if both termini pass the
    observability test, as when an occluder hides the full height of the
    vehicle's end), the end whose centre candidate agrees with the prediction
    is used. Without it the end nearer to the camera is preferred.

    ``g`` are the (smoothed) ground contacts, ``heading`` the body headings
    (rad), ``envs`` the per-frame envelopes of :func:`mask_envelope_px` (or
    ``None``). Returns ``None`` when fewer than ``max(5, n/4)`` frames yield
    an envelope measurement.
    """
    n = len(t)
    F = _frame_geometry(camera, g, heading)
    raw: list[tuple[np.ndarray, ...] | None] = [None] * n
    for i, env in enumerate(envs):
        if env is None or not np.all(np.isfinite(g[i])):
            continue
        E = camera.image_to_ground(env[:, :2])
        fin = np.all(np.isfinite(E), axis=1)
        if fin.sum() < 4:
            continue
        rel = E[fin] - g[i]
        u = rel @ F.u_hat[i]
        v = rel @ F.v_hat[i]
        dr = rel @ F.ray_n[i]
        raw[i] = (u, v, dr - np.percentile(dr, 8), env[fin, 3] > 0.5)

    def analyse(L: float, W: float) -> list[dict[str, Any] | None]:
        out: list[dict[str, Any] | None] = [None] * n
        for i in range(n):
            if raw[i] is None:
                continue
            u, v, q, brd = raw[i]
            ext_ray = abs(L * F.cos_b[i]) + abs(W * F.sin_b[i])
            keep = (q > -Q_FRONT_M) & (q < ext_ray + Q_MARGIN_M)
            if keep.sum() < 3:
                continue
            idx = np.flatnonzero(keep)
            brk = (np.diff(idx) > 1) | (np.abs(np.diff(q[idx])) > RUN_JUMP_M)
            runs = [r for r in np.split(idx, np.flatnonzero(brk) + 1) if len(r) >= 2]
            if not runs:
                continue
            tr = np.concatenate(runs)
            ut, vt = u[tr], v[tr]
            drop_u, drop_v = u[~keep], v[~keep]

            lo_k, hi_k = int(tr[np.argmin(ut)]), int(tr[np.argmax(ut)])
            vlo_k, vhi_k = int(tr[np.argmin(vt)]), int(tr[np.argmax(vt)])
            out[i] = dict(
                u_lo=float(u[lo_k]),
                u_hi=float(u[hi_k]),
                obs_lo=_observed(u[lo_k], ut, drop_u, bool(brd[lo_k]), True),
                obs_hi=_observed(u[hi_k], ut, drop_u, bool(brd[hi_k]), False),
                v_lo=float(v[vlo_k]),
                v_hi=float(v[vhi_k]),
                vobs_lo=_observed(v[vlo_k], vt, drop_v, bool(brd[vlo_k]), True),
                vobs_hi=_observed(v[vhi_k], vt, drop_v, bool(brd[vhi_k]), False),
            )
        return out

    ev_ok = F.dep >= min_depression_deg
    w_rng = np.minimum(1.0, np.tan(np.radians(F.dep)) / np.tan(np.radians(15.0)))

    def fit_L(meas: list[dict[str, Any] | None]) -> tuple[float, int]:
        cand, wts = [], []
        for i, d in enumerate(meas):
            if d is None or not (d["obs_lo"] and d["obs_hi"]):
                continue
            if abs(F.sin_b[i]) < SINB_MIN or not ev_ok[i]:
                continue
            cand.append(d["u_hi"] - d["u_lo"])
            wts.append(abs(F.sin_b[i]) * w_rng[i])
        # physical floor: a vehicle is at least as long as the along-axis extent of its trusted contact; without
        # it the both-ends gate starves on long vehicles seen nose-on and the shrinkage collapses them onto the prior
        spans = [d["u_hi"] - d["u_lo"] for i, d in enumerate(meas) if d is not None and ev_ok[i]]
        floor = float(np.percentile(spans, 95)) if len(spans) >= 3 else 0.0
        if not cand:
            return float(np.clip(max(L_prior, floor), *L_CLIP)), 0
        wp = max(float(np.mean(wts)), 0.3)
        L_sh = _wmedian(cand + [L_prior] * L_PRIOR_K, wts + [wp] * L_PRIOR_K)
        return float(np.clip(max(L_sh, floor), *L_CLIP)), len(cand)

    def fit_W(meas: list[dict[str, Any] | None]) -> tuple[float, int]:
        cand, wts = [], []
        for i, d in enumerate(meas):
            if d is None or not ev_ok[i] or abs(F.sin_b[i]) < SINB_W_MIN:
                continue
            cand.append(d["v_hi"] - d["v_lo"])
            wts.append(abs(F.sin_b[i]) * w_rng[i])
        if len(cand) < 3:
            return float(np.clip(W_prior, *W_CLIP)), len(cand)
        o = np.argsort(cand)
        cs = np.asarray(cand, float)[o]
        ws = np.cumsum(np.asarray(wts, float)[o])
        W_ev = float(cs[min(int(np.searchsorted(ws, W_QUANT * ws[-1])), len(cs) - 1)])
        return float(np.clip(0.5 * (W_prior + float(np.clip(W_ev, 1.2, 3.0))), *W_CLIP)), len(cand)

    need = max(5, n // 4)
    m0 = analyse(L_prior, W_prior)
    if sum(d is not None for d in m0) < need:
        return None
    L, n_ev = fit_L(m0)
    W, n_wev = fit_W(m0)
    if L > 6.5:  # bus, truck, box van
        W = max(W, 2.30)
    meas = analyse(L, W)
    if sum(d is not None for d in meas) < need:
        meas = m0
    else:
        L2, n_ev2 = fit_L(meas)
        W2, n_wev2 = fit_W(meas)
        if L2 > 6.5:
            W2 = max(W2, 2.30)
        L, W, n_ev, n_wev = L2, W2, max(n_ev, n_ev2), max(n_wev, n_wev2)

    u_c = np.full(n, np.nan)
    v_c = np.full(n, np.nan)
    sig = np.full(n, SIG_FILL)
    kind = np.zeros(n, np.int64)
    for i, d in enumerate(meas):
        if d is None:
            continue
        s_cam = -1.0 if F.sin_b[i] >= 0 else 1.0  # side of the camera along v
        v_near = d["v_hi"] if s_cam > 0 else d["v_lo"]
        v_c[i] = v_near - s_cam * W / 2.0
        if abs(F.sin_b[i]) < SINB_MIN:
            continue
        near_is_lo = F.cos_b[i] > 0  # u points away from the camera
        if predict is not None and np.all(np.isfinite(predict[i])) and (d["u_hi"] - d["u_lo"]) < FULL_FRAC * L:
            # one end is hidden: keep the observed end whose centre candidate the motion prediction supports
            u_p = float((predict[i] - g[i]) @ F.u_hat[i])
            cands = [
                (abs(c - u_p), c)
                for c, o in ((d["u_lo"] + L / 2.0, d["obs_lo"]), (d["u_hi"] - L / 2.0, d["obs_hi"]))
                if o
            ]
            if cands:
                u_c[i], sig[i], kind[i] = min(cands)[1], SIG_ONE, 1
            continue
        if d["obs_lo"] and d["obs_hi"] and (d["u_hi"] - d["u_lo"]) >= FULL_FRAC * L:
            u_c[i], sig[i], kind[i] = 0.5 * (d["u_lo"] + d["u_hi"]), SIG_BOTH, 2
        elif d["obs_lo"] and (not d["obs_hi"] or near_is_lo):
            u_c[i], sig[i], kind[i] = d["u_lo"] + L / 2.0, SIG_ONE, 1
        elif d["obs_hi"]:
            u_c[i], sig[i], kind[i] = d["u_hi"] - L / 2.0, SIG_ONE, 1

    gp = gap_prior(L, W) if gap is None else float(gap)
    rho = 0.5 * (np.abs(L * F.cos_b) + np.abs(W * F.sin_b)) - gp * F.rng / max(camera.height, 0.5)
    model = g + rho[:, None] * F.ray_n
    ok = np.isfinite(u_c) & np.isfinite(v_c)
    c = model.copy()
    c[ok] = g[ok] + u_c[ok, None] * F.u_hat[ok] + v_c[ok, None] * F.v_hat[ok]
    if 0 < ok.sum() < n:
        r = c - model
        idx = np.flatnonzero(ok)
        for k in range(2):
            r[~ok, k] = np.interp(t[~ok], t[idx], r[idx, k])
        c = model + r
    n_env = int(sum(d is not None for d in meas))
    return AnchorResult(c, model, ok, kind, sig, L, W, gp, n_ev, n_wev, {"frames_with_envelope": n_env})


def freeze_slow_offsets(
    c: np.ndarray, g: np.ndarray, speed_mps: np.ndarray, slow_mps: float, prefer: np.ndarray | None = None
) -> np.ndarray:
    """Hold the centre's offset from the ground contact constant while the vehicle is not moving.

    A stopped vehicle cannot slide along its own axis, but per-frame estimates
    can still change which end they believe they see. Each slow run takes the
    median offset of its moving neighbours (or its own median when it has
    none); the speed comes from the ground-contact path, which is independent
    of the centre estimate being corrected. ``prefer`` marks the samples whose
    offsets a run's own median is taken from when available (e.g. accepted
    silhouette fits).
    """
    c = np.asarray(c, float).copy()
    g = np.asarray(g, float)
    if len(c) != len(g) or len(c) < 3:
        return c
    o = c - g
    slow = np.asarray(speed_mps, float) < slow_mps
    fin = np.all(np.isfinite(o), axis=1)
    if not fin.any():
        return c
    pref = fin if prefer is None else (fin & np.asarray(prefer, bool))
    if not pref.any():
        pref = fin
    if slow.all():
        return g + np.median(o[pref], axis=0)[None, :]
    n = len(o)
    i = 0
    while i < n:
        if not slow[i]:
            i += 1
            continue
        j = i
        while j < n and slow[j]:
            j += 1
        ref = [o[k] for k in (i - 1, j) if 0 <= k < n and fin[k]]
        run = o[i:j][pref[i:j]] if pref[i:j].any() else o[i:j][fin[i:j]]
        if ref:
            o[i:j] = np.median(np.stack(ref), axis=0)
        elif len(run):
            o[i:j] = np.median(run, axis=0)
        i = j
    return g + o


def clearance_band(
    camera: Camera, g: np.ndarray, heading: np.ndarray, L: float, W: float, gap: float, band_m: float = BAND_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Admissible radial offsets ``(lo, hi)`` (m) of the centre from the contact, and the unit ray directions."""
    F = _frame_geometry(camera, g, heading)
    gp = float(np.clip(gap, *GAP_CLIP))
    rho = 0.5 * (np.abs(L * F.cos_b) + np.abs(W * F.sin_b)) - gp * F.rng / max(camera.height, 0.5)
    return rho - band_m, rho + band_m, F.ray_n


def apply_clearance_band(
    camera: Camera,
    c: np.ndarray,
    g: np.ndarray,
    heading: np.ndarray,
    L: float,
    W: float,
    gap: float,
    band_m: float = BAND_M,
) -> np.ndarray:
    """Clamp the radial component of centres ``c`` into the clearance-aware band; the tangential part is kept."""
    lo, hi, ray_n = clearance_band(camera, g, heading, L, W, gap, band_m)
    c = np.asarray(c, float)
    rho = np.einsum("ij,ij->i", c - g, ray_n)
    return c + (np.clip(rho, lo, hi) - rho)[:, None] * ray_n

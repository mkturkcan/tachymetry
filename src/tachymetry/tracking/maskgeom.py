"""Vehicle footprint centres and headings from instance masks (the mask path of track hygiene).

For a vehicle track whose detections carry instance masks, the footprint is
estimated as follows (positions in world metres on the ground plane):

1. **Ground contact and reference heading.** The contact of each detection is
   its mask bottom-centre at ``z = 0`` (:func:`tachymetry.tracking.envelope.mask_contact_px`,
   set in :func:`tachymetry.tracking.measure.measure_ground`). The contact path
   is smoothed; its direction gives the reference heading, held at the nearest
   moving sample while the vehicle is slower than ``mask_slow_kmh``
   (:func:`~tachymetry.tracking.envelope.stable_heading`); stopped samples snap
   to the nearest street axis within ``street_snap_deg``
   (:class:`~tachymetry.tracking.boxfit.StreetAxes`). A vehicle that never
   moves chooses its heading by silhouette cost among the street axes, the
   principal axis of its envelope ground points and that axis's normal.
2. **Envelope anchor.** The truncation-aware ground envelope gives per-frame
   centres with an along-axis uncertainty, and evidence-gated ``L`` and ``W``
   (:func:`~tachymetry.tracking.envelope.envelope_anchor`). While the vehicle is
   stopped the offset from the contact is frozen.
3. **Box-silhouette fit.** Frames whose mask is large enough
   (``boxfit_min_mask_px``), not truncated by the frame and seen above a
   grazing depression angle (``boxfit_min_depression_deg``) are fitted with a
   calibrated upright box: shared ``L, W, H`` and clearance, per-frame centre and
   bounded heading correction, fused with the anchor along the vehicle axis
   (:func:`~tachymetry.tracking.boxfit.fit_track_boxes`). Poses are fitted at up
   to ``boxfit_max_hz`` while moving and ``boxfit_slow_hz`` while stopped; the
   frames between take the anchor plus the fit residual interpolated in time.
   The fit replaces the anchor on frames with a silhouette cost below
   ``boxfit_max_cost`` when such frames make up a quarter of the track.
4. **Occlusion.** A frame whose fitted silhouette the mask fills markedly less
   than usual while the box still covers the mask is occluded: its lowest
   visible mask point is the occluder's edge. The contact path is recomputed
   without occluded frames; through an occlusion the heading is interpolated
   between the clean stretches on either side, or held from the nearer one (for
   at most ``HOLD_S``) at a track end. The speed of occluded frames comes from
   the first-pass box centres, the hidden end of the vehicle is chosen by
   consistency with their motion, and occluded poses are re-fitted with the
   shared box at the held heading. Frames filled below ``HIDDEN_FILL`` of the
   usual are left unmeasured (interpolated inside a track, dropped at its ends).
5. **Consolidation.** Offsets are frozen again on the final centres while the
   vehicle is stopped. Tracks without an accepted fit are clamped into the
   clearance-aware admissibility band around the contact. The heading is the
   reference heading plus the silhouette-measured correction, smoothed in time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import HygieneConfig
from ..detections import DetectionSet
from ..geometry.camera import Camera
from ..geometry.silhouette import MaskRaster
from .boxfit import StreetAxes, fit_track_boxes, snap_to_axes
from .envelope import (
    apply_clearance_band,
    depression_deg,
    envelope_anchor,
    freeze_slow_offsets,
    gap_prior,
    mask_envelope_px,
    stable_heading,
)
from .measure import GroundObservations, dims_for
from .signal import running_median, smooth_kinematics, wrap

__all__ = ["MaskGeometry", "vehicle_geometry", "BOX_BOUNDS"]

#: box-fit ranges per class: ((L_min, L_max), (W_min, W_max), (H_min, H_max)), metres
BOX_BOUNDS = {
    "car": ((3.0, 6.8), (1.5, 2.4), (1.2, 2.4)),
    "bus": ((6.5, 20.0), (2.2, 2.9), (2.3, 4.6)),
    "truck": ((4.5, 20.0), (1.8, 2.9), (1.8, 4.4)),
}


@dataclass
class MaskGeometry:
    centre: np.ndarray  #: (n, 2) footprint centres
    heading: np.ndarray  #: (n,) body headings (rad)
    L: float
    W: float
    H: float
    gap: float
    box_ok: np.ndarray  #: (n,) centre and heading from an accepted silhouette fit
    anchored: np.ndarray  #: (n,) envelope anchor measured (one or both ends)
    measured: np.ndarray  #: (n,) False where an occluder hides too much of the vehicle to locate it
    source: str  #: ``boxfit``, ``envelope`` or ``ray``
    seconds: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


def _predict(t: np.ndarray, xy: np.ndarray, clean: np.ndarray, cfg: HygieneConfig) -> np.ndarray:
    """Motion prediction of the centre at every sample from the clean samples (NaN where none is possible).

    Clean stretches are smoothed; samples between two stretches take the time
    interpolation of their ends, samples before the first or after the last are
    extrapolated with the velocity at that end.
    """
    out = np.full((len(t), 2), np.nan)
    if clean.sum() < 3:
        return out
    pos, vel, _ = smooth_kinematics(
        t, xy, clean, cfg.median_window_s, max(1.5, cfg.smooth_window_s), 1, cfg.interp_gap_s, vel_gap_s=None
    )
    fin = np.all(np.isfinite(pos), axis=1) & np.all(np.isfinite(vel), axis=1)
    out[fin] = pos[fin]
    fi = np.flatnonzero(fin)
    if not len(fi):
        return out
    for i in np.flatnonzero(~fin):
        before, after = fi[fi < i], fi[fi > i]
        if len(before) and len(after):
            a, b = before[-1], after[0]
            w = (t[i] - t[a]) / max(t[b] - t[a], 1e-9)
            out[i] = (1 - w) * pos[a] + w * pos[b]
        else:
            k = before[-1] if len(before) else after[0]
            out[i] = pos[k] + vel[k] * (t[i] - t[k])
    return out


def _choose_heading(
    camera: Camera,
    rasters: list[MaskRaster | None],
    weights: np.ndarray,
    c0: np.ndarray,
    candidates: list[float],
    dims: tuple[float, float, float],
    n_frames: int = 6,
) -> float | None:
    """The candidate heading (modulo pi) whose box best explains the best-observed masks of a track.

    Each candidate is scored by the sum of the lowest two thirds of the costs of
    a short centre fit on ``n_frames`` well-observed frames, at the class's prior
    dimensions with a heading refinement of up to 10 degrees.
    """
    from ..geometry.silhouette import fit_centre

    idx = [i for i in np.argsort(-weights) if rasters[i] is not None][: n_frames * 2 : 2]
    if not idx:
        return None
    uniq: list[float] = []
    for c in candidates:
        if all(abs(float((c - u + np.pi / 2) % np.pi - np.pi / 2)) > np.radians(5.0) for u in uniq):
            uniq.append(float(c))
    best, best_cost = None, np.inf
    for c in uniq:
        costs = sorted(
            fit_centre(
                camera,
                rasters[i],
                c,
                dims[0],
                dims[1],
                dims[2],
                0.15,
                c0[i],
                max_iter=20,
                opt_heading=True,
                heading_lim_deg=10.0,
            )[1]
            for i in idx
        )
        k = max(1, (len(costs) * 2 + 2) // 3)
        if sum(costs[:k]) < best_cost:
            best, best_cost = c, sum(costs[:k])
    return best


def _pose_frames(t: np.ndarray, gate: np.ndarray, slow: np.ndarray, cfg: HygieneConfig) -> np.ndarray:
    """Frames to fit a pose on: gated frames at most ``boxfit_max_hz`` while moving, and at most
    ``boxfit_slow_hz`` (24 per run) while stopped. The others interpolate the fit's residual from the anchor."""
    pose = gate.copy()
    rate = 1.0 / max(float(np.median(np.diff(t))) if len(t) > 1 else 0.1, 1e-3)
    for mask, hz, cap in ((~slow, cfg.boxfit_max_hz, None), (slow, cfg.boxfit_slow_hz, 24)):
        step = max(1, int(round(rate / max(hz, 1e-3))))
        for a, b in _runs(mask):
            keep = np.zeros(b - a, bool)
            keep[np.flatnonzero(gate[a:b])[::step][:cap]] = True
            pose[a:b] &= keep
    return pose


def _track_kinematics(
    t: np.ndarray, xy: np.ndarray, ok: np.ndarray, cfg: HygieneConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Heading (NaN where slower than 1 m/s or undefined) and speed of a smoothed centre trajectory."""
    if ok.sum() < 4:
        return np.full(len(t), np.nan), np.full(len(t), np.nan)
    _, v, seg = smooth_kinematics(
        t, xy, ok, cfg.median_window_s, max(1.5, cfg.smooth_window_s), 1, cfg.interp_gap_s, vel_gap_s=None
    )
    sp = np.linalg.norm(v, axis=1)
    hd = np.where(sp > 1.0, np.arctan2(v[:, 1], v[:, 0]), np.nan)
    return hd, sp


def _contact_path(
    t: np.ndarray, contact: np.ndarray, use: np.ndarray, cfg: HygieneConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed ground-contact positions and the contact velocity (first-order fit over at least 1.5 s).

    Velocities of runs with fewer than four contacts are undefined (NaN), not zero.
    """
    g_s, _, _ = smooth_kinematics(
        t, contact, use, cfg.median_window_s, cfg.smooth_window_s, cfg.smooth_order, cfg.interp_gap_s
    )
    _, v_g, seg = smooth_kinematics(
        t, contact, use, cfg.median_window_s, max(1.5, cfg.smooth_window_s), 1, cfg.interp_gap_s
    )
    ok = np.asarray(use, bool) & np.all(np.isfinite(contact), axis=1)
    for k in np.unique(seg[seg >= 0]):
        if int((ok & (seg == k)).sum()) < 4:
            v_g[seg == k] = np.nan
    return g_s, v_g


def _reference_heading(
    t: np.ndarray,
    g_s: np.ndarray,
    v_g: np.ndarray,
    slow_mps: float,
    pca: list[np.ndarray],
    axes: StreetAxes | None,
    cfg: HygieneConfig,
    camera: Camera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Reference heading, contact speed and stopped flags; stopped stretches snap to the nearest street axis.

    Samples where the contact velocity is undefined take the speed of the
    nearest sample where it is.
    """
    hd, moved = stable_heading(t, v_g, slow_mps, 0.6, pca)
    if not np.isfinite(hd).any():
        ray = g_s - camera.center[:2]
        hd = np.arctan2(ray[:, 1], ray[:, 0])
    fin = np.all(np.isfinite(v_g), axis=1)
    speed = np.linalg.norm(np.nan_to_num(v_g), axis=1)
    if fin.any() and not fin.all():
        fi = np.flatnonzero(fin)
        speed[~fin] = speed[fi[np.argmin(np.abs(t[~fin, None] - t[None, fi]), axis=1)]]
    slow = speed < slow_mps
    hd_ref = hd.copy()
    if axes is not None:
        for a, b in _runs(slow):
            ax = axes.at(np.median(g_s[a:b], axis=0))
            hd_ref[a:b] = [snap_to_axes(h, ax, cfg.street_snap_deg) for h in hd[a:b]]
    return hd_ref, speed, slow, moved


#: a fitted silhouette filled below this fraction of the track's typical fill counts as occluded
OCCLUDED_FILL = 0.8
#: ... and below this fraction the visible remainder no longer locates the vehicle (sample left unmeasured)
HIDDEN_FILL = 0.55
#: ... provided the fitted box still covers the mask (share of mask pixels outside it below this)
OCCLUDED_MISS = 0.08
#: longest time (s) over which an occluded vehicle's heading and speed are carried from clean evidence
HOLD_S = 2.0


def _occluded(
    t: np.ndarray, fill: np.ndarray, miss: np.ndarray, ok: np.ndarray, pad_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Frames whose fitted silhouette is markedly less filled by the mask than usual (padded by ``pad_s``), and
    frames hidden so far that the visible remainder no longer locates the vehicle.

    The typical fill is the 75th percentile over the accepted fits; frames whose
    fit failed count when their fill is low. Moderately low fill counts only where
    the fitted box still covers the mask (miss share below ``OCCLUDED_MISS``): a
    box at a wrong heading, as in a sharp turn, lowers the fill too but leaves the
    mask uncovered. Below ``HIDDEN_FILL`` the frame is hidden either way. The
    padding covers the reach of the heading smoothing into neighbouring frames.
    """
    valid = ok & np.isfinite(fill)
    if valid.sum() < 5:
        return np.zeros(len(t), bool), np.zeros(len(t), bool)
    typ = float(np.percentile(fill[valid], 75))
    covered = np.isfinite(miss) & (miss < OCCLUDED_MISS)
    hidden = np.isfinite(fill) & (fill < HIDDEN_FILL * typ)
    occ = (covered & np.isfinite(fill) & (fill < OCCLUDED_FILL * typ)) | hidden
    if not occ.any():
        return occ, hidden
    ti = t[occ]
    return np.min(np.abs(t[:, None] - ti[None, :]), axis=1) <= pad_s, hidden


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    e = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(e[::2], e[1::2]))


def vehicle_geometry(
    rows: np.ndarray,
    dets: DetectionSet,
    obs: GroundObservations,
    camera: Camera,
    cfg: HygieneConfig,
    cls: str,
    use: np.ndarray,
    axes: StreetAxes | None = None,
    cache: dict[int, Any] | None = None,
) -> MaskGeometry | None:
    """Mask-based footprint geometry of one vehicle track, or ``None`` when too few samples have masks."""
    t_start = time.perf_counter()
    masks = dets.masks
    if masks is None:
        return None
    n = len(rows)
    cache = cache if cache is not None else {}
    image_size = dets.image_size if dets.image_size[0] else camera.image_size

    def crop_of(r: int) -> Any:
        key = int(r)
        if key not in cache:
            c = masks.crop(key)
            env = mask_envelope_px(c, image_size) if c is not None else None
            cache[key] = (env, float(c[2].sum()) if c is not None else 0.0)
        return cache[key]

    has = np.array([masks.has(int(r)) for r in rows]) & use
    if has.sum() < max(3, cfg.mask_min_share * max(int(use.sum()), 1)):
        return None
    t = dets.time[rows].astype(np.float64)
    contact = obs.contact[rows]
    slow_mps = cfg.mask_slow_kmh / 3.6
    g_s, v_g = _contact_path(t, contact, use, cfg)
    bad = ~np.all(np.isfinite(g_s), axis=1)
    g_s[bad] = contact[bad]

    info = [crop_of(r) if has[i] else (None, 0.0) for i, r in enumerate(rows)]
    envs = [e for e, _ in info]
    px = np.array([a for _, a in info])
    pca = []
    for e in envs:
        if e is not None:
            p = camera.image_to_ground(e[:, :2])
            pca.append(p[np.all(np.isfinite(p), axis=1)])
    hd_ref, speed, slow, moved = _reference_heading(t, g_s, v_g, slow_mps, pca, axes, cfg, camera)

    L0, W0, H0 = dims_for(cls)
    anc = envelope_anchor(camera, t, g_s, hd_ref, envs, L0, W0, None, cfg.boxfit_min_depression_deg)
    if anc is not None:
        centre, L, W, gap = anc.centre.copy(), anc.L, anc.W, anc.gap
        sigma, anchored, source = anc.sigma, anc.measured, "envelope"
    else:
        L, W, gap = L0, W0, gap_prior(L0, W0)
        ray = g_s - camera.center[:2]
        rng = np.maximum(np.linalg.norm(ray, axis=1), 1e-6)
        rhat = ray / rng[:, None]
        b = hd_ref - np.arctan2(rhat[:, 1], rhat[:, 0])
        rho = np.maximum(0.5 * (np.abs(L * np.cos(b)) + np.abs(W * np.sin(b))) - gap * rng / max(camera.height, 0.5), 0)
        centre = g_s + rho[:, None] * rhat
        sigma, anchored, source = None, np.zeros(n, bool), "ray"
    centre = freeze_slow_offsets(centre, g_s, speed, slow_mps)
    H = H0
    heading = hd_ref.copy()
    box_ok = np.zeros(n, bool)
    occl = np.zeros(n, bool)
    measured = np.ones(n, bool)
    fit_info: dict[str, Any] = {}
    if cfg.boxfit:
        dep = depression_deg(camera, g_s)
        gate = has & ~obs.truncated[rows] & ~obs.big[rows] & (px >= cfg.boxfit_min_mask_px)
        gate &= dep >= cfg.boxfit_min_depression_deg
        pose = _pose_frames(t, gate, slow, cfg)
        if gate.sum() >= 3:
            rasters: list[MaskRaster | None] = [None] * n
            for i in np.flatnonzero(gate):
                rasters[i] = MaskRaster(masks.crop(int(rows[i])), cfg.boxfit_down, image_size)
            weights = dets.score[rows].astype(np.float64) * np.sqrt(px)
            bounds = BOX_BOUNDS.get(cls, BOX_BOUNDS["car"])
            if not moved:
                # a vehicle that never moves has no trajectory heading: choose among the street axes, the principal
                # axis of its envelope and its normal by silhouette cost, before the bounded refinement
                cands = [float(hd_ref[0]), float(hd_ref[0]) + np.pi / 2]
                if axes is not None:
                    cands += axes.at(np.median(g_s, axis=0))
                best = _choose_heading(camera, rasters, weights, centre, cands, (L0, W0, H0))
                if best is not None and abs(float(wrap(best - hd_ref[0]))) > 1e-6:
                    hd_ref[:] = best
                    anc = envelope_anchor(camera, t, g_s, hd_ref, envs, L0, W0, None, cfg.boxfit_min_depression_deg)
                    if anc is not None:
                        centre, L, W, gap, sigma, anchored = (
                            freeze_slow_offsets(anc.centre, g_s, speed, slow_mps),
                            anc.L,
                            anc.W,
                            anc.gap,
                            anc.sigma,
                            anc.measured,
                        )
                    heading = hd_ref.copy()
            bf = fit_track_boxes(
                camera,
                rasters,
                hd_ref,
                centre,
                weights,
                (L, W, H0),
                bounds,
                cfg.boxfit_dim_frames,
                centre if sigma is not None else None,
                sigma,
                pose_frames=pose,
            )
            if bf is not None:
                ok = bf.ok(cfg.boxfit_max_cost)
                n_pose = max(int(pose.sum()), 1)
                expected = ok.sum() / n_pose * gate.sum()
                fit_info = dict(bf.info, seconds=round(bf.seconds, 3), ok=int(ok.sum()), pose=n_pose)
                if ok.sum() >= min(5, n_pose) and expected >= max(5, n // 4):
                    occl, hidden = _occluded(t, bf.fill, bf.miss, ok, 0.5 * max(1.5, cfg.smooth_window_s))
                    good = ok & ~occl
                    if occl.any() and good.sum() >= 3:
                        # Occluded frames: the lowest visible mask point is the occluder's edge, so the contact path,
                        # and the reference heading and speed smoothed from it, are corrupted there and up to half
                        # a smoothing window around. They are recomputed without those contacts (the heading is held
                        # through the occlusion) and the poses re-fitted with the shared box; occluded frames keep
                        # the reference heading, since their silhouette no longer determines it.
                        _, v_g2 = _contact_path(t, contact, use & ~occl, cfg)
                        hd2, speed2, slow2, _ = _reference_heading(t, g_s, v_g2, slow_mps, pca, axes, cfg, camera)
                        # Within HOLD_S of clean contacts the heading is held from them; farther away, and for the
                        # speed of every occluded frame, the trajectory of the box centres fitted in the first
                        # pass is used (it stays on the vehicle while the contact path follows the occluder).
                        hd_b, sp_b = _track_kinematics(t, bf.centre, ok, cfg)
                        clean = np.flatnonzero(np.all(np.isfinite(v_g2), axis=1))
                        far = np.ones(n, bool)
                        if len(clean):
                            far = np.min(np.abs(t[:, None] - t[clean][None, :]), axis=1) > HOLD_S
                        hb = far & np.isfinite(hd_b)
                        hd2[hb] = hd_b[hb]
                        hd2[far & ~hb] = hd_ref[far & ~hb]
                        # an occlusion between two clean stretches (for example during a turn) interpolates the
                        # heading between them instead of holding the nearer one
                        sp_c = np.linalg.norm(np.nan_to_num(v_g2), axis=1)
                        gi = np.flatnonzero(np.all(np.isfinite(v_g2), axis=1) & (sp_c > slow_mps))
                        if len(gi) >= 2:
                            inside = occl & ~far & (t > t[gi[0]]) & (t < t[gi[-1]])
                            th = np.unwrap(np.arctan2(v_g2[gi, 1], v_g2[gi, 0]))
                            hd2[inside] = np.interp(t[inside], t[gi], th)
                        sb = occl & np.isfinite(sp_b)
                        speed2[sb] = sp_b[sb]
                        slow2 = speed2 < slow_mps
                        speed, slow = speed2, slow2
                        pose = _pose_frames(t, gate, slow, cfg)
                        changed = np.abs(wrap(hd2 - hd_ref)) > np.radians(1.0)
                        hd_ref = hd2
                        pred = _predict(t, bf.centre, good, cfg)
                        pred[~occl] = np.nan
                        anc2 = envelope_anchor(
                            camera, t, g_s, hd_ref, envs, bf.L, bf.W, bf.gap, cfg.boxfit_min_depression_deg, pred
                        )
                        a2 = anc2.centre if anc2 is not None else centre
                        s2 = anc2.sigma if anc2 is not None else sigma
                        bf2 = fit_track_boxes(
                            camera,
                            rasters,
                            hd_ref,
                            a2,
                            weights,
                            (bf.L, bf.W, bf.H),
                            bounds,
                            anchors=a2 if s2 is not None else None,
                            anchor_sigma=s2,
                            pose_frames=(gate & occl) | (pose & changed),
                            fixed_dims=(bf.L, bf.W, bf.H, bf.gap),
                            refine_heading=~occl,
                        )
                        if bf2 is not None:
                            r2 = np.isfinite(bf2.cost)
                            bf.centre[r2], bf.heading[r2], bf.cost[r2] = bf2.centre[r2], bf2.heading[r2], bf2.cost[r2]
                            bf.fill[r2], bf.miss[r2] = bf2.fill[r2], bf2.miss[r2]
                            ok = bf.ok(cfg.boxfit_max_cost)
                        if anc2 is not None:
                            centre[occl] = anc2.centre[occl]
                        # frames first fitted in the second pass are judged on their own fill as well
                        _, hidden2 = _occluded(t, bf.fill, bf.miss, ok, 0.5 * max(1.5, cfg.smooth_window_s))
                        hidden = hidden | hidden2
                        measured = ~hidden
                        fit_info["occluded"] = int(occl.sum())
                        fit_info["hidden"] = int(hidden.sum())
                    box_ok = ok
                    # gated frames without a pose fit take the anchor plus the fit residual interpolated in time
                    oi = np.flatnonzero(ok)
                    rest = np.flatnonzero(gate & ~ok & ~np.isfinite(bf.cost))
                    if len(oi) >= 2 and len(rest):
                        near = rest[np.min(np.abs(t[rest, None] - t[oi][None, :]), axis=1) <= 0.5]
                        res = bf.centre[oi] - centre[oi]
                        for ax in range(2):
                            centre[near, ax] += np.interp(t[near], t[oi], res[:, ax])
                    centre[ok] = bf.centre[ok]
                    L, W, H, gap = bf.L, bf.W, bf.H, bf.gap
                    source = "boxfit"
                    d = np.full(n, np.nan)
                    d[ok] = wrap(bf.heading[ok] - hd_ref[ok])
                    oi = np.flatnonzero(ok)
                    if len(oi) >= 3:
                        d[oi] = running_median(t[oi], d[oi, None], 0.6)[:, 0]
                    heading = hd_ref + np.interp(t, t[oi], d[oi])
    centre = freeze_slow_offsets(centre, g_s, speed, slow_mps, prefer=box_ok if box_ok.any() else None)
    # The band bounds anchor-only centres. With an accepted box fit it is not applied: the frames between fits are
    # tied to them through the interpolated residual, and the band's support-point model of the contact does not
    # hold when the lowest mask point is a tyre inset from the vehicle's corner. Occluded contacts are displaced.
    free = ~box_ok & ~occl & (source != "boxfit")
    if free.any():
        centre[free] = apply_clearance_band(camera, centre[free], g_s[free], heading[free], L, W, gap)
    if slow.all():  # stopped throughout: one axis (the travel direction is unknown)
        ref = box_ok if box_ok.any() else np.ones(n, bool)
        heading[:] = float(np.angle(np.mean(np.exp(2j * heading[ref]))) / 2)
    return MaskGeometry(
        centre,
        wrap(heading),
        float(L),
        float(W),
        float(H),
        float(gap),
        box_ok & measured,
        anchored,
        measured,
        source,
        time.perf_counter() - t_start,
        {"moved": bool(moved), "box": fit_info, "masked": int(has.sum())},
    )

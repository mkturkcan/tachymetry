"""Per-track repair: from raw associations to clean metric trajectories.

Stages, in order (all thresholds in seconds and metres):

1. **Validity** of each detection (below the horizon, ground sampling distance
   within bounds) and one detection per frame.
2. **Presplit** at long gaps and at physically impossible steps of the robust
   (running-median) ground path. All resulting runs are kept.
3. **Identity-switch split**, only when two independent signals agree: the box
   jumps away from its constant-velocity prediction *and* the ground position
   violates the local kinematics.
4. **End trimming** of truncated, frame-filling and far-field-collapsed boxes.
   Truncated boxes inside a track keep their time stamps but contribute no
   measurement.
5. **Centre estimation**: for vehicles with instance masks, the mask path of
   :mod:`tachymetry.tracking.maskgeom` (mask ground contact, truncation-aware
   envelope anchor and calibrated box-silhouette fit with shared dimensions);
   otherwise the trusted cuboid footprint centroid where available, and
   elsewhere the ground contact pushed along the viewing ray by half the
   footprint's ray extent, minus the parallax of the lowest visible edge, with
   the footprint-model residual interpolated through gaps.
6. **Hampel repair** of isolated outliers.
7. **Gap-aware zero-phase smoothing** (uniform-time resampling, edge-replicated
   median, Savitzky-Golay with analytic derivative).
8. **End-fold trimming** of hook-shaped track ends.
9. **Kinematics** (speed, heading held through stops), **hold-still** freezing of
   stopped runs, and **parked** classification.
10. **Physical validity**: sustained impossible speed or acceleration, path
    reversal, and the centre-inside-own-box referee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import SUPER_CLASS, HygieneConfig
from ..detections import DetectionSet
from ..geometry.camera import Camera
from .boxfit import StreetAxes
from .maskgeom import vehicle_geometry
from .measure import CENTROID_HEIGHTS, CLEARANCE_PRIORS, DIM_BOUNDS, GroundObservations, dims_for
from .model import Flags, Track
from .signal import fill_angles, hampel, running_median, segments_by_gap, smooth_kinematics, wrap

__all__ = ["TrackContext", "class_vote", "process_raw_track", "HygieneStats"]


@dataclass
class TrackContext:
    dets: DetectionSet
    obs: GroundObservations
    camera: Camera
    cfg: HygieneConfig
    axes: StreetAxes | None = None  #: street axes from moving traffic (stopped vehicles' headings)
    mask_cache: dict[int, Any] = field(default_factory=dict)  #: per-detection mask envelopes

    @property
    def image_size(self) -> tuple[int, int]:
        return self.dets.image_size if self.dets.image_size[0] else self.camera.image_size


@dataclass
class HygieneStats:
    raw_tracks: int = 0
    splits_gap_or_teleport: int = 0
    splits_identity: int = 0
    trimmed_samples: int = 0
    dropped_short: int = 0
    repaired_samples: int = 0
    folds_trimmed: int = 0
    invalid: int = 0
    parked: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


#: Lowest weighted evidence share that decides the class of a track (detectors with class evidence).
CLASS_SHARE_MIN = 0.20
#: Two-wheeler flag of a vehicle track: share of its detections covered by a confident bike detection,
#: and the least number of such detections.
BIKE_FLAG_SHARE = 0.5
BIKE_FLAG_MIN_FRAMES = 4
#: Unresolved tracks whose car fit reaches this length are fitted again as trucks (size fallback).
SIZE_FALLBACK_CAR_L = 6.7
#: ... and are trucks when that fit is longer than this.
SIZE_FALLBACK_TRUCK_L = 8.0
#: Unresolved tracks whose car fit stays at the lower bounds of a car (length and width at most these)
#: are smaller than any car, and become two-wheelers (the small-vehicle size rule).
SIZE_FALLBACK_SMALL_LW = (3.15, 1.65)


def view_weights(dets: DetectionSet, rows: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Quality weight of each detection for class votes: sqrt(box area) times centrality.

    ``centrality = (1 - 0.8 |x_mid - W/2| / (W/2)) (0.4 + 0.6 y_bottom / H)``, so that large,
    mid-frame, near-field views dominate and far-field views, where vehicles look alike, barely vote.
    """
    w, h = image_size
    b = dets.xyxy[rows].astype(np.float64)
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1.0)
    xm = 0.5 * (b[:, 0] + b[:, 2])
    central = (1 - 0.8 * np.clip(np.abs(xm - w / 2) / (w / 2), 0, 1)) * (0.4 + 0.6 * np.clip(b[:, 3] / h, 0, 1))
    return np.sqrt(area) * np.maximum(central, 0.05)


def class_vote(dets: DetectionSet, rows: np.ndarray, image_size: tuple[int, int]) -> tuple[str, dict[str, Any]]:
    """Class of a track from its detections. Returns ``(class, info)``.

    With class evidence (SAM 3), each detection votes for every class prompt with
    its view weight (:func:`view_weights`) times the score of the matching evidence
    detection, and the shares are the votes divided by the total weight of the track,
    frames without a match included. The class is the argmax over the shares of the
    slots two-wheeler, ``bus``, ``truck``, ``car``, ``van`` and ``suv``
    (:data:`~tachymetry.detect.sam3.CLASS_PROMPTS`; a detection's two-wheeler score is
    the larger of its ``bike`` and ``motorcycle`` evidence; ``car``, ``van`` and ``suv``
    fold into ``car``; a two-wheeler is a motorcycle when the ``motorcycle`` share
    exceeds the ``bike`` share, else a bicycle) when that share reaches
    :data:`CLASS_SHARE_MIN`.
    Otherwise a track with a confident bike detection covering more than half of
    its detections (at least four) is a two-wheeler, and any other track a car,
    subject to the size fallback of :func:`process_raw_track`: a car fit at the car
    length bound is refitted as a truck, and one at the smallest car (see
    :data:`SIZE_FALLBACK_SMALL_LW`) makes the track a two-wheeler (``info["rule"]`` is
    ``"evidence"``, ``"bike_flag"`` or ``"size"``; ``meta["class_rule"]`` becomes
    ``"size_small"`` in the last case).

    Without evidence, the per-detection classes vote with the view weight times the
    detection score (``info["rule"] == "detections"``).
    """
    names = dets.class_names
    wt = view_weights(dets, rows, image_size)
    cls_rows = [names[int(c)] for c in dets.cls[rows]]
    if dets.evidence is not None and dets.evidence_names and "person" not in cls_rows:
        from ..detect.sam3 import CLASS_PROMPTS, EVIDENCE_COLUMNS, class_scores, fold_class

        ev = np.zeros((len(rows), len(EVIDENCE_COLUMNS)), np.float64)
        for j, n in enumerate(EVIDENCE_COLUMNS):
            if n in dets.evidence_names:
                ev[:, j] = dets.evidence[rows, dets.evidence_names.index(n)]
        total = max(float(wt.sum()), 1e-9)
        slots = (wt[:, None] * class_scores(ev)).sum(axis=0) / total
        col = {n: k for k, n in enumerate(EVIDENCE_COLUMNS)}
        bike = float((wt * ev[:, col["bike"]]).sum() / total)
        moto = float((wt * ev[:, col["motorcycle"]]).sum() / total)
        shares = {("two_wheeler" if p == "bike" else p): round(float(v), 4) for p, v in zip(CLASS_PROMPTS, slots)}
        shares.update(bike=round(bike, 4), motorcycle=round(moto, 4))
        info: dict[str, Any] = {"shares": shares}
        if float(slots.max()) >= CLASS_SHARE_MIN:
            info["rule"] = "evidence"
            return str(fold_class(slots, bike, moto)), info
        cover = ev[:, col["bike_cover"]] > 0
        if cover.sum() >= BIKE_FLAG_MIN_FRAMES and cover.mean() > BIKE_FLAG_SHARE:
            info["rule"] = "bike_flag"
            return ("motorcycle" if moto > bike else "bicycle"), info
        info["rule"] = "size"
        return "car", info
    votes: dict[str, float] = {}
    for name, v in zip(cls_rows, dets.score[rows] * wt):
        votes[name] = votes.get(name, 0.0) + float(v)
    tot = sum(votes.values()) or 1.0
    shares_d = {k: v / tot for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}
    return next(iter(shares_d)), {"rule": "detections", "shares": shares_d}


def _box_centres(b: np.ndarray) -> np.ndarray:
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])


def _split_raw(rows: np.ndarray, ctx: TrackContext, sup: str, stats: HygieneStats) -> list[np.ndarray]:
    cfg = ctx.cfg
    d, o = ctx.dets, ctx.obs
    t = d.time[rows]
    if len(rows) < 2:
        return [rows]
    c = o.contact[rows]
    cm = running_median(t, c, cfg.median_window_s) if len(rows) >= 3 else c
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(cm, axis=0), axis=1)
    s = np.maximum(cfg.position_noise_m, 3.0 * np.nan_to_num(o.gsd[rows], nan=1.0))
    sig = np.sqrt(s[:-1] ** 2 + s[1:] ** 2)
    vmax = cfg.teleport_speed_mps.get(sup, 55.0)
    cut = (dt > cfg.max_gap_s) | (step > vmax * dt + 3.0 * sig)
    stats.splits_gap_or_teleport += int(cut.sum())
    # identity switch: box jump AND kinematic violation
    b = d.xyxy[rows].astype(np.float64)
    bc = _box_centres(b)
    bw = np.maximum(b[:, 2] - b[:, 0], 20.0 * ctx.image_size[1] / 1080.0)
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1.0)
    trunc = o.truncated[rows] | o.big[rows]
    lo_a, hi_a = cfg.switch_area
    for i in range(len(rows) - 1):
        if cut[i] or dt[i] > 1.0:
            continue
        j = int(np.searchsorted(t, t[i] - 0.5, side="left"))
        v_img = (bc[i] - bc[j]) / (t[i] - t[j]) if j < i else np.zeros(2)
        jump = np.linalg.norm(bc[i + 1] - (bc[i] + v_img * dt[i])) / bw[i]
        ratio = area[i + 1] / area[i]
        boxy = jump > cfg.switch_jump or (not trunc[i] and not trunc[i + 1] and not (lo_a <= ratio <= hi_a))
        if not boxy:
            continue
        k = int(np.searchsorted(t, t[i] - 1.0, side="left"))
        vg = (cm[i] - cm[k]) / (t[i] - t[k]) if k < i else np.zeros(2)
        resid = np.linalg.norm(cm[i + 1] - (cm[i] + vg * dt[i]))
        if resid > max(2.0, 3.0 * sig[i]):
            cut[i] = True
            stats.splits_identity += 1
    return [r for r in np.split(rows, np.flatnonzero(cut) + 1) if len(r)]


def _trim_ends(rows: np.ndarray, ctx: TrackContext, stats: HygieneStats) -> np.ndarray:
    cfg = ctx.cfg
    o, d = ctx.obs, ctx.dets
    bad = o.truncated[rows] | o.big[rows]
    n = len(rows)
    a, b = 0, n
    while a < b and bad[a]:
        a += 1
    while b > a and bad[b - 1]:
        b -= 1
    rows2 = rows[a:b]
    # far-field collapse at either end: box height far below its inward running median
    if len(rows2) >= 6:
        t = d.time[rows2]
        hgt = (d.xyxy[rows2, 3] - d.xyxy[rows2, 1]).astype(np.float64)
        n2 = len(rows2)

        def collapsed(i: int, inward: int) -> bool:
            if inward > 0:
                sel = (t > t[i] + 0.1) & (t <= t[i] + 1.0)
            else:
                sel = (t < t[i] - 0.1) & (t >= t[i] - 1.0)
            if sel.sum() < 3:
                return False
            return hgt[i] < cfg.far_shrink * float(np.median(hgt[sel]))

        a2, b2 = 0, n2
        while a2 < n2 - 3 and collapsed(a2, +1):
            a2 += 1
        while b2 > a2 + 3 and collapsed(b2 - 1, -1):
            b2 -= 1
        rows2 = rows2[a2:b2]
    # confidence collapse at either end: when the object it followed is occluded or leaves, a track can
    # continue on a neighbouring or phantom detection (for example from one parked car to the next) at a
    # small fraction of its usual score
    if len(rows2) >= 6:
        s = d.score[rows2]
        thr = min(cfg.end_score_max, cfg.end_score_ratio * float(np.median(s)))
        a3, b3 = 0, len(rows2)
        while a3 < b3 - 3 and s[a3] < thr:
            a3 += 1
        while b3 > a3 + 3 and s[b3 - 1] < thr:
            b3 -= 1
        rows2 = rows2[a3:b3]
    stats.trimmed_samples += n - len(rows2)
    return rows2


def _dims(rows: np.ndarray, ctx: TrackContext, cls: str, use: np.ndarray) -> tuple[float, float, float]:
    L0, W0, H0 = dims_for(cls)
    o = ctx.obs
    sel = use & o.has_footprint[rows]
    if cls == "person" or sel.sum() < 3:
        return L0, W0, H0
    g = o.gsd[rows][sel]
    near = g <= np.percentile(g, 75)
    Lv = o.length[rows][sel][near]
    Wv = o.width[rows][sel][near]
    if len(Lv) < 3:
        return L0, W0, H0
    lo_l, hi_l, lo_w, hi_w = DIM_BOUNDS.get(cls, (0.3, 25, 0.3, 4))
    L = float(np.clip(np.percentile(Lv, 70), lo_l, hi_l))
    W = float(np.clip(np.percentile(Wv, 60), lo_w, hi_w))
    return L, W, H0


def _box_ratios(
    rows: np.ndarray,
    ctx: TrackContext,
    centre: np.ndarray,
    heading: np.ndarray,
    L: float,
    W: float,
    H: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Width and height of each detector box over those of the projection of the track's own 3D box.

    The expected box is the image bounding box of an upright ``L x W x H`` box at
    the sample's centre and heading. Ratios are NaN where the projection fails.
    """
    cam = ctx.camera
    b = ctx.dets.xyxy[rows].astype(np.float64)
    rw = np.full(len(rows), np.nan)
    rh = np.full(len(rows), np.nan)
    c, s = np.cos(heading), np.sin(heading)
    for i in range(len(rows)):
        f = np.array([c[i], s[i]]) * L / 2
        lf = np.array([-s[i], c[i]]) * W / 2
        base = np.array([centre[i] + f + lf, centre[i] - f + lf, centre[i] - f - lf, centre[i] + f - lf])
        pts = np.vstack([np.column_stack([base, np.zeros(4)]), np.column_stack([base, np.full(4, H)])])
        uv = cam.project(pts)
        if not np.all(np.isfinite(uv)):
            continue
        ew, eh = np.ptp(uv[:, 0]), np.ptp(uv[:, 1])
        if ew > 0 and eh > 0:
            rw[i] = (b[i, 2] - b[i, 0]) / ew
            rh[i] = (b[i, 3] - b[i, 1]) / eh
    return rw, rh


def _partial_visibility(
    rows: np.ndarray,
    ctx: TrackContext,
    centre: np.ndarray,
    heading: np.ndarray,
    L: float,
    W: float,
    H: float,
    has_fp: np.ndarray,
) -> np.ndarray:
    """Samples at which the vehicle is partly hidden (occluded or outside the frame).

    Three tests, with the box ratios of :func:`_box_ratios`:

    * a sample without a 3D footprint whose detector box covers less than
      ``min_ratio`` = 0.55 of the expected box in width or height;
    * a sample whose 3D footprint is shorter than ``partial_footprint_ratio``
      times the vehicle's length while its box is also smaller than usual, so
      the footprint is a guess at hidden corners;
    * occlusion onset or exit at either end of the track: the run of samples,
      contiguous with the end, whose box height or width ratio falls below
      ``end_shrink`` times the track's median ratio (a vehicle passing behind
      a tree or a building keeps moving while its box stops growing or shrinks);
      applied to vehicles that move more than 3 m, and to at most a quarter of
      the track at each end.
    """
    cfg = ctx.cfg
    rw, rh = _box_ratios(rows, ctx, centre, heading, L, W, H)
    out = ((rw < 0.55) | (rh < 0.55)) & ~has_fp

    def rel(x: np.ndarray) -> np.ndarray:
        m = np.nanmedian(x) if np.isfinite(x).any() else np.nan
        return x / m if np.isfinite(m) and m > 0 else np.ones_like(x)

    qw, qh = rel(rw), rel(rh)
    smaller = (qw < cfg.end_shrink) | (qh < cfg.end_shrink)
    fl = ctx.obs.length[rows] / max(L, 1e-6)
    out |= has_fp & np.isfinite(fl) & (fl < cfg.partial_footprint_ratio) & smaller
    # end runs only for vehicles that move (a parked vehicle's box shrinks only when something passes in
    # front of it), and never more than half of the track
    contact = ctx.obs.contact[rows]
    k = max(1, min(5, len(rows) // 4))
    moved = float(np.linalg.norm(np.nanmedian(contact[-k:], axis=0) - np.nanmedian(contact[:k], axis=0)))
    if np.isfinite(moved) and moved > 3.0:
        n = len(rows)
        a, b = 0, n
        while a < n // 4 and smaller[a]:
            a += 1
        while b > max(a, n - n // 4) and smaller[b - 1]:
            b -= 1
        out[:a] = True
        out[b:] = True
    return out


def _centres(
    rows: np.ndarray,
    ctx: TrackContext,
    cls: str,
    use: np.ndarray,
    L: float,
    W: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Footprint centres, the motion/footprint heading used, and a footprint flag per sample.

    The model centre is the ground contact pushed along the viewing ray by half
    the footprint's ray extent, less the parallax of the lowest visible edge.
    It assumes that the lowest point of the object in the image is its side
    facing the camera, which holds while the camera's nadir lies well below the
    image. In steep views it does not: the lowest image point is merely one side
    of the object. There, the centroid of the object's silhouette, intersected
    with the horizontal plane at the class's centroid height
    (``CENTROID_HEIGHTS``), lies above the footprint centre, with an error of
    the height mismatch times the cotangent of the depression angle. The model
    therefore moves from the ray offset to the centroid as the depression angle
    of the sample rises through ``steep_view_deg`` (60 to 80 degrees), with a
    smoothstep weight so that a track crossing the band has continuous position
    and velocity. Oblique cameras, whose samples stay below 60 degrees, are
    unaffected.
    """
    cfg, o, cam = ctx.cfg, ctx.obs, ctx.camera
    t = ctx.dets.time[rows]
    contact = o.contact[rows]
    pc, vc, _ = smooth_kinematics(
        t, contact, use, cfg.median_window_s, max(1.5, cfg.smooth_window_s), 1, cfg.interp_gap_s
    )
    sp = np.linalg.norm(np.nan_to_num(vc), axis=1)
    hd = np.where(sp > 1.0, np.arctan2(vc[:, 1], vc[:, 0]), np.nan)
    fh = o.heading[rows]
    has_fp = o.has_footprint[rows] & use
    if np.isfinite(hd).sum() == 0 and np.isfinite(fh).any():
        # axial (mod pi) circular mean of the footprint headings
        a2 = np.nanmean(np.exp(2j * fh[np.isfinite(fh)]))
        hd[:] = np.angle(a2) / 2
    hd = fill_angles(hd)
    ray = contact - cam.center[:2]
    dist = np.maximum(np.linalg.norm(ray, axis=1), 1e-6)
    rhat = ray / dist[:, None]
    if not np.isfinite(hd).any():
        hd = np.arctan2(rhat[:, 1], rhat[:, 0])
    b = hd - np.arctan2(rhat[:, 1], rhat[:, 0])
    ext = np.abs(L * np.cos(b)) + np.abs(W * np.sin(b))
    clearance = CLEARANCE_PRIORS.get(cls, 0.08)
    overshoot = dist * clearance / max(cam.height, 0.5)
    rho = np.maximum(0.5 * ext - overshoot, 0.0)
    model = contact + rhat * rho[:, None]
    lo, hi = (float(v) for v in cfg.steep_view_deg)
    dep = np.degrees(np.arctan2(max(cam.height, 1e-6), dist))
    x = np.clip((dep - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    w = x * x * (3.0 - 2.0 * x)
    if np.any(w > 0) and o.centroid_px is not None:
        g_c = cam.image_to_ground(o.centroid_px[rows], z=CENTROID_HEIGHTS.get(cls, 0.5 * dims_for(cls)[2]))
        sel = (w > 0) & np.all(np.isfinite(g_c), axis=1)
        model[sel] = (1.0 - w[sel, None]) * model[sel] + w[sel, None] * g_c[sel]
    centre = model.copy()
    if has_fp.any():
        fp = o.center[rows]
        resid = fp - model
        idx = np.flatnonzero(has_fp)
        for ax in range(2):
            ri = np.interp(t, t[idx], resid[idx, ax])
            # decay the borrowed residual far from any footprint sample
            near_t = np.min(np.abs(t[:, None] - t[idx][None, :]), axis=1)
            ri *= np.clip(1.0 - (near_t - 1.0) / 2.0, 0.0, 1.0)
            centre[:, ax] = model[:, ax] + ri
        centre[has_fp] = fp[has_fp]
    return centre, hd, has_fp


def _fold_trim(t: np.ndarray, pos: np.ndarray, speed: np.ndarray, cfg: HygieneConfig) -> tuple[int, int]:
    """Indices ``[a, b)`` that remove hook-shaped ends."""
    n = len(t)
    if n < 6:
        return 0, n
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    lim = np.cos(np.radians(cfg.fold_deg))
    move = cfg.move_speed_kmh / 3.6

    def end_fold(sv: np.ndarray, p: np.ndarray, sp: np.ndarray) -> int:
        """Number of samples to cut from the end of (s, p)."""
        total = sv[-1]
        if total < 3 * cfg.fold_dist_m or sp[-1] < move:
            return 0
        k = int(np.searchsorted(sv, total - cfg.fold_dist_m, side="right")) - 1
        j = int(np.searchsorted(sv, sv[k] - cfg.fold_dist_m, side="right")) - 1
        if j < 0 or k <= j or k >= len(sv) - 1:
            return 0
        d_end = p[-1] - p[k]
        d_prev = p[k] - p[j]
        ne, npv = np.linalg.norm(d_end), np.linalg.norm(d_prev)
        if ne < 0.3 or npv < 0.3:
            return 0
        if float(d_end @ d_prev) / (ne * npv) < lim:
            return len(sv) - 1 - k
        return 0

    cut_end = end_fold(s, pos, speed)
    cut_start = end_fold((s[-1] - s)[::-1], pos[::-1], speed[::-1])
    return cut_start, n - cut_end


def _hold_still(
    t: np.ndarray, pos: np.ndarray, vel: np.ndarray, raw: np.ndarray, cfg: HygieneConfig, flags: np.ndarray
) -> None:
    sp = np.linalg.norm(vel, axis=1)
    stop = sp < cfg.stop_speed_kmh / 3.6
    if not stop.any():
        return
    edges = np.flatnonzero(np.diff(np.concatenate([[0], stop.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if t[b - 1] - t[a] < 0.6:
            continue
        r = raw[a:b]
        r = r[np.all(np.isfinite(r), axis=1)]
        ref = r if len(r) >= 2 else pos[a:b]
        med = np.median(ref, axis=0)
        if np.sqrt(np.mean(np.sum((ref - med) ** 2, axis=1))) > 0.6:
            continue
        pos[a:b] = med
        vel[a:b] = 0.0
        flags[a:b] |= Flags.HELD


def _validity(tr: Track, ctx: TrackContext) -> str:
    """Empty string when physically valid, else the reason."""
    cfg = ctx.cfg
    t, sp = tr.t, tr.speed
    if len(t) < 3:
        return ""
    vmax = cfg.impossible_speed_mps.get(tr.super_cls, 60.0)

    def sustained(mask: np.ndarray) -> bool:
        if not mask.any():
            return False
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
        return any(t[b - 1] - t[a] >= cfg.accel_duration_s for a, b in zip(edges[::2], edges[1::2]))

    if sustained(sp > vmax):
        return "impossible_speed"
    spm = running_median(t, sp[:, None], 0.4)[:, 0]
    acc = np.gradient(spm, t) if len(np.unique(t)) == len(t) else np.zeros_like(spm)
    if tr.super_cls != "person" and sustained(np.abs(acc) > cfg.max_accel_mps2):
        return "impossible_acceleration"
    # path reversal on a 0.4 s lattice
    if tr.duration > 1.2:
        tau = np.arange(t[0], t[-1], 0.4)
        P = np.column_stack([np.interp(tau, t, tr.xy[:, 0]), np.interp(tau, t, tr.xy[:, 1])])
        dP = np.diff(P, axis=0)
        n = np.linalg.norm(dP, axis=1)
        for i in range(len(dP) - 1):
            if n[i] > cfg.reversal_min_m and n[i + 1] > cfg.reversal_min_m:
                cosang = float(dP[i] @ dP[i + 1]) / (n[i] * n[i + 1])
                if cosang < np.cos(np.radians(cfg.reversal_deg)):
                    return "path_reversal"
    # referee: the footprint centre must project inside the object's own box
    obs = (tr.flags & Flags.OBSERVED) > 0
    if obs.sum() >= 10 and tr.super_cls != "person":
        cam = ctx.camera
        pts = np.column_stack([tr.xy[obs], np.full(obs.sum(), 0.3 * tr.height)])
        uv = cam.project(pts)
        b = tr.bbox[obs]
        mw = cfg.inside_box_margin * (b[:, 2] - b[:, 0])
        mh = cfg.inside_box_margin * (b[:, 3] - b[:, 1])
        inside = (
            (uv[:, 0] >= b[:, 0] - mw)
            & (uv[:, 0] <= b[:, 2] + mw)
            & (uv[:, 1] >= b[:, 1] - mh)
            & (uv[:, 1] <= b[:, 3] + mh)
        )
        if np.mean(inside) < cfg.inside_box_share:
            return "centre_outside_box"
    return ""


def _build(rows: np.ndarray, ctx: TrackContext, cls: str, stats: HygieneStats) -> Track | None:
    cfg = ctx.cfg
    d, o = ctx.dets, ctx.obs
    sup = SUPER_CLASS.get(cls, cls)
    t = d.time[rows].astype(np.float64)
    if len(rows) < 3 or t[-1] - t[0] < cfg.min_track_s:
        stats.dropped_short += 1
        return None
    use = ~(o.truncated[rows] | o.big[rows])
    if use.sum() < 3:
        stats.dropped_short += 1
        return None
    L, W, H = _dims(rows, ctx, cls, use)
    geo = None
    if cfg.footprints and sup == "vehicle" and cfg.mask_geometry and d.masks is not None:
        geo = vehicle_geometry(rows, d, o, ctx.camera, cfg, cls, use, ctx.axes, ctx.mask_cache)
    if geo is not None:
        # occlusion and truncation are handled by the shared-dimension box fit and the envelope anchor, so partly
        # visible samples are kept rather than trimmed
        hidden = use & ~geo.measured
        use = use & geo.measured
        if use.sum() < 3:
            stats.dropped_short += 1
            return None
        centre, hd_model, has_fp = geo.centre, geo.heading, geo.box_ok & use
        L, W, H = geo.L, geo.W, geo.H
    elif cfg.footprints:
        centre, hd_model, has_fp = _centres(rows, ctx, cls, use, L, W)
        if sup == "vehicle" and cfg.trim:
            partial = _partial_visibility(rows, ctx, centre, hd_model, L, W, H, has_fp)
            if partial.any():
                # partly visible ends are trimmed; partly visible samples inside the track are interpolated
                a, b = 0, len(rows)
                while a < b and (partial[a] or not use[a]):
                    a += 1
                while b > a and (partial[b - 1] or not use[b - 1]):
                    b -= 1
                if a > 0 or b < len(rows):
                    stats.trimmed_samples += len(rows) - (b - a)
                    return _build(rows[a:b], ctx, cls, stats) if b - a >= 3 else None
                use = use & ~partial
                if use.sum() < 3:
                    stats.dropped_short += 1
                    return None
                centre, hd_model, has_fp = _centres(rows, ctx, cls, use, L, W)
    else:
        centre = o.contact[rows].copy()
        hd_model = np.zeros(len(rows))
        has_fp = np.zeros(len(rows), bool)
    flags = np.full(len(rows), Flags.OBSERVED, np.uint8)
    flags[~use] = Flags.INTERPOLATED | Flags.TRUNCATED
    if geo is not None:
        flags[hidden] = Flags.INTERPOLATED  # hidden behind an occluder, not truncated by the frame
    flags[has_fp] |= Flags.FOOTPRINT
    if geo is not None:
        flags[has_fp] |= Flags.SILHOUETTE
    meas = centre.copy()
    meas[~use] = np.nan
    ok = use.copy()
    if ok.sum() >= 5 and cfg.repair:
        rep_xy, rep = hampel(t[ok], meas[ok], cfg.hampel_window_s, cfg.hampel_sigma, cfg.hampel_min_m)
        idx = np.flatnonzero(ok)
        meas[idx] = rep_xy
        flags[idx[rep]] |= Flags.REPAIRED
        stats.repaired_samples += int(rep.sum())
    pos, vel, _ = smooth_kinematics(
        t, meas, ok, cfg.median_window_s, cfg.smooth_window_s, cfg.smooth_order, cfg.interp_gap_s
    )
    good = np.all(np.isfinite(pos), axis=1)
    if good.sum() < 3:
        stats.dropped_short += 1
        return None
    sel = np.flatnonzero(good)
    a, b = _fold_trim(t[sel], pos[sel], np.linalg.norm(vel[sel], axis=1), cfg)
    if cfg.trim and (a > 0 or b < len(sel)):
        stats.folds_trimmed += 1
        sel = sel[a:b]
    if len(sel) < 3 or t[sel[-1]] - t[sel[0]] < cfg.min_track_s:
        stats.dropped_short += 1
        return None
    rows, t, pos, vel = rows[sel], t[sel], pos[sel], vel[sel]
    centre, meas, flags, hd_model, has_fp = centre[sel], meas[sel], flags[sel], hd_model[sel], has_fp[sel]
    sp = np.linalg.norm(vel, axis=1)
    move = cfg.move_speed_kmh / 3.6
    if geo is not None:
        heading = hd_model.copy()  # trajectory heading with the silhouette-measured correction
    else:
        heading = np.where(sp > move, np.arctan2(vel[:, 1], vel[:, 0]), np.nan)
        if np.isfinite(heading).sum() == 0:
            heading = hd_model.copy()
    heading = fill_angles(heading)
    if cfg.hold_still:
        _hold_still(t, pos, vel, meas, cfg, flags)
    # parked: little motion over its whole life
    sp = np.linalg.norm(vel, axis=1)
    moving_share = float(np.mean(sp > move))
    raw_ok = meas[np.all(np.isfinite(meas), axis=1)]
    ref = raw_ok if len(raw_ok) >= 3 else pos
    med = np.median(ref, axis=0)
    rog = float(np.sqrt(np.mean(np.sum((ref - med) ** 2, axis=1))))
    net = float(np.linalg.norm(pos[-1] - pos[0]))
    path = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else 0.0
    # image-space stillness: box centre wanders less than a third of its own width (immune to the
    # far-field amplification that makes ground positions of distant parked cars jitter)
    bb = d.xyxy[rows].astype(np.float64)
    bc = np.column_stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2])
    bw = float(np.median(np.maximum(bb[:, 2] - bb[:, 0], 1.0)))
    img_net = float(
        np.linalg.norm(np.median(bc[: max(3, len(bc) // 5)], axis=0) - np.median(bc[-max(3, len(bc) // 5) :], axis=0))
    )
    img_still = (
        img_net < 0.35 * bw and float(np.percentile(np.linalg.norm(bc - np.median(bc, axis=0), axis=1), 90)) < 0.6 * bw
    )
    jitter = net < 3.0 and path > 4.0 * max(net, 0.5)
    state = "moving"
    if (
        cfg.parked
        and sup != "person"
        and (
            (rog < cfg.parked_radius_m and moving_share < cfg.parked_moving_share)
            or (moving_share < 0.3 and net < 3.0)
            or (jitter and moving_share < 0.6)
            or (img_still and net < 6.0)
        )
    ):
        state = "parked"
        stats.parked += 1
        pos[:] = med
        vel[:] = 0.0
        flags |= Flags.HELD
        fh = o.heading[rows]
        if geo is not None:
            ref = has_fp if has_fp.any() else np.ones(len(rows), bool)
            heading[:] = np.angle(np.mean(np.exp(2j * hd_model[ref]))) / 2
        elif np.isfinite(fh).any():
            heading[:] = np.angle(np.nanmean(np.exp(2j * fh[np.isfinite(fh)]))) / 2
        else:
            heading[:] = float(np.angle(np.mean(np.exp(1j * heading))))
    if sup == "person" and moving_share < 0.25 and net < 2.0:
        state = "stationary"
    tr = Track(
        id=-1,
        cls=cls,
        super_cls=sup,
        t=t,
        frame=d.frame[rows].astype(np.int64),
        rows=rows.astype(np.int64),
        xy=pos,
        vel=vel,
        heading=wrap(heading),
        raw_xy=meas,
        contact=o.contact[rows].copy(),
        bbox=d.xyxy[rows].astype(np.float64),
        score=d.score[rows].astype(np.float64),
        flags=flags,
        length=L,
        width=W,
        height=H,
        state=state,
        cuboid=None if d.cuboid is None else d.cuboid[rows].copy(),
    )
    if geo is not None:
        tr.meta["geometry"] = {
            "source": geo.source,
            "clearance_m": round(geo.gap, 3),
            "silhouette_share": round(float(np.mean(has_fp)), 3),
            "occluded_samples": int(geo.info.get("box", {}).get("occluded", 0)),
            "hidden_samples": int(geo.info.get("box", {}).get("hidden", 0)),
            "seconds": round(geo.seconds, 3),
        }
    reason = _validity(tr, ctx) if (state == "moving" and cfg.validity) else ""
    if reason:
        tr.verdict, tr.reason = "invalid", reason
        stats.invalid += 1
    return tr


def process_raw_track(rows: np.ndarray, ctx: TrackContext, stats: HygieneStats | None = None) -> list[Track]:
    """Turn one raw association (detection rows) into zero or more clean tracks."""
    stats = stats if stats is not None else HygieneStats()
    d, o = ctx.dets, ctx.obs
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows[o.valid[rows]]
    if len(rows) < 2:
        return []
    order = np.lexsort((-d.score[rows], d.time[rows]))
    rows = rows[order]
    _, first = np.unique(d.frame[rows], return_index=True)
    rows = rows[np.sort(first)]
    cls, _ = class_vote(d, rows, ctx.image_size)
    sup = SUPER_CLASS.get(cls, cls)
    out = []
    segments = _split_raw(rows, ctx, sup, stats) if ctx.cfg.split else [rows]
    for seg in segments:
        if ctx.cfg.trim:
            seg = _trim_ends(seg, ctx, stats)
        if len(seg) < 3:
            stats.dropped_short += 1
            continue
        seg_cls, info = class_vote(d, seg, ctx.image_size)
        tr = _build(seg, ctx, seg_cls, stats)
        rule = info.get("rule")
        if tr is not None and rule == "size" and tr.length >= SIZE_FALLBACK_CAR_L:
            # no class evidence and a car fit at its length bound: a heavy vehicle the prompts did not see well
            alt = _build(seg, ctx, "truck", HygieneStats())
            if alt is not None and alt.length > SIZE_FALLBACK_TRUCK_L:
                tr = alt
        elif (
            tr is not None
            and rule == "size"
            and tr.length <= SIZE_FALLBACK_SMALL_LW[0]
            and tr.width <= SIZE_FALLBACK_SMALL_LW[1]
        ):
            # no class evidence and a car fit at the smallest car: a two-wheeler (or a pedal three-wheeler)
            sh = info.get("shares", {})
            small = "motorcycle" if sh.get("motorcycle", 0.0) > sh.get("bike", 0.0) else "bicycle"
            alt = _build(seg, ctx, small, HygieneStats())
            if alt is not None:
                tr, rule = alt, "size_small"
        if tr is not None:
            tr.meta["class_rule"] = rule
            if "shares" in info and info.get("rule") != "detections":
                tr.meta["class_shares"] = info["shares"]
            out.append(tr)
    return out


def _unused() -> None:  # pragma: no cover - keeps linters quiet about re-exported helpers
    _ = segments_by_gap

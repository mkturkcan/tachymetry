"""Cross-track reasoning: stitching, duplicates and false-positive verdicts.

Run on the output of :mod:`tachymetry.tracking.hygiene`, in this order:

1. **Stitching** re-joins fragments of one object (occlusion, missed detections):
   kinematic prediction from the end of one track must reach the start of the
   next within a radius that grows with the gap; heading and speed must agree;
   a track that left through the image border is never continued.
2. **Duplicates**: two concurrent tracks of one object (part-and-whole boxes).
3. **Riders**: persons on a bicycle or motorcycle are one road user with it; fast
   "pedestrians" without a detected two-wheeler become cyclists.
4. **Passengers / reflections**: person boxes contained in a vehicle box.
5. **Shadows**: person tracks moving in lockstep with a vehicle.
6. **Ghosts**: short-lived or low-confidence tracks that never go anywhere.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..config import SUPER_CLASS, FilterConfig
from ..detections import DetectionSet
from .model import Flags, Track

__all__ = ["stitch_candidates", "mark_duplicates", "mark_passengers_and_riders", "mark_shadows", "mark_ghosts"]


def _exited(tr: Track, at_end: bool, image_size: tuple[int, int]) -> bool:
    """Did the object leave (or enter) through the image border at this end?"""
    w, h = image_size
    i = -1 if at_end else 0
    trunc = (tr.flags[i] & Flags.TRUNCATED) > 0
    b = tr.bbox[i]
    m = max(3.0, 0.01 * np.hypot(w, h))
    near = b[0] <= m or b[1] <= m or b[2] >= w - 1 - m or b[3] >= h - 1 - m
    return bool(trunc or near)


def stitch_candidates(tracks: list[Track], cfg: FilterConfig, image_size: tuple[int, int]) -> list[tuple[int, int]]:
    """Pairs ``(i, j)``: track ``j`` continues track ``i``. One-to-one (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    ok = [k for k, tr in enumerate(tracks) if tr.verdict in ("ok", "invalid")]
    if len(ok) < 2:
        return []
    ends = [(k, tracks[k]) for k in ok]
    n = len(ends)
    cost = np.full((n, n), 1e6)
    for a, (i, A) in enumerate(ends):
        if _exited(A, True, image_size):
            continue
        va = A.vel[-1]
        sa = float(np.linalg.norm(va))
        for b, (j, B) in enumerate(ends):
            if i == j or A.super_cls != B.super_cls:
                continue
            gap = B.t[0] - A.t[-1]
            if gap <= 0 or gap > cfg.stitch_max_gap_s:
                continue
            if _exited(B, False, image_size) and gap > 0.5:
                continue
            pred = A.xy[-1] + va * gap
            radius = cfg.stitch_radius_m + 0.25 * sa * gap
            dist = float(np.linalg.norm(B.xy[0] - pred))
            if dist > radius:
                continue
            vb = B.vel[0]
            sb = float(np.linalg.norm(vb))
            if sa > 1.5 and sb > 1.5:
                cosang = float(va @ vb) / (sa * sb)
                if cosang < np.cos(np.radians(cfg.stitch_max_turn_deg)):
                    continue
                if abs(sa - sb) > max(3.0, 0.5 * max(sa, sb)):
                    continue
            if A.state == "parked" or B.state == "parked":
                if not (A.state == B.state == "parked" and dist < 1.5):
                    continue
            cost[a, b] = dist / radius + 0.2 * gap / cfg.stitch_max_gap_s
    r, c = linear_sum_assignment(cost)
    return [(ends[x][0], ends[y][0]) for x, y in zip(r, c) if cost[x, y] < 1.2]


def _common(a: Track, b: Track) -> tuple[np.ndarray, np.ndarray]:
    _, ia, ib = np.intersect1d(a.frame, b.frame, assume_unique=False, return_indices=True)
    return ia, ib


def _ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1.0)
    return inter / area


def _pairs(tracks: list[Track], pred: Callable[[Track, Track], bool]) -> list[tuple[int, int]]:
    out = []
    spans = [(tr.t[0], tr.t[-1]) for tr in tracks]
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            if i == j:
                continue
            if spans[i][1] < spans[j][0] or spans[j][1] < spans[i][0]:
                continue
            if pred(tracks[i], tracks[j]):
                out.append((i, j))
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def mark_duplicates(tracks: list[Track], cfg: FilterConfig) -> int:
    """Concurrent tracks of one object: sustained box overlap (range-independent) or co-located footprints."""
    n = 0
    live = [tr for tr in tracks if tr.is_ok]
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if not (a.is_ok and b.is_ok) or a.super_cls != b.super_cls:
                continue
            ia, ib = _common(a, b)
            if len(ia) < 3:
                continue
            if a.t[ia[-1]] - a.t[ia[0]] < cfg.duplicate_min_s:
                continue
            iou = _iou(a.bbox[ia], b.bbox[ib])
            ov = np.maximum(_ioa(a.bbox[ia], b.bbox[ib]), _ioa(b.bbox[ib], a.bbox[ia]))
            dist = np.linalg.norm(a.xy[ia] - b.xy[ib], axis=1)
            hit = (iou > 0.5) | ((dist < cfg.duplicate_distance_m) & (ov > 0.6))
            if hit.mean() < cfg.duplicate_share:
                continue
            loser = a if (a.flags & Flags.OBSERVED).sum() < (b.flags & Flags.OBSERVED).sum() else b
            other = b if loser is a else a
            if len(ia) / max(len(loser), 1) >= 0.5:
                loser.verdict, loser.reason = "duplicate", f"duplicate of {other.id}"
                n += 1
    return n


def mark_passengers_and_riders(
    tracks: list[Track], cfg: FilterConfig, dets: DetectionSet | None = None
) -> dict[str, int]:
    """Person tracks that are not pedestrians. The first rule that applies decides.

    1. **Rider**, on a two-wheeler, when any of these holds:

       a. a confident ``bike`` detection overlaps the person's box (``bike_ride``
          evidence of SAM 3, :mod:`tachymetry.detect.sam3`) in more than
          ``rider_share`` of its detections, at least ``rider_min_frames``;
       b. the person's box overlaps the box of a two-wheeler track, enlarged by
          15 % on each side, by more than ``rider_overlap`` of its own area in more
          than ``rider_share`` of their common frames (at least
          ``rider_min_frames``);
       c. its ground position lies within ``rider_distance_m`` of a vehicle or
          two-wheeler track in more than half of its sampled frames (every second
          detection, at least ``rider_colocation_min`` samples);
       d. its median speed exceeds ``rider_speed_kmh``. A person moving that fast
          for at least ``cyclist_min_duration_s`` with no partner from b or c
          becomes a cyclist (class ``bicycle``) instead: a two-wheeler that no
          prompt detected.

    2. **Passenger**, when the person's box lies inside a vehicle detection box of
       the same frame (intersection over the person's area above
       ``passenger_ioa``) in more than ``passenger_share`` of its detections:
       occupants of cars and buses, and reflections. Without ``dets`` the boxes of
       the vehicle tracks are used.
    """
    counts = {"passenger": 0, "rider": 0, "cyclist_inferred": 0}
    persons = [tr for tr in tracks if tr.is_ok and tr.super_cls == "person"]
    vehicles = [tr for tr in tracks if tr.is_ok and tr.super_cls == "vehicle"]
    wheels = [tr for tr in tracks if tr.is_ok and tr.super_cls == "two_wheeler"]
    ride = dets.evidence_column("bike_ride") if dets is not None else None
    veh_boxes: dict[int, np.ndarray] = {}
    if dets is not None and len(dets):
        sup = np.array([SUPER_CLASS.get(c, c) for c in dets.class_names])[dets.cls]
        rows_v = np.flatnonzero(sup != "person")
        for f in np.unique(dets.frame[rows_v]):
            veh_boxes[int(f)] = dets.xyxy[rows_v[dets.frame[rows_v] == f]].astype(np.float64)
    for p in persons:
        obs = (p.flags & Flags.OBSERVED) > 0
        if not obs.any():
            continue
        reason = ""
        partner: Track | None = None
        # a. bike evidence under the person
        if ride is not None:
            r = ride[p.rows[obs]] > 0
            if r.sum() >= cfg.rider_min_frames and r.mean() > cfg.rider_share:
                reason = "bike detection under the person"
        # b. overlap with a tracked two-wheeler
        for wtr in wheels:
            ip, iw = _common(p, wtr)
            if len(ip) < cfg.rider_min_frames:
                continue
            wb = wtr.bbox[iw].copy()
            ex = 0.15 * (wb[:, 2] - wb[:, 0])
            ey = 0.15 * (wb[:, 3] - wb[:, 1])
            wb[:, 0] -= ex
            wb[:, 2] += ex
            wb[:, 1] -= ey
            wb[:, 3] += ey
            if (_ioa(p.bbox[ip], wb) > cfg.rider_overlap).mean() > cfg.rider_share:
                partner, reason = wtr, reason or f"on two-wheeler {wtr.id}"
                break
        # c. co-located with a vehicle or two-wheeler on the ground
        if partner is None:
            sample = np.zeros(len(p), bool)
            sample[np.flatnonzero(obs)[::2]] = True
            present = np.zeros(len(p), bool)  # some vehicle is in the scene at this sample
            hit = np.zeros(len(p), bool)  # ... and one of them within rider_distance_m
            hits: dict[int, int] = {}
            for v in wheels + vehicles:
                ip, iv = _common(p, v)
                keep = sample[ip]
                if not keep.any():
                    continue
                ip, iv = ip[keep], iv[keep]
                close = np.linalg.norm(p.xy[ip] - v.xy[iv], axis=1) < cfg.rider_distance_m
                present[ip] = True
                hit[ip[close]] = True
                if close.any():
                    hits[v.id] = int(close.sum())
            tot = int(present.sum())
            if tot >= cfg.rider_colocation_min and hit.sum() / tot > 0.5:
                vid = max(hits, key=hits.get)
                partner = next(v for v in wheels + vehicles if v.id == vid)
                reason = reason or f"co-located with {partner.cls} {partner.id}"
        # d. cycling speed
        fast = len(p.speed_kmh) > 0 and float(np.median(p.speed_kmh)) > cfg.rider_speed_kmh
        if not reason and fast:
            if p.duration >= cfg.cyclist_min_duration_s:
                p.cls, p.super_cls = "bicycle", "two_wheeler"
                p.length, p.width, p.height = 1.75, 0.6, 1.6
                p.meta["inferred"] = "person moving at cycling speed"
                p.meta["class_rule"] = "inferred"
                counts["cyclist_inferred"] += 1
                continue
            reason = "moves at cycling speed"
        if reason:
            p.verdict, p.reason = "rider", reason
            if partner is not None:
                partner.meta["rider"] = p.id
                p.meta["partner"] = partner.id
            counts["rider"] += 1
            continue
        # 2. passengers: the box sits inside a vehicle box for most of the person's detections
        inside = np.zeros(len(p), bool)
        if dets is not None:
            for k in np.flatnonzero(obs):
                vb = veh_boxes.get(int(p.frame[k]))
                if vb is not None:
                    inside[k] = bool(
                        (_ioa(np.repeat(p.bbox[k : k + 1], len(vb), axis=0), vb) > cfg.passenger_ioa).any()
                    )
        else:
            for v in vehicles:
                ip, iv = _common(p, v)
                if len(ip):
                    inside[ip] |= _ioa(p.bbox[ip], v.bbox[iv]) > cfg.passenger_ioa
        if inside[obs].mean() > cfg.passenger_share:
            p.verdict, p.reason = "passenger", "box contained in a vehicle box"
            counts["passenger"] += 1
    return counts


def mark_shadows(tracks: list[Track], cfg: FilterConfig) -> int:
    n = 0
    persons = [tr for tr in tracks if tr.is_ok and tr.super_cls == "person"]
    vehicles = [tr for tr in tracks if tr.is_ok and tr.super_cls == "vehicle" and tr.state == "moving"]
    for p in persons:
        for v in vehicles:
            ip, iv = _common(p, v)
            if len(ip) < 3 or p.t[ip[-1]] - p.t[ip[0]] < cfg.shadow_min_s / 2:
                continue
            both = (p.speed_kmh[ip] > 8.0) & (v.speed_kmh[iv] > 8.0)
            if both.mean() < 0.8:
                continue
            off = p.xy[ip] - v.xy[iv]
            if np.linalg.norm(off.mean(axis=0)) < 4.0 and np.sqrt(off.var(axis=0).sum()) < 2 * cfg.shadow_offset_std_m:
                p.verdict, p.reason = "shadow", f"moves in lockstep with vehicle {v.id}"
                n += 1
                break
    return n


def mark_ghosts(tracks: list[Track], cfg: FilterConfig) -> int:
    """Short-lived tracks, and static low-confidence "vehicles" (parked-vehicle hallucinations).

    A real parked vehicle seen by a cuboid detector receives trusted 3D
    footprints; a shadow, sign or road marking detected as a car does not, and
    its confidence stays mediocre.
    """
    n = 0
    for tr in tracks:
        if not tr.is_ok:
            continue
        obs = (tr.flags & Flags.OBSERVED) > 0
        score = float(np.mean(tr.score[obs])) if obs.any() else 0.0
        fp_share = float(np.mean((tr.flags & Flags.FOOTPRINT) > 0))
        cuboid_detector = tr.cuboid is not None
        if tr.duration < cfg.ghost_min_duration_s:
            tr.verdict, tr.reason = "ghost", "too short"
        elif score < cfg.ghost_min_score and tr.path_length < 2.0:
            tr.verdict, tr.reason = "ghost", "low confidence and static"
        elif (
            tr.state == "parked"
            and tr.super_cls == "vehicle"
            and (
                (cuboid_detector and fp_share < 0.15 and score < cfg.ghost_parked_cuboid_score)
                or score < cfg.ghost_parked_score
            )
        ):
            tr.verdict, tr.reason = "ghost", "static low-confidence vehicle without a 3D footprint"
        else:
            continue
        n += 1
    return n

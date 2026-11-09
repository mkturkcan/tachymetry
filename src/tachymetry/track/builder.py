"""From detections to clean tracks: association, hygiene and verdicts."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ..config import PipelineConfig
from ..geometry.camera import Camera
from ..types import DetectionSet
from .associate import ByteTracker
from .hygiene import HygieneStats, TrackContext, process_raw_track
from .measure import GroundObservations, measure_ground
from .model import Track
from .verdicts import mark_duplicates, mark_ghosts, mark_passengers_and_riders, mark_shadows, stitch_candidates

__all__ = ["associate", "build_tracks"]


def associate(dets: DetectionSet, obs: GroundObservations | None, cfg: PipelineConfig) -> np.ndarray:
    """Track id for every detection row (``-1`` for unassociated detections)."""
    tracker = ByteTracker(cfg.tracker, dets.class_names, dets.image_size)
    ids = np.full(len(dets), -1, np.int64)
    if len(dets) == 0:
        return ids
    order = np.argsort(dets.frame, kind="stable")
    fr = dets.frame[order]
    starts = np.searchsorted(fr, dets.processed_frames, side="left")
    ends = np.searchsorted(fr, dets.processed_frames, side="right")
    fps = dets.fps or 30.0
    for f, a, b in zip(dets.processed_frames, starts, ends):
        rows = order[a:b]
        t = float(dets.time[rows[0]]) if len(rows) else float(f) / fps
        if obs is not None and len(rows):
            ground = obs.contact[rows].copy()
            ground[~obs.valid[rows]] = np.nan
            gsd = obs.gsd[rows]
        else:
            ground, gsd = None, None
        out = tracker.step(t, dets.xyxy[rows].astype(np.float64), dets.score[rows], dets.cls[rows], ground, gsd)
        ids[rows] = out
    return ids


def _group(ids: np.ndarray) -> list[np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for row, tid in enumerate(ids):
        if tid >= 0:
            groups[int(tid)].append(row)
    return [np.asarray(v, np.int64) for _, v in sorted(groups.items())]


def build_tracks(dets: DetectionSet, camera: Camera, cfg: PipelineConfig) -> tuple[list[Track], dict[str, Any]]:
    """Run association, per-track hygiene, stitching and verdicts. Returns ``(tracks, report)``."""
    obs = measure_ground(dets, camera, cfg.hygiene)
    ids = associate(dets, obs, cfg)
    ctx = TrackContext(dets, obs, camera, cfg.hygiene)
    stats = HygieneStats()
    raw = _group(ids)
    stats.raw_tracks = len(raw)
    tracks: list[Track] = []
    for rows in raw:
        tracks.extend(process_raw_track(rows, ctx, stats))

    stitched = 0
    if cfg.filters.stitch and len(tracks) > 1:
        pairs = stitch_candidates(tracks, cfg.filters, ctx.image_size)
        if pairs:
            nxt = {i: j for i, j in pairs}
            prv = {j: i for i, j in pairs}
            used: set[int] = set()
            merged: list[Track] = []
            for k in range(len(tracks)):
                if k in used or k in prv:
                    continue
                chain = [k]
                while chain[-1] in nxt and nxt[chain[-1]] not in used:
                    chain.append(nxt[chain[-1]])
                used.update(chain)
                if len(chain) == 1:
                    merged.append(tracks[k])
                    continue
                rows = np.concatenate([tracks[c].rows for c in chain])
                sub = HygieneStats()
                new = process_raw_track(rows, ctx, sub)
                stitched += len(chain) - 1
                merged.extend(new if new else [tracks[c] for c in chain])
            for k in range(len(tracks)):  # tracks only reachable as successors in a cycle (defensive)
                if k not in used:
                    merged.append(tracks[k])
            tracks = merged

    tracks.sort(key=lambda tr: (tr.t[0], tr.t[-1]))
    for k, tr in enumerate(tracks, start=1):
        tr.id = k
    verdicts = {}
    if cfg.filters.enabled:
        verdicts = {
            "duplicates": mark_duplicates(tracks, cfg.filters),
            **mark_passengers_and_riders(tracks, cfg.filters),
            "shadows": mark_shadows(tracks, cfg.filters),
            "ghosts": mark_ghosts(tracks, cfg.filters),
        }
    report = {
        "hygiene": stats.as_dict(),
        "stitched_fragments": stitched,
        "verdicts": verdicts,
        "verdict_counts": dict(Counter(tr.verdict for tr in tracks)),
        "ok_by_class": dict(Counter(tr.cls for tr in tracks if tr.is_ok)),
        "ok_by_state": dict(Counter(f"{tr.cls}:{tr.state}" for tr in tracks if tr.is_ok)),
        "detections": int(len(dets)),
        "associated_detections": int((ids >= 0).sum()),
    }
    return tracks, report

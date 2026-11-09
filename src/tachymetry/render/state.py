"""Continuous-time track state for rendering at the video frame rate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..track.model import Flags, Track

__all__ = ["TrackState", "StateIndex"]


@dataclass
class TrackState:
    track: Track
    xy: np.ndarray
    heading: float
    speed_kmh: float
    bbox: np.ndarray
    observed: bool
    trail: np.ndarray  # (k, 2) recent positions, oldest first


class StateIndex:
    """Interpolates the ``ok`` tracks at arbitrary times."""

    def __init__(self, tracks: list[Track], trail_s: float = 3.0, max_dt: float | None = None):
        self.tracks = [tr for tr in tracks if tr.is_ok and len(tr) >= 2]
        self.trail_s = trail_s
        self.t0 = np.array([tr.t[0] for tr in self.tracks]) if self.tracks else np.zeros(0)
        self.t1 = np.array([tr.t[-1] for tr in self.tracks]) if self.tracks else np.zeros(0)
        self.max_dt = max_dt

    def at(self, t: float) -> list[TrackState]:
        out = []
        if not self.tracks:
            return out
        live = np.flatnonzero((self.t0 <= t) & (self.t1 >= t))
        for k in live:
            tr = self.tracks[k]
            i = int(np.searchsorted(tr.t, t, side="right"))
            i = min(max(i, 1), len(tr.t) - 1)
            a, b = i - 1, i
            span = tr.t[b] - tr.t[a]
            if self.max_dt is not None and span > self.max_dt:
                continue
            w = 0.0 if span <= 0 else float(np.clip((t - tr.t[a]) / span, 0, 1))
            xy = tr.xy[a] * (1 - w) + tr.xy[b] * w
            dh = (tr.heading[b] - tr.heading[a] + np.pi) % (2 * np.pi) - np.pi
            hd = float(tr.heading[a] + w * dh)
            sp = float(tr.speed_kmh[a] * (1 - w) + tr.speed_kmh[b] * w)
            bb = tr.bbox[a] * (1 - w) + tr.bbox[b] * w
            obs = bool(tr.flags[a] & Flags.OBSERVED) or bool(tr.flags[b] & Flags.OBSERVED)
            lo = int(np.searchsorted(tr.t, t - self.trail_s, side="left"))
            trail = np.vstack([tr.xy[lo:b], xy[None]]) if b > lo else xy[None]
            out.append(TrackState(tr, xy, hd, sp, bb, obs, trail))
        return out

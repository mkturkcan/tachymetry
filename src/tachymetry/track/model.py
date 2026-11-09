"""Track data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Track", "Flags", "VERDICTS", "STATES"]


class Flags:
    """Per-sample bit flags."""

    OBSERVED = 1  #: the position is backed by a detection at this sample
    INTERPOLATED = 2  #: no usable measurement; position from neighbours
    REPAIRED = 4  #: an outlier replaced by the local median
    TRUNCATED = 8  #: the box touched the image border
    HELD = 16  #: position frozen during a stop (speed set to zero)
    FOOTPRINT = 32  #: centre measured from a trusted cuboid footprint


#: Track-level verdicts. Only ``ok`` tracks appear in default exports and renders.
VERDICTS = ("ok", "ghost", "passenger", "rider", "duplicate", "shadow", "invalid")
STATES = ("moving", "parked")


@dataclass
class Track:
    """A cleaned, metric trajectory."""

    id: int
    cls: str
    super_cls: str
    t: np.ndarray  # (n,) seconds
    frame: np.ndarray  # (n,) source frame index
    rows: np.ndarray  # (n,) detection row (into the run's DetectionSet)
    xy: np.ndarray  # (n, 2) smoothed footprint centre (m)
    vel: np.ndarray  # (n, 2) m/s
    heading: np.ndarray  # (n,) radians, CCW from world +x
    raw_xy: np.ndarray  # (n, 2) measured centre (NaN where not measured)
    contact: np.ndarray  # (n, 2) ground contact under the box bottom
    bbox: np.ndarray  # (n, 4) image box
    score: np.ndarray  # (n,)
    flags: np.ndarray  # (n,) uint8
    length: float
    width: float
    height: float
    verdict: str = "ok"
    reason: str = ""
    state: str = "moving"
    cuboid: np.ndarray | None = None  # (n, 8, 3) raw cuboid corners when available
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def speed(self) -> np.ndarray:
        """Speed in m/s."""
        return np.linalg.norm(self.vel, axis=1)

    @property
    def speed_kmh(self) -> np.ndarray:
        return self.speed * 3.6

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    @property
    def path_length(self) -> float:
        if len(self.xy) < 2:
            return 0.0
        return float(np.nansum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1)))

    @property
    def net_displacement(self) -> float:
        if len(self.xy) < 2:
            return 0.0
        return float(np.linalg.norm(self.xy[-1] - self.xy[0]))

    @property
    def is_ok(self) -> bool:
        return self.verdict == "ok"

    def moving_mask(self, threshold_kmh: float = 3.0) -> np.ndarray:
        return self.speed_kmh > threshold_kmh

    def summary(self) -> dict[str, Any]:
        sp = self.speed_kmh
        mv = sp[self.moving_mask()] if len(sp) else sp
        obs = (self.flags & Flags.OBSERVED) > 0
        return {
            "id": self.id,
            "class": self.cls,
            "verdict": self.verdict,
            "reason": self.reason,
            "state": self.state,
            "t_start": float(self.t[0]),
            "t_end": float(self.t[-1]),
            "duration_s": round(self.duration, 3),
            "n_samples": int(len(self)),
            "observed_share": round(float(obs.mean()), 3) if len(obs) else 0.0,
            "footprint_share": round(float(((self.flags & Flags.FOOTPRINT) > 0).mean()), 3) if len(obs) else 0.0,
            "length_m": round(self.length, 2),
            "width_m": round(self.width, 2),
            "height_m": round(self.height, 2),
            "path_m": round(self.path_length, 2),
            "net_m": round(self.net_displacement, 2),
            "speed_mean_kmh": round(float(np.mean(mv)), 2) if len(mv) else 0.0,
            "speed_median_kmh": round(float(np.median(mv)), 2) if len(mv) else 0.0,
            "speed_p85_kmh": round(float(np.percentile(mv, 85)), 2) if len(mv) else 0.0,
            "speed_max_kmh": round(float(np.max(mv)), 2) if len(mv) else 0.0,
            "moving_share": round(float(np.mean(self.moving_mask())), 3) if len(sp) else 0.0,
            "start_xy": [round(float(v), 3) for v in self.xy[0]],
            "end_xy": [round(float(v), 3) for v in self.xy[-1]],
            "mean_score": round(float(np.mean(self.score)), 3) if len(self.score) else 0.0,
        }

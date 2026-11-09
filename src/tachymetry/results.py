"""Run results and exports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .detections import DetectionSet
from .geometry.camera import Camera, _json_default
from .track.model import Flags, Track

__all__ = ["TrackSet", "RunResult"]

SAMPLE_FIELDS = [
    "track_id",
    "class",
    "state",
    "verdict",
    "frame",
    "time_s",
    "x_m",
    "y_m",
    "vx_mps",
    "vy_mps",
    "speed_kmh",
    "heading_deg",
    "observed",
    "interpolated",
    "held",
    "footprint",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
]
SUMMARY_FIELDS = [
    "id",
    "class",
    "state",
    "verdict",
    "reason",
    "t_start",
    "t_end",
    "duration_s",
    "n_samples",
    "observed_share",
    "footprint_share",
    "length_m",
    "width_m",
    "height_m",
    "path_m",
    "net_m",
    "speed_mean_kmh",
    "speed_median_kmh",
    "speed_p85_kmh",
    "speed_max_kmh",
    "moving_share",
    "mean_score",
]


@dataclass
class TrackSet:
    """All tracks of one video, with exports. Iterating yields only ``ok`` tracks."""

    tracks: list[Track]
    camera: Camera | None = None
    fps: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.ok)

    def __len__(self) -> int:
        return len(self.ok)

    @property
    def ok(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.is_ok]

    def by_id(self, tid: int) -> Track:
        for tr in self.tracks:
            if tr.id == tid:
                return tr
        raise KeyError(tid)

    def select(
        self, cls: str | Iterable[str] | None = None, state: str | None = None, include_rejected: bool = False
    ) -> list[Track]:
        classes = {cls} if isinstance(cls, str) else (set(cls) if cls is not None else None)
        pool = self.tracks if include_rejected else self.ok
        return [tr for tr in pool if (classes is None or tr.cls in classes) and (state is None or tr.state == state)]

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        ok = self.ok
        moving = [tr for tr in ok if tr.state == "moving"]
        speeds = (
            np.concatenate([tr.speed_kmh[tr.moving_mask()] for tr in moving if tr.super_cls != "person"])
            if moving
            else np.zeros(0)
        )
        return {
            "tracks": len(ok),
            "by_class": dict(Counter(tr.cls for tr in ok)),
            "moving_by_class": dict(Counter(tr.cls for tr in moving)),
            "parked_by_class": dict(Counter(tr.cls for tr in ok if tr.state == "parked")),
            "rejected": dict(Counter(tr.verdict for tr in self.tracks if not tr.is_ok)),
            "vehicle_speed_kmh": {
                "median": round(float(np.median(speeds)), 2) if len(speeds) else None,
                "p85": round(float(np.percentile(speeds, 85)), 2) if len(speeds) else None,
                "max": round(float(np.max(speeds)), 2) if len(speeds) else None,
            },
        }

    # ---------------------------------------------------------------- exports
    def _rows(self, include_rejected: bool = False) -> Iterable[list[Any]]:
        for tr in self.tracks if include_rejected else self.ok:
            sp = tr.speed_kmh
            hd = np.degrees(tr.heading)
            for k in range(len(tr)):
                f = int(tr.flags[k])
                b = tr.bbox[k]
                yield [
                    tr.id,
                    tr.cls,
                    tr.state,
                    tr.verdict,
                    int(tr.frame[k]),
                    round(float(tr.t[k]), 4),
                    round(float(tr.xy[k, 0]), 3),
                    round(float(tr.xy[k, 1]), 3),
                    round(float(tr.vel[k, 0]), 3),
                    round(float(tr.vel[k, 1]), 3),
                    round(float(sp[k]), 2),
                    round(float(hd[k]), 1),
                    int(bool(f & Flags.OBSERVED)),
                    int(bool(f & Flags.INTERPOLATED)),
                    int(bool(f & Flags.HELD)),
                    int(bool(f & Flags.FOOTPRINT)),
                    round(float(b[0]), 1),
                    round(float(b[1]), 1),
                    round(float(b[2]), 1),
                    round(float(b[3]), 1),
                    round(float(tr.score[k]), 3),
                ]

    def to_csv(self, path: str | Path, include_rejected: bool = False) -> Path:
        """One row per track sample (metres, m/s, km/h, degrees)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(SAMPLE_FIELDS)
            w.writerows(self._rows(include_rejected))
        return path

    def summary_csv(self, path: str | Path, include_rejected: bool = True) -> Path:
        """One row per track."""
        path = Path(path)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            w.writeheader()
            for tr in self.tracks if include_rejected else self.ok:
                w.writerow(tr.summary())
        return path

    def to_dataframe(self, include_rejected: bool = False) -> Any:
        """Samples as a :class:`pandas.DataFrame` (requires ``pandas``)."""
        import pandas as pd

        return pd.DataFrame(list(self._rows(include_rejected)), columns=SAMPLE_FIELDS)

    def to_parquet(self, path: str | Path, include_rejected: bool = False) -> Path:
        """Samples as Parquet (requires ``pyarrow``)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = list(self._rows(include_rejected))
        cols = list(zip(*rows)) if rows else [[] for _ in SAMPLE_FIELDS]
        table = pa.table({k: list(v) for k, v in zip(SAMPLE_FIELDS, cols)})
        pq.write_table(table, str(path))
        return Path(path)

    def to_json(self, path: str | Path, include_rejected: bool = False) -> Path:
        """Compact JSON: per-track summary plus sample arrays."""
        doc = {
            "format": "tachymetry.tracks/1",
            "fps": self.fps,
            "meta": self.meta,
            "summary": self.summary(),
            "camera": None if self.camera is None else self.camera.to_dict(),
            "tracks": [],
        }
        for tr in self.tracks if include_rejected else self.ok:
            doc["tracks"].append(
                {
                    **tr.summary(),
                    "t": np.round(tr.t, 4).tolist(),
                    "frame": tr.frame.tolist(),
                    "xy": np.round(tr.xy, 3).tolist(),
                    "speed_kmh": np.round(tr.speed_kmh, 2).tolist(),
                    "heading_deg": np.round(np.degrees(tr.heading), 1).tolist(),
                    "flags": tr.flags.tolist(),
                    "bbox": np.round(tr.bbox, 1).tolist(),
                }
            )
        Path(path).write_text(json.dumps(doc, default=_json_default))
        return Path(path)

    def to_geojson(self, path: str | Path) -> Path:
        """Trajectories as GeoJSON LineStrings (WGS84 when the camera is georeferenced, else local metres)."""
        georef = (self.camera.meta.get("georef") or {}) if self.camera is not None else {}
        origin = georef.get("origin") if georef.get("type") == "latlon" else None
        feats = []
        for tr in self.ok:
            xy = tr.xy
            if origin is not None:
                from .calib.correspondences import local_to_latlon

                ll = local_to_latlon(xy, tuple(origin))
                coords = [[round(float(lon), 7), round(float(lat), 7)] for lat, lon in ll]
            else:
                coords = np.round(xy, 3).tolist()
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": tr.summary(),
                }
            )
        doc = {"type": "FeatureCollection", "features": feats}
        if origin is None:
            doc["crs_note"] = "local metric frame of the calibrated camera (x east/right, y north/up)"
        Path(path).write_text(json.dumps(doc, default=_json_default))
        return Path(path)

    def to_mot(self, path: str | Path) -> Path:
        """MOTChallenge text format (image boxes of ``ok`` tracks, 1-based frames)."""
        with Path(path).open("w") as fh:
            for tr in self.ok:
                for k in range(len(tr)):
                    if not tr.flags[k] & Flags.OBSERVED:
                        continue
                    x1, y1, x2, y2 = tr.bbox[k]
                    fh.write(
                        f"{int(tr.frame[k]) + 1},{tr.id},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},"
                        f"{tr.score[k]:.3f},-1,-1,-1\n"
                    )
        return Path(path)

    def export(self, out_dir: str | Path, formats: Iterable[str] = ("csv", "json")) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for fmt in formats:
            if fmt == "csv":
                written["csv"] = self.to_csv(out_dir / "tracks.csv")
                written["summary_csv"] = self.summary_csv(out_dir / "tracks_summary.csv")
            elif fmt == "json":
                written["json"] = self.to_json(out_dir / "tracks.json")
            elif fmt == "geojson":
                written["geojson"] = self.to_geojson(out_dir / "tracks.geojson")
            elif fmt == "mot":
                written["mot"] = self.to_mot(out_dir / "tracks_mot.txt")
            elif fmt == "parquet":
                written["parquet"] = self.to_parquet(out_dir / "tracks.parquet")
            else:
                raise ValueError(f"unknown export format {fmt!r}")
        return written


@dataclass
class RunResult:
    """Everything a pipeline run produced for one video."""

    source: Path
    tracks: TrackSet
    camera: Camera
    detections: DetectionSet
    out_dir: Path | None
    report: dict[str, Any] = field(default_factory=dict)
    background: np.ndarray | None = None

    def summary(self) -> dict[str, Any]:
        return self.tracks.summary()

    def render(self, path: str | Path | None = None, layout: str | None = None, **kwargs: Any) -> Path:
        """Render an annotated video (see :func:`tachymetry.render.render_video`)."""
        from .render.video import render_video

        if path is None:
            if self.out_dir is None:
                raise ValueError("pass a path (this run has no output directory)")
            path = self.out_dir / "overlay.mp4"
        return render_video(self, path, layout=layout, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.summary()
        return f"RunResult({self.source.name}: {s['tracks']} tracks {s['by_class']})"

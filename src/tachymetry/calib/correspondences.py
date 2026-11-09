"""Image-to-world correspondences for manual calibration.

World coordinates can be given in three ways:

* **metres** in a local right-handed frame (``x`` east/right, ``y`` north/up);
* **pixels of a reference image** (orthophoto, satellite tile, site plan) with
  a known ground sampling distance (``gsd`` metres per pixel); image ``y``
  points down, which this module flips into a right-handed frame;
* **latitude/longitude** (WGS84), converted to a local east-north frame.

Files are YAML, JSON or CSV. YAML/JSON layout::

    image_size: [1920, 1080]          # optional, checked against the video
    world_units: metres               # metres | reference_px | latlon
    gsd: 0.1                          # only for reference_px
    points:
      - {name: kerb_ne, image: [1102.5, 640.0], world: [12.4, 8.1]}
      - ...
    lines:                            # optional polylines straight in the world
      - [[100, 900], [400, 860], [700, 835], [1000, 822]]

CSV: header ``u,v,x,y`` (or ``u,v,lat,lon``), one point per row.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Correspondences", "latlon_to_local", "local_to_latlon"]

_EARTH_R = 6378137.0


def latlon_to_local(
    latlon: np.ndarray, origin: tuple[float, float] | None = None
) -> tuple[np.ndarray, tuple[float, float]]:
    """WGS84 ``(lat, lon)`` degrees to local east-north metres (local tangent plane).

    Accurate to millimetres over the few hundred metres a camera observes.
    """
    ll = np.asarray(latlon, dtype=np.float64).reshape(-1, 2)
    if origin is None:
        origin = (float(ll[:, 0].mean()), float(ll[:, 1].mean()))
    lat0, lon0 = map(math.radians, origin)
    lat, lon = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    x = (lon - lon0) * math.cos(lat0) * _EARTH_R
    y = (lat - lat0) * _EARTH_R
    return np.column_stack([x, y]), origin


def local_to_latlon(xy: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    """Inverse of :func:`latlon_to_local`."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    lat0, lon0 = map(math.radians, origin)
    lat = lat0 + xy[:, 1] / _EARTH_R
    lon = lon0 + xy[:, 0] / (_EARTH_R * math.cos(lat0))
    return np.column_stack([np.degrees(lat), np.degrees(lon)])


@dataclass
class Correspondences:
    """Point (and optional line) correspondences between raw image pixels and the ground plane."""

    image: np.ndarray  # (N, 2) raw pixels
    world: np.ndarray  # (N, 2) metres, right-handed
    names: list[str] | None = None
    lines: list[np.ndarray] = field(default_factory=list)  # raw-pixel polylines straight in the world
    image_size: tuple[int, int] | None = None
    georef: dict[str, Any] = field(default_factory=dict)  # how to map local metres back to the source frame

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image, dtype=np.float64).reshape(-1, 2)
        self.world = np.asarray(self.world, dtype=np.float64).reshape(-1, 2)
        if len(self.image) != len(self.world):
            raise ValueError("image and world point counts differ")
        self.lines = [np.asarray(l, dtype=np.float64).reshape(-1, 2) for l in self.lines]

    def __len__(self) -> int:
        return len(self.image)

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_reference(cls, image_pts: Any, reference_px: Any, gsd: float, **kw: Any) -> Correspondences:
        """World points given as pixels of a north-up reference image with ``gsd`` m/px."""
        ref = np.asarray(reference_px, dtype=np.float64).reshape(-1, 2)
        world = np.column_stack([ref[:, 0] * gsd, -ref[:, 1] * gsd])
        georef = {"type": "reference_px", "gsd": float(gsd)}
        return cls(np.asarray(image_pts), world, georef=georef, **kw)

    @classmethod
    def from_latlon(
        cls, image_pts: Any, latlon: Any, origin: tuple[float, float] | None = None, **kw: Any
    ) -> Correspondences:
        world, origin = latlon_to_local(np.asarray(latlon), origin)
        return cls(np.asarray(image_pts), world, georef={"type": "latlon", "origin": list(origin)}, **kw)

    @classmethod
    def load(cls, path: str | Path) -> Correspondences:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return cls._from_csv(path)
        text = path.read_text(encoding="utf-8")
        if suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Correspondences:
        units = data.get("world_units", "metres")
        pts = data.get("points", [])
        img = np.array([p["image"] for p in pts], dtype=np.float64).reshape(-1, 2)
        wld = np.array([p["world"] for p in pts], dtype=np.float64).reshape(-1, 2)
        names = [str(p.get("name", i)) for i, p in enumerate(pts)]
        lines = [np.asarray(l, dtype=np.float64) for l in data.get("lines", [])]
        size = tuple(data["image_size"]) if data.get("image_size") else None
        common = dict(names=names, lines=lines, image_size=size)
        if units in ("metres", "meters", "m"):
            return cls(img, wld, **common)
        if units in ("reference_px", "pixels", "px"):
            return cls.from_reference(img, wld, float(data["gsd"]), **common)
        if units in ("latlon", "wgs84"):
            origin = tuple(data["origin"]) if data.get("origin") else None
            return cls.from_latlon(img, wld, origin=origin, **common)
        raise ValueError(f"unknown world_units {units!r}")

    @classmethod
    def _from_csv(cls, path: Path) -> Correspondences:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{path} has no rows")
        keys = {k.strip().lower() for k in rows[0]}
        img = np.array([[float(r["u"]), float(r["v"])] for r in rows])
        names = [r.get("name", str(i)) for i, r in enumerate(rows)]
        if {"lat", "lon"} <= keys:
            ll = np.array([[float(r["lat"]), float(r["lon"])] for r in rows])
            return cls.from_latlon(img, ll, names=names)
        wld = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        return cls(img, wld, names=names)

    # ------------------------------------------------------------------- output
    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size) if self.image_size else None,
            "world_units": "metres",
            "points": [
                {"name": n, "image": i.tolist(), "world": w.tolist()}
                for n, i, w in zip(self.names or [str(k) for k in range(len(self))], self.image, self.world)
            ],
            "lines": [l.tolist() for l in self.lines],
            "georef": self.georef,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def subset(self, idx: Sequence[int] | np.ndarray) -> Correspondences:
        idx = np.asarray(idx)
        names = None if self.names is None else [self.names[i] for i in np.arange(len(self))[idx]]
        return Correspondences(self.image[idx], self.world[idx], names, self.lines, self.image_size, self.georef)

    def handedness(self) -> float:
        """Sign of the image-to-world Jacobian determinant (negative for a right-handed world)."""
        from ..geometry.homography import dlt_homography

        H = dlt_homography(self.image, self.world)
        c = self.image.mean(axis=0)
        eps = 1.0
        from ..geometry.homography import apply_homography

        p = apply_homography(H, np.array([c, c + [eps, 0], c + [0, eps]]))
        a, b = p[1] - p[0], p[2] - p[0]
        return float(np.sign(a[0] * b[1] - a[1] * b[0]))

"""Per-detection ground measurements.

For each detection this computes, in world metres:

* the **ground contact**: the ground point under the bottom-centre of the box,
  i.e. the vehicle edge nearest the camera;
* the **footprint** from predicted cuboid corners when trusted (UrbanOmniDetect-2):
  its four ground corners, centroid, heading (front face = corners 0 and 3),
  length and width;
* the **ground sampling distance** at the contact (metres per pixel), which
  measures how trustworthy the localisation is, and quality flags:
  ``valid`` (below the horizon, finite, GSD within bounds), ``truncated``
  (touching the image border) and ``big`` (degenerate, frame-filling boxes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import HygieneConfig
from ..detections import DetectionSet
from ..geometry.camera import Camera

__all__ = ["GroundObservations", "measure_ground", "DIM_PRIORS", "CLEARANCE_PRIORS", "dims_for"]

#: (length, width, height) priors in metres per class
DIM_PRIORS = {
    "car": (4.6, 1.85, 1.5),
    "bus": (12.0, 2.55, 3.2),
    "truck": (8.0, 2.5, 3.2),
    "motorcycle": (2.0, 0.8, 1.4),
    "bicycle": (1.75, 0.6, 1.6),
    "person": (0.5, 0.5, 1.7),
}
#: plausible footprint bounds (L_min, L_max, W_min, W_max)
DIM_BOUNDS = {
    "car": (3.0, 6.8, 1.45, 2.4),
    "bus": (7.0, 20.0, 2.2, 2.9),
    "truck": (4.5, 20.0, 1.8, 2.9),
    "motorcycle": (1.3, 2.8, 0.45, 1.1),
    "bicycle": (1.2, 2.4, 0.35, 0.9),
    "person": (0.25, 1.0, 0.25, 1.0),
}
#: height of the lowest visible edge (bumper or wheel bottom) used for parallax
CLEARANCE_PRIORS = {"car": 0.08, "bus": 0.25, "truck": 0.25, "motorcycle": 0.05, "bicycle": 0.05, "person": 0.0}


def dims_for(cls_name: str) -> tuple[float, float, float]:
    return DIM_PRIORS.get(cls_name, (4.6, 1.85, 1.5))


@dataclass
class GroundObservations:
    """Column-aligned with a :class:`~tachymetry.detections.DetectionSet`."""

    contact: np.ndarray  # (M, 2)
    gsd: np.ndarray  # (M,)
    valid: np.ndarray  # (M,) bool
    truncated: np.ndarray  # (M,) bool
    big: np.ndarray  # (M,) bool
    quad: np.ndarray  # (M, 4, 2) NaN without a trusted footprint
    center: np.ndarray  # (M, 2) footprint centroid, NaN without a trusted footprint
    heading: np.ndarray  # (M,) radians, NaN without a trusted footprint
    length: np.ndarray  # (M,)
    width: np.ndarray  # (M,)

    @property
    def has_footprint(self) -> np.ndarray:
        return np.isfinite(self.center[:, 0])


def measure_ground(dets: DetectionSet, camera: Camera, cfg: HygieneConfig) -> GroundObservations:
    """Ground measurements for every detection in ``dets``."""
    m = len(dets)
    w, h = dets.image_size if dets.image_size[0] else camera.image_size
    if (w, h) != camera.image_size:
        camera = camera.resized((w, h))
    xyxy = dets.xyxy.astype(np.float64)
    bottom = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]])
    contact = camera.image_to_ground(bottom) if m else np.zeros((0, 2))
    gsd = camera.ground_sampling_distance(bottom) if m else np.zeros(0)
    below = camera.below_horizon(bottom, margin_px=2.0) if m else np.zeros(0, bool)
    valid = below & np.all(np.isfinite(contact), axis=1) & np.isfinite(gsd) & (gsd <= cfg.max_gsd_m)
    if cfg.max_range_m is not None and m:
        valid &= camera.distance_to(contact) <= cfg.max_range_m
    margin = max(3.0, cfg.edge_margin_frac * float(np.hypot(w, h)))
    truncated = (
        (xyxy[:, 0] <= margin)
        | (xyxy[:, 1] <= margin)
        | (xyxy[:, 2] >= w - 1 - margin)
        | (xyxy[:, 3] >= h - 1 - margin)
    )
    big = ((xyxy[:, 2] - xyxy[:, 0]) > cfg.big_box_frac * w) | ((xyxy[:, 3] - xyxy[:, 1]) > cfg.big_box_frac * h)

    quad = np.full((m, 4, 2), np.nan)
    center = np.full((m, 2), np.nan)
    heading = np.full(m, np.nan)
    length = np.full(m, np.nan)
    width = np.full(m, np.nan)
    if dets.cuboid is not None and m:
        cub = dets.cuboid
        ok = np.all(np.isfinite(cub[:, :4, :2]), axis=(1, 2))
        idx = np.flatnonzero(ok)
        if len(idx):
            pts = cub[idx, :4, :2].reshape(-1, 2).astype(np.float64)
            g = camera.image_to_ground(pts).reshape(-1, 4, 2)
            gok = np.all(np.isfinite(g), axis=(1, 2))
            idx, g = idx[gok], g[gok]
            front = 0.5 * (g[:, 0] + g[:, 3])
            rear = 0.5 * (g[:, 1] + g[:, 2])
            L = 0.5 * (np.linalg.norm(g[:, 0] - g[:, 1], axis=1) + np.linalg.norm(g[:, 3] - g[:, 2], axis=1))
            W = 0.5 * (np.linalg.norm(g[:, 0] - g[:, 3], axis=1) + np.linalg.norm(g[:, 1] - g[:, 2], axis=1))
            ax = front - rear
            e_l = g[:, 1] - g[:, 0]
            e_w = g[:, 3] - g[:, 0]
            cosang = np.abs(np.sum(e_l * e_w, axis=1)) / np.maximum(
                np.linalg.norm(e_l, axis=1) * np.linalg.norm(e_w, axis=1), 1e-9
            )
            names = np.array(dets.class_names)[dets.cls[idx]]
            plaus = np.ones(len(idx), bool)
            for k, c in enumerate(names):
                lo_l, hi_l, lo_w, hi_w = DIM_BOUNDS.get(str(c), (0.3, 25, 0.3, 4))
                plaus[k] = (0.6 * lo_l <= L[k] <= 1.4 * hi_l) and (0.6 * lo_w <= W[k] <= 1.4 * hi_w)
            good = plaus & (cosang < 0.35) & valid[idx]
            idx, g = idx[good], g[good]
            quad[idx] = g
            center[idx] = g.mean(axis=1)
            heading[idx] = np.arctan2(ax[good, 1], ax[good, 0])
            length[idx] = L[good]
            width[idx] = W[good]
    return GroundObservations(contact, gsd, valid, truncated, big, quad, center, heading, length, width)

"""Camera-view overlay: metric 3D boxes, ground trails and speed chips."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..geometry.camera import Camera
from .state import TrackState
from .style import CLASS_COLORS, HALO, blend_poly, color_for, polyline_aa
from .text import draw_chip

__all__ = ["draw_camera_overlay", "box_corners"]


def box_corners(xy: np.ndarray, heading: float, L: float, W: float, H: float) -> np.ndarray:
    """``(8, 3)`` corners of an upright box: 0-3 on the ground (front-left first), 4-7 on top."""
    c, s = math.cos(heading), math.sin(heading)
    f = np.array([c, s]) * L / 2
    lft = np.array([-s, c]) * W / 2
    base = np.array([xy + f + lft, xy - f + lft, xy - f - lft, xy + f - lft])
    return np.vstack([np.column_stack([base, np.zeros(4)]), np.column_stack([base, np.full(4, H)])])


def _edges(cam: Camera, corners: np.ndarray, pairs: Sequence[tuple[int, int]], n: int = 6) -> list[np.ndarray]:
    out = []
    for a, b in pairs:
        t = np.linspace(0, 1, n)[:, None]
        uv = cam.project(corners[a] + t * (corners[b] - corners[a]))
        if np.all(np.isfinite(uv)):
            out.append(uv)
    return out


def _fading_trail(img: np.ndarray, cam: Camera, trail: np.ndarray, col: tuple[int, int, int], lw: int) -> None:
    """Ground trail that fades out towards its oldest end."""
    import cv2

    p = cam.ground_to_image(trail)
    p = p[np.all(np.isfinite(p), axis=1)]
    n = len(p)
    if n < 2:
        return
    x0, y0 = np.maximum(np.floor(p.min(axis=0)).astype(int) - lw - 2, 0)
    x1, y1 = np.minimum(np.ceil(p.max(axis=0)).astype(int) + lw + 3, [img.shape[1], img.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    acc = roi.astype(np.float32)
    for k in range(1, n):
        a = 0.85 * k / (n - 1)
        layer = roi.copy()
        q = np.round((p[k - 1 : k + 1] - [x0, y0]) * 16).astype(np.int32)
        cv2.polylines(layer, [q], False, col, lw, cv2.LINE_AA, shift=4)
        mask = np.any(layer != roi, axis=2)
        acc[mask] = acc[mask] * (1 - a) + layer[mask].astype(np.float32) * a
    roi[...] = np.clip(acc, 0, 255).astype(np.uint8)


def _speed_text(kmh: float, units: str) -> str:
    if units == "mph":
        return f"{kmh / 1.609344:.0f} mph"
    if units == "m/s":
        return f"{kmh / 3.6:.1f} m/s"
    return f"{kmh:.0f} km/h"


def draw_camera_overlay(
    img: np.ndarray,
    cam: Camera,
    states: Sequence[TrackState],
    draw: str = "cuboid",
    units: str = "km/h",
    show_parked: bool = True,
    show_ids: bool = False,
    trail: bool = True,
    max_labels: int = 14,
    label_min_h: int = 22,
) -> np.ndarray:
    """Draw tracked objects on a camera frame (in place) and return it."""
    import cv2

    h, w = img.shape[:2]
    scale = max(1.0, h / 1080.0)
    lw = max(1, int(round(1.5 * scale)))
    # draw far objects first so near ones overlap them
    order = sorted(states, key=lambda s: -float(np.linalg.norm(s.xy - cam.center[:2])))
    chips = []
    for s in order:
        tr = s.track
        parked = tr.state == "parked"
        if parked and not show_parked:
            continue
        col = color_for(tr.cls, parked)
        moving = s.speed_kmh > 3.0 and not parked
        if tr.super_cls == "person":
            g = cam.ground_to_image(s.xy[None])[0]
            if not np.all(np.isfinite(g)):
                continue
            if trail and len(s.trail) > 2 and moving:
                _fading_trail(img, cam, s.trail[-max(3, len(s.trail) * 2 // 5) :], col, lw)
            r = max(3, int(round(0.012 * (s.bbox[3] - s.bbox[1]) + 3 * scale)))
            cv2.circle(img, tuple(np.round(g).astype(int)), r + 1, HALO, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.round(g).astype(int)), r, col, -1, cv2.LINE_AA)
            continue
        if trail and moving and len(s.trail) > 2:
            _fading_trail(img, cam, s.trail[-max(3, len(s.trail) * 2 // 5) :], col, lw + 1)
        if draw == "box":
            x1, y1, x2, y2 = s.bbox
            polyline_aa(img, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), col, lw, closed=True)
        else:
            cr = box_corners(s.xy, s.heading, tr.length, tr.width, tr.height if draw == "cuboid" else 0.0)
            base = _edges(cam, cr, [(0, 1), (1, 2), (2, 3), (3, 0)])
            if len(base) == 4:
                ring = np.vstack(base)
                blend_poly(img, ring, col, 0.10 if parked else 0.22)
            if draw == "cuboid" and not parked:
                for e in _edges(cam, cr, [(4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]):
                    polyline_aa(img, e, col, 1, halo=True)
            for e in base:
                polyline_aa(img, e, col, lw + (0 if parked else 1), halo=not parked)
        x1, y1, x2, y2 = s.bbox
        if moving and (y2 - y1) >= label_min_h * scale:
            chips.append(((y2 - y1), s))
    chips.sort(key=lambda c: -c[0])
    for _, s in chips[:max_labels]:
        tr = s.track
        text = _speed_text(s.speed_kmh, units)
        if show_ids:
            text = f"{tr.id}  {text}"
        cr = box_corners(s.xy, s.heading, tr.length, tr.width, tr.height)
        uv = cam.project(cr)
        top = np.nanmin(uv[:, 1]) if np.all(np.isfinite(uv)) else s.bbox[1]
        left = np.nanmin(uv[:, 0]) if np.all(np.isfinite(uv)) else s.bbox[0]
        size = int(round(13 * scale))
        draw_chip(
            img,
            text,
            (int(max(left, 0)), int(max(top - 6 * scale, size + 8))),
            size=size,
            fg=(20, 16, 14),
            bg=CLASS_COLORS.get(tr.cls, (230, 230, 230)),
            alpha=0.92,
        )
    return img

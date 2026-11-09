"""Visual verification of a calibration.

* :func:`draw_ground_grid` projects a metric grid onto the raw image (edges are
  densely sampled, so the grid bends correctly under lens distortion).
* :func:`warp_to_bev` orthorectifies the image onto the ground plane.
* :func:`preview` composes both with a parameter panel; :func:`compare` lays
  several candidate calibrations side by side.

A calibration is only trustworthy once its grid lies flat on the road and the
bird's-eye view shows parallel lane lines and rectangular crosswalks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..geometry.camera import Camera

__all__ = ["draw_ground_grid", "warp_to_bev", "bev_extent", "preview", "compare"]

AMBER = (71, 181, 255)  # BGR of #ffb547
CYAN = (255, 227, 111)  # BGR of #6fe3ff
INK = (13, 11, 10)


def bev_extent(
    camera: Camera,
    max_range: float | None = None,
    margin: float = 2.0,
    max_gsd: float = 0.30,
) -> tuple[float, float, float, float]:
    """Ground rectangle ``(x_min, x_max, y_min, y_max)`` covering the usable visible ground.

    Usable means imaged at a ground sampling distance of at most ``max_gsd``
    metres per pixel (and within ``max_range`` of the camera when given), so the
    extent adapts to low pole cameras and high zoom cameras alike.
    """
    w, h = camera.image_size
    uu, vv = np.meshgrid(np.linspace(0, w - 1, 96), np.linspace(0, h - 1, 54))
    uv = np.stack([uu.ravel(), vv.ravel()], 1)
    g = camera.image_to_ground(uv)
    gsd = camera.ground_sampling_distance(uv)
    ok = np.all(np.isfinite(g), axis=1) & np.isfinite(gsd) & (gsd <= max_gsd)
    if max_range is not None:
        ok &= camera.distance_to(g) <= max_range
    g = g[ok]
    if len(g) < 4:
        C = camera.center
        return C[0] - 10, C[0] + 10, C[1] - 10, C[1] + 10
    lo = np.percentile(g, 0.5, axis=0) - margin
    hi = np.percentile(g, 99.5, axis=0) + margin
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def warp_to_bev(
    image: np.ndarray,
    camera: Camera,
    extent: tuple[float, float, float, float] | None = None,
    resolution: float = 0.05,
    max_side: int = 1600,
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    """Orthorectify ``image`` onto the ground. Returns ``(bev, extent, resolution)``.

    BEV rows run from ``y_max`` (top) to ``y_min``; columns from ``x_min`` to ``x_max``.
    """
    import cv2

    ext = extent or bev_extent(camera)
    x0, x1, y0, y1 = ext
    res = max(resolution, max(x1 - x0, y1 - y0) / max_side)
    W = int(math.ceil((x1 - x0) / res))
    H = int(math.ceil((y1 - y0) / res))
    xs = x0 + (np.arange(W) + 0.5) * res
    ys = y1 - (np.arange(H) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    uv = camera.ground_to_image(np.stack([gx.ravel(), gy.ravel()], 1))
    mx = uv[:, 0].reshape(H, W).astype(np.float32)
    my = uv[:, 1].reshape(H, W).astype(np.float32)
    bad = ~np.isfinite(mx) | ~np.isfinite(my)
    mx[bad] = -1
    my[bad] = -1
    bev = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(10, 11, 13))
    return bev, ext, res


def _poly(camera: Camera, a: np.ndarray, b: np.ndarray, n: int = 60) -> np.ndarray | None:
    t = np.linspace(0, 1, n)[:, None]
    pts = camera.ground_to_image(a + t * (b - a))
    ok = np.all(np.isfinite(pts), axis=1)
    if ok.sum() < 2:
        return None
    return pts[ok]


def draw_ground_grid(
    image: np.ndarray,
    camera: Camera,
    spacing: float = 1.0,
    major: float = 5.0,
    max_range: float | None = None,
    color: tuple[int, int, int] = AMBER,
    alpha: float = 0.85,
    horizon: bool = True,
    max_gsd: float = 0.30,
) -> np.ndarray:
    """Overlay a metric ground grid (minor every ``spacing`` m, major every ``major`` m)."""
    import cv2

    out = image.copy()
    overlay = image.copy()
    h, w = image.shape[:2]
    C = camera.center[:2]
    ext = bev_extent(camera, max_range=max_range, margin=0, max_gsd=max_gsd)
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    if span > 160:  # keep the grid legible on wide views
        spacing, major = max(spacing, 5.0), max(major, 25.0)
    elif span > 70:
        spacing, major = max(spacing, 2.0), max(major, 10.0)
    x0 = math.floor(ext[0] / major) * major
    x1 = math.ceil(ext[1] / major) * major
    y0 = math.floor(ext[2] / major) * major
    y1 = math.ceil(ext[3] / major) * major
    scale = max(1, int(round(max(h, w) / 1280)))
    for val in np.arange(x0, x1 + 1e-9, spacing):
        is_major = abs(val / major - round(val / major)) < 1e-6
        p = _poly(camera, np.array([val, y0]), np.array([val, y1]))
        if p is None:
            continue
        p = p[camera.ground_sampling_distance(p) <= max_gsd]
        if len(p) >= 2:
            cv2.polylines(
                overlay,
                [np.round(p * 4).astype(np.int32)],
                False,
                color,
                (2 if is_major else 1) * scale,
                cv2.LINE_AA,
                shift=2,
            )
    for val in np.arange(y0, y1 + 1e-9, spacing):
        is_major = abs(val / major - round(val / major)) < 1e-6
        p = _poly(camera, np.array([x0, val]), np.array([x1, val]))
        if p is None:
            continue
        p = p[camera.ground_sampling_distance(p) <= max_gsd]
        if len(p) >= 2:
            cv2.polylines(
                overlay,
                [np.round(p * 4).astype(np.int32)],
                False,
                color,
                (2 if is_major else 1) * scale,
                cv2.LINE_AA,
                shift=2,
            )
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if horizon:
        l = camera.horizon()
        if abs(l[1]) > 1e-9:
            xs = np.array([0.0, w - 1.0])
            ys = -(l[0] * xs + l[2]) / l[1]
            raw = camera.lens.distort(np.column_stack([xs, ys]))
            if np.all(np.isfinite(raw)):
                cv2.line(
                    out,
                    tuple(np.round(raw[0]).astype(int)),
                    tuple(np.round(raw[1]).astype(int)),
                    CYAN,
                    scale,
                    cv2.LINE_AA,
                )
    _ = C
    return out


def _panel(text_lines: Sequence[str], width: int, height: int) -> np.ndarray:
    from ..render.text import draw_text

    panel = np.full((height, width, 3), INK, np.uint8)
    y = 18
    for i, line in enumerate(text_lines):
        size = 22 if i == 0 else 16
        panel = draw_text(
            panel,
            line,
            (18, y),
            size=size,
            color=(236, 235, 232) if i == 0 else (163, 156, 155),
            weight="semibold" if i == 0 else "regular",
        )
        y += size + 12
    return panel


def preview(
    image: np.ndarray,
    camera: Camera,
    title: str | None = None,
    spacing: float = 1.0,
    major: float = 5.0,
    max_range: float | None = None,
    bev_height: int | None = None,
    max_gsd: float = 0.30,
) -> np.ndarray:
    """Side-by-side preview: camera view with grid | bird's-eye view with grid, plus a caption."""
    import cv2

    h, w = image.shape[:2]
    left = draw_ground_grid(image, camera, spacing, major, max_range, max_gsd=max_gsd)
    ext = bev_extent(camera, max_range=max_range, max_gsd=max_gsd)
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    if span > 160:
        spacing, major = max(spacing, 5.0), max(major, 25.0)
    elif span > 70:
        spacing, major = max(spacing, 2.0), max(major, 10.0)
    bev, ext, res = warp_to_bev(image, camera, ext, resolution=0.04, max_side=1400)
    # BEV grid
    gx0 = math.floor(ext[0] / major) * major
    gy0 = math.floor(ext[2] / major) * major
    for val in np.arange(gx0, ext[1], spacing):
        col = int(round((val - ext[0]) / res))
        if 0 <= col < bev.shape[1]:
            major_line = abs(val / major - round(val / major)) < 1e-6
            cv2.line(bev, (col, 0), (col, bev.shape[0] - 1), AMBER, 2 if major_line else 1, cv2.LINE_AA)
    for val in np.arange(gy0, ext[3], spacing):
        row = int(round((ext[3] - val) / res))
        if 0 <= row < bev.shape[0]:
            major_line = abs(val / major - round(val / major)) < 1e-6
            cv2.line(bev, (0, row), (bev.shape[1] - 1, row), AMBER, 2 if major_line else 1, cv2.LINE_AA)
    # camera footprint marker
    C = camera.center
    cpx = (int(round((C[0] - ext[0]) / res)), int(round((ext[3] - C[1]) / res)))
    if 0 <= cpx[0] < bev.shape[1] and 0 <= cpx[1] < bev.shape[0]:
        cv2.circle(bev, cpx, 9, CYAN, -1, cv2.LINE_AA)
    target_h = bev_height or h
    s = target_h / bev.shape[0]
    bev = cv2.resize(bev, (max(1, int(bev.shape[1] * s)), target_h), interpolation=cv2.INTER_AREA)
    body = np.concatenate([left, np.full((h, 8, 3), INK, np.uint8), bev], axis=1)
    yaw, pitch, roll = camera.angles
    lines = [
        title or f"Calibration: {camera.meta.get('calibration_method', camera.lens.kind)}",
        f"focal {camera.focal:.0f} px (HFOV {camera.fov[0]:.1f} deg)    height {camera.height:.2f} m    "
        f"pitch {pitch:.2f} deg    roll {roll:.2f} deg    lens {camera.lens.kind}",
        f"grid: {spacing:g} m minor, {major:g} m major"
        + (f", within {max_range:g} m of the camera" if max_range else ""),
    ]
    return np.concatenate([_panel(lines, body.shape[1], 104), body], axis=0)


def compare(
    image: np.ndarray, cameras: Sequence[tuple[str, Camera]], max_range: float | None = None, cols: int = 2
) -> np.ndarray:
    """Grid of :func:`preview` panels, one per ``(name, camera)``."""
    import cv2

    tiles = [preview(image, cam, title=name, max_range=max_range) for name, cam in cameras]
    tw = max(t.shape[1] for t in tiles)
    th = max(t.shape[0] for t in tiles)
    tiles = [
        cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, tw - t.shape[1], cv2.BORDER_CONSTANT, value=INK) for t in tiles
    ]
    rows = [
        np.concatenate(
            tiles[i : i + cols] + [np.full((th, tw, 3), INK, np.uint8)] * (cols - len(tiles[i : i + cols])), axis=1
        )
        for i in range(0, len(tiles), cols)
    ]
    return np.concatenate(rows, axis=0)

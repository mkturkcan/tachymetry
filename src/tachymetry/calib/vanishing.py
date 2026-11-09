"""Camera rotation and focal length from orthogonal vanishing points.

For two vanishing points ``v1, v2`` of orthogonal ground directions and a
principal point ``c`` (square pixels), the focal length is
``f^2 = -(v1 - c) . (v2 - c)`` and the rotation columns are the normalised
back-projected directions. A vertical vanishing point ``v3`` can replace either
horizontal one. The resulting camera is defined up to the metric scale of the
world (the camera height), which :mod:`tachymetry.calib.scale` fixes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..geometry.camera import CalibrationError, Camera
from ..geometry.lens import IdentityLens, Lens
from .lines import Segments, VanishingPoint, estimate_vanishing_points

__all__ = ["focal_from_vps", "camera_from_vps", "select_orthogonal_pair", "classify_vertical"]


def focal_from_vps(v1: np.ndarray, v2: np.ndarray, pp: np.ndarray) -> float | None:
    """Focal length implied by two orthogonal (finite) vanishing points, or ``None``."""
    if abs(v1[2]) < 1e-12 or abs(v2[2]) < 1e-12:
        return None
    a = v1[:2] / v1[2] - pp
    b = v2[:2] / v2[2] - pp
    f2 = -float(a @ b)
    return math.sqrt(f2) if f2 > 0 else None


def _direction(v: np.ndarray, Kinv: np.ndarray) -> np.ndarray:
    d = Kinv @ v
    return d / np.linalg.norm(d)


def classify_vertical(
    vps: Sequence[VanishingPoint], image_size: tuple[int, int], max_angle_deg: float = 30.0
) -> tuple[list[VanishingPoint], list[VanishingPoint]]:
    """Split vanishing points into near-vertical (poles, facades) and other candidates."""
    w, h = image_size
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    vertical, other = [], []
    for vp in vps:
        if vp.finite:
            d = vp.xy - c
            ang = math.degrees(math.atan2(abs(d[0]), abs(d[1]) + 1e-9))  # 0 = straight up/down
            far_below = d[1] > 1.5 * h  # vertical VPs of a down-looking camera lie far below the image
            (vertical if (ang < max_angle_deg and far_below) else other).append(vp)
        else:
            d = vp.point[:2]
            ang = math.degrees(math.atan2(abs(d[0]), abs(d[1]) + 1e-9))
            (vertical if ang < max_angle_deg else other).append(vp)
    return vertical, other


def select_orthogonal_pair(
    vps: Sequence[VanishingPoint],
    image_size: tuple[int, int],
    vertical: VanishingPoint | None = None,
    hfov_range: tuple[float, float] = (15.0, 130.0),
) -> tuple[VanishingPoint, VanishingPoint, float] | None:
    """Best pair of horizontal vanishing points that are mutually orthogonal for a plausible focal length."""
    w, h = image_size
    pp = np.array([(w - 1) / 2, (h - 1) / 2])
    f_lo = (w / 2) / math.tan(math.radians(hfov_range[1] / 2))
    f_hi = (w / 2) / math.tan(math.radians(hfov_range[0] / 2))
    best = None
    for i in range(len(vps)):
        for j in range(i + 1, len(vps)):
            f = focal_from_vps(vps[i].point, vps[j].point, pp)
            if f is None or not (f_lo <= f <= f_hi):
                continue
            score = math.sqrt(vps[i].support * vps[j].support)
            if vertical is not None:
                K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
                Ki = np.linalg.inv(K)
                n = np.cross(_direction(vps[i].point, Ki), _direction(vps[j].point, Ki))
                n /= np.linalg.norm(n)
                cosang = abs(float(n @ _direction(vertical.point, Ki)))
                score *= cosang**8  # heavily favour pairs consistent with the vertical direction
            if best is None or score > best[3]:
                best = (vps[i], vps[j], f, score)
    return None if best is None else best[:3]


def camera_from_vps(
    v_x: np.ndarray,
    v_y: np.ndarray | None,
    image_size: tuple[int, int],
    v_z: np.ndarray | None = None,
    focal: float | None = None,
    lens: Lens | None = None,
    height: float = 1.0,
) -> Camera:
    """Build a camera whose world ``+x`` points at ``v_x`` and whose ground is orthogonal to ``v_z``.

    Supply two of ``v_x``, ``v_y`` (horizontal, orthogonal) and ``v_z`` (vertical).
    ``focal`` overrides the focal length implied by the vanishing points. The
    camera centre is placed at ``(0, 0, height)``.
    """
    w, h = image_size
    pp = np.array([(w - 1) / 2, (h - 1) / 2])
    if focal is None:
        pairs = [(v_x, v_y), (v_x, v_z), (v_y, v_z)]
        for a, b in pairs:
            if a is not None and b is not None:
                focal = focal_from_vps(a, b, pp)
                if focal is not None:
                    break
    if focal is None:
        raise CalibrationError("vanishing points do not determine a real focal length; supply focal=")
    K = np.array([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    dx = _direction(v_x, Ki)
    if v_z is not None:
        dz = _direction(v_z, Ki)
    elif v_y is not None:
        dz = np.cross(dx, _direction(v_y, Ki))
    else:
        raise CalibrationError("need a second vanishing point")
    # world up must point towards the top of the image (camera -y); the ground must be in front
    if dz[1] > 0:
        dz = -dz
    dx = dx - (dx @ dz) * dz
    dx /= np.linalg.norm(dx)
    if dx[2] < 0:  # world +x points away from the camera (into the scene)
        dx = -dx
    dy = np.cross(dz, dx)
    Rwc = np.column_stack([dx, dy, dz])  # columns: world axes in camera coordinates
    U, _, Vt = np.linalg.svd(Rwc)
    R = U @ Vt
    C = np.array([0.0, 0.0, height])
    cam = Camera(image_size, K, R, -R @ C, lens=lens or IdentityLens(image_size))
    if not (0.0 < cam.pitch < 90.0):
        raise CalibrationError(f"implausible pitch {cam.pitch:.1f} deg from vanishing points")
    return cam


def vanishing_points_from_segments(
    segs: Segments,
    image_size: tuple[int, int],
    all_segs: Segments | None = None,
    threshold_deg: float = 1.5,
    seed: int = 0,
):
    """Horizontal candidates from marking segments plus a vertical VP from all-image segments."""
    horiz = estimate_vanishing_points(segs, n_points=4, threshold_deg=threshold_deg, seed=seed)
    vert_cands: list[VanishingPoint] = []
    if all_segs is not None and len(all_segs):
        d = all_segs.directions
        near_vertical = np.abs(d[:, 1]) > math.cos(math.radians(25))
        sv = all_segs.subset(near_vertical)
        if len(sv) >= 4:
            vert_cands = estimate_vanishing_points(
                sv, n_points=1, threshold_deg=threshold_deg, seed=seed, min_support_frac=0.2
            )
    vertical, others = classify_vertical(horiz, image_size)
    vertical = vert_cands or vertical
    return others, (vertical[0] if vertical else None)

"""Detector-agnostic detection post-processing."""

from __future__ import annotations

import numpy as np

__all__ = ["cuboid_trust", "suppress_nested", "box_area", "box_ioa", "nms_by_group"]


def box_area(xyxy: np.ndarray) -> np.ndarray:
    return np.maximum(xyxy[..., 2] - xyxy[..., 0], 1.0) * np.maximum(xyxy[..., 3] - xyxy[..., 1], 1.0)


def box_ioa(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection over the area of ``a``: ``(len(a), len(b))``."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / box_area(a)[:, None]


def cuboid_trust(xyxy: np.ndarray, cuboid: np.ndarray, min_conf: float = 0.5, expand: float = 0.75) -> np.ndarray:
    """Boolean trust flag per detection for predicted cuboid corners ``(N, 8, 3)``.

    A cuboid is trusted when all corners are finite, the weakest corner
    confidence reaches ``min_conf``, all corners lie inside the box expanded by
    ``expand`` box sizes, and no corner collapsed to the image origin.
    """
    if cuboid is None or len(cuboid) == 0:
        return np.zeros(len(xyxy), dtype=bool)
    xy = cuboid[..., :2]
    finite = np.all(np.isfinite(cuboid), axis=(1, 2))
    conf_ok = np.nanmin(cuboid[..., 2], axis=1) >= min_conf
    w = np.maximum(xyxy[:, 2] - xyxy[:, 0], 1.0)[:, None]
    h = np.maximum(xyxy[:, 3] - xyxy[:, 1], 1.0)[:, None]
    inside = (
        (xy[..., 0] >= xyxy[:, None, 0] - expand * w)
        & (xy[..., 0] <= xyxy[:, None, 2] + expand * w)
        & (xy[..., 1] >= xyxy[:, None, 1] - expand * h)
        & (xy[..., 1] <= xyxy[:, None, 3] + expand * h)
    )
    not_origin = np.abs(xy).sum(axis=-1) > 1e-6
    return finite & conf_ok & np.all(inside, axis=1) & np.all(not_origin, axis=1)


def _poly_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection of two convex quads over the smaller area."""
    import cv2

    ha = cv2.convexHull(a.astype(np.float32))
    hb = cv2.convexHull(b.astype(np.float32))
    aa, ab = cv2.contourArea(ha), cv2.contourArea(hb)
    small = min(aa, ab)
    if small <= 1e-6:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(ha, hb)
    return float(inter) / small


def suppress_nested(
    xyxy: np.ndarray,
    cls: np.ndarray,
    vehicle_ids: set[int],
    cuboid: np.ndarray | None,
    trusted: np.ndarray | None,
    containment: float = 0.85,
    ground_overlap: float = 0.30,
) -> np.ndarray:
    """Keep-mask that removes vehicle boxes nested in a larger vehicle box (e.g. a truck cab).

    NMS-free heads can report a whole vehicle and one of its parts. A nested box is
    dropped when it lies ``containment`` inside a larger vehicle box *and* the two
    stand on the same ground: their trusted ground footprints overlap by
    ``ground_overlap`` (of the smaller), or, without cuboids, their bottom edges
    are within 15 % of the larger box height. A car in front of a bus is nested
    in the image but not on the ground, so it survives.
    """
    n = len(xyxy)
    keep = np.ones(n, dtype=bool)
    veh = np.array([int(c) in vehicle_ids for c in cls], dtype=bool)
    if veh.sum() < 2:
        return keep
    area = box_area(xyxy)
    ioa = box_ioa(xyxy, xyxy)
    order = np.argsort(area)  # smallest first
    for a in order:
        if not veh[a]:
            continue
        for b in np.argsort(-area):
            if b == a or not veh[b] or not keep[b] or area[b] <= area[a]:
                continue
            if ioa[a, b] < containment:
                continue
            same_ground = False
            if cuboid is not None and trusted is not None and trusted[a] and trusted[b]:
                same_ground = _poly_overlap(cuboid[a, :4, :2], cuboid[b, :4, :2]) >= ground_overlap
            else:
                hb = xyxy[b, 3] - xyxy[b, 1]
                same_ground = abs(xyxy[a, 3] - xyxy[b, 3]) <= 0.15 * hb
            if same_ground:
                keep[a] = False
                break
    return keep


def nms_by_group(xyxy: np.ndarray, score: np.ndarray, groups: np.ndarray, iou_thresh: float = 0.8) -> np.ndarray:
    """Greedy NMS within groups (e.g. vehicle super-class) for near-identical boxes. Returns a keep mask."""
    keep = np.ones(len(score), dtype=bool)
    order = np.argsort(-score)
    area = box_area(xyxy)
    for ii, i in enumerate(order):
        if not keep[i]:
            continue
        for j in order[ii + 1 :]:
            if not keep[j] or groups[j] != groups[i]:
                continue
            x1 = max(xyxy[i, 0], xyxy[j, 0])
            y1 = max(xyxy[i, 1], xyxy[j, 1])
            x2 = min(xyxy[i, 2], xyxy[j, 2])
            y2 = min(xyxy[i, 3], xyxy[j, 3])
            inter = max(x2 - x1, 0) * max(y2 - y1, 0)
            if inter / max(area[i] + area[j] - inter, 1e-9) > iou_thresh:
                keep[j] = False
    return keep

"""Detector-agnostic detection post-processing.

:func:`prefilter` and :func:`refine` are the per-frame steps that the
built-in back-ends apply, in this order:

1. class floors (``DetectorConfig.class_conf``) and the width limit;
2. near-duplicate suppression within each super-class (:func:`nms_by_group`);
3. with instance masks, duplicate suppression by mask overlap (:func:`suppress_mask_duplicates`);
4. with cuboids, cuboid trust (:func:`cuboid_trust`);
5. nested-vehicle suppression (:func:`suppress_nested`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..config import ROAD_CLASSES, SUPER_CLASS, VEHICLE_CLASSES

if TYPE_CHECKING:
    from ..config import DetectorConfig
    from ..detections import MaskCrop

__all__ = [
    "cuboid_trust",
    "suppress_nested",
    "suppress_mask_duplicates",
    "mask_overlap",
    "box_area",
    "box_ioa",
    "nms_by_group",
    "super_groups",
    "prefilter",
    "refine",
    "DUPLICATE_IOU",
    "MASK_DUPLICATE_IOU",
]

#: Box IoU above which two detections of one super-class are near-duplicates (:func:`nms_by_group`).
DUPLICATE_IOU = 0.8
#: Mask IoU above which two detections of one super-class are duplicates (:func:`suppress_mask_duplicates`).
MASK_DUPLICATE_IOU = 0.5


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


def super_groups(cls: np.ndarray, class_names: Sequence[str] = ROAD_CLASSES) -> np.ndarray:
    """Super-class index per detection (vehicle, two-wheeler, person; other labels keep their own group)."""
    supers = sorted(set(SUPER_CLASS.values()))
    lut = {
        i: (supers.index(SUPER_CLASS[c]) if c in SUPER_CLASS else len(supers) + i) for i, c in enumerate(class_names)
    }
    return np.array([lut.get(int(c), len(supers) + len(class_names) + int(c)) for c in cls], dtype=np.int64)


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


def mask_overlap(a: MaskCrop, b: MaskCrop) -> tuple[int, int, int]:
    """``(intersection, area_a, area_b)`` in pixels of two mask crops ``(x0, y0, mask)``."""
    ax, ay, ma = a
    bx, by, mb = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + ma.shape[1], bx + mb.shape[1]), min(ay + ma.shape[0], by + mb.shape[0])
    inter = 0
    if x1 > x0 and y1 > y0:
        inter = int(
            np.count_nonzero(ma[y0 - ay : y1 - ay, x0 - ax : x1 - ax] & mb[y0 - by : y1 - by, x0 - bx : x1 - bx])
        )
    return inter, int(np.count_nonzero(ma)), int(np.count_nonzero(mb))


def suppress_mask_duplicates(
    xyxy: np.ndarray,
    score: np.ndarray,
    masks: Sequence[MaskCrop | None],
    groups: np.ndarray | None = None,
    iou_thresh: float = MASK_DUPLICATE_IOU,
) -> np.ndarray:
    """Keep-mask that removes the lower-scored of two detections whose instance masks overlap.

    Two detections are duplicates when their mask IoU exceeds ``iou_thresh``:
    open-vocabulary detectors prompted with several class names report one
    vehicle as both ``car`` and ``truck``, and set-prediction detectors report a
    second, lower-scored mask of one vehicle with a looser box. With ``groups``,
    only detections of the same group are compared. Detections without a mask
    are kept.
    """
    n = len(score)
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    # masks lie inside their crop windows (box plus 2 px): only pairs whose padded boxes intersect are compared
    xyxy = np.asarray(xyxy, np.float64)
    pad = np.array([-3.0, -3.0, 3.0, 3.0])
    cand = box_ioa(xyxy + pad, xyxy + pad) > 0
    order = np.argsort(-np.asarray(score), kind="stable")
    for ii, i in enumerate(order):
        if not keep[i] or masks[i] is None:
            continue
        for j in order[ii + 1 :]:
            if not keep[j] or masks[j] is None or not cand[i, j]:
                continue
            if groups is not None and groups[i] != groups[j]:
                continue
            inter, a, b = mask_overlap(masks[i], masks[j])
            if inter / max(a + b - inter, 1) > iou_thresh:
                keep[j] = False
    return keep


def suppress_nested(
    xyxy: np.ndarray,
    cls: np.ndarray,
    vehicle_ids: set[int],
    cuboid: np.ndarray | None,
    trusted: np.ndarray | None,
    containment: float = 0.85,
    ground_overlap: float = 0.30,
    score: np.ndarray | None = None,
    masks: Sequence[MaskCrop | None] | None = None,
    mask_containment: float = 0.5,
) -> np.ndarray:
    """Keep-mask that removes vehicle boxes nested in a larger vehicle box (e.g. a truck cab).

    Detectors can report a whole vehicle and one of its parts. A nested box is
    dropped when it lies ``containment`` inside a larger vehicle box *and* the two
    stand on the same ground: their trusted ground footprints overlap by
    ``ground_overlap`` (of the smaller), or, without cuboids, their bottom edges
    are within 15 % of the larger box height. A car in front of a bus is nested
    in the image but not on the ground, so it survives.

    With ``score``, a larger box only suppresses a nested box that it outscores
    or ties: set-prediction detectors (RF-DETR, SAM 3) report low-scored boxes
    around groups of parked cars, which must not remove the cars. With instance
    ``masks``, the nested box must in addition lie on the larger detection's
    mask (``mask_containment`` of its own mask pixels); a vehicle that occludes
    the larger one is not part of its mask and survives.
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
            if score is not None and score[b] < score[a]:
                continue
            same_ground = False
            if cuboid is not None and trusted is not None and trusted[a] and trusted[b]:
                same_ground = _poly_overlap(cuboid[a, :4, :2], cuboid[b, :4, :2]) >= ground_overlap
            else:
                hb = xyxy[b, 3] - xyxy[b, 1]
                same_ground = abs(xyxy[a, 3] - xyxy[b, 3]) <= 0.15 * hb
            if same_ground and masks is not None and masks[a] is not None and masks[b] is not None:
                inter, area_a, _ = mask_overlap(masks[a], masks[b])
                same_ground = inter >= mask_containment * max(area_a, 1)
            if same_ground:
                keep[a] = False
                break
    return keep


def nms_by_group(
    xyxy: np.ndarray, score: np.ndarray, groups: np.ndarray, iou_thresh: float = DUPLICATE_IOU
) -> np.ndarray:
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


def prefilter(
    xyxy: np.ndarray,
    score: np.ndarray,
    cls: np.ndarray,
    image_width: int,
    cfg: DetectorConfig,
    class_names: Sequence[str] = ROAD_CLASSES,
) -> np.ndarray:
    """Indices that pass the class floors, the width limit and near-duplicate suppression."""
    keep = np.ones(len(score), dtype=bool)
    for name, floor in cfg.class_conf.items():
        if name in class_names:
            keep &= ~((cls == class_names.index(name)) & (score < float(floor)))
    keep &= (xyxy[:, 2] - xyxy[:, 0]) <= cfg.max_box_width_frac * image_width
    idx = np.flatnonzero(keep)
    nk = nms_by_group(xyxy[idx], score[idx], super_groups(cls[idx], class_names), DUPLICATE_IOU)
    return idx[nk]


def refine(
    xyxy: np.ndarray,
    score: np.ndarray,
    cls: np.ndarray,
    cfg: DetectorConfig,
    class_names: Sequence[str] = ROAD_CLASSES,
    cuboid: np.ndarray | None = None,
    masks: Sequence[MaskCrop | None] | None = None,
    score_aware: bool = False,
    mask_duplicate_iou: float | None = None,
    mask_duplicate_groups: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Indices that survive mask-duplicate, cuboid-trust and nested-vehicle steps, and the cuboids to keep.

    ``mask_duplicate_iou`` enables :func:`suppress_mask_duplicates` (within
    super-classes when ``mask_duplicate_groups``). Untrusted cuboids are
    replaced by NaN in the returned cuboid array (rows follow the returned
    indices). ``score_aware`` passes the scores to :func:`suppress_nested`.
    """
    idx = np.arange(len(score))
    if masks is not None and mask_duplicate_iou is not None and len(idx) > 1:
        groups = super_groups(cls, class_names) if mask_duplicate_groups else None
        idx = idx[suppress_mask_duplicates(xyxy, score, masks, groups, mask_duplicate_iou)]
    vehicle_ids = {class_names.index(c) for c in VEHICLE_CLASSES if c in class_names}
    sub_masks = None if masks is None else [masks[int(i)] for i in idx]
    sub_score = score[idx] if score_aware else None
    if cuboid is None:
        nk = suppress_nested(
            xyxy[idx],
            cls[idx],
            vehicle_ids,
            None,
            None,
            cfg.nested_containment,
            cfg.nested_ground_overlap,
            score=sub_score,
            masks=sub_masks,
        )
        return idx[nk], None
    cub = cuboid[idx]
    trusted = cuboid_trust(xyxy[idx], cub, cfg.cuboid_conf)
    nk = suppress_nested(
        xyxy[idx],
        cls[idx],
        vehicle_ids,
        cub,
        trusted,
        cfg.nested_containment,
        cfg.nested_ground_overlap,
        score=sub_score,
        masks=sub_masks,
    )
    cub = cub[nk].copy()
    cub[~trusted[nk]] = np.nan  # untrusted cuboids never reach geometry
    return idx[nk], cub

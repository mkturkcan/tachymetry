"""Visual identity: one palette for all renderers and figures.

Colour encodes class only; structure is drawn in neutral hairlines; moving
objects are saturated, parked objects are muted, and nothing competes with
the footage.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CLASS_COLORS",
    "PARKED",
    "INK",
    "TEXT",
    "MUTED",
    "FAINT",
    "AMBER",
    "CYAN",
    "HALO",
    "color_for",
    "blend_poly",
    "polyline_aa",
]

# BGR
CLASS_COLORS = {
    "car": (246, 150, 96),
    "bus": (140, 196, 86),
    "truck": (98, 110, 232),
    "motorcycle": (250, 139, 167),
    "bicycle": (250, 139, 167),
    "person": (74, 178, 240),
}
PARKED = (160, 152, 150)
INK = (13, 11, 10)
PANEL = (23, 19, 17)
TEXT = (232, 235, 236)
MUTED = (163, 156, 155)
FAINT = (118, 110, 108)
AMBER = (71, 181, 255)
CYAN = (255, 227, 111)
HALO = (14, 10, 8)
LABELS = {
    "car": "Car",
    "bus": "Bus",
    "truck": "Truck",
    "motorcycle": "Motorcycle",
    "bicycle": "Bicycle",
    "person": "Pedestrian",
}
PLURALS = {
    "car": "cars",
    "bus": "buses",
    "truck": "trucks",
    "motorcycle": "motorcycles",
    "bicycle": "bicycles",
    "person": "pedestrians",
}


def color_for(cls: str, parked: bool = False) -> tuple[int, int, int]:
    return PARKED if parked else CLASS_COLORS.get(cls, (160, 160, 170))


def blend_poly(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    """Alpha-filled polygon (in place), clipped to its bounding box for speed."""
    import cv2

    p = np.round(np.asarray(pts, dtype=np.float64) * 16).astype(np.int32)
    x0, y0 = np.maximum(p.min(axis=0) // 16 - 2, 0)
    x1, y1 = np.minimum(p.max(axis=0) // 16 + 3, [img.shape[1], img.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.fillPoly(over, [p - np.array([x0 * 16, y0 * 16])], color, cv2.LINE_AA, shift=4)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)


def polyline_aa(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
    closed: bool = False,
    halo: bool = True,
) -> None:
    import cv2

    p = np.round(np.asarray(pts, dtype=np.float64) * 16).astype(np.int32)
    if halo:
        cv2.polylines(img, [p], closed, HALO, thickness + 2, cv2.LINE_AA, shift=4)
    cv2.polylines(img, [p], closed, color, thickness, cv2.LINE_AA, shift=4)

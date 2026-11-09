"""Instance masks for synthetic traffic: vehicle silhouettes, occluders and frame truncation.

Each agent is drawn as the union of the convex image hulls of a few upright
boxes that approximate its body, so that its silhouette is less angular than
its footprint box and its lowest visible edge lies above the road between the
wheels:

* cars: a body from the clearance height to 55 % of the roof height over the
  full footprint, a cabin over the rear 55 % of the length and 85 % of the
  width up to the roof, and four wheels reaching the ground;
* buses and trucks: a body from the clearance height to the roof, and wheels;
* persons and two-wheelers: a single box.

Silhouettes are painted far to near into a label image, so nearer agents hide
farther ones. **Occluders** are fixed image-space blobs (a tree crown as a
union of discs) in front of all agents, and the image border truncates
silhouettes. An agent's mask is what remains visible of its silhouette, with
optional boundary noise, and its detection box is the bounding box of that
mask, as a detector would report it. Ground truth stays the footprint box of
the agent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry.camera import Camera
from .scene import Box3D

__all__ = ["Occluder", "agent_parts", "tree_occluder", "render_label_image"]

#: clearance (m) of the lowest body edge between the wheels, per class
BODY_CLEARANCE = {"car": 0.22, "bus": 0.40, "truck": 0.45, "bicycle": 0.0, "motorcycle": 0.0, "person": 0.0}


@dataclass
class Occluder:
    """An image-space occluder: the union of discs ``(u, v, radius)`` in pixels."""

    discs: Sequence[tuple[float, float, float]]

    def draw(self, label: np.ndarray, value: int) -> None:
        import cv2

        for u, v, r in self.discs:
            cv2.circle(label, (int(round(u)), int(round(v))), int(round(r)), int(value), -1)


def tree_occluder(center: tuple[float, float], radius: float, seed: int = 0) -> Occluder:
    """A tree-crown-like blob around ``center`` (pixels): a disc of ``radius`` plus lobes on its rim."""
    rng = np.random.default_rng(seed)
    cu, cv = center
    discs = [(cu, cv, radius)]
    for a in np.linspace(0, 2 * math.pi, 9, endpoint=False) + rng.uniform(0, 0.6):
        r = radius * rng.uniform(0.35, 0.6)
        d = radius * rng.uniform(0.7, 1.0)
        discs.append((cu + d * math.cos(a), cv + d * math.sin(a), r))
    return Occluder(discs)


def agent_parts(cls: str, x: float, y: float, heading: float, L: float, W: float, H: float) -> list[Box3D]:
    """Boxes approximating an agent's body (``heading`` in radians, footprint centre ``(x, y)``)."""
    deg = math.degrees(heading)
    gap = BODY_CLEARANCE.get(cls, 0.0)
    if cls in ("person", "bicycle", "motorcycle"):
        return [Box3D(x, y, deg, L, W, H)]
    f = np.array([math.cos(heading), math.sin(heading)])
    lft = np.array([-math.sin(heading), math.cos(heading)])
    c = np.array([x, y])
    parts = []
    if cls == "car":
        parts.append(Box3D(x, y, deg, L, W, 0.55 * H - gap, clearance=gap))
        cab = c - 0.12 * L * f
        parts.append(Box3D(cab[0], cab[1], deg, 0.55 * L, 0.85 * W, 0.45 * H, clearance=0.55 * H))
        wheel_in = 0.8
    else:
        parts.append(Box3D(x, y, deg, L, W, H - gap, clearance=gap))
        wheel_in = 1.6 if cls == "bus" else 1.2
    for sl in (1, -1):
        for sw in (1, -1):
            p = c + sl * (L / 2 - wheel_in) * f + sw * (W / 2 - 0.12) * lft
            parts.append(Box3D(p[0], p[1], deg, 0.7 if cls == "car" else 1.0, 0.24, gap + 0.15))
    return parts


def render_label_image(
    camera: Camera,
    agents: Sequence[tuple[int, list[Box3D], float]],
    occluders: Sequence[Occluder] = (),
) -> np.ndarray:
    """Label image (``int32``, ``-1`` background, ``-2`` occluder) of agents ``(label, parts, depth)``."""
    import cv2

    w, h = camera.image_size
    label = np.full((h, w), -1, np.int32)
    for lab, parts, _ in sorted(agents, key=lambda a: -a[2]):
        for b in parts:
            uv = camera.project(b.corners())
            if not np.all(np.isfinite(uv)):
                continue
            hull = cv2.convexHull(np.round(uv * 16).astype(np.int32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(label, hull, int(lab), cv2.LINE_8, shift=4)
    for oc in occluders:
        oc.draw(label, -2)
    return label

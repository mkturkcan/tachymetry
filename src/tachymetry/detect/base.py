"""The detector interface and helpers shared by the back-ends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import ROAD_CLASSES

if TYPE_CHECKING:
    from ..detections import MaskCrop

__all__ = [
    "Detector",
    "DetectorOutput",
    "COCO_TO_ROAD",
    "COCO_CATEGORY_TO_ROAD",
    "resolve_device",
    "road_class_lut",
    "empty_output",
]

#: Per-image detector output: ``(xyxy, score, class_index, cuboid_or_None, masks_or_None[, evidence])``.
DetectorOutput = tuple[np.ndarray, np.ndarray, np.ndarray, "np.ndarray | None", "list[MaskCrop | None] | None"]

#: Contiguous COCO class index (80 classes, as in Ultralytics checkpoints) -> road-user name.
COCO_TO_ROAD = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
#: COCO category id (1-90, as in RF-DETR checkpoints) -> road-user name.
COCO_CATEGORY_TO_ROAD = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}


class Detector:
    """Base class. ``__call__`` maps a list of BGR images to per-image detections.

    Each image yields ``(xyxy, score, cls, cuboid, masks)``: boxes ``(N, 4)``
    ``float32`` in input pixels, scores ``(N,)`` ``float32``, ``cls`` ``(N,)``
    ``int16`` indices into :attr:`class_names`, projected cuboid corners
    ``(N, 8, 3)`` or ``None``, and one instance-mask crop per detection
    (:func:`~tachymetry.detections.crop_mask`) or ``None``. Four-element
    tuples without masks are accepted as well. A sixth element, class evidence
    ``(N, E)`` with columns :attr:`evidence_names`, is optional (SAM 3).
    """

    class_names: tuple[str, ...] = ROAD_CLASSES
    provides_cuboids: bool = False
    provides_masks: bool = False
    evidence_names: tuple[str, ...] = ()  #: columns of the optional class evidence
    name: str = "detector"

    def __call__(self, images: Sequence[np.ndarray]) -> list[DetectorOutput]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classes": list(self.class_names),
            "cuboids": self.provides_cuboids,
            "masks": self.provides_masks,
        }


def empty_output(cuboids: bool = False, masks: bool = False) -> DetectorOutput:
    """Output of an image without detections."""
    return (
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.float32),
        np.zeros(0, np.int16),
        np.zeros((0, 8, 3), np.float32) if cuboids else None,
        [] if masks else None,
    )


def resolve_device(device: str) -> str:
    """``"auto"`` becomes ``cuda:0``, ``mps`` or ``cpu`` (in this order of preference)."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def road_class_lut(labels: Mapping[int, str], classes: Sequence[str]) -> dict[int, int]:
    """Model label id -> index into :data:`~tachymetry.config.ROAD_CLASSES`, for labels named like a kept class.

    Labels outside ``classes`` (traffic lights, animals, ...) are absent from the result and dropped.
    """
    out = {}
    for idx, label in labels.items():
        name = str(label).strip().lower()
        if name in ROAD_CLASSES and name in classes:
            out[int(idx)] = ROAD_CLASSES.index(name)
    return out

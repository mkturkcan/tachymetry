"""Detector back-ends.

All back-ends map their native labels into the shared road-user taxonomy
(:data:`tachymetry.config.ROAD_CLASSES`) and return, per image,
``(xyxy, score, class_index, cuboid_or_None, masks_or_None)``.

``sam3`` (default)
    SAM 3 (Meta) through Hugging Face ``transformers``: ``vehicle`` and
    ``person`` prompts for boxes and masks, class-evidence prompts for the
    vehicle class (:mod:`tachymetry.detect.sam3`). The ``facebook/sam3``
    checkpoint is gated.
``rfdetr-seg-{n,s,m,l}``
    RF-DETR-Seg (Roboflow): COCO-trained real-time DETR with an instance
    segmentation head; boxes and masks. Apache-2.0 weights, downloaded on first
    use, about nine times faster than SAM 3 with lower recall of parked and
    distant vehicles.
``rfdetr-{n,s,m,l}``
    RF-DETR detection models; boxes only.
``uod2-{n,s,m,l,x}``
    UrbanOmniDetect-2: boxes and the eight projected corners of a 3D cuboid per
    road user. Needs the optional ``uod2`` extra (Ultralytics).
``precomputed:<path>``
    Detections produced elsewhere (``.npz`` written by Tachymetry), loaded by the pipeline.
"""

from __future__ import annotations

from ..config import DetectorConfig
from .base import COCO_CATEGORY_TO_ROAD, COCO_TO_ROAD, Detector, DetectorOutput, resolve_device
from .rfdetr import RFDETR_MODELS, RFDETRDetector, rfdetr_weights
from .sam3 import Sam3Detector, sam3_download
from .uod2 import UOD2_REPO, UOD2_SIZES, UOD2Detector, uod2_checkpoint

__all__ = [
    "Detector",
    "DetectorOutput",
    "RFDETRDetector",
    "Sam3Detector",
    "UOD2Detector",
    "build_detector",
    "detector_names",
    "download_weights",
    "uod2_checkpoint",
    "UOD2_REPO",
    "COCO_TO_ROAD",
    "COCO_CATEGORY_TO_ROAD",
    "resolve_device",
]


def detector_names() -> list[str]:
    """Names accepted by :func:`build_detector`, in order of preference."""
    seg = [n for n in RFDETR_MODELS if n.startswith("rfdetr-seg-")]
    box = [n for n in RFDETR_MODELS if not n.startswith("rfdetr-seg-")]
    return ["sam3", *reversed(seg), *reversed(box), *(f"uod2-{s}" for s in reversed(UOD2_SIZES))]


def build_detector(cfg: DetectorConfig, device: str = "auto") -> Detector:
    """Instantiate the detector named by ``cfg.name``."""
    low = cfg.name.strip().lower()
    if low in RFDETR_MODELS:
        return RFDETRDetector(cfg, device, variant=low)
    if low == "sam3":
        return Sam3Detector(cfg, device)
    if low.startswith("uod2") or low.startswith("urbanomnidetect"):
        return UOD2Detector(cfg, device, size=low.split("-")[-1] if "-" in low else "x")
    if low.startswith("precomputed:"):
        raise ValueError("precomputed detections are loaded by the pipeline (Pipeline.detect), not built")
    raise ValueError(
        f"unknown detector {cfg.name!r}; choose from {', '.join(detector_names())} or precomputed:<.npz file or folder>"
    )


def download_weights(name: str) -> str:
    """Fetch the weights of detector ``name`` into its cache ahead of time; returns their location."""
    low = name.strip().lower()
    if low in RFDETR_MODELS:
        return str(rfdetr_weights(low))
    if low == "sam3":
        return sam3_download()
    if low.startswith("uod2") or low.startswith("urbanomnidetect"):
        return str(uod2_checkpoint(low.split("-")[-1] if "-" in low else "x"))
    raise ValueError(f"unknown detector {name!r}; choose from {', '.join(detector_names())}")

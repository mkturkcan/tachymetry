"""Object detection back-ends and detection post-processing."""

from .backends import (
    COCO_CATEGORY_TO_ROAD,
    COCO_TO_ROAD,
    UOD2_REPO,
    Detector,
    DetectorOutput,
    RFDETRDetector,
    Sam3Detector,
    UOD2Detector,
    build_detector,
    detector_names,
    download_weights,
    uod2_checkpoint,
)
from .postprocess import box_ioa, cuboid_trust, nms_by_group, suppress_mask_duplicates, suppress_nested

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
    "cuboid_trust",
    "suppress_nested",
    "suppress_mask_duplicates",
    "nms_by_group",
    "box_ioa",
]

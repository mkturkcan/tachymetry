"""UrbanOmniDetect-2: boxes and projected 3D cuboid corners (optional; needs Ultralytics).

UrbanOmniDetect-2 (Turkcan, 2026) is a YOLO26 hybrid detect-and-pose network
that also regresses the eight projected corners of a 3D cuboid for each road
user, with no camera calibration. Checkpoints download from
``huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2`` on first use. The
network runs on Ultralytics, which Tachymetry installs only with the ``uod2``
extra (``pip install "tachymetry[uod2]"``).
"""

from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ROAD_CLASSES, DetectorConfig
from .base import COCO_TO_ROAD, Detector, DetectorOutput, empty_output, resolve_device, road_class_lut
from .postprocess import prefilter, refine

__all__ = ["UOD2Detector", "UOD2_REPO", "UOD2_SIZES", "uod2_checkpoint"]

UOD2_REPO = "mehmetkeremturkcan/UrbanOmniDetect-2"
UOD2_SIZES = ("n", "s", "m", "l", "x")
#: Input size (letterboxed long side) when ``DetectorConfig.imgsz`` is ``None``.
UOD2_IMGSZ = 960
#: Score threshold when ``DetectorConfig.conf`` is ``None``.
UOD2_CONF = 0.10

_MISSING = (
    "UrbanOmniDetect-2 runs on Ultralytics, which is not installed. Install the optional extra with "
    'pip install "tachymetry[uod2]", or use the default detector (sam3) or rfdetr-seg-l.'
)


def _ultralytics() -> Any:
    try:
        import ultralytics
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING) from exc
    return ultralytics


_SHIM_CLASS: Any = None


def _hybrid_class() -> Any:
    """``HybridPoseModel26``: the training-time subclass the checkpoints were pickled with.

    It differs from Ultralytics' ``PoseModel`` only in its training criterion, so
    inference needs nothing beyond the class being importable while unpickling.
    """
    global _SHIM_CLASS
    if _SHIM_CLASS is None:
        _ultralytics()
        from ultralytics.nn.tasks import PoseModel

        class HybridPoseModel26(PoseModel):  # noqa: D401 - unpickling shim
            """Inference stand-in for ``v2.hybrid_v2.HybridPoseModel26``."""

        HybridPoseModel26.__module__ = __name__
        HybridPoseModel26.__qualname__ = "HybridPoseModel26"
        _SHIM_CLASS = HybridPoseModel26
        globals()["HybridPoseModel26"] = HybridPoseModel26
    return _SHIM_CLASS


@contextlib.contextmanager
def _uod2_unpickling() -> Iterator[None]:
    """Temporarily expose ``v2.hybrid_v2.HybridPoseModel26`` to the unpickler, then clean up."""
    cls = _hybrid_class()
    added: list[str] = []
    parent = sys.modules.get("v2")
    if parent is None:
        parent = types.ModuleType("v2")
        sys.modules["v2"] = parent
        added.append("v2")
    had_attr = hasattr(parent, "hybrid_v2")
    if "v2.hybrid_v2" not in sys.modules:
        mod = types.ModuleType("v2.hybrid_v2")
        mod.HybridPoseModel26 = cls  # type: ignore[attr-defined]
        sys.modules["v2.hybrid_v2"] = mod
        added.append("v2.hybrid_v2")
        if not had_attr:
            parent.hybrid_v2 = mod  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)
        if not had_attr and "v2" in sys.modules and hasattr(sys.modules["v2"], "hybrid_v2"):
            with contextlib.suppress(AttributeError):
                delattr(sys.modules["v2"], "hybrid_v2")


def uod2_checkpoint(size: str = "x") -> Path:
    """Local path of an UrbanOmniDetect-2 checkpoint, downloading it on first use."""
    if size not in UOD2_SIZES:
        raise ValueError(f"UrbanOmniDetect-2 sizes are {UOD2_SIZES}, got {size!r}")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(UOD2_REPO, f"checkpoints/urbanomnidetect_v2_{size}_640.pt"))


class UOD2Detector(Detector):
    """UrbanOmniDetect-2 through Ultralytics: boxes and projected cuboid corners."""

    provides_cuboids = True

    def __init__(self, cfg: DetectorConfig, device: str = "auto", size: str = "x", weights: str | Path | None = None):
        _ultralytics()  # before downloading weights
        from ultralytics import YOLO

        weights = weights or cfg.weights or uod2_checkpoint(size)
        self.cfg = cfg
        self.device = resolve_device(device)
        self.name = f"UrbanOmniDetect-2 ({size.upper()})"
        self.imgsz = int(cfg.imgsz or UOD2_IMGSZ)
        self.conf = float(UOD2_CONF if cfg.conf is None else cfg.conf)
        with _uod2_unpickling():
            self.model = YOLO(str(weights), task="pose")
        names = getattr(self.model, "names", None) or {}
        labels = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        # checkpoints name their labels; COCO_TO_ROAD covers unnamed COCO-indexed heads
        labels = {i: (n if str(n).lower() in ROAD_CLASSES else COCO_TO_ROAD.get(int(i), n)) for i, n in labels.items()}
        self._map = road_class_lut(labels, cfg.classes)
        self.class_names = ROAD_CLASSES
        self.half = bool(cfg.half and self.device.startswith("cuda"))

    def __call__(self, images: Sequence[np.ndarray]) -> list[DetectorOutput]:
        if not len(images):
            return []
        results = self.model.predict(
            list(images), imgsz=self.imgsz, conf=self.conf, device=self.device, half=self.half, verbose=False
        )
        out: list[DetectorOutput] = []
        for img, r in zip(images, results):
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(empty_output(cuboids=True))
                continue
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            score = boxes.conf.cpu().numpy().astype(np.float32)
            raw_cls = boxes.cls.cpu().numpy().astype(int)
            cub = np.full((len(xyxy), 8, 3), np.nan, np.float32)
            if getattr(r, "keypoints", None) is not None and r.keypoints.data is not None:
                kd = r.keypoints.data.cpu().numpy().astype(np.float32)
                if kd.ndim == 3 and kd.shape[0] == len(xyxy) and kd.shape[1] >= 8:
                    cub = (
                        kd[:, :8, :3]
                        if kd.shape[2] >= 3
                        else np.concatenate([kd[:, :8, :2], np.ones((len(kd), 8, 1), np.float32)], axis=2)
                    )
            known = np.array([c in self._map for c in raw_cls], dtype=bool)
            cls = np.array([self._map.get(c, -1) for c in raw_cls], dtype=np.int16)
            xyxy, score, cls, cub = xyxy[known], score[known], cls[known], cub[known]
            idx = prefilter(xyxy, score, cls, img.shape[1], self.cfg, self.class_names)
            xyxy, score, cls, cub = xyxy[idx], score[idx], cls[idx], cub[idx]
            idx, cub = refine(xyxy, score, cls, self.cfg, self.class_names, cuboid=cub)
            out.append((xyxy[idx], score[idx], cls[idx], cub, None))
        return out

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "device": self.device, "imgsz": self.imgsz, "conf": self.conf, "half": self.half}

"""Detector back-ends.

All back-ends map their native labels into the shared road-user taxonomy
(:data:`tachymetry.config.ROAD_CLASSES`) and return, per image,
``(xyxy, score, class_index, cuboid_or_None)``.

``uod2-{n,s,m,l,x}``
    UrbanOmniDetect-2 (Turkcan, 2026): a YOLO26 hybrid detect-and-pose network
    that also regresses the eight projected corners of a 3D cuboid for each
    road user, with no camera calibration. Checkpoints download from
    ``huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2`` on first use.
``yolo:<weights>``
    Any Ultralytics detection or segmentation checkpoint (``yolo11x.pt``,
    ``yolo26l.pt``, a fine-tuned ``.pt``, an exported ``.engine``).
``precomputed:<path>``
    Detections produced elsewhere (``.npz`` written by Tachymetry or MOT text).
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
from .postprocess import cuboid_trust, nms_by_group, suppress_nested

__all__ = ["Detector", "UltralyticsDetector", "build_detector", "UOD2_REPO", "uod2_checkpoint", "COCO_TO_ROAD"]

UOD2_REPO = "mehmetkeremturkcan/UrbanOmniDetect-2"
UOD2_SIZES = ("n", "s", "m", "l", "x")
#: COCO class id -> road-user name
COCO_TO_ROAD = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class Detector:
    """Base class. ``__call__`` maps a list of BGR images to per-image detections."""

    class_names: tuple[str, ...] = ROAD_CLASSES
    provides_cuboids: bool = False
    name: str = "detector"

    def __call__(
        self, images: Sequence[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "classes": list(self.class_names), "cuboids": self.provides_cuboids}


# ----------------------------------------------------------- UrbanOmniDetect-2
_SHIM_CLASS: Any = None


def _hybrid_class() -> Any:
    """``HybridPoseModel26``: the training-time subclass the checkpoints were pickled with.

    It differs from Ultralytics' ``PoseModel`` only in its training criterion, so
    inference needs nothing beyond the class being importable while unpickling.
    """
    global _SHIM_CLASS
    if _SHIM_CLASS is None:
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


def _resolve_device(device: str) -> str:
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


class UltralyticsDetector(Detector):
    """Ultralytics-based detector (UrbanOmniDetect-2 or any YOLO checkpoint)."""

    def __init__(
        self,
        weights: str | Path,
        cfg: DetectorConfig,
        device: str = "auto",
        cuboids: bool = False,
        name: str | None = None,
    ):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = _resolve_device(device)
        self.provides_cuboids = cuboids
        self.name = name or Path(str(weights)).name
        task = "pose" if cuboids else None
        if cuboids:
            with _uod2_unpickling():
                self.model = YOLO(str(weights), task="pose")
        else:
            self.model = YOLO(str(weights), task=task) if task else YOLO(str(weights))
        names = getattr(self.model, "names", None) or {}
        self._map = {}
        for idx, label in names.items() if isinstance(names, dict) else enumerate(names):
            label = str(label).lower()
            road = COCO_TO_ROAD.get(int(idx)) if label not in ROAD_CLASSES else label
            if label in ROAD_CLASSES:
                road = label
            if road in cfg.classes:
                self._map[int(idx)] = ROAD_CLASSES.index(road)
        self.class_names = ROAD_CLASSES
        self._vehicle_ids = {ROAD_CLASSES.index(c) for c in ("car", "bus", "truck")}
        sup = {"car": 1, "bus": 1, "truck": 1, "bicycle": 2, "motorcycle": 2, "person": 3}
        self._group = {ROAD_CLASSES.index(c): g for c, g in sup.items()}
        self._floors = {ROAD_CLASSES.index(k): float(v) for k, v in cfg.class_conf.items() if k in ROAD_CLASSES}
        self.half = bool(cfg.half and self.device.startswith("cuda"))

    def __call__(
        self, images: Sequence[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
        if not len(images):
            return []
        results = self.model.predict(
            list(images), imgsz=self.cfg.imgsz, conf=self.cfg.conf, device=self.device, half=self.half, verbose=False
        )
        out = []
        for img, r in zip(images, results):
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                out.append(
                    (
                        np.zeros((0, 4), np.float32),
                        np.zeros(0, np.float32),
                        np.zeros(0, np.int16),
                        np.zeros((0, 8, 3), np.float32) if self.provides_cuboids else None,
                    )
                )
                continue
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            score = boxes.conf.cpu().numpy().astype(np.float32)
            raw_cls = boxes.cls.cpu().numpy().astype(int)
            cub = None
            if self.provides_cuboids and getattr(r, "keypoints", None) is not None and r.keypoints.data is not None:
                kd = r.keypoints.data.cpu().numpy().astype(np.float32)
                if kd.ndim == 3 and kd.shape[0] == len(xyxy) and kd.shape[1] >= 8:
                    cub = (
                        kd[:, :8, :3]
                        if kd.shape[2] >= 3
                        else np.concatenate([kd[:, :8, :2], np.ones((len(kd), 8, 1), np.float32)], axis=2)
                    )
            keep = np.array([c in self._map for c in raw_cls], dtype=bool)
            cls = np.array([self._map.get(c, -1) for c in raw_cls], dtype=np.int16)
            for cid, floor in self._floors.items():
                keep &= ~((cls == cid) & (score < floor))
            w = img.shape[1]
            keep &= (xyxy[:, 2] - xyxy[:, 0]) <= self.cfg.max_box_width_frac * w
            xyxy, score, cls = xyxy[keep], score[keep], cls[keep]
            if cub is not None:
                cub = cub[keep]
            grp = np.array([self._group.get(int(c), int(c)) for c in cls], dtype=np.int64)
            nk = nms_by_group(xyxy, score, grp, 0.8)
            xyxy, score, cls = xyxy[nk], score[nk], cls[nk]
            if cub is not None:
                cub = cub[nk]
                trusted = cuboid_trust(xyxy, cub, self.cfg.cuboid_conf)
                nk = suppress_nested(
                    xyxy,
                    cls,
                    self._vehicle_ids,
                    cub,
                    trusted,
                    self.cfg.nested_containment,
                    self.cfg.nested_ground_overlap,
                )
                xyxy, score, cls, cub, trusted = xyxy[nk], score[nk], cls[nk], cub[nk], trusted[nk]
                cub = cub.copy()
                cub[~trusted] = np.nan  # untrusted cuboids never reach geometry
            else:
                nk = suppress_nested(
                    xyxy,
                    cls,
                    self._vehicle_ids,
                    None,
                    None,
                    self.cfg.nested_containment,
                    self.cfg.nested_ground_overlap,
                )
                xyxy, score, cls = xyxy[nk], score[nk], cls[nk]
            out.append((xyxy, score, cls, cub))
        return out

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "device": self.device,
            "imgsz": self.cfg.imgsz,
            "conf": self.cfg.conf,
            "half": self.half,
        }


def build_detector(cfg: DetectorConfig, device: str = "auto") -> Detector:
    """Instantiate the detector named by ``cfg.name``."""
    name = cfg.name.strip()
    low = name.lower()
    if low.startswith("uod2") or low.startswith("urbanomnidetect"):
        size = low.split("-")[-1] if "-" in low else "x"
        weights = cfg.weights or uod2_checkpoint(size)
        return UltralyticsDetector(weights, cfg, device, cuboids=True, name=f"UrbanOmniDetect-2 ({size.upper()})")
    if low.startswith("yolo:") or low.endswith((".pt", ".engine", ".onnx")):
        weights = cfg.weights or (name.split(":", 1)[1] if ":" in name else name)
        return UltralyticsDetector(weights, cfg, device, cuboids=False)
    raise ValueError(
        f"unknown detector {cfg.name!r}; use 'uod2-x' or 'yolo:<weights>' "
        "(precomputed detections are loaded by the pipeline)"
    )

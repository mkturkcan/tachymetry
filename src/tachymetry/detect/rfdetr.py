"""RF-DETR detection and instance-segmentation models (Roboflow, Apache-2.0 weights).

RF-DETR is a real-time DETR with a DINOv2 backbone, trained on COCO. The
``rfdetr-seg-*`` models add an instance-segmentation head; the ``rfdetr-*``
models predict boxes only. Weights download to ``~/.roboflow/models`` (or
``$RF_HOME``) on first use. Only the Nano to Large sizes are offered; the
XLarge and 2XLarge detection weights ship under the Roboflow Platform Model
License.

Frames are resized to a square of ``imgsz`` pixels without preserving the
aspect ratio, as in RF-DETR training (by default twice the training
resolution, at most 1008), and run in batches through the model's export graph
(FP16 weights on CUDA). One detection is kept per query: its highest-scoring
road class. Mask logits stay at the mask-head resolution until
:func:`~tachymetry.detect.masks.mask_crops` samples them inside each box.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ROAD_CLASSES, DetectorConfig
from .base import COCO_CATEGORY_TO_ROAD, Detector, DetectorOutput, empty_output, resolve_device, road_class_lut
from .masks import mask_crops
from .postprocess import MASK_DUPLICATE_IOU, prefilter, refine

__all__ = ["RFDETRDetector", "RFDETR_MODELS", "RFDETR_IMGSZ", "RFDETR_CONF", "rfdetr_weights"]

#: Tachymetry name -> ``rfdetr`` model class (Apache-2.0 weights only).
RFDETR_MODELS = {
    "rfdetr-seg-n": "RFDETRSegNano",
    "rfdetr-seg-s": "RFDETRSegSmall",
    "rfdetr-seg-m": "RFDETRSegMedium",
    "rfdetr-seg-l": "RFDETRSegLarge",
    "rfdetr-n": "RFDETRNano",
    "rfdetr-s": "RFDETRSmall",
    "rfdetr-m": "RFDETRMedium",
    "rfdetr-l": "RFDETRLarge",
}
#: Largest default input side: ``imgsz=None`` selects twice the variant's training resolution, at most this,
#: rounded up to the model's patch grid (``rfdetr-seg-l``: 1008; ``rfdetr-l``: 1024; ``rfdetr-seg-n``: 624).
RFDETR_IMGSZ = 1008
#: Score threshold when ``DetectorConfig.conf`` is ``None``.
RFDETR_CONF = 0.15


@contextlib.contextmanager
def _quiet(logger_name: str = "rf-detr") -> Iterator[None]:
    """Silence ``rfdetr`` info and warning messages (weights checks, legacy checkpoint layout) while loading."""
    log = logging.getLogger(logger_name)
    level = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(level)


def rfdetr_weights(variant: str = "rfdetr-seg-l") -> Path:
    """Local path of an RF-DETR checkpoint, downloading it on first use."""
    import rfdetr

    if variant not in RFDETR_MODELS:
        raise ValueError(f"RF-DETR models are {sorted(RFDETR_MODELS)}, got {variant!r}")
    with _quiet():
        model = getattr(rfdetr, RFDETR_MODELS[variant])()
    return Path(str(model.model_config.pretrain_weights))


def _label_names(model: Any, num_slots: int) -> dict[int, str]:
    """Logit slot -> class name of an RF-DETR model (sparse COCO category ids for the released weights)."""
    try:
        from rfdetr.utilities.class_names import prediction_labels

        return {int(k): str(v) for k, v in prediction_labels(model).class_id_to_name.items()}
    except Exception:  # pragma: no cover - fallback for other rfdetr releases
        names = [str(n) for n in model.class_names]
        if num_slots > len(names) and names[:4] == ["person", "bicycle", "car", "motorcycle"]:
            return dict(COCO_CATEGORY_TO_ROAD)
        return dict(enumerate(names))


class RFDETRDetector(Detector):
    """RF-DETR or RF-DETR-Seg, mapped to the road-user classes."""

    def __init__(self, cfg: DetectorConfig, device: str = "auto", variant: str = "rfdetr-seg-l"):
        import torch

        try:
            import rfdetr
        except ImportError as exc:  # pragma: no cover - core dependency
            raise ImportError("RF-DETR is not installed: pip install rfdetr") from exc
        if variant not in RFDETR_MODELS:
            raise ValueError(f"RF-DETR models are {sorted(RFDETR_MODELS)}, got {variant!r}")
        self.cfg = cfg
        self.variant = variant
        self.device = torch.device(resolve_device(device))
        self.conf = float(RFDETR_CONF if cfg.conf is None else cfg.conf)
        self.half = bool(cfg.half and self.device.type == "cuda")
        cls = getattr(rfdetr, RFDETR_MODELS[variant])
        kwargs: dict[str, Any] = {}
        if cfg.weights:
            kwargs["pretrain_weights"] = str(cfg.weights)
        with _quiet():
            defaults = cls.get_model_config(cls.__new__(cls))  # the variant's default configuration, no weights
            self.block = int(defaults.patch_size) * int(defaults.num_windows)
            side = int(round(float(cfg.imgsz or min(2 * int(defaults.resolution), RFDETR_IMGSZ))))
            self.imgsz = max(self.block, -(-side // self.block) * self.block)  # round up to the patch grid
            self.rf = cls(resolution=self.imgsz, **kwargs)
        # export mode: plain tensor in, (boxes, logits[, masks]) out; FP16 weights on CUDA
        self.net = self.rf.model.model.to(self.device).eval()
        if self.half:
            self.net.half()
        self.net.export()
        mc = self.rf.model_config
        self.provides_masks = bool(getattr(mc, "segmentation_head", False))
        size = variant.rsplit("-", 1)[-1].upper()
        self.name = f"RF-DETR-Seg ({size})" if variant.startswith("rfdetr-seg-") else f"RF-DETR ({size})"
        num_slots = int(getattr(mc, "num_classes", 90)) + 1
        lut = road_class_lut(_label_names(self.rf, num_slots), cfg.classes)
        self._cols = sorted(lut)  # logit slots of the kept road classes
        self._road = np.array([lut[c] for c in self._cols], np.int16)
        self._mean = torch.tensor(self.rf.means, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.rf.stds, device=self.device).view(1, 3, 1, 1)
        self.class_names = ROAD_CLASSES

    # ------------------------------------------------------------------ inference
    def _forward(self, images: Sequence[np.ndarray]) -> tuple[Any, ...]:
        """``(boxes, logits[, mask logits])`` for same-sized BGR images."""
        import torch
        import torch.nn.functional as F  # noqa: N812

        x = torch.from_numpy(np.ascontiguousarray(np.stack(images)))
        if self.device.type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        x = x.permute(0, 3, 1, 2).flip(1).float().div_(255.0)  # BGR uint8 -> RGB float
        x = F.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        with torch.inference_mode():
            return tuple(self.net(x.half() if self.half else x))

    def __call__(self, images: Sequence[np.ndarray]) -> list[DetectorOutput]:
        if not len(images):
            return []
        shapes = {im.shape for im in images}
        if len(shapes) > 1:  # mixed sizes: one batch per size, in input order
            out: list[DetectorOutput | None] = [None] * len(images)
            for shape in shapes:
                idx = [i for i, im in enumerate(images) if im.shape == shape]
                for i, r in zip(idx, self._run([images[i] for i in idx])):
                    out[i] = r
            return out  # type: ignore[return-value]
        return self._run(list(images))

    def _run(self, images: list[np.ndarray]) -> list[DetectorOutput]:
        import torch

        if not self._cols:
            return [empty_output(masks=self.provides_masks) for _ in images]
        pred = self._forward(images)
        with torch.inference_mode():
            prob = pred[1][..., self._cols].float().sigmoid()  # (B, Q, R)
            best, ridx = prob.max(dim=-1)
            score_all = best.cpu().numpy()
            ridx_all = ridx.cpu().numpy()
            boxes_all = pred[0].float().cpu().numpy()  # (B, Q, 4) cx, cy, w, h in [0, 1]
            masks_all = pred[2] if len(pred) > 2 else None  # (B, Q, h, w) logits
            h, w = images[0].shape[:2]
            out: list[DetectorOutput] = []
            for b in range(len(images)):
                q = np.flatnonzero(score_all[b] > self.conf)
                cx, cy, bw, bh = boxes_all[b, q].T
                xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1) * [w, h, w, h]
                xyxy = np.clip(xyxy, 0, [w, h, w, h]).astype(np.float32).reshape(-1, 4)
                score = score_all[b, q].astype(np.float32)
                cls = self._road[ridx_all[b, q]]
                idx = prefilter(xyxy, score, cls, w, self.cfg, self.class_names)
                xyxy, score, cls, q = xyxy[idx], score[idx], cls[idx], q[idx]
                masks = None
                if masks_all is not None:
                    sel = torch.as_tensor(q, device=masks_all.device, dtype=torch.long)
                    masks = mask_crops(masks_all[b].index_select(0, sel), xyxy, (w, h), threshold=0.0)
                idx, _ = refine(
                    xyxy,
                    score,
                    cls,
                    self.cfg,
                    self.class_names,
                    masks=masks,
                    score_aware=True,
                    mask_duplicate_iou=MASK_DUPLICATE_IOU,
                )
                out.append(
                    (xyxy[idx], score[idx], cls[idx], None, None if masks is None else [masks[int(i)] for i in idx])
                )
        return out

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model": RFDETR_MODELS[self.variant],
            "device": str(self.device),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "half": self.half,
        }

"""SAM 3 (Meta, 2025) through Hugging Face ``transformers``: text-prompted detection and segmentation.

Each road class is a text prompt (``car``, ``bus``, ``truck``, ``motorcycle``,
``bicycle``, ``person``). The image encoder runs once per frame and the
prompt-conditioned DETR decoder once per (frame, prompt) pair, in batches.
SAM 3 is a set predictor without non-maximum suppression; duplicates across
prompts (one vehicle found by ``car`` and by ``truck``) are removed by mask
overlap, keeping the higher score. The checkpoint ``facebook/sam3`` is gated
on Hugging Face.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import ROAD_CLASSES, DetectorConfig
from .base import Detector, DetectorOutput, resolve_device
from .masks import mask_crops
from .postprocess import MASK_DUPLICATE_IOU, prefilter, refine

__all__ = ["Sam3Detector", "SAM3_REPO", "SAM3_PROMPTS", "SAM3_CONF", "SAM3_ACCESS_HELP", "load_sam3", "sam3_download"]

SAM3_REPO = "facebook/sam3"
#: Road class -> text prompt.
SAM3_PROMPTS = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}
#: Score threshold (class score times presence score) when ``DetectorConfig.conf`` is ``None``.
SAM3_CONF = 0.30
#: (frame, prompt) pairs per decoder pass; bounds memory (about 8 GB at 8 pairs and batch 8, BF16).
PAIRS_PER_PASS = 8

SAM3_ACCESS_HELP = (
    "SAM 3 weights ({repo}) are gated on Hugging Face. Request access at https://huggingface.co/{repo} "
    "and wait for the approval, then authenticate this machine with `hf auth login` (or set the HF_TOKEN "
    "environment variable) and run again."
)


def _is_gated(exc: BaseException) -> bool:
    """True when ``exc`` (or its cause chain) reports a gated or unauthorised Hugging Face repository."""
    try:
        from huggingface_hub.errors import GatedRepoError
    except ImportError:  # pragma: no cover
        GatedRepoError = ()  # type: ignore[assignment,misc]  # noqa: N806
    seen: BaseException | None = exc
    while seen is not None:
        if GatedRepoError and isinstance(seen, GatedRepoError):
            return True
        text = str(seen).lower()
        if "gated repo" in text or "401 client error" in text or "403 client error" in text:
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def load_sam3(repo: str = SAM3_REPO, dtype: Any = None) -> tuple[Any, Any]:
    """``(processor, model)`` of a SAM 3 checkpoint, with a clear error when access is not granted."""
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("SAM 3 needs transformers 5.1 or newer: pip install -U transformers") from exc
    try:
        processor = Sam3Processor.from_pretrained(repo)
        model = Sam3Model.from_pretrained(repo, dtype=dtype) if dtype is not None else Sam3Model.from_pretrained(repo)
    except Exception as exc:
        if _is_gated(exc):
            raise PermissionError(SAM3_ACCESS_HELP.format(repo=repo)) from exc
        raise
    return processor, model


def sam3_download(repo: str = SAM3_REPO) -> str:
    """Download the ``transformers`` files of a SAM 3 checkpoint; returns the local folder."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    except Exception as exc:
        if _is_gated(exc):
            raise PermissionError(SAM3_ACCESS_HELP.format(repo=repo)) from exc
        raise


class Sam3Detector(Detector):
    """SAM 3 prompted with the road-class names: boxes, scores and instance masks."""

    provides_masks = True

    def __init__(self, cfg: DetectorConfig, device: str = "auto", repo: str | None = None):
        import torch

        self.cfg = cfg
        self.repo = str(cfg.weights or repo or SAM3_REPO)
        self.device = torch.device(resolve_device(device))
        cuda = self.device.type == "cuda"
        self.half = bool(cfg.half and cuda)
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.half else None
        self.processor, model = load_sam3(self.repo, self.dtype)
        self.model = model.to(self.device).eval()
        self.conf = float(SAM3_CONF if cfg.conf is None else cfg.conf)
        size = getattr(self.processor.image_processor, "size", None) or {"height": 1008, "width": 1008}
        self.imgsz = (int(size["height"]), int(size["width"]))
        ip = self.processor.image_processor
        self._mean = torch.tensor(ip.image_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(ip.image_std, device=self.device).view(1, 3, 1, 1)
        self.classes = [c for c in ROAD_CLASSES if c in cfg.classes and c in SAM3_PROMPTS]
        self.prompts = [SAM3_PROMPTS[c] for c in self.classes]
        self._cls = np.array([ROAD_CLASSES.index(c) for c in self.classes], np.int16)
        self.class_names = ROAD_CLASSES
        self.name = "SAM 3"
        text = self.processor(text=self.prompts, return_tensors="pt") if self.prompts else None
        self._text_mask = None
        self._text = None
        if text is not None:
            with torch.inference_mode():
                ids = text["input_ids"].to(self.device)
                self._text_mask = text["attention_mask"].to(self.device)
                self._text = self.model.get_text_features(input_ids=ids, attention_mask=self._text_mask).pooler_output

    def _pixels(self, images: Sequence[np.ndarray]) -> Any:
        import torch
        import torch.nn.functional as F  # noqa: N812

        x = torch.from_numpy(np.ascontiguousarray(np.stack(images)))
        x = x.pin_memory().to(self.device, non_blocking=True) if self.device.type == "cuda" else x.to(self.device)
        x = x.permute(0, 3, 1, 2).flip(1).float().div_(255.0)  # BGR uint8 -> RGB float
        x = F.interpolate(x, size=self.imgsz, mode="bilinear", align_corners=False, antialias=True)
        x = (x - self._mean) / self._std
        return x.to(self.dtype) if self.dtype is not None else x

    def __call__(self, images: Sequence[np.ndarray]) -> list[DetectorOutput]:
        if not len(images):
            return []
        shapes = {im.shape for im in images}
        if len(shapes) > 1:
            out: list[DetectorOutput | None] = [None] * len(images)
            for shape in shapes:
                idx = [i for i, im in enumerate(images) if im.shape == shape]
                for i, r in zip(idx, self._run([images[i] for i in idx])):
                    out[i] = r
            return out  # type: ignore[return-value]
        return self._run(list(images))

    def _run(self, images: list[np.ndarray]) -> list[DetectorOutput]:
        import torch

        h, w = images[0].shape[:2]
        n_img, n_prompt = len(images), len(self.prompts)
        boxes: list[list[np.ndarray]] = [[] for _ in images]
        scores: list[list[np.ndarray]] = [[] for _ in images]
        classes: list[list[np.ndarray]] = [[] for _ in images]
        maps: list[list[Any]] = [[] for _ in images]
        if n_prompt:
            amp = torch.autocast("cuda", dtype=self.dtype) if self.half else contextlib.nullcontext()
            with torch.inference_mode(), amp:
                vision = self.model.get_vision_features(pixel_values=self._pixels(images))
                pairs = [(i, p) for i in range(n_img) for p in range(n_prompt)]
                for start in range(0, len(pairs), PAIRS_PER_PASS):
                    chunk = pairs[start : start + PAIRS_PER_PASS]
                    ii = torch.as_tensor([i for i, _ in chunk], device=self.device)
                    pp = torch.as_tensor([p for _, p in chunk], device=self.device)
                    sub = copy.copy(vision)
                    sub.fpn_hidden_states = tuple(t.index_select(0, ii) for t in vision.fpn_hidden_states)
                    sub.fpn_position_encoding = tuple(t.index_select(0, ii) for t in vision.fpn_position_encoding)
                    res = self.model(
                        vision_embeds=sub,
                        text_embeds=self._text.index_select(0, pp),
                        attention_mask=self._text_mask.index_select(0, pp),
                    )
                    score = res.pred_logits.float().sigmoid()
                    if res.presence_logits is not None:
                        score = score * res.presence_logits.float().sigmoid()
                    keep = score > self.conf
                    score_np, keep_np = score.cpu().numpy(), keep.cpu().numpy()
                    box_np = res.pred_boxes.float().cpu().numpy()  # (n, Q, 4) x1, y1, x2, y2 in [0, 1]
                    for k, (i, p) in enumerate(chunk):
                        q = np.flatnonzero(keep_np[k])
                        if not len(q):
                            continue
                        xyxy = np.clip(box_np[k, q] * [w, h, w, h], 0, [w, h, w, h]).astype(np.float32)
                        boxes[i].append(xyxy)
                        scores[i].append(score_np[k, q].astype(np.float32))
                        classes[i].append(np.full(len(q), self._cls[p], np.int16))
                        sel = torch.as_tensor(q, device=self.device, dtype=torch.long)
                        maps[i].append(res.pred_masks[k].index_select(0, sel).float().sigmoid())
                    del res
        out: list[DetectorOutput] = []
        for i in range(n_img):
            if not boxes[i]:
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int16), None, []))
                continue
            xyxy, score, cls = np.concatenate(boxes[i]), np.concatenate(scores[i]), np.concatenate(classes[i])
            mp = torch.cat(maps[i])
            idx = prefilter(xyxy, score, cls, w, self.cfg, self.class_names)
            xyxy, score, cls = xyxy[idx], score[idx], cls[idx]
            masks = mask_crops(mp.index_select(0, torch.as_tensor(idx, device=mp.device)), xyxy, (w, h), 0.5)
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
            out.append((xyxy[idx], score[idx], cls[idx], None, [masks[int(j)] for j in idx]))
        return out

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model": self.repo,
            "prompts": dict(zip(self.classes, self.prompts)),
            "device": str(self.device),
            "imgsz": list(self.imgsz),
            "conf": self.conf,
            "half": self.half,
        }

"""SAM 3 (Meta, 2025) through Hugging Face ``transformers``: text-prompted detection and segmentation.

The default detector of Tachymetry. Detection follows a two-level prompt scheme:

* **detection prompts** ``vehicle`` and ``person`` give the boxes and instance
  masks that are tracked (score threshold 0.4);
* **class-evidence prompts** ``car``, ``van``, ``suv``, ``bus``, ``truck``,
  ``bike`` and ``motorcycle`` run in the same forward pass. They do not create
  tracked objects (except two-wheelers that the ``vehicle`` prompt missed); they
  vote for the class of the detections they overlap.

Every detection carries an evidence vector (:data:`EVIDENCE_COLUMNS`): for each
class prompt, the highest score of that prompt's detections whose box overlaps
it with an IoU above 0.5, counted from 0.35 for ``car``, ``van`` and ``suv`` and
from 0.40 for the others; and two two-wheeler flags from ``bike`` detections
scoring at least 0.5: ``bike_cover`` (the bike box lies within the detection,
intersection over the bike box above 0.6) and ``bike_ride`` (the detection,
a person, overlaps the bike box enlarged by 15 % on each side, intersection over
the person box above 0.3). The class of a *track* is decided from these vectors
by a quality-weighted vote over its frames (:func:`tachymetry.tracking.hygiene.class_vote`),
and riders and passengers are separated from pedestrians with the flags
(:mod:`tachymetry.tracking.verdicts`). The per-detection class set here is the
same rule applied to a single frame.

The image encoder runs once per frame and the prompt-conditioned DETR decoder
once per (frame, prompt) pair, in batches. SAM 3 is a set predictor without
non-maximum suppression; duplicates are removed by mask overlap, keeping the
higher score. The checkpoint ``facebook/sam3`` is gated on Hugging Face.
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

__all__ = [
    "Sam3Detector",
    "SAM3_REPO",
    "SAM3_PROMPTS",
    "SAM3_EVIDENCE",
    "SAM3_CONF",
    "SAM3_EVIDENCE_MIN",
    "CLASS_PROMPTS",
    "EVIDENCE_COLUMNS",
    "SAM3_ACCESS_HELP",
    "load_sam3",
    "sam3_download",
    "assign_classes",
    "class_evidence",
    "class_scores",
    "fold_class",
    "evidence_class",
]

SAM3_REPO = "facebook/sam3"
#: Detection prompts: text prompt -> road class (vehicles get their class from the evidence prompts).
SAM3_PROMPTS = {"vehicle": "car", "person": "person"}
#: Class-evidence prompts: text prompt -> road class.
SAM3_EVIDENCE = {
    "car": "car",
    "van": "car",
    "suv": "car",
    "bus": "bus",
    "truck": "truck",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
}
#: Score threshold of the detection prompts (class score times presence score) when ``DetectorConfig.conf`` is ``None``.
SAM3_CONF = 0.40
#: Lowest score at which a class-evidence detection votes, per prompt.
SAM3_EVIDENCE_MIN = {
    "car": 0.35,
    "van": 0.35,
    "suv": 0.35,
    "bus": 0.40,
    "truck": 0.40,
    "bike": 0.40,
    "motorcycle": 0.40,
}
#: Box IoU above which an evidence detection votes for the class of a detection.
SAM3_EVIDENCE_IOU = 0.5
#: Lowest ``bike`` score for the two-wheeler flags ``bike_cover`` and ``bike_ride``.
SAM3_BIKE_FLAG_MIN = 0.5
#: ``bike_cover``: intersection over the bike box above which a bike lies within a detection.
SAM3_BIKE_COVER = 0.6
#: ``bike_ride``: enlargement of the bike box on each side, and intersection over the person box.
SAM3_RIDE_EXPAND = 0.15
SAM3_RIDE_OVERLAP = 0.3
#: Intersection over the smaller box above which a two-wheeler evidence detection belongs to a vehicle detection.
SAM3_CLAIM_OVERLAP = 0.6
#: The class slots of the vote, in tie-break order. The two-wheeler slot ``bike`` takes the larger
#: of the ``bike`` and ``motorcycle`` evidence of a detection; ``motorcycle`` against ``bike`` then
#: splits two-wheelers into motorcycles and bicycles.
CLASS_PROMPTS = ("bike", "bus", "truck", "car", "van", "suv")
#: Columns of the per-detection evidence vector.
EVIDENCE_COLUMNS = (*SAM3_EVIDENCE, "bike_cover", "bike_ride")
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
        # the vehicle prompt always runs (vehicle classes come from the evidence); person only when requested
        detect = {p: c for p, c in SAM3_PROMPTS.items() if c != "person" or "person" in cfg.classes}
        evidence = {p: c for p, c in SAM3_EVIDENCE.items() if c in cfg.classes}
        self.prompts = list(detect) + list(evidence)
        self.n_detect = len(detect)
        self._thr = [self.conf] * len(detect) + [SAM3_EVIDENCE_MIN[p] for p in evidence]
        self.evidence_names = EVIDENCE_COLUMNS
        self._cls = np.array([ROAD_CLASSES.index(c) for c in list(detect.values()) + list(evidence.values())], np.int16)
        self.classes = sorted({*detect.values(), *evidence.values()}, key=ROAD_CLASSES.index)
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
        prompt_of: list[list[np.ndarray]] = [[] for _ in images]
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
                    thr = torch.as_tensor([self._thr[p] for _, p in chunk], device=score.device).view(-1, 1)
                    keep = score > thr
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
                        prompt_of[i].append(np.full(len(q), p, np.int16))
                        sel = torch.as_tensor(q, device=self.device, dtype=torch.long)
                        maps[i].append(res.pred_masks[k].index_select(0, sel).float().sigmoid())
                    del res
        out: list[DetectorOutput] = []
        k_ev = len(EVIDENCE_COLUMNS)

        def empty() -> DetectorOutput:
            z = np.zeros(0, np.float32)
            return (np.zeros((0, 4), np.float32), z, np.zeros(0, np.int16), None, [], np.zeros((0, k_ev), np.float32))

        for i in range(n_img):
            if not boxes[i]:
                out.append(empty())
                continue
            xyxy, score, cls = np.concatenate(boxes[i]), np.concatenate(scores[i]), np.concatenate(classes[i])
            prm = np.concatenate(prompt_of[i])
            mp = torch.cat(maps[i])
            person = self.class_names.index("person")
            rows, score, cls, ev = assign_classes(xyxy, score, cls, prm, self.n_detect, self.prompts, self.conf)
            xyxy = xyxy[rows]
            mp = mp.index_select(0, torch.as_tensor(rows, device=mp.device, dtype=torch.long))
            if not len(xyxy):
                out.append(empty())
                continue
            # class floors apply to what a prompt detected: a vehicle-prompt detection is kept at the
            # vehicle threshold whatever class its evidence suggests (two-wheelers included)
            vehicle_prompt = (prm[rows] < self.n_detect) & (cls != person)
            floor_cls = np.where(vehicle_prompt, self.class_names.index("car"), cls).astype(np.int16)
            idx = prefilter(xyxy, score, floor_cls, w, self.cfg, self.class_names)
            xyxy, score, cls, ev = xyxy[idx], score[idx], cls[idx], ev[idx]
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
            out.append((xyxy[idx], score[idx], cls[idx], None, [masks[int(j)] for j in idx], ev[idx]))
        return out

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "model": self.repo,
            "prompts": self.prompts[: self.n_detect],
            "evidence_prompts": self.prompts[self.n_detect :],
            "evidence_columns": list(EVIDENCE_COLUMNS),
            "device": str(self.device),
            "imgsz": list(self.imgsz),
            "conf": self.conf,
            "half": self.half,
        }


def _pairwise(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersection ``(len(a), len(b))`` and the areas of the boxes ``a`` and ``b``."""
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    area_a = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-9)
    area_b = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)
    return iw * ih, area_a, area_b


def class_evidence(xyxy: np.ndarray, ev_xyxy: np.ndarray, ev_score: np.ndarray, ev_prompt: Sequence[str]) -> np.ndarray:
    """Evidence vectors ``(len(xyxy), len(EVIDENCE_COLUMNS))`` of the boxes ``xyxy`` from the evidence
    detections of the same frame (boxes ``ev_xyxy``, scores ``ev_score``, prompt names ``ev_prompt``)."""
    out = np.zeros((len(xyxy), len(EVIDENCE_COLUMNS)), np.float32)
    if not len(xyxy) or not len(ev_xyxy):
        return out
    a = np.asarray(xyxy, np.float64)
    b = np.asarray(ev_xyxy, np.float64)
    sc = np.asarray(ev_score, np.float64)
    prm = np.asarray(ev_prompt)
    inter, area_a, area_b = _pairwise(a, b)
    iou = inter / (area_a[:, None] + area_b[None, :] - inter)
    for j, name in enumerate(SAM3_EVIDENCE):
        sel = (prm == name) & (sc >= SAM3_EVIDENCE_MIN[name])
        if sel.any():
            hit = iou[:, sel] > SAM3_EVIDENCE_IOU
            out[:, j] = np.where(hit, sc[sel][None, :], 0.0).max(axis=1)
    bikes = (prm == "bike") & (sc >= SAM3_BIKE_FLAG_MIN)
    if bikes.any():
        bb, bs = b[bikes], sc[bikes]
        cover = inter[:, bikes] / area_b[bikes][None, :] > SAM3_BIKE_COVER
        out[:, EVIDENCE_COLUMNS.index("bike_cover")] = np.where(cover, bs[None, :], 0.0).max(axis=1)
        ex = SAM3_RIDE_EXPAND * (bb[:, 2:] - bb[:, :2])
        big = np.column_stack([bb[:, :2] - ex, bb[:, 2:] + ex])
        inter_r, area_p, _ = _pairwise(a, big)
        ride = inter_r / area_p[:, None] > SAM3_RIDE_OVERLAP
        out[:, EVIDENCE_COLUMNS.index("bike_ride")] = np.where(ride, bs[None, :], 0.0).max(axis=1)
    return out


def class_scores(ev: np.ndarray) -> np.ndarray:
    """Scores of the class slots :data:`CLASS_PROMPTS` from evidence vectors ``(..., len(EVIDENCE_COLUMNS))``."""
    ev = np.asarray(ev, np.float64)
    col = {n: k for k, n in enumerate(EVIDENCE_COLUMNS)}
    two = np.maximum(ev[..., col["bike"]], ev[..., col["motorcycle"]])
    return np.stack([two if p == "bike" else ev[..., col[p]] for p in CLASS_PROMPTS], axis=-1)


def fold_class(scores: np.ndarray, bike: float, motorcycle: float) -> str | None:
    """Road class from class-slot scores (or weighted shares): the argmax over :data:`CLASS_PROMPTS`
    (first slot on ties), with ``car``, ``van`` and ``suv`` folded into ``car`` and the two-wheeler slot
    a ``motorcycle`` when ``motorcycle > bike``, else a ``bicycle``; ``None`` when all scores are 0."""
    k = int(np.argmax(scores))
    if float(scores[k]) <= 0.0:
        return None
    if CLASS_PROMPTS[k] == "bike":
        return "motorcycle" if motorcycle > bike else "bicycle"
    return SAM3_EVIDENCE[CLASS_PROMPTS[k]]


def evidence_class(ev: np.ndarray) -> str | None:
    """Road class from one evidence vector (:func:`fold_class` of its :func:`class_scores`)."""
    col = {n: k for k, n in enumerate(EVIDENCE_COLUMNS)}
    return fold_class(class_scores(ev), float(ev[col["bike"]]), float(ev[col["motorcycle"]]))


def assign_classes(
    xyxy: np.ndarray,
    score: np.ndarray,
    cls: np.ndarray,
    prompt: np.ndarray,
    n_detect: int,
    prompts: Sequence[str],
    conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows, scores, classes and evidence vectors of one frame's detections.

    The rows kept are the detection-prompt results and the two-wheeler evidence
    detections that no vehicle detection covers (intersection over the smaller
    box below 0.6) and that score at least ``conf``. Each row's class is the
    single-frame rule :func:`evidence_class`; vehicle detections without
    evidence are cars.
    """
    names = list(ROAD_CLASSES)
    det = np.flatnonzero(prompt < n_detect)
    ev = np.flatnonzero(prompt >= n_detect)
    ev_prompt = [prompts[int(prompt[k])] for k in ev]
    rows = list(det)
    two = [k for k, p in zip(ev, ev_prompt) if p in ("bike", "motorcycle") and score[k] >= conf]
    if two:
        veh = [r for r in det if names[int(cls[r])] != "person"]
        claimed = np.zeros(len(two), bool)
        if veh:
            inter, area_t, area_v = _pairwise(xyxy[two].astype(np.float64), xyxy[veh].astype(np.float64))
            claimed = (inter / np.minimum(area_t[:, None], area_v[None, :]) >= SAM3_CLAIM_OVERLAP).any(axis=1)
        rows.extend(k for k, c in zip(two, claimed) if not c)
    rows_a = np.asarray(rows, np.int64)
    evidence = class_evidence(xyxy[rows_a], xyxy[ev], score[ev], ev_prompt)
    out_cls = np.empty(len(rows_a), np.int16)
    col = {n: k for k, n in enumerate(EVIDENCE_COLUMNS)}
    for i, r in enumerate(rows_a):
        if prompt[r] < n_detect and names[int(cls[r])] == "person":
            out_cls[i] = cls[r]
        elif prompt[r] >= n_detect:  # an unclaimed two-wheeler
            moto = evidence[i, col["motorcycle"]] > evidence[i, col["bike"]]
            out_cls[i] = names.index("motorcycle" if moto else "bicycle")
        else:
            out_cls[i] = names.index(evidence_class(evidence[i]) or "car")
    return rows_a, score[rows_a].astype(np.float32), out_cls, evidence

"""Core data containers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROAD_CLASSES

__all__ = ["FrameDetections", "DetectionSet", "MaskStore", "crop_mask", "crop_window"]


MaskCrop = tuple[int, int, np.ndarray]  # (x0, y0, bool mask of shape (h, w)) in full-resolution pixels


def crop_window(xyxy: np.ndarray, shape: tuple[int, int], pad: int = 2) -> tuple[int, int, int, int]:
    """Pixel window ``(x0, y0, x1, y1)`` (end exclusive) of a box plus ``pad``, clipped to ``shape = (h, w)``."""
    h, w = shape
    x0 = int(max(0, np.floor(xyxy[0]) - pad))
    y0 = int(max(0, np.floor(xyxy[1]) - pad))
    x1 = int(min(w, np.ceil(xyxy[2]) + pad + 1))
    y1 = int(min(h, np.ceil(xyxy[3]) + pad + 1))
    return x0, y0, x1, y1


def crop_mask(mask: np.ndarray, xyxy: np.ndarray, pad: int = 2) -> MaskCrop | None:
    """Crop a full-frame boolean instance mask to its box (plus ``pad`` pixels)."""
    x0, y0, x1, y1 = crop_window(xyxy, mask.shape, pad)
    if x1 <= x0 or y1 <= y0:
        return None
    m = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=bool)
    return (x0, y0, m) if m.any() else None


@dataclass
class MaskStore:
    """Instance masks of a :class:`DetectionSet`, cropped to their boxes and bit-packed.

    Mask ``i`` covers the full-resolution image rectangle starting at
    ``origin[i] = (x0, y0)`` with ``shape[i] = (h, w)``. Detections without a
    mask (box-only detectors, or a mask that was empty) have ``offset[i] = -1``.
    """

    bits: np.ndarray  # (B,) uint8, np.packbits of each crop, concatenated
    offset: np.ndarray  # (M,) int64 byte offset into ``bits``; -1 = no mask
    origin: np.ndarray  # (M, 2) int32 (x0, y0)
    shape: np.ndarray  # (M, 2) int32 (h, w)

    def __len__(self) -> int:
        return len(self.offset)

    @classmethod
    def from_crops(cls, crops: list[MaskCrop | None]) -> MaskStore:
        chunks, offset, origin, shape = [], [], [], []
        pos = 0
        for c in crops:
            if c is None:
                offset.append(-1)
                origin.append((0, 0))
                shape.append((0, 0))
                continue
            x0, y0, m = c
            b = np.packbits(np.asarray(m, bool).ravel())
            chunks.append(b)
            offset.append(pos)
            origin.append((int(x0), int(y0)))
            shape.append(m.shape)
            pos += len(b)
        return cls(
            np.concatenate(chunks).astype(np.uint8) if chunks else np.zeros(0, np.uint8),
            np.asarray(offset, np.int64),
            np.asarray(origin, np.int32).reshape(-1, 2),
            np.asarray(shape, np.int32).reshape(-1, 2),
        )

    def has(self, i: int) -> bool:
        return bool(self.offset[i] >= 0)

    def crop(self, i: int) -> MaskCrop | None:
        """``(x0, y0, mask)`` of detection ``i``, or ``None``."""
        o = int(self.offset[i])
        if o < 0:
            return None
        h, w = (int(v) for v in self.shape[i])
        n = h * w
        m = np.unpackbits(self.bits[o : o + (n + 7) // 8], count=n).astype(bool).reshape(h, w)
        return int(self.origin[i, 0]), int(self.origin[i, 1]), m

    def full(self, i: int, image_size: tuple[int, int]) -> np.ndarray | None:
        """Full-frame boolean mask of detection ``i`` (``image_size`` is ``(width, height)``)."""
        c = self.crop(i)
        if c is None:
            return None
        x0, y0, m = c
        out = np.zeros((image_size[1], image_size[0]), bool)
        h = min(m.shape[0], image_size[1] - y0)
        w = min(m.shape[1], image_size[0] - x0)
        out[y0 : y0 + h, x0 : x0 + w] = m[:h, :w]
        return out

    def crops(self, idx: np.ndarray) -> list[MaskCrop | None]:
        return [self.crop(int(i)) for i in idx]

    def subset(self, idx: np.ndarray) -> MaskStore:
        return MaskStore.from_crops(self.crops(np.asarray(idx).reshape(-1)))


@dataclass
class FrameDetections:
    """Detections in a single frame.

    ``cuboid`` holds the eight projected cuboid corners ``(N, 8, 3)`` as
    ``x, y, confidence`` when the detector provides them (UrbanOmniDetect-2):
    corners 0-3 touch the ground, 4-7 are the roof, and corners 0 and 3 form the
    front face. ``masks`` holds one instance-mask crop per detection
    (see :func:`crop_mask`) when the detector segments, else ``None``.
    ``evidence`` holds per-detection class evidence ``(N, E)`` when the detector
    provides it (SAM 3: the best matching score of each class prompt, see
    :data:`tachymetry.detect.sam3.EVIDENCE_COLUMNS`), else ``None``.
    """

    frame: int
    time: float
    xyxy: np.ndarray
    score: np.ndarray
    cls: np.ndarray
    cuboid: np.ndarray | None = None
    masks: list[MaskCrop | None] | None = None
    evidence: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.score)

    @classmethod
    def empty(cls, frame: int, time: float, with_cuboid: bool = False) -> FrameDetections:
        return cls(
            frame,
            time,
            np.zeros((0, 4), np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.int16),
            np.zeros((0, 8, 3), np.float32) if with_cuboid else None,
        )

    def subset(self, keep: np.ndarray) -> FrameDetections:
        keep = np.asarray(keep)
        idx = np.flatnonzero(keep) if keep.dtype == bool else keep.reshape(-1).astype(np.int64)
        return FrameDetections(
            self.frame,
            self.time,
            self.xyxy[idx],
            self.score[idx],
            self.cls[idx],
            None if self.cuboid is None else self.cuboid[idx],
            None if self.masks is None else [self.masks[int(i)] for i in idx],
            None if self.evidence is None else self.evidence[idx],
        )


@dataclass
class DetectionSet:
    """All detections of a video in columnar form (one row per detection)."""

    frame: np.ndarray  # (M,) int
    time: np.ndarray  # (M,) float seconds
    xyxy: np.ndarray  # (M, 4) float32
    score: np.ndarray  # (M,) float32
    cls: np.ndarray  # (M,) int16 index into class_names
    cuboid: np.ndarray | None = None  # (M, 8, 3) float32
    class_names: tuple[str, ...] = ROAD_CLASSES
    image_size: tuple[int, int] = (0, 0)
    fps: float = 0.0
    processed_frames: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    meta: dict[str, Any] = field(default_factory=dict)
    masks: MaskStore | None = None  #: instance masks when the detector segments
    evidence: np.ndarray | None = None  #: (M, E) float32 class evidence when the detector provides it
    evidence_names: tuple[str, ...] = ()  #: column names of ``evidence``

    def __len__(self) -> int:
        return len(self.score)

    def evidence_column(self, name: str) -> np.ndarray | None:
        """One column of :attr:`evidence` (zeros where a row has none), or ``None`` without it."""
        if self.evidence is None or name not in self.evidence_names:
            return None
        return self.evidence[:, self.evidence_names.index(name)]

    @classmethod
    def from_frames(
        cls,
        frames: list[FrameDetections],
        class_names: tuple[str, ...],
        image_size: tuple[int, int],
        fps: float,
        meta: dict[str, Any] | None = None,
        evidence_names: tuple[str, ...] = (),
    ) -> DetectionSet:
        has_cub = any(f.cuboid is not None for f in frames)
        has_masks = any(f.masks is not None for f in frames)
        has_ev = bool(evidence_names) and any(f.evidence is not None for f in frames)
        rows = [f for f in frames if len(f)]
        if rows:
            frame = np.concatenate([np.full(len(f), f.frame, np.int64) for f in rows])
            time = np.concatenate([np.full(len(f), f.time, np.float64) for f in rows])
            xyxy = np.concatenate([f.xyxy for f in rows]).astype(np.float32)
            score = np.concatenate([f.score for f in rows]).astype(np.float32)
            clsv = np.concatenate([f.cls for f in rows]).astype(np.int16)
            cub = (
                np.concatenate(
                    [f.cuboid if f.cuboid is not None else np.full((len(f), 8, 3), np.nan, np.float32) for f in rows]
                ).astype(np.float32)
                if has_cub
                else None
            )
        else:
            frame = np.zeros(0, np.int64)
            time = np.zeros(0)
            xyxy = np.zeros((0, 4), np.float32)
            score = np.zeros(0, np.float32)
            clsv = np.zeros(0, np.int16)
            cub = np.zeros((0, 8, 3), np.float32) if has_cub else None
        processed = np.array(sorted({f.frame for f in frames}), np.int64)
        masks = None
        if has_masks:
            crops: list[MaskCrop | None] = []
            for f in rows:
                crops.extend(f.masks if f.masks is not None else [None] * len(f))
            masks = MaskStore.from_crops(crops)
        evidence = None
        if has_ev:
            k = len(evidence_names)
            evidence = (
                np.concatenate(
                    [
                        f.evidence.astype(np.float32) if f.evidence is not None else np.zeros((len(f), k), np.float32)
                        for f in rows
                    ]
                )
                if rows
                else np.zeros((0, k), np.float32)
            )
        return cls(
            frame,
            time,
            xyxy,
            score,
            clsv,
            cub,
            tuple(class_names),
            tuple(image_size),
            float(fps),
            processed,
            dict(meta or {}),
            masks,
            evidence,
            tuple(evidence_names) if has_ev else (),
        )

    def frames(self) -> Iterator[FrameDetections]:
        """Iterate per processed frame (including frames without detections), in order."""
        order = np.argsort(self.frame, kind="stable")
        fr = self.frame[order]
        bounds = np.searchsorted(fr, self.processed_frames)
        ends = np.searchsorted(fr, self.processed_frames, side="right")
        times = {}
        if len(self.frame):
            times = dict(zip(self.frame.tolist(), self.time.tolist()))
        for f, a, b in zip(self.processed_frames, bounds, ends):
            idx = order[a:b]
            t = times.get(int(f), float(f) / self.fps if self.fps else 0.0)
            yield FrameDetections(
                int(f),
                t,
                self.xyxy[idx],
                self.score[idx],
                self.cls[idx],
                None if self.cuboid is None else self.cuboid[idx],
                None if self.masks is None else self.masks.crops(idx),
                None if self.evidence is None else self.evidence[idx],
            )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = dict(
            frame=self.frame,
            time=self.time,
            xyxy=self.xyxy,
            score=self.score,
            cls=self.cls,
            processed_frames=self.processed_frames,
            class_names=np.array(self.class_names),
            image_size=np.array(self.image_size),
            fps=np.array(self.fps),
            meta=np.array(json.dumps(self.meta, default=str)),
        )
        if self.cuboid is not None:
            arrays["cuboid"] = self.cuboid
        if self.masks is not None:
            arrays.update(
                mask_bits=self.masks.bits,
                mask_offset=self.masks.offset,
                mask_origin=self.masks.origin,
                mask_shape=self.masks.shape,
            )
        if self.evidence is not None:
            arrays.update(evidence=self.evidence.astype(np.float16), evidence_names=np.array(self.evidence_names))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> DetectionSet:
        z = np.load(path, allow_pickle=False)
        return cls(
            z["frame"],
            z["time"],
            z["xyxy"],
            z["score"],
            z["cls"],
            z["cuboid"] if "cuboid" in z else None,
            tuple(str(s) for s in z["class_names"]),
            tuple(int(v) for v in z["image_size"]),
            float(z["fps"]),
            z["processed_frames"],
            json.loads(str(z["meta"])) if "meta" in z else {},
            MaskStore(z["mask_bits"], z["mask_offset"], z["mask_origin"], z["mask_shape"])
            if "mask_bits" in z
            else None,
            z["evidence"].astype(np.float32) if "evidence" in z else None,
            tuple(str(s) for s in z["evidence_names"]) if "evidence_names" in z else (),
        )

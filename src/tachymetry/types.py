"""Core data containers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ROAD_CLASSES

__all__ = ["FrameDetections", "DetectionSet"]


@dataclass
class FrameDetections:
    """Detections in a single frame.

    ``cuboid`` holds the eight projected cuboid corners ``(N, 8, 3)`` as
    ``x, y, confidence`` when the detector provides them (UrbanOmniDetect-2):
    corners 0-3 touch the ground, 4-7 are the roof, and corners 0 and 3 form the
    front face.
    """

    frame: int
    time: float
    xyxy: np.ndarray
    score: np.ndarray
    cls: np.ndarray
    cuboid: np.ndarray | None = None

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
        return FrameDetections(
            self.frame,
            self.time,
            self.xyxy[keep],
            self.score[keep],
            self.cls[keep],
            None if self.cuboid is None else self.cuboid[keep],
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

    def __len__(self) -> int:
        return len(self.score)

    @classmethod
    def from_frames(
        cls,
        frames: list[FrameDetections],
        class_names: tuple[str, ...],
        image_size: tuple[int, int],
        fps: float,
        meta: dict[str, Any] | None = None,
    ) -> DetectionSet:
        has_cub = any(f.cuboid is not None for f in frames)
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
        )
        if self.cuboid is not None:
            arrays["cuboid"] = self.cuboid
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
        )

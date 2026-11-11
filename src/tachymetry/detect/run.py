"""Run a detector over a video."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..detections import DetectionSet, FrameDetections
from ..io.video import VideoSource, prefetch
from .backends import Detector

__all__ = ["detect_video"]


def detect_video(
    source: VideoSource,
    detector: Detector,
    rate_hz: float = 15.0,
    batch: int = 8,
    progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
) -> DetectionSet:
    """Detect every ``stride``-th frame (``stride`` from ``rate_hz``) with batched inference."""
    stride = source.stride_for(rate_hz)
    total = source.frame_count if source.frame_count > 0 else 0
    expected = (total + stride - 1) // stride if total else 0
    if max_frames is not None:
        expected = min(expected, max_frames) if expected else max_frames
    frames: list[FrameDetections] = []
    buf: list[tuple[int, float, np.ndarray]] = []
    t0 = time.time()

    def flush() -> None:
        if not buf:
            return
        outs = detector([b[2] for b in buf])
        for (idx, t, _), res in zip(buf, outs):
            xyxy, score, cls, cub = res[:4]
            masks = res[4] if len(res) > 4 else None  # detectors without masks may return four elements
            evidence = res[5] if len(res) > 5 else None  # class evidence (SAM 3)
            frames.append(
                FrameDetections(idx, t, xyxy, score, cls, cub, None if masks is None else list(masks), evidence)
            )
        buf.clear()
        if progress is not None:
            progress(len(frames), expected)

    for n, (idx, t, img) in enumerate(prefetch(source.frames(stride=stride), depth=2 * batch)):
        if max_frames is not None and n >= max_frames:
            break
        buf.append((idx, t, img))
        if len(buf) >= batch:
            flush()
    flush()
    meta = {
        "detector": detector.describe(),
        "stride": stride,
        "rate_hz": source.fps / stride,
        "seconds": round(time.time() - t0, 2),
        "source": str(source.path),
    }
    evidence_names = tuple(getattr(detector, "evidence_names", ()) or ())
    return DetectionSet.from_frames(frames, detector.class_names, source.size, source.fps, meta, evidence_names)

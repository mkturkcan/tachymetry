"""Video input.

:class:`VideoSource` reads video files with OpenCV, or folders of images. Files
that OpenCV opens but cannot decode (for example AV1 in builds without that
decoder), or that report an implausible frame rate, are read through the FFmpeg
binary shipped with ``imageio-ffmpeg``. Frame times are derived from the frame
index and the container frame rate, which suits constant-frame-rate traffic
footage and avoids OpenCV's unreliable timestamp queries.
"""

from __future__ import annotations

import os
import queue
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

__all__ = ["VideoSource", "VIDEO_EXTENSIONS", "IMAGE_EXTENSIONS", "find_videos", "prefetch"]

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ts",
    ".mts",
    ".webm",
    ".ogv",
    ".ogg",
    ".wmv",
    ".flv",
    ".h264",
    ".hevc",
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def find_videos(inputs: Sequence[str | Path], recursive: bool = True) -> list[Path]:
    """Expand files, directories and glob patterns into a sorted list of video sources.

    A directory that contains only images is treated as one image-sequence source.
    """
    out: list[Path] = []
    for item in inputs:
        s = str(item)
        p = Path(s)
        if any(ch in s for ch in "*?["):
            import glob

            hits = [Path(q) for q in glob.glob(s, recursive=True)]
            out.extend(
                sorted((q for q in hits if q.is_file() and q.suffix.lower() in VIDEO_EXTENSIONS), key=_natural_key)
            )
        elif p.is_dir():
            vids = sorted(
                (
                    q
                    for q in (p.rglob("*") if recursive else p.iterdir())
                    if q.is_file() and q.suffix.lower() in VIDEO_EXTENSIONS
                ),
                key=_natural_key,
            )
            if vids:
                out.extend(vids)
            elif any(q.suffix.lower() in IMAGE_EXTENSIONS for q in p.iterdir()):
                out.append(p)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"no such file, directory or pattern: {s}")
    seen: set[str] = set()
    uniq = []
    for p in out:
        r = os.path.abspath(p)
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


class VideoSource:
    """Random-access and sequential reader for a video file or an image folder."""

    def __init__(self, path: str | Path, fps: float | None = None, backend: str = "auto"):
        import cv2

        self.path = Path(path)
        self._images: list[Path] | None = None
        self._cap = None
        self._ff: _FFmpegReader | None = None
        self.backend = "images"
        if self.path.is_dir():
            self._images = sorted(
                (q for q in self.path.iterdir() if q.suffix.lower() in IMAGE_EXTENSIONS), key=_natural_key
            )
            if not self._images:
                raise ValueError(f"{self.path} contains no images")
            first = cv2.imread(str(self._images[0]))
            if first is None:
                raise ValueError(f"cannot read {self._images[0]}")
            self.height, self.width = first.shape[:2]
            self.fps = float(fps or 10.0)
            self.frame_count = len(self._images)
        else:
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            cap = cv2.VideoCapture(str(self.path)) if backend in ("auto", "opencv") else None
            usable = False
            if cap is not None and cap.isOpened():
                f = cap.get(cv2.CAP_PROP_FPS)
                ok, _ = cap.read()
                # OpenCV builds without some decoders (e.g. AV1) open the file but return no frames, and some
                # containers report a nonsensical rate; both are handled by the FFmpeg reader.
                usable = ok and bool(f) and np.isfinite(f) and 1.0 <= f <= 1000.0
                if usable:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._cap = cap
                    self.backend = "opencv"
                    self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.fps = float(fps or f)
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.frame_count = n if n > 0 else -1
                else:
                    cap.release()
            if not usable:
                if backend == "opencv":
                    raise ValueError(f"OpenCV cannot decode {self.path}")
                self._ff = _FFmpegReader(self.path)
                self.backend = "ffmpeg"
                self.width, self.height = self._ff.width, self._ff.height
                self.fps = float(fps or self._ff.fps)
                self.frame_count = self._ff.frame_count(self.fps)
        self._pos = 0

    # ------------------------------------------------------------ metadata
    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.frame_count > 0 else float("nan")

    def stride_for(self, rate_hz: float | None) -> int:
        """Frame stride that analyses about ``rate_hz`` frames per second (``0``/``None`` = every frame)."""
        if not rate_hz or rate_hz <= 0 or rate_hz >= self.fps:
            return 1
        return max(1, int(round(self.fps / rate_hz)))

    def time_of(self, index: int) -> float:
        return index / self.fps

    # ---------------------------------------------------------------- read
    def read(self, index: int) -> np.ndarray | None:
        """Random access to frame ``index`` (seeks to the frame)."""
        import cv2

        if self._images is not None:
            if not 0 <= index < len(self._images):
                return None
            return cv2.imread(str(self._images[index]))
        if self._ff is not None:
            return next(self._ff.iterate(index / self.fps), None)
        assert self._cap is not None
        if index != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self._cap.read()
        self._pos = index + 1 if ok else self._pos
        return img if ok else None

    def frames(
        self, stride: int = 1, start: int = 0, end: int | None = None
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """Sequential ``(index, time, image)`` for every ``stride``-th frame (decodes, never seeks per frame)."""
        import cv2

        if self._images is not None:
            stop = len(self._images) if end is None else min(end, len(self._images))
            for i in range(start, stop, stride):
                img = cv2.imread(str(self._images[i]))
                if img is not None:
                    yield i, self.time_of(i), img
            return
        if self._ff is not None:
            for k, img in enumerate(self._ff.iterate(start / self.fps)):
                i = start + k
                if end is not None and i >= end:
                    break
                if k % stride == 0:
                    yield i, self.time_of(i), img
            return
        assert self._cap is not None
        if start != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            self._pos = start
        i = start
        while end is None or i < end:
            if (i - start) % stride == 0:
                ok, img = self._cap.read()
                if not ok:
                    break
                self._pos = i + 1
                yield i, self.time_of(i), img
            else:
                if not self._cap.grab():
                    break
                self._pos = i + 1
            i += 1

    def sample_frames(self, n: int, max_side: int | None = None) -> list[np.ndarray]:
        """``n`` frames evenly spread over the video (for background estimation)."""
        import cv2

        total = self.frame_count if self.frame_count > 0 else 3000
        idx = np.unique(np.linspace(0, max(total - 1, 0), n).round().astype(int))
        out = []
        for i in idx:
            img = self.read(int(i))
            if img is None:
                continue
            if max_side:
                s = max_side / max(img.shape[:2])
                if s < 1:
                    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            out.append(img)
        return out

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"VideoSource({self.path.name}, {self.width}x{self.height} @ {self.fps:.2f} fps, {self.frame_count} frames)"
        )


class _FFmpegReader:
    """Decoder using the FFmpeg binary bundled with ``imageio-ffmpeg`` (AV1 through dav1d, unusual containers)."""

    def __init__(self, path: Path):
        import imageio_ffmpeg

        self.path = path
        gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="bgr24")
        try:
            meta = next(gen)
        finally:
            gen.close()
        self.width, self.height = (int(v) for v in meta["size"])
        rate = meta.get("fps") or 30.0
        self.fps = float(rate) if 1.0 <= float(rate) <= 1000.0 else 30.0
        self.duration = float(meta.get("duration") or 0.0)

    def frame_count(self, fps: float) -> int:
        return int(round(self.duration * fps)) if self.duration > 0 else -1

    def iterate(self, start_s: float = 0.0) -> Iterator[np.ndarray]:
        import imageio_ffmpeg

        params = ["-ss", f"{start_s:.4f}"] if start_s > 0 else []
        gen = imageio_ffmpeg.read_frames(str(self.path), pix_fmt="bgr24", input_params=params)
        try:
            next(gen)
            for buf in gen:
                yield np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3).copy()
        finally:
            gen.close()


def prefetch(iterator: Iterator, depth: int = 16) -> Iterator:
    """Run an iterator in a background thread (overlaps decoding with inference)."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()
    error: list[BaseException] = []

    def worker() -> None:
        try:
            for item in iterator:
                q.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            error.append(exc)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item
    t.join()
    if error:
        raise error[0]

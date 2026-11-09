"""Video output: H.264 through FFmpeg (bundled by ``imageio-ffmpeg``), OpenCV as a fallback."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["VideoWriter", "ffmpeg_executable"]


def ffmpeg_executable() -> str | None:
    """Path to an FFmpeg binary: ``imageio-ffmpeg``'s bundled build, else one on ``PATH``."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class VideoWriter:
    """Write BGR frames to an H.264 MP4 (``yuv420p``, ``+faststart``) that plays everywhere.

    Frame sizes are padded to even dimensions as H.264 requires.
    """

    def __init__(self, path: str | Path, fps: float, crf: int = 18, preset: str = "medium"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = preset
        self._gen: Any = None
        self._cv: Any = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

    def _open(self, w: int, h: int) -> None:
        self._size = (w, h)
        exe = ffmpeg_executable()
        if exe is not None:
            try:
                import imageio_ffmpeg

                self._gen = imageio_ffmpeg.write_frames(
                    str(self.path),
                    (w, h),
                    fps=self.fps,
                    codec="libx264",
                    pix_fmt_in="bgr24",
                    pix_fmt_out="yuv420p",
                    quality=None,
                    macro_block_size=2,
                    ffmpeg_log_level="error",
                    output_params=[
                        "-crf",
                        str(self.crf),
                        "-preset",
                        self.preset,
                        "-movflags",
                        "+faststart",
                        "-colorspace",
                        "bt709",
                        "-color_primaries",
                        "bt709",
                        "-color_trc",
                        "bt709",
                    ],
                )
                self._gen.send(None)
                return
            except Exception:
                self._gen = None
        import cv2

        self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not self._cv.isOpened():
            raise RuntimeError(f"cannot open a video writer for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if w % 2 or h % 2:
            frame = np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
            h, w = frame.shape[:2]
        if self._size is None:
            self._open(w, h)
        elif (w, h) != self._size:
            raise ValueError(f"frame size changed from {self._size} to {(w, h)}")
        if self._gen is not None:
            self._gen.send(np.ascontiguousarray(frame))
        else:
            self._cv.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

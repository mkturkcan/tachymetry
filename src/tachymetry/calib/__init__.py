"""Camera calibration: manual correspondences, automatic methods and lens rectification.

Methods
-------
``manual``    :func:`calibrate_manual` from point (and line) correspondences
``lines``     :func:`calibrate_from_lines` from road-marking and vertical vanishing points
``cuboids``   :func:`calibrate_from_cuboids` from learned vehicle cuboids (UrbanOmniDetect-2)
``auto``      :func:`calibrate_auto` fuses the available cues and ranks the candidates
``depth``     monocular metric depth plus road segmentation (``pip install "tachymetry[depth]"``)
``geocalib``  single-image gravity and focal length (``pip install "tachymetry[geocalib]"``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .auto import calibrate_auto
from .correspondences import Correspondences, latlon_to_local, local_to_latlon
from .from_cuboids import calibrate_from_cuboids
from .from_lines import ScaleCue, calibrate_from_lines
from .lines import background_image, estimate_lens_plumbline, estimate_lens_vp_consistency
from .manual import calibrate_manual, fit_camera
from .preview import compare, draw_ground_grid, preview, warp_to_bev
from .result import CalibrationResult

__all__ = [
    "calibrate_source",
    "calibrate_auto",
    "calibrate_manual",
    "calibrate_from_lines",
    "calibrate_from_cuboids",
    "fit_camera",
    "Correspondences",
    "CalibrationResult",
    "ScaleCue",
    "background_image",
    "preview",
    "compare",
    "draw_ground_grid",
    "warp_to_bev",
    "estimate_lens_plumbline",
    "estimate_lens_vp_consistency",
    "latlon_to_local",
    "local_to_latlon",
]


def _load_image_or_background(source: str | Path, n: int = 41) -> tuple[np.ndarray, Any]:
    import cv2

    from ..io.video import IMAGE_EXTENSIONS, VideoSource

    p = Path(source)
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"cannot read image {p}")
        return img, None
    src = VideoSource(p)
    return background_image(src.sample_frames(n)), src


def calibrate_source(
    source: str | Path,
    method: str = "auto",
    points: str | Path | None = None,
    scale: Any = None,
    lens: Any = None,
    detector: str = "uod2-x",
    device: str = "auto",
    seconds: float = 90.0,
    rate_hz: float = 5.0,
    **kwargs: Any,
) -> CalibrationResult:
    """Calibrate the camera of a video (or a single image).

    For ``auto`` and ``cuboids`` the detector runs on up to ``seconds`` of video
    at ``rate_hz`` to collect vehicle footprints.
    """
    img, src = _load_image_or_background(source)
    size = (img.shape[1], img.shape[0])
    if method == "manual":
        if points is None:
            raise ValueError("manual calibration needs points=<correspondence file>")
        corr = Correspondences.load(points)
        return calibrate_manual(corr, corr.image_size or size, lens=lens or "auto", **kwargs)
    if method == "lines":
        return calibrate_from_lines(img, scale=scale, lens=lens if lens is not None else "auto", **kwargs)
    if method == "depth":
        from .depth import calibrate_from_depth

        return calibrate_from_depth(img, **kwargs)
    if method == "geocalib":
        from .geocalib_backend import calibrate_geocalib

        return calibrate_geocalib(img, **kwargs)
    if method in ("auto", "cuboids"):
        dets = None
        if src is not None:
            from ..config import DetectorConfig
            from ..detect.backends import build_detector
            from ..detect.run import detect_video

            det = build_detector(DetectorConfig(name=detector), device)
            dets = detect_video(src, det, rate_hz=rate_hz, max_frames=int(seconds * rate_hz))
        if method == "cuboids":
            if dets is None:
                raise ValueError("cuboid calibration needs a video")
            return calibrate_from_cuboids(dets, **kwargs)
        from ..config import CalibrationConfig

        return calibrate_auto(img, dets, CalibrationConfig(), lens=lens, scale=scale)
    raise ValueError(f"unknown calibration method {method!r}")

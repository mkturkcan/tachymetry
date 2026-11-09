"""Public sample videos for demos, documentation and tests.

Every clip is downloaded on demand from its original host (Hugging Face or
Wikimedia Commons) into a local cache and is never redistributed with this
repository. Each entry records its licence and attribution; see
:func:`attribution`.

=====================  ===============================================  ===============
name                   scene                                            licence
=====================  ===============================================  ===============
intersection-day       signalised urban intersection, elevated camera   CC BY 4.0
intersection-day-2     same camera, another time of day                 CC BY 4.0
intersection-day-3     same camera, dense traffic                       CC BY 4.0
intersection-turn      same camera, turning traffic                     CC BY 4.0
intersection-night     same camera, infrared night view                 CC BY 4.0
motorway-overpass      divided motorway seen from an overpass           MIT
dhaka-roundabout       dense mixed traffic, straight-down drone view    CC BY 4.0
=====================  ===============================================  ===============

``dhaka-roundabout`` is a 30 s excerpt (about 390 MB, 3840x2160) cut from a
15 min recording; only the excerpt is transferred. It is filmed straight down,
so it is processed with a known camera rather than automatic calibration (see
the drone-footage example in the documentation).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Sample",
    "SAMPLES",
    "sample_video",
    "sample_camera",
    "sample_overrides",
    "attribution",
    "SAMPLE_ATTRIBUTION",
    "cache_dir",
]


@dataclass(frozen=True)
class Sample:
    scene: str
    license: str
    credit: str
    source: str  # "hf" (whole file), "hf-excerpt" (a time window, fetched with range requests) or "url"
    location: str  # Hugging Face "<repo>::<path>" or a direct URL
    page: str
    start_s: float = 0.0  # excerpt start (hf-excerpt only)
    duration_s: float = 0.0  # excerpt length (hf-excerpt only)
    camera: dict[str, Any] | None = field(default=None, hash=False)  # known camera (Camera.from_pose arguments)
    overrides: dict[str, Any] | None = field(default=None, hash=False)  # configuration the clip needs


_SVBRD = "Ryan-xiangyu-ut/svbrd-llm-roadside-video-av::svbrd-llm-roadside-video-av"
_SVBRD_PAGE = "https://huggingface.co/datasets/Ryan-xiangyu-ut/svbrd-llm-roadside-video-av"
_SVBRD_CREDIT = "svbrd-llm-roadside-video-av by Ryan-xiangyu-ut (Hugging Face)"
_PARS = "parsafaramarzi/Small-Crowd-and-Traffic-Video-Footages"
_DRINF = "hamimmahmud0/DRINF"

SAMPLES: dict[str, Sample] = {
    "intersection-day": Sample(
        "signalised urban intersection, elevated camera",
        "CC BY 4.0",
        _SVBRD_CREDIT,
        "hf",
        f"{_SVBRD}/Non_AV/2/2.mp4",
        _SVBRD_PAGE,
        # lens field of view from GeoCalib on four clips of this camera (focal 2053-2359 px), which the
        # earlier cuboid calibrations matched (2094-2314 px)
        overrides={"calibration.hfov_deg": 47.5},
    ),
    "intersection-day-2": Sample(
        "signalised urban intersection, another time of day",
        "CC BY 4.0",
        _SVBRD_CREDIT,
        "hf",
        f"{_SVBRD}/Non_AV/5/5.mp4",
        _SVBRD_PAGE,
        camera={
            "file": "svbrd-intersection.json"
        },  # same camera as intersection-day (backgrounds register within 4 px)
    ),
    "intersection-day-3": Sample(
        "signalised urban intersection, dense traffic",
        "CC BY 4.0",
        _SVBRD_CREDIT,
        "hf",
        f"{_SVBRD}/Non_AV/7/7.mp4",
        _SVBRD_PAGE,
        camera={
            "file": "svbrd-intersection.json"
        },  # same camera as intersection-day (backgrounds register within 4 px)
    ),
    "intersection-turn": Sample(
        "signalised urban intersection, turning traffic",
        "CC BY 4.0",
        _SVBRD_CREDIT,
        "hf",
        f"{_SVBRD}/clear/270/2025-09-24-sunny-northbound-right-turn-1.mp4",
        _SVBRD_PAGE,
        camera={
            "file": "svbrd-intersection.json"
        },  # same camera as intersection-day (backgrounds register within 4 px)
    ),
    "intersection-night": Sample(
        "signalised urban intersection, infrared night view",
        "CC BY 4.0",
        _SVBRD_CREDIT,
        "hf",
        f"{_SVBRD}/clear-night/100/09252025 - 5-1.mp4",
        _SVBRD_PAGE,
        camera={
            "file": "svbrd-intersection.json"
        },  # same camera as intersection-day (backgrounds register within 4 px)
    ),
    "motorway-overpass": Sample(
        "divided motorway seen from an overpass",
        "MIT",
        "Small-Crowd-and-Traffic-Video-Footages by parsafaramarzi (Hugging Face)",
        "hf",
        f"{_PARS}::cartraffic03.mp4",
        f"https://huggingface.co/datasets/{_PARS}",
        # lens field of view from GeoCalib; checked with the lane dashes (2.7-3.1 m dashes, 8.5-9.2 m periods
        # from near to far) and the lane width (3.43-3.45 m)
        overrides={"calibration.hfov_deg": 59.0},
    ),
    "dhaka-roundabout": Sample(
        "dense mixed traffic at a roundabout, straight-down drone view",
        "CC BY 4.0",
        "DRINF: Dhaka Road Intersection Nadir Footage by Hamim Mahmud (Hugging Face, "
        "doi:10.57967/hf/10301); 30 s excerpt from 2:20",
        "hf-excerpt",
        f"{_DRINF}::Nadirs/Doyel Chattor/Sep 3, 2026/DJI_0403_merged.mp4",
        f"https://huggingface.co/datasets/{_DRINF}",
        start_s=140.0,
        duration_s=30.0,
        # straight down from 111.7 m (flight-log altitude); DJI Mini 2 4K horizontal field of view 68.4 deg
        camera={"image_size": (3840, 2160), "focal": 2825.2, "height": 111.7, "pitch": 90.0, "yaw": 90.0},
        overrides={"detector.imgsz": 2560},
    ),
}

SAMPLE_ATTRIBUTION = f"Sample footage: {_SVBRD_CREDIT}, licensed CC BY 4.0."


def cache_dir() -> Path:
    """Folder for downloaded samples: ``$TACHYMETRY_CACHE`` or ``~/.cache/tachymetry``."""
    root = os.environ.get("TACHYMETRY_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "tachymetry"


def attribution(name: str) -> str:
    """Attribution line for a sample, as its licence requires."""
    s = SAMPLES[name]
    return f"{s.credit}, {s.license}, {s.page}"


def sample_camera(name: str):
    """The known camera of a sample, or ``None`` when the sample is calibrated automatically.

    Straight-down drone clips carry one (automatic calibration is designed for
    oblique views; see the drone-footage example in the documentation), and
    clips of a camera that was calibrated once on another clip share its file.
    """
    s = SAMPLES[name]
    if s.camera is None:
        return None
    from .geometry.camera import Camera

    kw = dict(s.camera)
    if "file" in kw:  # a camera calibrated once and shipped with the package
        from importlib.resources import files

        return Camera.load(str(files("tachymetry.assets").joinpath("cameras", kw["file"])))
    size = tuple(kw.pop("image_size"))
    return Camera.from_pose(size, meta={"calibration_method": "known", "source": f"sample {name}"}, **kw)


def sample_overrides(name: str) -> dict[str, Any]:
    """Configuration overrides a sample needs (dotted keys), e.g. a larger detector input for 4K aerial video."""
    return dict(SAMPLES[name].overrides or {})


def sample_video(name: str = "intersection-day") -> Path:
    """Download (once) and return the local path of a sample clip. See :data:`SAMPLES`."""
    if name not in SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(SAMPLES)}")
    s = SAMPLES[name]
    if s.source == "hf":
        from huggingface_hub import hf_hub_download

        repo, path = s.location.split("::", 1)
        return Path(hf_hub_download(repo, path, repo_type="dataset"))
    if s.source == "hf-excerpt":
        return _excerpt(name, s)
    from . import __version__

    ext = s.location.rsplit(".", 1)[-1].split("?")[0]
    target = cache_dir() / "samples" / f"{name}.{ext}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            s.location, headers={"User-Agent": f"tachymetry/{__version__} (+https://github.com/mkturkcan/tachymetry)"}
        )
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(target)
    return target


def _excerpt(name: str, s: Sample) -> Path:
    """Cut ``[start_s, start_s + duration_s)`` out of a remote video without downloading the whole file.

    FFmpeg reads the container index and the requested span with HTTP range
    requests and copies the video stream as is (no re-encoding).
    """
    import imageio_ffmpeg
    from huggingface_hub import hf_hub_url

    target = cache_dir() / "samples" / f"{name}.mp4"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    repo, path = s.location.split("::", 1)
    url = hf_hub_url(repo, path, repo_type="dataset")
    tmp = target.with_name(target.stem + ".part.mp4")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
        "-ss", f"{s.start_s:.3f}", "-i", url, "-t", f"{s.duration_s:.3f}",
        "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", str(tmp),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    tmp.replace(target)
    return target

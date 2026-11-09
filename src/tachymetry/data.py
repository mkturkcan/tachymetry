"""Public sample videos for demos, documentation and tests.

The clips come from the roadside-camera collection
``Ryan-xiangyu-ut/svbrd-llm-roadside-video-av`` on Hugging Face, released under
CC BY 4.0: an elevated PTZ camera over a signalised intersection in Austin,
Texas (1920x1080, 30 fps). They are downloaded on demand into the Hugging Face
cache and never redistributed with this repository.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["SAMPLES", "sample_video", "SAMPLE_ATTRIBUTION"]

_REPO = "Ryan-xiangyu-ut/svbrd-llm-roadside-video-av"
_ROOT = "svbrd-llm-roadside-video-av"
SAMPLE_ATTRIBUTION = "Sample footage: svbrd-llm-roadside-video-av (Ryan-xiangyu-ut, Hugging Face), licensed CC BY 4.0."

SAMPLES = {
    "intersection-day": f"{_ROOT}/Non_AV/2/2.mp4",
    "intersection-day-2": f"{_ROOT}/Non_AV/5/5.mp4",
    "intersection-day-3": f"{_ROOT}/Non_AV/7/7.mp4",
    "intersection-turn": f"{_ROOT}/clear/270/2025-09-24-sunny-northbound-right-turn-1.mp4",
    "intersection-night": f"{_ROOT}/clear-night/100/09252025 - 5-1.mp4",
}


def sample_video(name: str = "intersection-day") -> Path:
    """Download (once) and return the path of a sample clip. See :data:`SAMPLES`."""
    if name not in SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(SAMPLES)}")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(_REPO, SAMPLES[name], repo_type="dataset"))

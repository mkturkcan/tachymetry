"""Tachymetry: metric trajectories, speeds and counts from traffic-camera video.

The public API in three calls::

    import tachymetry as tm

    run = tm.track("clip.mp4", out="runs/clip")            # detect, calibrate, track, export
    run.render()                                           # annotated video next to the outputs
    tm.track_many(["a.mp4", "b.mp4"], out="runs")          # parallel over the available GPUs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._version import __version__
from .config import PipelineConfig
from .geometry.camera import CalibrationError, Camera

__all__ = [
    "__version__",
    "track",
    "track_many",
    "calibrate",
    "Camera",
    "PipelineConfig",
    "Pipeline",
    "CalibrationError",
    "load_run",
]


def __getattr__(name: str) -> Any:  # lazy imports keep `import tachymetry` fast
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    raise AttributeError(name)


def _config(config: PipelineConfig | str | Path | None, overrides: dict[str, Any]) -> PipelineConfig:
    cfg = PipelineConfig.load(config) if isinstance(config, (str, Path)) else (config or PipelineConfig())
    return cfg.override({k.replace("__", "."): v for k, v in overrides.items()}) if overrides else cfg


def track(
    source: str | Path,
    out: str | Path | None = None,
    camera: Any = "auto",
    config: PipelineConfig | str | Path | None = None,
    render: bool | str | Path = False,
    device: str | None = None,
    **overrides: Any,
):
    """Detect, calibrate, track and export one video.

    Parameters
    ----------
    source:
        Video file or folder of frames.
    out:
        Output directory (tracks, camera, previews, cached detections). ``None`` keeps everything in memory.
    camera:
        ``"auto"``, a :class:`Camera`, a camera JSON, a correspondence file, or a directory of cameras.
    config:
        A :class:`PipelineConfig` or YAML/JSON path.
    render:
        ``True`` writes ``overlay.mp4`` into ``out``; a path writes it there.
    overrides:
        Dotted config overrides with ``__`` as separator, e.g. ``detector__imgsz=1280``.

    Returns
    -------
    tachymetry.results.RunResult
    """
    from .pipeline import Pipeline

    return Pipeline(_config(config, overrides), camera=camera, device=device).run(source, out, render=render)


def track_many(
    sources: Any,
    out: str | Path = "runs",
    camera: Any = "auto",
    config: PipelineConfig | str | Path | None = None,
    devices: str | list[str] = "auto",
    workers_per_device: int | str = "auto",
    render: bool = False,
    overwrite: bool = False,
    **overrides: Any,
):
    """Process many videos in parallel across the available GPUs (or CPU cores).

    ``sources`` may mix files, directories and glob patterns. Returns a
    :class:`tachymetry.batch.BatchReport`.
    """
    from .batch import run_batch

    return run_batch(
        sources,
        out,
        camera=camera,
        config=_config(config, overrides),
        devices=devices,
        workers_per_device=workers_per_device,
        render=render,
        overwrite=overwrite,
    )


def calibrate(source: str | Path, method: str = "auto", points: str | Path | None = None, **kwargs: Any):
    """Calibrate a camera from a video or image.

    ``method``: ``"auto"`` (fused evidence; needs a cuboid detector for vehicle scale),
    ``"lines"`` (road markings), ``"manual"`` (``points=`` correspondence file),
    ``"depth"`` or ``"geocalib"``. Returns a :class:`tachymetry.calib.CalibrationResult`.
    """
    from .calib import calibrate_source

    return calibrate_source(source, method=method, points=points, **kwargs)


def load_run(out_dir: str | Path):
    """Reload a finished run directory (tracks are re-built from cached detections)."""
    from .pipeline import Pipeline

    out_dir = Path(out_dir)
    cfg = PipelineConfig.load(out_dir / "config.yaml")
    import json

    src = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["video"]["source"]
    return Pipeline(cfg, camera=Camera.load(out_dir / "camera.json")).run(src, out_dir)

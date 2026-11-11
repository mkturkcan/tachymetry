"""The end-to-end pipeline for one video.

``detect -> (calibrate) -> associate -> repair -> verdicts -> export -> (render)``

Detections and the background image are cached in the run directory, so
re-running with different tracking or rendering settings never touches the
GPU again.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .calib.correspondences import Correspondences
from .config import PipelineConfig
from .detections import DetectionSet
from .geometry.camera import CalibrationError, Camera, _json_default
from .io.video import VideoSource
from .results import RunResult, TrackSet

__all__ = ["Pipeline", "resolve_camera"]

ProgressFn = Callable[[str, int, int], None]

#: Automatic calibrations that place the camera higher than this are reported as implausible.
MAX_PLAUSIBLE_HEIGHT_M = 250.0


#: Version of the stored detection format; part of the cache key, so that older caches are detected again.
#: 2: SAM 3 detections carry class evidence (``detections.npz`` ``evidence``).
DETECTIONS_VERSION = 2


def _detector_key(cfg: PipelineConfig) -> str:
    d = cfg.detector
    blob = json.dumps(
        {
            "version": DETECTIONS_VERSION,
            "name": d.name,
            "weights": d.weights,
            "imgsz": d.imgsz,
            "conf": d.conf,
            "class_conf": d.class_conf,
            "classes": list(d.classes),
            "cuboid_conf": d.cuboid_conf,
            "nested": [d.nested_containment, d.nested_ground_overlap],
            "rate": cfg.rate_hz,
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def resolve_camera(spec: Any, video: Path | None = None) -> Camera | str | None:
    """Turn a camera specification into a :class:`Camera` (or ``"auto"``).

    Accepts a :class:`Camera`, ``"auto"``, a camera JSON, a correspondence file
    (YAML/JSON/CSV with ``points``; calibrated on the fly), or a directory
    holding ``<video stem>.json`` / ``<video stem>.camera.json`` files.
    """
    if spec is None or isinstance(spec, Camera) or spec == "auto":
        return spec
    p = Path(spec)
    if p.is_dir():
        if video is None:
            raise ValueError("a camera directory needs the video path to pick a file")
        for name in (f"{video.stem}.camera.json", f"{video.stem}.json", "camera.json"):
            if (p / name).exists():
                return resolve_camera(p / name, video)
        return "auto"
    if not p.exists():
        raise FileNotFoundError(f"camera file not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if "K" in data and "R" in data:
            return Camera.from_dict(data)
    corr = Correspondences.load(p)
    from .calib.manual import calibrate_manual

    size = corr.image_size
    if size is None and video is not None:
        with VideoSource(video) as src:
            size = src.size
    return calibrate_manual(corr, size).camera


def _load_precomputed(location: str, video: Path) -> DetectionSet:
    """Detections produced elsewhere: an ``.npz`` file, or a folder with ``<video stem>.npz``."""
    p = Path(location)
    if p.is_dir():
        for cand in (p / f"{video.stem}.npz", p / video.stem / "detections.npz"):
            if cand.exists():
                return DetectionSet.load(cand)
        raise FileNotFoundError(f"no precomputed detections for {video.name} in {p}")
    return DetectionSet.load(p)


class Pipeline:
    """Detection, calibration, tracking and export for single videos.

    Parameters
    ----------
    config:
        A :class:`~tachymetry.config.PipelineConfig`, a YAML/JSON path, or ``None`` for defaults.
    camera:
        ``"auto"`` (calibrate each video automatically), a :class:`Camera`, a
        camera JSON, a correspondence file, or a directory of per-video cameras.
    device:
        ``"auto"``, ``"cpu"``, ``"cuda:0"``, ``"mps"``, ...
    """

    def __init__(
        self,
        config: PipelineConfig | str | Path | None = None,
        camera: Any = "auto",
        device: str | None = None,
        detector: Any = None,
    ):
        if isinstance(config, (str, Path)):
            config = PipelineConfig.load(config)
        self.config = (config or PipelineConfig()).with_detector_profile()
        if device is not None:
            self.config.device = device
        self.camera_spec = camera
        self._detector = detector

    @property
    def detector(self) -> Any:
        if self._detector is None:
            from .detect.backends import build_detector

            self._detector = build_detector(self.config.detector, self.config.device)
        return self._detector

    # ------------------------------------------------------------------ steps
    def detect(
        self,
        src: VideoSource,
        out_dir: Path | None = None,
        progress: ProgressFn | None = None,
        max_frames: int | None = None,
        use_cache: bool = True,
        write: bool = True,
    ) -> DetectionSet:
        cfg = self.config
        if cfg.detector.name.startswith("precomputed:"):
            return _load_precomputed(cfg.detector.name.split(":", 1)[1], src.path)
        cache = out_dir / "detections.npz" if out_dir is not None else None
        key = _detector_key(cfg)
        if use_cache and cache is not None and cache.exists():
            meta_p = cache.with_suffix(".json")
            if meta_p.exists() and json.loads(meta_p.read_text(encoding="utf-8")).get("key") == key:
                return DetectionSet.load(cache)
        from .detect.run import detect_video

        cb = (lambda done, total: progress("detect", done, total)) if progress else None
        dets = detect_video(src, self.detector, cfg.rate_hz, cfg.detector.batch, cb, max_frames=max_frames)
        if write and cache is not None and cfg.keep_detections:
            dets.save(cache)
            cache.with_suffix(".json").write_text(
                json.dumps({"key": key, **dets.meta}, indent=2, default=_json_default), encoding="utf-8"
            )
        return dets

    def background(
        self, src: VideoSource, out_dir: Path | None = None, use_cache: bool = True, write: bool = True
    ) -> np.ndarray:
        import cv2

        from .calib.lines import background_image

        cache = out_dir / "background.jpg" if out_dir is not None else None
        if use_cache and cache is not None and cache.exists():
            img = cv2.imread(str(cache))
            if img is not None and (img.shape[1], img.shape[0]) == src.size:
                return img
        bg = background_image(src.sample_frames(self.config.calibration.background_frames))
        if write and cache is not None:
            cv2.imwrite(str(cache), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return bg

    def calibrate(
        self,
        src: VideoSource,
        dets: DetectionSet,
        bg: np.ndarray,
        out_dir: Path | None,
        use_cache: bool = True,
        write: bool = True,
    ) -> Camera:
        spec = resolve_camera(self.camera_spec, src.path)
        if isinstance(spec, Camera):
            cam = spec if spec.image_size == src.size else spec.resized(src.size)
            if write and out_dir is not None:
                cam.save(out_dir / "camera.json")
            return cam
        if use_cache and out_dir is not None and (out_dir / "camera.json").exists():
            cached = Camera.load(out_dir / "camera.json")
            if cached.image_size == src.size and str(cached.meta.get("calibration_method", "")).startswith("auto"):
                return cached
        from .calib.auto import calibrate_auto

        res = calibrate_auto(bg, dets, self.config.calibration)
        if res.camera.meta.get("metric") is False:
            raise CalibrationError(
                f"{src.path.name}: automatic calibration found no reliable metric camera (no instance masks or "
                "cuboids, or the car silhouettes did not fit the estimated orientation). Give the lens field of view "
                "(--set calibration.hfov_deg=<degrees>), a scale cue (--set calibration.scale=height=<metres> or "
                "lane_width=<metres>), or calibrate manually (tachymetry calibrate pick / manual)."
            )
        if res.camera.height > MAX_PLAUSIBLE_HEIGHT_M:
            note = (
                f"{src.path.name}: automatic calibration placed the camera {res.camera.height:.0f} m above the ground, "
                "which is implausible for traffic footage. Straight-down (nadir) drone views are a known failure "
                "case: supply the camera instead (altitude and field of view, see the drone-footage example)."
            )
            warnings.warn(note, stacklevel=2)
            res.camera.meta.setdefault("warnings", []).append(note)
        if write and out_dir is not None:
            import cv2

            from .calib.preview import preview

            res.details.pop("candidate_cameras", None)
            res.save(out_dir / "camera.json")
            cv2.imwrite(
                str(out_dir / "calibration_preview.jpg"),
                preview(bg, res.camera, title=f"Automatic calibration: {res.method}"),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
        return res.camera

    # -------------------------------------------------------------------- run
    def run(
        self,
        source: str | Path,
        out_dir: str | Path | None = None,
        render: bool | str | Path = False,
        progress: ProgressFn | None = None,
        max_frames: int | None = None,
        use_cache: bool = True,
        write: bool = True,
    ) -> RunResult:
        """Process one video. Returns a :class:`~tachymetry.results.RunResult`.

        With ``use_cache=False`` cached detections, background and camera in
        ``out_dir`` are recomputed. With ``write=False`` caches in ``out_dir``
        are read but nothing is written there.
        """
        from .tracking.builder import build_tracks

        t0 = time.time()
        timings: dict[str, float] = {}
        out = Path(out_dir) if out_dir is not None else None
        if out is not None and write:
            out.mkdir(parents=True, exist_ok=True)
            self.config.save(out / "config.yaml")
        with VideoSource(source, fps=self.config.fps) as src:
            t = time.time()
            dets = self.detect(src, out, progress, max_frames=max_frames, use_cache=use_cache, write=write)
            timings["detect_s"] = round(time.time() - t, 2)
            t = time.time()
            bg = self.background(src, out, use_cache=use_cache, write=write)
            camera = self.calibrate(src, dets, bg, out, use_cache=use_cache, write=write)
            timings["calibrate_s"] = round(time.time() - t, 2)
            t = time.time()
            if progress:
                progress("track", 0, 1)
            tracks, report = build_tracks(dets, camera, self.config)
            timings["track_s"] = round(time.time() - t, 2)
            meta = {
                "source": str(src.path),
                "fps": src.fps,
                "frames": src.frame_count,
                "size": list(src.size),
                "duration_s": round(src.duration, 3) if np.isfinite(src.duration) else None,
            }
            ts = TrackSet(tracks, camera, src.fps, meta)
            result = RunResult(Path(source), ts, camera, dets, out, report, bg)
            if out is not None and write:
                ts.export(out, self.config.exports)
                summary = {
                    "tachymetry": __version__,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "video": meta,
                    "detector": dets.meta.get("detector"),
                    "camera": camera.to_dict()["summary"],
                    "calibration": camera.meta.get("calibration_method"),
                    "summary": ts.summary(),
                    "report": report,
                    "timings": timings,
                }
                (out / "summary.json").write_text(
                    json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
                )
            if render:
                t = time.time()
                path = Path(render) if isinstance(render, (str, Path)) and not isinstance(render, bool) else None
                result.render(path, layout=self.config.render.layout, cfg=self.config.render, progress=progress)
                timings["render_s"] = round(time.time() - t, 2)
        timings["total_s"] = round(time.time() - t0, 2)
        result.report["timings"] = timings
        if out is not None and write and (out / "summary.json").exists():
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            s["timings"] = timings
            (out / "summary.json").write_text(json.dumps(s, indent=2, default=_json_default), encoding="utf-8")
        return result

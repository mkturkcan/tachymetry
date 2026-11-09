"""Parallel processing of many videos across devices.

Design:

* One worker process per slot from :func:`plan_workers`; each worker pins one
  device (``CUDA_VISIBLE_DEVICES`` is set before CUDA initialises, so each
  worker sees a single GPU) and loads the detector once for all its jobs.
* Processes use the ``spawn`` start method on all platforms, which is the
  only method available on Windows and is also the safe choice with CUDA.
* Jobs are dispatched longest-first (by file size) to balance the makespan.
* A job that fails is recorded with its traceback and never stops the batch;
  finished jobs are skipped on re-runs unless ``overwrite=True`` (resume).
* Progress from all workers is aggregated in the parent process.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

from ..config import PipelineConfig
from ..io.video import find_videos
from .devices import discover_devices, parse_devices, plan_workers

__all__ = ["run_batch", "BatchReport", "output_dir_for"]

_WORKER: dict[str, Any] = {}


def output_dir_for(video: Path, root: Path, common: Path | None) -> Path:
    """A collision-free run directory: the video's path relative to the common input root."""
    try:
        rel = Path(os.path.abspath(video)).relative_to(common) if common is not None else Path(video.name)
    except ValueError:
        rel = Path(video.name)
    parts = list(rel.with_suffix("").parts) if video.is_file() else list(rel.parts)
    return root / "__".join(parts)


@dataclass
class BatchReport:
    """Outcome of a batch run."""

    out_dir: Path
    jobs: list[dict[str, Any]] = field(default_factory=list)
    workers: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> list[dict[str, Any]]:
        return [j for j in self.jobs if j["status"] in ("done", "cached")]

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [j for j in self.jobs if j["status"] == "failed"]

    def save(self) -> Path:
        p = self.out_dir / "batch_summary.json"
        p.write_text(
            json.dumps(
                {"workers": self.workers, "seconds": round(self.seconds, 2), "jobs": self.jobs}, indent=2, default=str
            ),
            encoding="utf-8",
        )
        import csv

        with (self.out_dir / "batch_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "video",
                    "status",
                    "out_dir",
                    "tracks",
                    "vehicles_moving",
                    "pedestrians",
                    "median_speed_kmh",
                    "seconds",
                    "device",
                    "error",
                ]
            )
            for j in self.jobs:
                s = j.get("summary") or {}
                mv = s.get("moving_by_class", {})
                w.writerow(
                    [
                        j["video"],
                        j["status"],
                        j["out_dir"],
                        s.get("tracks"),
                        sum(v for k, v in mv.items() if k in ("car", "bus", "truck")),
                        mv.get("person"),
                        (s.get("vehicle_speed_kmh") or {}).get("median"),
                        j.get("seconds"),
                        j.get("device"),
                        (j.get("error") or "").splitlines()[-1] if j.get("error") else "",
                    ]
                )
        return p

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BatchReport({len(self.ok)} done, {len(self.failed)} failed, {self.seconds:.0f} s, workers={self.workers})"
        )


# ------------------------------------------------------------------- worker side
def _init_worker(
    slots: Any, cfg_dict: dict[str, Any], camera: Any, render: bool, progress_q: Any, overwrite: bool = False
) -> None:
    device = slots.get()
    if device.startswith("cuda:"):
        idx = int(device.split(":", 1)[1])
        parent = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
        # logical index in the parent process -> physical device id
        os.environ["CUDA_VISIBLE_DEVICES"] = parent[idx] if idx < len(parent) else str(idx)
        local = "cuda:0"
    else:
        local = device
        if device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    _WORKER.update(overwrite=overwrite)
    _WORKER.update(device=device, local=local, cfg=cfg_dict, camera=camera, render=render, q=progress_q, pipeline=None)


def _job(video: str, out_dir: str, job_id: int) -> dict[str, Any]:
    from ..pipeline import Pipeline

    t0 = time.time()
    q = _WORKER["q"]
    try:
        if _WORKER["pipeline"] is None:
            cfg = PipelineConfig.from_dict(_WORKER["cfg"])
            _WORKER["pipeline"] = Pipeline(cfg, camera=_WORKER["camera"], device=_WORKER["local"])
        pipe = _WORKER["pipeline"]

        def progress(stage: str, done: int, total: int) -> None:
            try:
                q.put_nowait((job_id, stage, done, total))
            except Exception:
                pass

        q.put_nowait((job_id, "start", 0, 1))
        res = pipe.run(
            video, out_dir, render=_WORKER["render"], progress=progress, use_cache=not _WORKER.get("overwrite")
        )
        q.put_nowait((job_id, "done", 1, 1))
        return {
            "status": "done",
            "summary": res.summary(),
            "seconds": round(time.time() - t0, 2),
            "device": _WORKER["device"],
            "camera": res.camera.describe(),
            "calibration": res.camera.meta.get("calibration_method"),
        }
    except Exception:
        q.put_nowait((job_id, "failed", 1, 1))
        return {
            "status": "failed",
            "error": traceback.format_exc(),
            "seconds": round(time.time() - t0, 2),
            "device": _WORKER["device"],
        }


# ------------------------------------------------------------------- parent side
def run_batch(
    sources: Any,
    out: str | Path = "runs",
    camera: Any = "auto",
    config: PipelineConfig | None = None,
    devices: str | Sequence[str] = "auto",
    workers_per_device: int | str = "auto",
    render: bool = False,
    overwrite: bool = False,
    quiet: bool = False,
) -> BatchReport:
    """Run the pipeline over many videos in parallel. See the module docstring."""
    from concurrent.futures import ProcessPoolExecutor

    cfg = config or PipelineConfig()
    srcs = [sources] if isinstance(sources, (str, Path)) else list(sources)
    videos = find_videos(srcs)
    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)
    if not videos:
        raise FileNotFoundError(f"no videos found in {srcs}")
    try:
        common = Path(os.path.commonpath([os.path.abspath(v.parent) for v in videos]))
    except ValueError:
        common = None
    jobs = []
    for v in videos:
        od = output_dir_for(v, out_root, common)
        cached = (od / "summary.json").exists() and not overwrite
        jobs.append(
            {
                "video": str(v),
                "out_dir": str(od),
                "status": "cached" if cached else "pending",
                "size": v.stat().st_size if v.is_file() else 0,
            }
        )
    pending = sorted([j for j in jobs if j["status"] == "pending"], key=lambda j: -j["size"])
    devs = parse_devices(devices) if devices != "auto" else discover_devices()
    slots = plan_workers(devs, workers_per_device, n_jobs=max(1, len(pending)))
    report = BatchReport(out_root, jobs, slots)
    t0 = time.time()
    for j in jobs:
        if j["status"] == "cached":
            try:
                j["summary"] = json.loads((Path(j["out_dir"]) / "summary.json").read_text(encoding="utf-8")).get(
                    "summary"
                )
            except Exception:
                j["summary"] = None
    if pending:
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        slot_q = manager.Queue()
        for s in slots:
            slot_q.put(s)
        progress_q = manager.Queue()
        cam_spec = camera if not hasattr(camera, "to_dict") else camera.to_dict()
        if isinstance(cam_spec, dict):
            from ..geometry.camera import Camera

            cam_spec = Camera.from_dict(cam_spec)
        bar = _Progress(len(pending), quiet)
        with ProcessPoolExecutor(
            max_workers=len(slots),
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(slot_q, cfg.to_dict(), cam_spec, render, progress_q, overwrite),
        ) as pool:
            futs = {pool.submit(_job, j["video"], j["out_dir"], k): (k, j) for k, j in enumerate(pending)}
            remaining = set(futs)
            while remaining:
                _drain(progress_q, bar, pending)
                done_now = [f for f in remaining if f.done()]
                for f in done_now:
                    remaining.discard(f)
                    k, j = futs[f]
                    try:
                        j.update(f.result())
                    except Exception:
                        j.update({"status": "failed", "error": traceback.format_exc()})
                    bar.finish(k, Path(j["video"]).name, j["status"])
                if remaining:
                    time.sleep(0.2)
        _drain(progress_q, bar, pending)
        bar.close()
        manager.shutdown()
    report.seconds = time.time() - t0
    report.save()
    return report


def _drain(q: Any, bar: _Progress, pending: list[dict[str, Any]]) -> None:
    while True:
        try:
            job_id, stage, done, total = q.get_nowait()
        except (Empty, EOFError, OSError):
            return
        bar.update(job_id, Path(pending[job_id]["video"]).name, stage, done, total)


class _Progress:
    """Rich live progress when available, plain log lines otherwise."""

    def __init__(self, n: int, quiet: bool):
        self.n = n
        self.quiet = quiet
        self.done = 0
        self.rich = None
        self.tasks: dict[int, Any] = {}
        if quiet:
            return
        try:
            from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

            self.rich = Progress(
                TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()
            )
            self.rich.start()
            self.total = self.rich.add_task("videos", total=n)
        except Exception:
            self.rich = None

    def update(self, job: int, name: str, stage: str, done: int, total: int) -> None:
        if self.rich is None:
            return
        if job not in self.tasks:
            self.tasks[job] = self.rich.add_task(f"{name[:28]:28s} {stage}", total=max(total, 1))
        self.rich.update(
            self.tasks[job], description=f"{name[:28]:28s} {stage:7s}", completed=done, total=max(total, 1)
        )

    def finish(self, job: int, name: str, status: str) -> None:
        self.done += 1
        if self.rich is not None:
            if job in self.tasks:
                self.rich.remove_task(self.tasks.pop(job))
            self.rich.update(self.total, completed=self.done)
        elif not self.quiet:
            print(f"[{self.done}/{self.n}] {status:6s} {name}", flush=True)

    def close(self) -> None:
        if self.rich is not None:
            self.rich.stop()

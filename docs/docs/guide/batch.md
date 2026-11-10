# Batch processing on multiple GPUs

`tachymetry run` and `tm.track_many` (`batch/scheduler.py`, `run_batch`) process many videos in parallel worker processes spread over the selected devices. Each video goes through the same [pipeline](pipeline.md) as `tm.track` and gets its own run directory.

```bash
tachymetry run "videos/**/*.mp4" --out runs --devices auto --render
tachymetry devices                          # devices and the worker plan
```

```python
import tachymetry as tm

if __name__ == "__main__":
    report = tm.track_many(["videos/a/", "videos/b/*.mp4"], out="runs", devices="cuda:0,cuda:1")
    print(report)
```

A worked example with two public clips is in [Examples: batch processing](../examples/batch.md).

## Inputs

`io.find_videos` expands each input: a file is used as is; a directory is searched recursively for files with a video extension (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.mpg`, `.mpeg`, `.ts`, `.mts`, `.webm`, `.wmv`, `.flv`, `.h264`, `.hevc`), and a directory that holds images but no videos becomes one image-sequence source (10 fps unless `VideoSource(path, fps=...)` is used directly); a pattern containing `*`, `?` or `[` is expanded with Python's `glob` module in recursive mode, where `**` also matches nested folders. Quote patterns on the command line so that the library, not the shell, expands them (this also makes `**` work in shells without recursive globbing, and on Windows, where shells do not expand patterns). Files are sorted in natural order and duplicates removed.

## Devices and workers

`--devices` (`devices=`) accepts `auto` (all CUDA GPUs, else Apple MPS, else CPU), `cpu`, `cuda` (all GPUs), indices such as `0,2`, or names such as `cuda:1,cuda:3`. `plan_workers` then assigns worker processes:

| Device | Workers per device with `--workers-per-device auto` |
|---|---|
| CUDA GPU | $\min(2, \max(1, \lfloor \text{free GB} / g \rfloor))$, with the per-worker budget $g$ = 9 GB for SAM 3 and 6 GB for the other detectors (`batch.devices.worker_gb`); 2 if free memory is unknown |
| Apple MPS | 1 |
| CPU | $\max(1, \lfloor \text{CPU threads} / 4 \rfloor)$ |

Slots are ordered round-robin across devices (`cuda:0`, `cuda:1`, ..., `cuda:0`, `cuda:1`, ...) before the list is truncated, so a batch with fewer videos than slots still spreads over all selected GPUs: two videos on two GPUs run on one GPU each. Two workers per GPU overlap decoding, tracking and rendering in one process with inference in the other. When GPUs are used, the total number of workers is capped at $\max(\text{number of devices}, \min(\text{workers}, \lfloor\text{CPU threads}/2\rfloor))$ so that each worker keeps CPU threads for decoding and tracking, and it never exceeds the number of pending videos. An integer `--workers-per-device N` overrides the rule. `tachymetry devices` prints the devices with their free memory and the resulting plan. The SAM 3 budget covers its measured peak of 7.3 GB of GPU memory per worker at the default batch of 8 frames ([Detection](detection.md#frame-rate-and-decoding)), so two SAM 3 workers share a GPU only when it has at least 18 GB free. `tachymetry devices` prints the plan with the 6 GB budget.

## Scheduling

* **Processes.** Workers are started with the `spawn` method on all platforms, which is the only method on Windows and the safe one with CUDA. Each worker takes one device slot; for a CUDA slot it sets `CUDA_VISIBLE_DEVICES` to that GPU before CUDA initialises and then uses `cuda:0` internally, and it sets `OMP_NUM_THREADS=4` unless already set. The pipeline and its detector are created once per worker and reused for all its videos.
* **Order.** Pending videos are dispatched longest first by file size, which shortens the makespan when durations differ.
* **Failures.** An exception in one video is recorded with its traceback (`status: failed`) and does not stop the batch. The command exits with status 1 if a video failed.
* **Resume.** A video whose run directory already contains `summary.json` is skipped (`status: cached`) and its summary is read from disk, unless `--overwrite` (`overwrite=True`) is given. `--overwrite` processes such videos again from scratch: the cached detections, background and automatic camera in the run directory are ignored and rewritten (`Pipeline.run(..., use_cache=False)`). `summary.json` is written after export and before rendering, so a run interrupted during rendering counts as finished; re-run it with `--overwrite` or render it with `tachymetry render`.
* **Progress.** Per-video progress from all workers is aggregated in the parent process (a `rich` live display, or plain log lines).

!!! note "Device indices and `CUDA_VISIBLE_DEVICES`"
    Device names are resolved in the parent process. Each worker translates the parent's logical index into the physical device id before CUDA initialises: if the parent runs with `CUDA_VISIBLE_DEVICES=2,3`, `--devices cuda:1` selects physical GPU 3 (`batch.scheduler._init_worker`).

## Cameras in batch mode

`--camera` (`camera=`) applies to all videos of the batch: `auto` calibrates each video separately; a camera JSON or correspondence file is used for all videos (resized to each video's resolution); a directory is searched per video for `<stem>.camera.json`, `<stem>.json` or `camera.json`, and videos without a match are calibrated automatically (`pipeline.resolve_camera`).

## Output layout

Each video gets the directory `output_dir_for(video, out, common)`: its path relative to the deepest directory common to all inputs, without the extension, with path separators replaced by `__`. Inputs `videos/a/clip.mp4` and `videos/b/clip.mp4` therefore map to `runs/a__clip` and `runs/b__clip`, and a folder of clips maps to one directory per file stem.

The output root also receives `batch_summary.json` (workers, wall time, and per video: `video`, `out_dir`, `status`, `size`, `summary`, `seconds`, `device`, `camera`, `calibration`, `error`) and `batch_summary.csv` with the columns `video, status, out_dir, tracks, vehicles_moving, pedestrians, median_speed_kmh, seconds, device, error`.

`track_many` returns a `BatchReport` with `jobs` (the dictionaries above), `ok` (done or cached), `failed`, `workers`, `seconds` and `out_dir`.

## Windows

* Install the CUDA build of PyTorch before Tachymetry, for example `pip install torch --index-url https://download.pytorch.org/whl/cu128`; the default PyPI wheel on Windows is CPU-only ([Installation](../installation.md)).
* Because workers are spawned, a script that calls `tm.track_many` is imported again by each worker. Put the call under `if __name__ == "__main__":`, as in the example above; without the guard each worker would start a batch of its own. The command line `tachymetry run` needs no guard.
* Neither `cmd` nor PowerShell expands glob patterns; Tachymetry does. Passing a folder, for example `tachymetry run D:\videos --out runs`, is the simplest form, since folders are searched recursively.
* Linux and macOS use the same `spawn` method, so a script that works on one platform behaves the same on the others.

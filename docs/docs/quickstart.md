# Quick start

## Demo on the public sample clip

```bash
tachymetry demo
```

The demo downloads the clip `intersection-day` from [svbrd-llm-roadside-video-av](https://huggingface.co/datasets/Ryan-xiangyu-ut/svbrd-llm-roadside-video-av) (CC BY 4.0; an elevated camera over a signalised intersection, 1920 x 1080 at 30 fps, 31 s), runs detection with SAM 3, automatic calibration with the 47.5° field of view registered for the clip, and tracking, renders `overlay.mp4` and writes everything to `demo_run/`. SAM 3 needs an approved access request and `hf auth login` once ([Installation](installation.md#access-to-sam-3)). Other clips are selected with `--sample` and the output directory with `-o`. The seven registered clips are five recordings of the intersection, a motorway seen from an overpass and a straight-down drone view of mixed traffic ([Sample clips](samples.md)); each runs with the camera or field of view registered for it, for example `tachymetry demo --sample dhaka-roundabout` ([Drone footage](examples/drone.md)).

## Your own videos

```bash
tachymetry run clip.mp4 --render              # one video
tachymetry run videos/ --out runs --render    # all video files under a folder, on all available GPUs
tachymetry run "videos/**/*.mp4" --devices cuda:0,cuda:1
```

Each video gets a run directory under `--out` (default `runs/`) containing:

| File | Content |
|---|---|
| `tracks.csv` | One row per track sample: position (m), velocity (m/s), speed (km/h), heading, flags, image box |
| `tracks_summary.csv` | One row per track: class, state (moving or parked), verdict, duration, dimensions, path length, speed statistics |
| `tracks.json` | Per-track summaries with sample arrays and the camera |
| `camera.json`, `camera.report.json` | The calibrated camera and the calibration diagnostics |
| `calibration_preview.jpg` | Metric ground grid on the camera image and the orthorectified view (automatic calibration) |
| `summary.json` | Counts by class, speed statistics, hygiene and verdict report, timings |
| `detections.npz`, `background.jpg`, `config.yaml` | Caches and the configuration used; re-runs reuse them |
| `overlay.mp4` | The annotated video (with `--render`) |

All columns are defined in [Outputs and file formats](guide/outputs.md).

## Python

```python
import tachymetry as tm

run = tm.track("clip.mp4", out="runs/clip", render=True)
print(run.summary())                       # counts by class, moving vs parked, speeds
df = run.tracks.to_dataframe()             # one row per track sample, in metres
cars = df[(df["class"] == "car") & (df["state"] == "moving")]
print(cars.groupby("track_id")["speed_kmh"].median().describe())
```

`tm.track` returns a `RunResult` whose `tracks` attribute is a `TrackSet`; iterating it yields the tracks with verdict `ok`, and `select(cls=..., state=...)` filters them. A finished run directory is reloaded with `tm.load_run("runs/clip")`, which rebuilds the tracks from the cached detections.

## Calibration

`camera="auto"` (the default) calibrates each video automatically. Give the lens's horizontal field of view when it is known (`--hfov`, `calibration.hfov_deg`); for a fixed camera, calibrate once on its clearest clip and reuse the camera file for its other clips ([Calibration verification](guide/calibration.md#calibration-verification)):

```bash
tachymetry calibrate auto clip.mp4 --hfov 60 -o camera.json           # writes camera_preview.jpg as well
tachymetry calibrate lines clip.mp4 --scale height=8.5 -o camera.json # road markings and a known camera height
tachymetry calibrate pick clip.mp4 --reference ortho.png --gsd 0.1    # click correspondences in the browser
tachymetry calibrate manual clip.mp4 --points points.yaml -o camera.json
tachymetry run clip.mp4 --camera camera.json
```

[Calibration](guide/calibration.md) explains how to choose a method and how to read the preview.

## Settings

Any configuration key can be overridden on the command line or as a keyword argument, with `__` in place of the dot:

```bash
tachymetry run clip.mp4 --detector rfdetr-seg-l --rate 10 --set hygiene.max_gsd_m=0.25
tachymetry config --dump defaults.yaml       # every key with its default
tachymetry run clip.mp4 --config my.yaml     # partial files override the defaults
```

```python
run = tm.track("clip.mp4", detector__name="rfdetr-seg-l", hygiene__max_gsd_m=0.25)
```

The keys are listed in the [configuration reference](guide/configuration.md).

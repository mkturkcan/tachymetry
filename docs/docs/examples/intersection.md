# End-to-end: a signalised intersection

This example runs the full pipeline on the public sample clip `intersection-day`, an elevated camera over a signalised intersection (1920×1080, 30 fps, 31 s), from the Hugging Face dataset [svbrd-llm-roadside-video-av](https://huggingface.co/datasets/Ryan-xiangyu-ut/svbrd-llm-roadside-video-av) (CC BY 4.0). The clip is downloaded once into the Hugging Face cache by `tachymetry.data.sample_video`. All outputs below were produced by the commands shown, on one NVIDIA RTX PRO 6000 Blackwell Max-Q GPU.

## Command line

```bash
tachymetry demo --out demo_run
```

`tachymetry demo` downloads the clip, runs detection (SAM 3), automatic calibration with the sample's lens field of view, tracking, export and rendering, and writes everything to `demo_run/`. `--device cuda:1` selects a GPU, `--sample` another clip ([Sample clips](../samples.md)).

## Python

```python
import tachymetry as tm
from tachymetry.data import attribution, sample_overrides, sample_video

clip = sample_video("intersection-day")  # downloaded once into the Hugging Face cache
print(attribution("intersection-day"))
print(sample_overrides("intersection-day"))  # the lens field of view of this camera

run = tm.track(clip, out="runs/intersection-day", render=True, calibration__hfov_deg=47.5)
print(run.camera.describe())
print(run.camera.meta["calibration_method"])
print(run.summary())
print(run.report["timings"])
```

```text
svbrd-llm-roadside-video-av by Ryan-xiangyu-ut (Hugging Face), CC BY 4.0, https://huggingface.co/datasets/Ryan-xiangyu-ut/svbrd-llm-roadside-video-av
{'calibration.hfov_deg': 47.5}
f=2181.8px (hfov 47.5 deg), height=27.21 m, pitch=17.88 deg, roll=-0.01 deg, yaw=134.0 deg, lens=identity
auto:fov+vp
{'tracks': 50, 'by_class': {'person': 8, 'car': 29, 'bus': 2, 'bicycle': 2, 'truck': 9}, 'moving_by_class': {'bus': 2, 'car': 17, 'bicycle': 2, 'person': 7, 'truck': 3}, 'parked_by_class': {'car': 12, 'truck': 6}, 'rejected': {'rider': 2, 'ghost': 2, 'duplicate': 1}, 'vehicle_speed_kmh': {'median': 24.65, 'p85': 34.62, 'max': 58.59}}
{'detect_s': 55.88, 'calibrate_s': 23.29, 'track_s': 41.34, 'render_s': 24.5, 'total_s': 145.22}
```

The run uses the default detector, SAM 3, whose instance masks give the vehicle footprints ([Vehicle footprints from masks](../guide/footprints.md)) and the metric scale of the calibration. `tachymetry demo` applies the sample's overrides itself; with `tm.track`, the field of view is passed as `calibration__hfov_deg`.

Calibration:

* *Candidate.* The known field of view (47.5°) and the main ground vanishing point give the orientation (`fov+vp`, [Automatic calibration](../guide/auto-calibration.md)): a camera pitched 17.9° down.
* *Scale.* The vehicle silhouettes place it 27.2 m above the road. The camera packaged for the five intersection clips, calibrated once, is 27.7 m high; see [Camera calibration](../guide/calibration.md) for the spread between calibrations of one camera.

The run directory now holds the files described in [Outputs](../guide/outputs.md):

```text
runs/intersection-day/
  background.jpg  calibration_preview.jpg  camera.json  camera.report.json  config.yaml
  detections.json  detections.npz  overlay.mp4  summary.json  tracks.csv  tracks.json  tracks_summary.csv
```

![The rendered side layout of the sample clip](../../media/layout_side.jpg)

!!! note "One camera for the five intersection clips"
    The five `intersection-*` samples come from one camera pose. `tachymetry.data.sample_camera` returns the camera of this intersection for the other four clips. Pass it, or the `camera.json` of this run, when several runs must share one calibration, as in the [batch example](batch.md).

## Analysing the outputs

`tracks.csv` holds one row per track and processed frame and `tracks_summary.csv` one row per track, including rejected tracks. The script below reads both with pandas and computes counts, the rejected tracks, spot-speed percentiles of moving motor vehicles, and a directional count at a screen line.

```python
import numpy as np
import pandas as pd

samples = pd.read_csv("runs/intersection-day/tracks.csv")  # one row per track and processed frame
tracks = pd.read_csv("runs/intersection-day/tracks_summary.csv")  # one row per track, rejected ones included

# Road users by class and state (accepted tracks only)
ok = tracks[tracks.verdict == "ok"]
print(pd.crosstab(ok["class"], ok["state"]))

# Rejected tracks and the rule that rejected them
print(tracks.loc[tracks.verdict != "ok", ["id", "class", "verdict", "reason"]].to_string(index=False))

# Spot speeds of moving motor vehicles (samples above 3 km/h)
veh = samples[samples["class"].isin(["car", "bus", "truck"]) & (samples.state == "moving")]
spot = veh.loc[veh.speed_kmh > 3, "speed_kmh"]
print(spot.describe(percentiles=[0.15, 0.5, 0.85]).round(1).to_string())

# A screen line across the main road: x = -45 m, restricted to 35 m < y < 60 m
counts = {"+x": 0, "-x": 0}
for tid, g in veh.groupby("track_id"):
    g = g[(g.y_m > 35) & (g.y_m < 60)]
    x = g.x_m.to_numpy()
    if len(x) > 1 and x.min() < -45 < x.max():
        counts["+x" if x[-1] > x[0] else "-x"] += 1
print(counts)
```

```text
state    moving  parked  stationary
class
bicycle       2       0           0
bus           2       0           0
car          17      12           0
person        7       0           1
truck         3       6           0
 id  class   verdict                reason
  7 person     rider  riding two-wheeler 6
 29 person     ghost             too short
 33  truck duplicate       duplicate of 23
 43 person     rider riding two-wheeler 41
 46    car     ghost             too short
count    2177.0
mean       25.2
std         9.6
min         3.2
15%        15.0
50%        24.6
85%        34.6
max        58.6
{'+x': 8, '-x': 2}
```

Notes on the result:

* The two riders are the persons on the two bicycles; they are counted once, as cyclists ([Verdicts](../guide/verdicts.md#riders-inferred-cyclists-and-passengers)).
* Twelve parked cars and six parked trucks are reported with zero speed; they do not enter the speed statistics.
* The world frame of an automatic calibration has $+x$ along the best-supported lane-line direction. Without cuboid footprints the origin stays below the camera, so the main road runs along $x$ at $y \approx 41$ to $53$ m in this clip (traffic in $+x$ near 42 m, in $-x$ near 52 m). For counts at a specific location, define the screen line in this frame, or calibrate manually against a georeferenced reference to work in map coordinates ([Manual calibration](../guide/manual-calibration.md)).
* The spot speeds are samples at 15 Hz, so longer tracks weigh more. For one value per vehicle use `tracks_summary.csv` (`speed_median_kmh`, `speed_p85_kmh`).

## Changing settings without re-detecting

Detections and the automatic calibration are cached in the run directory ([Pipeline](../guide/pipeline.md#caching-and-re-runs)). Re-running with other tracking settings repeats only the CPU stages, of which the mask geometry takes most of the time:

```python
import tachymetry as tm
from tachymetry.data import sample_video

pipe = tm.Pipeline()
pipe.config.hygiene.smooth_window_s = 0.8
pipe.config.exports = ("csv", "json", "geojson")
run = pipe.run(sample_video("intersection-day"), "runs/intersection-day")
print(run.report["timings"], run.summary()["vehicle_speed_kmh"])
```

```text
{'detect_s': 0.03, 'calibrate_s': 0.01, 'track_s': 40.92, 'total_s': 41.18} {'median': 24.81, 'p85': 34.85, 'max': 57.8}
```

A shorter smoothing window follows speed changes faster at the cost of more noise ([Bird's-eye view and speed](../guide/bev-speed.md#speed)); on this clip the median and 85th-percentile vehicle speeds move by 0.16 and 0.23 km/h.

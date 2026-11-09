# Batch processing

This example processes two public sample clips in parallel with the batch scheduler ([Batch processing on multiple GPUs](../guide/batch.md)): `intersection-day` (1920×1080, 31 s) and `intersection-turn` (a shorter clip of the same dataset). The outputs below were produced on a machine with four NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, restricted to one of them with `--devices cuda:1`.

## Collect the clips

```python
import shutil
from pathlib import Path

from tachymetry.data import sample_video

Path("videos").mkdir(exist_ok=True)
for name in ("intersection-day", "intersection-turn"):
    shutil.copy(sample_video(name), Path("videos") / f"{name}.mp4")
```

## Command line

```bash
tachymetry run videos/ --out runs --devices cuda:1
```

```text
Tachymetry: 2 of 2 videos processed in 25 s
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ video                 ┃ status ┃ tracks ┃ moving vehicles ┃ parked ┃ pedestrians ┃ median km/h ┃ seconds ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ intersection-day.mp4  │ done   │     43 │              22 │     10 │           8 │       24.55 │   23.17 │
│ intersection-turn.mp4 │ done   │     23 │              10 │     12 │           1 │       21.54 │   17.33 │
└───────────────────────┴────────┴────────┴─────────────────┴────────┴─────────────┴─────────────┴─────────┘
```

With free memory well above 12 GB, `plan_workers` started two worker processes on the GPU, one per clip, so the batch took 25 s of wall time for 40.5 s of per-video processing (detection, calibration, tracking and export; no rendering). Without `--devices`, all available GPUs are used. The output root holds one directory per video, named after the file stem, plus the batch summaries:

```text
runs/
  batch_summary.csv  batch_summary.json
  intersection-day/   background.jpg  calibration_preview.jpg  camera.json  camera.report.json  config.yaml
                      detections.json  detections.npz  summary.json  tracks.csv  tracks.json  tracks_summary.csv
  intersection-turn/  (same files)
```

```text
video,status,out_dir,tracks,vehicles_moving,pedestrians,median_speed_kmh,seconds,device,error
videos/intersection-day.mp4,done,runs/intersection-day,43,22,8,24.55,23.17,cuda:1,
videos/intersection-turn.mp4,done,runs/intersection-turn,23,10,1,21.54,17.33,cuda:1,
```

Add `--render` for an `overlay.mp4` per video, `--formats csv,json,geojson` for more exports, and `--set KEY=VALUE` (repeatable) for any configuration field ([Configuration](../guide/configuration.md)).

## Python

```python
import tachymetry as tm

if __name__ == "__main__":
    report = tm.track_many(["videos/"], out="runs", devices="cuda:1")
    print(report)
    for job in report.jobs:
        print(f'{job["status"]:6s} {job["out_dir"]}  {job["summary"]["tracks"]} tracks')
```

Run after the command above, both videos already have a `summary.json`, so both are skipped and their summaries are read from disk:

```text
BatchReport(2 done, 0 failed, 0 s, workers=['cuda:1'])
cached runs/intersection-day  43 tracks
cached runs/intersection-turn  23 tracks
```

The `if __name__ == "__main__":` guard is required: workers are started with the `spawn` method and import the script again ([Windows](../guide/batch.md#windows)). Pass `overwrite=True` (or `--overwrite`) to process finished videos again; this re-runs tracking and export from the cached detections and camera in each run directory, which took 3.2 s and 1.0 s for the two clips.

## Reading the results together

```python
import json
from pathlib import Path

import pandas as pd

rows = []
for summary in sorted(Path("runs").glob("*/summary.json")):
    s = json.loads(summary.read_text())
    rows.append({"video": summary.parent.name, "tracks": s["summary"]["tracks"],
                 "camera_height_m": s["camera"]["height_m"], "hfov_deg": s["camera"]["hfov_deg"],
                 "median_kmh": s["summary"]["vehicle_speed_kmh"]["median"]})
print(pd.DataFrame(rows).to_string(index=False))

samples = pd.concat([pd.read_csv(p).assign(video=p.parent.name) for p in Path("runs").glob("*/tracks.csv")])
print(samples.groupby(["video", "class"]).track_id.nunique().unstack(fill_value=0))
```

```text
            video  tracks  camera_height_m  hfov_deg  median_kmh
 intersection-day      43          28.3806    46.547       24.55
intersection-turn      23          25.8893    49.500       21.54
class              bicycle  bus  car  person  truck
video
intersection-day         2    2   30       9      0
intersection-turn        0    0   20       1      2
```

With `camera="auto"` each video is calibrated independently, so clips of different views or zoom settings get their own cameras (here 28.4 m and 25.9 m above the road, with 46.5° and 49.5° horizontal fields of view).

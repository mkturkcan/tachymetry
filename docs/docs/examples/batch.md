# Batch processing

This example processes two public sample clips in parallel with the batch scheduler ([Batch processing on multiple GPUs](../guide/batch.md)): `intersection-day` (1920×1080, 31 s) and `intersection-turn` (a 6.6 s clip of the same camera). The outputs below were produced on a machine with four NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, restricted to one of them with `--devices cuda:1`.

## Collect the clips

Both clips come from one camera pose, so the batch shares one camera instead of calibrating each clip on its own ([Sample clips](../samples.md)). `sample_camera` returns the camera packaged for this intersection:

```python
import shutil
from pathlib import Path

from tachymetry.data import sample_camera, sample_video

Path("videos").mkdir(exist_ok=True)
for name in ("intersection-day", "intersection-turn"):
    shutil.copy(sample_video(name), Path("videos") / f"{name}.mp4")
sample_camera("intersection-turn").save("intersection.camera.json")  # both clips share one camera pose
```

## Command line

```bash
tachymetry run videos/ --out runs --devices cuda:1 --camera intersection.camera.json
```

```text
Tachymetry: 2 of 2 videos processed in 117 s
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ video                 ┃ status ┃ tracks ┃ moving vehicles ┃ parked ┃ pedestrians ┃ median km/h ┃ seconds ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ intersection-day.mp4  │ done   │     49 │              22 │     17 │           7 │       24.83 │  115.14 │
│ intersection-turn.mp4 │ done   │     32 │              11 │     18 │           2 │       20.33 │   51.49 │
└───────────────────────┴────────┴────────┴─────────────────┴────────┴─────────────┴─────────────┴─────────┘
```

With free memory well above 18 GB (two SAM 3 workers of 9 GB each), `plan_workers` started two worker processes on the GPU, one per clip, each with its own SAM 3 model. The batch took 117 s of wall time for 167 s of per-video processing (detection, tracking and export; no rendering, and no calibration since the camera is given). Without `--devices`, all available GPUs are used. The output root holds one directory per video, named after the file stem, plus the batch summaries. With a given camera, `camera.json` is a copy of it and no calibration report or preview is written:

```text
runs/
  batch_summary.csv  batch_summary.json
  intersection-day/   background.jpg  camera.json  config.yaml  detections.json  detections.npz
                      summary.json  tracks.csv  tracks.json  tracks_summary.csv
  intersection-turn/  (same files)
```

```text
video,status,out_dir,tracks,vehicles_moving,pedestrians,median_speed_kmh,seconds,device,error
videos/intersection-day.mp4,done,runs/intersection-day,49,22,7,24.83,115.14,cuda:1,
videos/intersection-turn.mp4,done,runs/intersection-turn,32,11,2,20.33,51.49,cuda:1,
```

Add `--render` for an `overlay.mp4` per video, `--formats csv,json,geojson` for more exports, and `--set KEY=VALUE` (repeatable) for configuration fields ([Configuration](../guide/configuration.md)).

## Python

```python
import tachymetry as tm

if __name__ == "__main__":
    report = tm.track_many(["videos/"], out="runs", camera="intersection.camera.json", devices="cuda:1")
    print(report)
    for job in report.jobs:
        print(f'{job["status"]:6s} {job["out_dir"]}  {job["summary"]["tracks"]} tracks')
```

Run after the command above, both videos already have a `summary.json`, so both are skipped and their summaries are read from disk:

```text
BatchReport(2 done, 0 failed, 0 s, workers=['cuda:1'])
cached runs/intersection-day  49 tracks
cached runs/intersection-turn  32 tracks
```

The `if __name__ == "__main__":` guard is required: workers are started with the `spawn` method and import the script again ([Windows](../guide/batch.md#windows)). Pass `overwrite=True` (or `--overwrite`) to process finished videos again from scratch, ignoring the cached detections, background and camera in each run directory; this repeats detection. To change only tracking or export settings, run the pipeline on the existing run directory instead, which reuses the caches ([Pipeline](../guide/pipeline.md#caching-and-re-runs)).

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
 intersection-day      49          27.7095      47.5       24.83
intersection-turn      32          27.7095      47.5       20.33
class              bicycle  bus  car  person  truck
video
intersection-day         2    2   28       8      9
intersection-turn        0    0   27       3      2
```

Both runs used the shared camera (27.7 m above the road, 47.5° horizontal field of view). With `camera="auto"` each video is calibrated on its own, which suits clips of different cameras; for clips of one camera pose, calibrating each separately adds the per-clip calibration spread to the comparison ([Camera calibration](../guide/calibration.md)).

# Drone footage (straight-down views)

The sample `dhaka-roundabout` is 30 s of dense mixed traffic at a roundabout, recorded by a DJI drone that hovers 111.7 m above its take-off point and looks straight down. The video is 3840 x 2160 at 23.976 fps, from [DRINF](https://huggingface.co/datasets/hamimmahmud0/DRINF) by Hamim Mahmud, CC BY 4.0. Straight-down (nadir) views are processed with a known camera instead of automatic calibration. This page derives that camera from the flight log and the lens's field of view, runs the pipeline, explains why automatic calibration is not used here, and checks the result against the scene.

<figure markdown="span">
  ![Straight-down drone view of a roundabout with tracked road users](../../media/gallery/dhaka-roundabout.jpg)
  <figcaption>`dhaka-roundabout` with Tachymetry's output. Known camera: pitch 90°, height 111.7 m, f = 2825.2 px.</figcaption>
</figure>

## One command

```bash
tachymetry demo --sample dhaka-roundabout -o runs/dhaka
```

The demo downloads the 30 s excerpt (about 390 MB), prints its attribution and the camera, and runs the pipeline with the sample's known camera (`tachymetry.data.sample_camera`) and its configuration override for cycle-rickshaws (`tachymetry.data.sample_overrides`, [below](#cycle-rickshaws)). The sections below make each step explicit.

## The camera

A straight-down pinhole camera over flat ground is fixed by three quantities:

- **Pitch** 90°, roll 0°. The yaw only orients the world frame. The default `yaw=90` puts world $+y$ along the image's up direction.
- **Height** $h$ above the road: the drone's relative altitude, read from its flight log (below).
- **Focal length** from the horizontal field of view: $f = (w/2) / \tan(\mathrm{HFOV}/2)$. The DJI Mini 2 records 4K video with a 68.4° horizontal field of view. `DZOOM 1.000` in the flight log confirms that no digital zoom was applied, so $f = 1920 / \tan 34.2° = 2825.2$ px.

For this geometry the ground sampling distance is $h / f$ = 3.95 cm per pixel. It is the same everywhere in the image, because a nadir pinhole camera images the ground plane with a uniform scale. `Camera.ground_sampling_distance` returns 0.03954 m at the centre and at both corners.

### Altitude from the DJI subtitle track

DJI drones record per-second telemetry as a subtitle stream, inside the MP4 or as a sidecar `.SRT` file. The DRINF recording carries it as a `mov_text` stream. Extract it with the FFmpeg binary that `imageio-ffmpeg` installs; `-ss` and `-t` select the excerpt window:

```bash
FFMPEG=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FFMPEG" -v error -ss 140 -i DJI_0403_merged.mp4 -t 30 -map 0:s:0 flight.srt
```

Each entry looks like this:

```text
F/2.8, SS 100.00, ISO 400, EV -0.3, DZOOM 1.000, GPS (...), D 45.70m, H 111.70m, H.S 0.00m/s, V.S -0.00m/s
```

| Field | Meaning |
|---|---|
| `H` | altitude above the take-off point |
| `D` | horizontal distance from the take-off point |
| `H.S` | horizontal speed |
| `V.S` | vertical speed |
| `DZOOM` | digital zoom factor |

The following function collects `H` over a time window:

```python
import re
import statistics


def srt_altitudes(path, start_s=0.0, end_s=float("inf")):
    """DJI relative altitude ("H", metres above the take-off point) of each subtitle entry in [start_s, end_s)."""
    text = open(path, encoding="utf-8").read()
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        t = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", block)
        h = re.search(r"\bH (-?\d+(?:\.\d+)?)m\b", block)
        if t and h:
            hh, mm, ss, ms = map(int, t.groups())
            s = 3600 * hh + 60 * mm + ss + ms / 1000
            if start_s <= s < end_s:
                out.append(float(h.group(1)))
    return out


alt = srt_altitudes("flight.srt")
print(len(alt), min(alt), statistics.median(alt), max(alt))
```

```text
32 111.4 111.55 111.8
```

The altitude varies between 111.4 and 111.8 m during the excerpt. The sample camera uses 111.7 m, the first reading. Distances and speeds scale with $h$, so the 0.3 m range corresponds to 0.3 %.

`H` is the barometric height above the take-off point. If the drone took off from a level other than the road, add the difference.

## Python

```python
import math

import tachymetry as tm
from tachymetry.data import attribution, sample_video

video = sample_video("dhaka-roundabout")       # 30 s excerpt, 3840 x 2160 at 23.976 fps, about 390 MB
print(attribution("dhaka-roundabout"))

w, h = 3840, 2160
hfov = 68.4                                    # DJI Mini 2, 4K video, no digital zoom (DZOOM 1.000)
focal = (w / 2) / math.tan(math.radians(hfov / 2))
cam = tm.Camera.from_pose((w, h), focal=focal, height=111.7, pitch=90.0)
print(cam.describe())
print(f"ground sampling distance {100 * cam.height / cam.focal:.2f} cm/px")
cam.save("cam.json")

run = tm.track(
    video,
    out="runs/dhaka",
    camera=cam,
    hygiene__small_vehicle_rule=True,          # cycle-rickshaws, which no class prompt recognises
)
print(run.summary()["by_class"])
print(run.report["timings"])
```

```text
DRINF: Dhaka Road Intersection Nadir Footage by Hamim Mahmud (Hugging Face, doi:10.57967/hf/10301); 30 s excerpt from 2:20, CC BY 4.0, https://huggingface.co/datasets/hamimmahmud0/DRINF
f=2825.2px (hfov 68.4 deg), height=111.70 m, pitch=90.00 deg, roll=0.00 deg, yaw=90.0 deg, lens=identity
ground sampling distance 3.95 cm/px
{'motorcycle': 208, 'car': 45, 'bicycle': 21, 'person': 32, 'truck': 8}
{'detect_s': 47.47, 'calibrate_s': 17.28, 'track_s': 100.95, 'total_s': 166.56}
```

On one NVIDIA RTX PRO 6000 GPU, SAM 3 detection took 47 s for the 360 analysed frames. The background median and the camera took 17 s, and tracking 101 s, most of it the mask geometry of the vehicle tracks on the CPU. The class counts are those of the accepted tracks ([class vote](../guide/tracking.md#class-vote)).

- **Motorcycles.** 175 of the 208 motorcycles are classed from their evidence. The other 33 are rickshaws that the small-vehicle rule labels motorcycles ([below](#cycle-rickshaws)). Seen from above, the `bike` prompt misses most motorcycles: `motorcycle` evidence is found on 51 % of the vehicle detections and `bike` evidence on 17 %. The two-wheeler score of a detection is the larger of the two ([class vote](../guide/tracking.md#class-vote)).
- **Riders.** 34 person tracks are riders and are not among the 32 persons: 31 ride a tracked two-wheeler, and 3 move at cycling speed for less than 2 s. 3 more person tracks move that fast for longer, without a two-wheeler, and become cyclists ([Verdicts](../guide/verdicts.md#riders-inferred-cyclists-and-passengers)).
- **Small road users.** In steep views, persons and two-wheelers are placed at their mask centroid projected at a class height, not at their lowest mask point (`hygiene.steep_view_deg`, 60° to 80° of depression). Their lowest point is a foot or a wheel, far from the centre seen from above.

### Cycle-rickshaws

Cycle-rickshaws and rickshaw vans receive almost no evidence from any class prompt, and the class scheme has no three-wheeler class. By default the size fallback of the class vote labels them cars. Their car fits stay at the smallest car, and `hygiene.small_vehicle_rule` makes such tracks two-wheelers, the nearest classes of the scheme. The rule is off by default; this sample turns it on (`sample_overrides("dhaka-roundabout")`), so the demo applies it. In the run above, 55 tracks have no usable evidence. The car fits of 51 of them stay at about 3.05 × 1.55 m and end in that branch, nearly all of them cycle-rickshaws and rickshaw vans; the other 4 remain cars. The rule labels 18 of the 51 bicycles and 33 motorcycles; the 33 are those that the `motorcycle` prompt scored above `bike`, both far below the 0.20 share the vote needs.

With the rule, the run gives 45 cars, 208 motorcycles and 21 bicycles in 314 accepted tracks. Without it, the same detections give 98 cars, 175 motorcycles and 3 bicycles in 316 tracks. The totals differ because fragments of a track are stitched together only when both are vehicles or both are two-wheelers, and one more track fails the plausibility checks with the rule. CNG auto-rickshaws (motorised three-wheelers) with `car` or `van` evidence are cars either way ([Limitations](../limitations.md#mixed-traffic-and-class-dependent-scale)).

## Command line

```bash
python -c "import math, tachymetry as tm; tm.Camera.from_pose((3840, 2160), focal=1920 / math.tan(math.radians(34.2)), height=111.7, pitch=90.0).save('cam.json')"
tachymetry run ~/.cache/tachymetry/samples/dhaka-roundabout.mp4 --camera cam.json --out runs --render \
    --set hygiene.small_vehicle_rule=true
```

The sample is stored under `$TACHYMETRY_CACHE/samples`, by default `~/.cache/tachymetry/samples`, once `sample_video("dhaka-roundabout")` or the demo has fetched it. A camera given with `--camera` is used as is; it is resized if the video's resolution differs. For a flight with a different altitude or lens, change `height` and the field of view.

## Why automatic calibration is not used

Automatic calibration ([Automatic calibration](../guide/auto-calibration.md)) assumes an oblique view. In a straight-down view, the road's parallel markings stay parallel in the image, so their vanishing points lie at infinity, and perspective does not change the shape of vehicle silhouettes across the image. The script below runs it on the detections of the run above:

```python
import warnings

import cv2

from tachymetry.calib import calibrate_auto
from tachymetry.detections import DetectionSet

dets = DetectionSet.load("runs/dhaka/detections.npz")
background = cv2.imread("runs/dhaka/background.jpg")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    res = calibrate_auto(background, dets)
print(res.method, res.camera.describe(), res.details["scale"])
for w in caught:
    print("warning:", str(w.message).split(";")[0])
```

```text
auto:lines f=10074.4px (hfov 21.6 deg), height=1.00 m, pitch=2.78 deg, roll=0.17 deg, yaw=-4.7 deg, lens=identity {'scale': 'relative'}
warning: car silhouettes do not fit this camera (cost 2.07)
warning: dashed lane markings change length along the road under this camera (spread 0.47 in log units): its pitch or focal length is likely wrong
```

The `lines` candidate reads nearly parallel lines as a distant vanishing point and returns a near-level camera with a long focal length. Both checks reject it:

- the car silhouettes do not fit it, so it keeps relative scale (height 1);
- the dashed markings change length along the road under it.

The scene contradicts such a camera. The roundabout's central island images as a circle: an ellipse fitted to its outline has an axis ratio of 0.98, while a level camera would image it as a flat ellipse. With UrbanOmniDetect-2, whose cuboid footprints are compressed along the image vertical in nadir views, the cuboid solver reads the compression as perspective instead. `Pipeline` warns when an automatic calibration places the camera more than 250 m above the ground ([Camera calibration](../guide/calibration.md#implausible-automatic-calibrations)).

## Checks of the known camera

The known camera is not compared with ground truth: the clip has none. The quantities below are independent of the calibration inputs and should be read as consistency checks.

| Check | Measured | Reference |
|---|---|---|
| Width of the swimming pool beside the roundabout, in the bird's-eye view | 24.3 m | 25 m if it is a standard 50 m pool; its far end lies outside the frame |
| Roundabout island in the image | circle, axis ratio 0.98; diameter 29.7 m | a circular island under a nadir view |
| Pedestrians, median of per-track median speeds (28 moving tracks) | 4.8 km/h | free walking speeds of 1.2 to 1.4 m/s (4.3 to 5.0 km/h) |
| Cars, median of per-track median speeds (28 moving tracks) | 11.1 km/h | congested roundabout |
| Motorcycles, median of per-track median speeds (187 moving tracks, some rickshaws included) | 12.6 km/h | congested roundabout |

The speeds are computed from `tracks_summary.csv` of the run above:

```python
import pandas as pd

s = pd.read_csv("runs/dhaka/tracks_summary.csv")
moving = s[(s.verdict == "ok") & (s.state == "moving")]
print(moving.groupby("class").speed_median_kmh.median().round(1))
print(moving["class"].value_counts().to_dict())
```

```text
class
bicycle       13.1
car           11.1
motorcycle    12.6
person         4.8
truck          7.7
Name: speed_median_kmh, dtype: float64
{'motorcycle': 187, 'car': 28, 'person': 28, 'bicycle': 17, 'truck': 6}
```

## Rendering straight-down views

A nadir camera has no horizon in the image: `Camera.below_horizon` accepts all finite pixels. The bird's-eye view is oriented by `Camera.ground_up()`, the ground direction that appears as up in the image. It sums the ground projections of the optical axis and of the image's up axis. The first projection vanishes for a nadir camera and the second for a level one, so the orientation is defined for all pitches in between. Detections without a metric track are drawn as corner brackets ([Rendering](../guide/rendering.md#what-is-drawn)).

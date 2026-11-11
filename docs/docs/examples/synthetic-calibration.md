# Synthetic calibration walkthrough

This example calibrates a rendered intersection whose camera is known, with each calibration method, and compares the estimates with the ground truth. Everything runs on a CPU in about 50 s and needs no downloads.

## Scene and evidence

`tachymetry.synthetic` renders a metric ground texture (asphalt, lane lines, zebra crosswalks, stop lines, kerbs, sidewalk joints) through a known camera with per-pixel mip-mapping, and adds shaded boxes for vehicles, signal poles and buildings (`synthetic/scene.py`). `synthetic.traffic.simulate` produces the detections of simulated traffic seen by the same camera, in two forms (`synthetic/traffic.py`):

- **Instance masks** (`TrafficSpec(masks=True)`), as reported by a segmenting detector such as the default SAM 3 (`synthetic/masks.py`).
    - *Silhouettes.* Each vehicle is drawn as the union of a few boxes: for cars, a body up to 55 % of the roof height, a cabin over the rear 55 % of the length and 85 % of the width, and four wheels.
    - *Occlusion.* Silhouettes are painted far to near, so each mask is the visible part of its vehicle, with one-pixel boundary noise; masks keeping less than 20 % of their silhouette are not detected.
    - *Boxes.* Each detection box is the bounding box of its mask.
- **Cuboid corners** (default `TrafficSpec`), as reported by UrbanOmniDetect-2: the projected corners of each vehicle's 3D box with corner noise, and jittered boxes.

Both forms include missed detections, label flicker and false positives.

The ground-truth camera is `default_camera((1280, 720))`: 8 m above the ground, pitch 32 degrees, roll 1.5 degrees, yaw 55 degrees, horizontal field of view 70 degrees ($f$ = 914.0 px), no lens distortion.

<figure markdown="span">
  ![Synthetic intersection rendered through the ground-truth camera](../../media/syn_scene.jpg)
  <figcaption>`make_scene(default_camera((1280, 720)), RoadLayout(), seed=0)`: a four-way intersection with two lanes per direction, crosswalks, stop lines, queued and parked vehicles, poles and buildings.</figcaption>
</figure>

## Calibrating with each method

```python
import numpy as np
from tachymetry.calib import Correspondences, calibrate_auto, calibrate_from_lines, calibrate_manual
from tachymetry.config import CalibrationConfig
from tachymetry.synthetic import RoadLayout, default_camera, make_scene
from tachymetry.synthetic.traffic import TrafficSpec, simulate

truth = default_camera((1280, 720))                                  # the camera to recover
scene = make_scene(truth, RoadLayout(), seed=0)                      # traffic-free image of a marked intersection
masks, _ = simulate(truth, TrafficSpec(duration_s=60, seed=0, masks=True))  # segmenting detector: instance masks
cuboids, _ = simulate(truth, TrafficSpec(duration_s=60, seed=0))     # cuboid detector (UrbanOmniDetect-2 path)

rng = np.random.default_rng(0)                                       # 12 surveyed points, 1 px click noise
uv = rng.uniform([0, 0], [1279, 719], size=(4000, 2))
uv = uv[truth.ground_sampling_distance(uv) <= 0.30][:12]
corr = Correspondences(uv + rng.normal(0, 1.0, uv.shape), truth.image_to_ground(uv), image_size=(1280, 720))

results = {
    "manual": calibrate_manual(corr),
    "lines": calibrate_from_lines(scene.image, scale="height=8"),
    "masks": calibrate_auto(scene.image, masks),
    "masks+fov": calibrate_auto(scene.image, masks, CalibrationConfig(hfov_deg=70)),
    "cuboids": calibrate_auto(scene.image, cuboids),
}
for name, res in results.items():
    cam = res.camera
    print(f"{name:9s} [{res.method:13s}] focal {100 * (cam.focal / truth.focal - 1):+6.2f} %   "
          f"pitch {cam.pitch - truth.pitch:+6.2f} deg   roll {cam.roll - truth.roll:+6.2f} deg   "
          f"height {cam.height:6.2f} m")
```

Output (Linux, NumPy 2.4, OpenCV 4.13; 45 s on a CPU, most of it simulating the masks and fitting the silhouettes):

```text
manual    [manual       ] focal  -0.04 %   pitch  +0.03 deg   roll  -0.05 deg   height   7.99 m
lines     [lines        ] focal  -0.39 %   pitch  -0.01 deg   roll  -0.09 deg   height   8.00 m
masks     [auto:lines   ] focal  -0.39 %   pitch  -0.01 deg   roll  -0.09 deg   height   7.71 m
masks+fov [auto:fov+vp  ] focal  +0.00 %   pitch  -0.11 deg   roll  -0.09 deg   height   7.75 m
cuboids   [auto:lines   ] focal  -0.39 %   pitch  -0.01 deg   roll  -0.09 deg   height   7.95 m
```

No chain of five or more dashes lies on the road surface of this scene, so the [dash check](../guide/calibration.md#dash-regularity) does not apply.

How each estimate is obtained:

- **manual** fits the physical camera to 12 correspondences, chooses the lens model by 10-fold cross-validation, and inherits its metric scale from the world coordinates ([Manual calibration](../guide/manual-calibration.md)).
- **lines** recovers focal length, pitch and roll from the vanishing points of the painted markings and the vertical structures. The 15 pole edges of the scene do not contradict right-angled streets, so the two street directions give the focal length ([Focal length and rotation](../guide/auto-calibration.md#road-markings-and-vanishing-points)). Its height is the scale cue (`scale="height=8"`), so its height error is zero by construction and its accuracy is visible in the focal length and angles.
- **masks** is the default path of `tachymetry run` with a segmenting detector. Without cuboids or a field of view, `lines` is the only candidate.
    - The metric scale comes from the vehicle silhouettes ([Automatic calibration](../guide/auto-calibration.md#metric-scale-from-vehicle-silhouettes)): factor 7.71 from 92 masks of 28 tracklets, mean cost 0.47.
    - The silhouette model of the calibration (beltline at 63 % of the height, cabin 55 % by 75 %) differs from the simulated car shape above, as a real car population would.
    - The height error (−3.6 %) is close to the one with the true field of view below (−3.1 %), so it comes mostly from the difference between the two car shapes; the silhouette scale also inherits any error of the `lines` orientation.
- **masks+fov** adds the true 70° field of view. The candidate built from it, `fov+vp` ($f$ = 914 px, pitch 31.89°, roll 1.41°), is selected, as a known field of view makes it the choice over `lines` ($f$ = 910.5 px). The focal length is then the true one; the pitch error is −0.11° and the scale error +2.4 %.
- **cuboids** is the UrbanOmniDetect-2 path. The candidates scored `lines` 0.326, `cuboids` 0.279 and `fused` 0.270 ([Ranking](../guide/auto-calibration.md#ranking)); `lines` was selected and scaled from 351 trusted car footprints to 7.95 m (−0.6 %).

Without a field of view, the automatic methods selected the `lines` orientation (focal length −0.39 %, pitch −0.01°); they differ only in the metric scale. With the field of view, `fov+vp` has the true focal length by construction. The table below measures the consequence on the ground plane directly.

## Measured errors

The table is produced by `tools/make_docs_media.py` on the same scene and traffic.

--8<-- "docs/media/synthetic_calibration.md"

**Protocol.**

- **Errors.** All errors are relative to the ground-truth camera. *Focal error* is $f / f_\mathrm{true} - 1$; *pitch*, *roll* and *height* errors are differences.
- **Ground comparison.** A grid of image points whose true ground sampling distance is at most 0.30 m per pixel is used: the region in which the tracker measures (`hygiene.max_gsd_m`). Each point is back-projected with the true camera and with the estimate.
- **Rigid alignment.** For methods whose world frame is arbitrary (yaw and origin chosen by the calibration), the estimated ground points are aligned to the true ones by the least-squares 2D rigid transform (rotation and translation, no scale). *Ground RMS after rigid alignment* is the remaining RMS distance in metres; it contains the scale error.
- **Scale error.** The scale factor of a separate least-squares similarity fit, minus one, i.e. the bias that all distances and speeds would inherit.
- **Rows.**
    - *Manual:* 12 ground points sampled with a fixed seed from the same region, projected through the true camera with 1.0 px Gaussian noise, and calibrated with `lens="auto"`.
    - *Road markings:* scaled with the true height of 8 m.
    - *Automatic rows:* `calibrate_auto` on the background image with 60 s of simulated instance masks, with the masks and the known 70° field of view (`CalibrationConfig(hfov_deg=70)`), and with 60 s of simulated cuboid detections, as in the script above.

## Previews

<figure markdown="span">
  ![Manual calibration preview](../../media/syn_calib_manual.jpg)
  <figcaption>Manual calibration from 12 noisy correspondences.</figcaption>
</figure>

<figure markdown="span">
  ![Road-marking calibration preview](../../media/syn_calib_lines.jpg)
  <figcaption>Calibration from road markings and vertical structures, scaled with the known height of 8 m.</figcaption>
</figure>

<figure markdown="span">
  ![Automatic calibration preview](../../media/syn_calib_auto.jpg)
  <figcaption>Automatic calibration (`calibrate_auto`) from the background image and 60 s of simulated instance masks, scaled from the vehicle silhouettes.</figcaption>
</figure>

## Variations

- `default_camera(size, height=..., pitch=..., hfov=..., position=..., yaw=...)` changes the viewpoint; `RoadLayout(kind="straight", lanes_ew=3, rotation_deg=20)` changes the road.
- `default_camera(size, lens=DivisionLens(size, k=-0.2))` renders through a distorted lens; `calibrate_from_lines(..., lens="auto")` then estimates the distortion (see [Lens rectification](lens-rectification.md)).
- `TrafficSpec(vehicles_per_min=..., parked=..., miss_prob=..., corner_noise=...)` controls the amount and quality of the evidence. With `masks=True`, `occluders=(tree_occluder(...),)` (`tachymetry.synthetic.masks`) adds image-space occluders in front of the traffic, and `mask_noise` and `min_visible` control the masks.

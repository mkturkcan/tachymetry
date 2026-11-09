# Technical notes and limitations

Tachymetry measures road users on a single calibrated ground plane from a single static camera. This page states the assumptions behind every metric output, quantifies where they break down, and lists what the track filters can and cannot detect. Notation follows [Camera calibration](guide/calibration.md#camera-model-and-conventions).

## Flat ground

Every position is the intersection of a pixel ray with the plane $z = 0$ (`Camera.image_to_ground`). A road surface point that lies $\Delta z$ above the calibrated plane, at horizontal distance $d$ from a camera at height $h$, is back-projected to a point displaced away from the camera by

$$
\delta = \frac{d\,\Delta z}{h - \Delta z} \approx \frac{d}{h}\,\Delta z .
$$

For a camera 8 m high, 0.2 m of unmodelled elevation at 40 m range displaces positions by 1.0 m. Sources of $\Delta z$ are grades that differ between the calibration region and the measurement region, road crowns (a 2 % cross slope raises the centre of a 3.5 m lane by 3.5 cm relative to its edge), kerbs and raised medians, bridges and ramps. Speeds are affected through the derivative of $\delta$ along the path: on a uniform grade that is not part of the calibrated plane, the error grows with range. Off-plane points also bias calibration: correspondences on kerb tops or sidewalks and vehicle footprints on a sloped approach pull the fitted plane.

Objects are reduced to a ground point. With a trusted cuboid the footprint centroid is used; otherwise the ground contact under the box bottom is pushed along the viewing ray by half the footprint extent, minus the parallax of the lowest visible edge, using class priors of the ground clearance (car 0.08 m, bus and truck 0.25 m, two-wheelers 0.05 m, persons 0; `track/measure.py`, `track/hygiene.py`). Errors of these priors scale with $d / h$ as above.

## Far field and ground sampling distance

The ground extent of one pixel along the viewing ray grows quadratically with range. On the optical axis of a camera at height $h$ with focal length $f$ (pixels), the along-ray ground sampling distance at horizontal distance $d$ is

$$
\mathrm{GSD}(d) \approx \frac{d^2 + h^2}{f\,h},
$$

so a fixed error of one pixel in the box bottom or a cuboid corner becomes a ground error proportional to $d^2 / (f h)$, and speed noise grows in the same way. `Camera.ground_sampling_distance` evaluates the worst-axis value from the image-to-ground Jacobian at any pixel. Measurements are discarded where it exceeds `hygiene.max_gsd_m` = 0.30 m per pixel; the calibration preview and the bird's-eye-view renderer use the same bound.

| Camera | $f$ (px) | $h$ (m) | Range at 0.05 m/px | at 0.10 m/px | at 0.30 m/px |
|---|---|---|---|---|---|
| Synthetic pole camera (`default_camera((1280, 720))`) | 914 | 8.0 | 17.6 m | 26.9 m | 43.0 m |
| Public sample clip, automatic calibration (`intersection-day`) | 2163 | 28.4 | 49.0 m | 73.2 m | 133.7 m |

Ranges are along the central image column, computed with `Camera.ground_sampling_distance`. Within the usable region, speed accuracy still varies by an order of magnitude between near and far field. The [benchmarks](benchmarks.md) report errors aggregated over the usable region; per-range errors for a specific installation can be obtained with `tachymetry.synthetic` at the calibrated viewpoint.

Other far-field effects: boxes of distant vehicles collapse when the detector loses them, which the end trimming handles only at track ends (`hygiene.far_shrink` = 0.5); cuboid footprints are less often trusted; and the class vote down-weights small, peripheral boxes.

## Occlusion

There is no explicit occlusion reasoning. Missing detections are bridged by the tracker (lost tracks coast for `tracker.max_lost_s` = 1.5 s), interpolated up to `hygiene.interp_gap_s` = 1.5 s, split at gaps above `hygiene.max_gap_s` = 5.0 s, and fragments are re-joined by stitching across gaps of up to `filters.stitch_max_gap_s` = 3.0 s when the start of the later fragment lies within 2.5 m plus 25 % of the predicted travel distance of the extrapolated end of the earlier one. Boxes that touch the image border are flagged as truncated and contribute no position. A box that is partially occluded inside the image is not flagged: its bottom edge, and with it the ground contact, shifts towards the camera or away from it, and only the robust smoothing and outlier repair (Hampel filter, 3 MAD and 0.8 m) limit the effect. Vehicles that are hidden for their whole passage are not counted; counts are lower bounds in dense queues seen from a low camera.

## Night and low light

Detection recall and calibration evidence both degrade at night. The cached runs of two public clips of the same intersection illustrate this (both at the default 15 Hz processing rate, UrbanOmniDetect-2 X):

| Clip | Duration | Detections | Tracks kept | Tracks by class | Calibration |
|---|---|---|---|---|---|
| `intersection-day` | 31.0 s | 23,575 | 44 | 31 car, 2 bus, 2 bicycle, 9 person | `auto:fused` |
| `intersection-night` | 18.5 s | 3,412 | 8 | 8 car | `auto:lines` |

Traffic volumes differ between the recordings, so the comparison is indicative only. At night no pedestrian or cyclist track was kept, and the calibration candidates disagreed more: the `lines`, `fused` and `cuboids` cameras of the night clip had pitches of 21.25, 19.78 and 19.60 degrees (`camera.report.json`). Headlight glare can displace the bottom edges of vehicle boxes, and road markings are visible mainly where they are lit. For cameras that record in both conditions, calibrate on daytime footage and pass the camera file to the night runs.

## Moving, PTZ and re-zoomed cameras

Each video is calibrated once and the camera is assumed static for the whole video. A pan, tilt or zoom change within a video invalidates all positions after the change; split such recordings at the preset change. Wind sway and vibration move the whole image, which appears as small common motion of all objects; parked-vehicle freezing and the hold-still rule remove it from stationary objects, but not from moving ones.

When a run directory is reused, an automatic `camera.json` cached there is reused if its image size matches (`Pipeline.calibrate`). If the camera moved between recordings that share an output directory, delete `camera.json` or use a new output directory (`--overwrite` alone re-processes the video but still reuses the cached camera). For batches of different cameras, pass a directory of per-video cameras (`--camera cameras/`, files named `<video stem>.camera.json` or `<video stem>.json`).

The principal point is fixed at the image centre for all automatic methods. Streams that are cropped or digitally zoomed off-centre should be calibrated manually with `principal_point=`.

## Calibration failure modes

| Method | Fails or degrades when | Signature |
|---|---|---|
| `manual` | points cluster in a small area (extrapolation), points are off the ground plane, the reference GSD is wrong, a point is mis-clicked | large `cv_rmse_m` relative to `rmse_m`; large individual `residuals_m`; wrong lane widths in the preview |
| `lines` | fewer than 8 marking segments (worn or no markings, snow, wet glare, night); only one marked direction and no vertical structure; non-orthogonal streets without a vertical vanishing point (the fallback assumes orthogonality) | `CalibrationError`; `details["assumes_orthogonal_streets"]`; converging lane lines in the bird's-eye view |
| `cuboids` | fewer than 40 trusted car footprints (light traffic, short clips, night, distant views); viewpoints far from the detector's training distribution | `CalibrationError`; large footprint right-angle error in `details["metrics"]` |
| `fused`, `auto` | the ranking prefers a candidate that is consistent with the evidence but wrong; the `lines` candidate scores a near-zero vanishing-point residual by construction | inspect `details["candidates"]` and `calib.compare` previews |
| `depth` | the depth network misjudges the scale of high-mounted views; focal length error couples into pitch; segmentation includes non-road pixels | `details["warnings"]` (height outside 2.5 to 150 m); low `inlier_ratio` |
| `geocalib` | low-texture or unusual views; no metric scale without `height` | large posterior uncertainties, low `score`; `meta["metric"] = False` |

Accept a calibration only after inspecting its preview (see [Verifying a calibration](guide/calibration.md#verifying-a-calibration)).

## Metric scale

Methods based on image geometry recover the camera up to scale; the scale then comes from one cue, and every distance and speed inherits its error.

- **Vehicle sizes** (default of automatic calibration) assume a median passenger-car footprint of 4.6 m by 1.85 m. The width estimate carries 80 % of the weight ([Metric scale](guide/calibration.md#metric-scale)), so a local fleet whose median car is 5 % narrower and 5 % shorter biases all distances and speeds by about +5 %. A systematic bias of the detector's cuboid corners (footprints regressed slightly too large or too small) propagates in the same way. The 12 % and 6 % dispersions that set the weights are assumptions in `calib/scale.py`, not measurements.
- **Depth networks** provide a metric scale that can be wrong by a large factor for viewpoints unlike their training data; within `calibrate_auto` it is replaced by the vehicle-size scale by default.
- **Relative scale.** Without any cue (`lines` without `scale`, `geocalib` without `height`, or `calibrate_auto` when fewer than 10 footprints exist), the camera has height 1, `meta["metric"]` is `false`, and positions and speeds are in arbitrary units. Check `camera.json` before using the outputs.
- The latitude/longitude conversion of manual calibration uses a sphere of radius 6,378,137 m, which scales north-south distances by +0.42 % and east-west distances by −0.08 % relative to the WGS84 ellipsoid at 30 degrees latitude ([Manual calibration](guide/manual-calibration.md#correspondence-files)).

## What the track filters can and cannot detect

The per-track referees and cross-track verdicts ([Association and track hygiene](guide/tracking.md), [Verdicts](guide/verdicts.md)) test physical plausibility and mutual consistency. They detect:

- sustained speeds above the class limit (vehicles 60 m/s, two-wheelers 25 m/s, persons 9 m/s) or accelerations above 12 m/s² for at least 0.6 s, and path reversals sharper than 150 degrees;
- tracks whose footprint centre projects outside their own box for more than 30 % of the observed samples;
- short-lived or low-confidence static tracks, static low-confidence "vehicles" without a 3D footprint, duplicate tracks of one object, passengers inside vehicles, riders on two-wheelers, and person tracks moving in lockstep with a vehicle.

They cannot detect:

- **A consistent calibration error.** A wrong scale or pitch changes all tracks coherently and leaves speeds plausible; only the calibration preview and independent ground truth reveal it.
- **Plausible identity swaps.** Two similar vehicles that exchange identities while moving side by side at similar speeds produce two plausible tracks.
- **Plausible false positives.** A persistent false detection with a trusted cuboid and a high score, or a reflection that moves like a vehicle, passes every test.
- **Missed objects.** Objects that are never detected produce no track; counts are lower bounds.
- **Class errors** that are consistent over a track (a van labelled as a truck); the class is a weighted vote over the track's detections.
- **Objects off the ground plane**, such as vehicles on an overpass, which are measured as if they were on the road below.

## Frame rate and timestamps

Frame times are derived from the frame index and the container frame rate (`io/video.py`): $t_k = k / \mathrm{fps}$, with fps read from the container and replaced by 30 when it is missing or outside 1 to 1000. Variable-frame-rate recordings, and recorders that drop frames without duplicating them, violate this assumption; a timing error of $\epsilon$ (relative) produces a speed error of $\epsilon$ (relative). Re-encode such files to a constant frame rate first. Folders of images are read at 10 frames per second (`VideoSource(path, fps=None)`), and the pipeline currently has no option to change this rate; pass a video file when the frame rate matters.

Detection runs on every `stride`-th frame with $\mathrm{stride} = \mathrm{round}(\mathrm{fps} / \texttt{rate\_hz})$ (default `rate_hz` = 15), so the effective rate is $\mathrm{fps}/\mathrm{stride}$: 15 Hz for 30 fps video and 12.5 Hz for 25 fps video. Thresholds are expressed in seconds and metres (`config.py`), so the configuration does not change with the rate; the number of samples per filter window does. Rolling-shutter skew is not modelled.

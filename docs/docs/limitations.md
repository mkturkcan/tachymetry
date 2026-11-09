# Technical notes and limitations

Tachymetry measures road users on a single calibrated ground plane from a single static camera. This page states the assumptions behind every metric output, quantifies where they break down, and lists what the track filters can and cannot detect. Notation follows [Camera calibration](guide/calibration.md#camera-model-and-conventions).

## Flat ground

Every position is the intersection of a pixel ray with the plane $z = 0$ (`Camera.image_to_ground`). A road surface point that lies $\Delta z$ above the calibrated plane, at horizontal distance $d$ from a camera at height $h$, is back-projected to a point displaced away from the camera by

$$
\delta = \frac{d\,\Delta z}{h - \Delta z} \approx \frac{d}{h}\,\Delta z .
$$

For a camera 8 m high, 0.2 m of unmodelled elevation at 40 m range displaces positions by 1.0 m. Sources of $\Delta z$ are grades that differ between the calibration region and the measurement region, road crowns (a 2 % cross slope raises the centre of a 3.5 m lane by 3.5 cm relative to its edge), kerbs and raised medians, bridges and ramps. Speeds are affected through the derivative of $\delta$ along the path: on a uniform grade that is not part of the calibrated plane, the error grows with range. Off-plane points also bias calibration: correspondences on kerb tops or sidewalks and vehicle footprints on a sloped approach pull the fitted plane.

Objects are reduced to a ground point. With a trusted cuboid the footprint centroid is used; otherwise the ground contact under the box bottom is pushed along the viewing ray by half the footprint extent, minus the parallax of the lowest visible edge, using class priors of the ground clearance (car 0.08 m, bus and truck 0.25 m, two-wheelers 0.05 m, persons 0; `tracking/measure.py`, `tracking/hygiene.py`). Errors of these priors scale with $d / h$ as above.

## Far field and ground sampling distance

The ground extent of one pixel along the viewing ray grows quadratically with range. On the optical axis of a camera at height $h$ with focal length $f$ (pixels), the along-ray ground sampling distance at horizontal distance $d$ is

$$
\mathrm{GSD}(d) \approx \frac{d^2 + h^2}{f\,h},
$$

so a fixed error of one pixel in the box bottom or a cuboid corner becomes a ground error proportional to $d^2 / (f h)$, and speed noise grows in the same way. `Camera.ground_sampling_distance` evaluates the worst-axis value from the image-to-ground Jacobian at any pixel. Measurements are discarded where it exceeds `hygiene.max_gsd_m` = 0.30 m per pixel; the calibration preview and the bird's-eye-view renderer use the same bound.

| Camera | $f$ (px) | $h$ (m) | Range at 0.05 m/px | at 0.10 m/px | at 0.30 m/px |
|---|---|---|---|---|---|
| Synthetic pole camera (`default_camera((1280, 720))`) | 914 | 8.0 | 17.6 m | 26.9 m | 43.0 m |
| Public sample clip, automatic calibration (`intersection-day`) | 2212 | 27.8 | 49.3 m | 73.5 m | 133.9 m |

Ranges are along the central image column, computed with `Camera.ground_sampling_distance`. Within the usable region, speed accuracy still varies by an order of magnitude between near and far field. The [benchmarks](benchmarks.md) report errors aggregated over the usable region; per-range errors for a specific installation can be obtained with `tachymetry.synthetic` at the calibrated viewpoint.

Other far-field effects: boxes of distant vehicles collapse when the detector loses them, which the end trimming handles only at track ends (`hygiene.far_shrink` = 0.5); cuboid footprints are less often trusted; and the class vote down-weights small, peripheral boxes.

## Occlusion

There is no explicit occlusion reasoning. Missing detections are bridged by the tracker (lost tracks coast for `tracker.max_lost_s` = 1.5 s), interpolated up to `hygiene.interp_gap_s` = 1.5 s, split at gaps above `hygiene.max_gap_s` = 5.0 s, and fragments are re-joined by stitching across gaps of up to `filters.stitch_max_gap_s` = 3.0 s when the start of the later fragment lies within 2.5 m plus 25 % of the predicted travel distance of the extrapolated end of the earlier one. Boxes that touch the image border are flagged as truncated and contribute no position. A box that is partially occluded inside the image is not flagged: its bottom edge, and with it the ground contact, shifts towards the camera or away from it, and only the robust smoothing and outlier repair (Hampel filter, 3 MAD and 0.8 m) limit the effect. Vehicles that are hidden for their whole passage are not counted; counts are lower bounds in dense queues seen from a low camera.

## Night and low light

Detection recall and calibration evidence both degrade at night. Runs on two public clips of the same intersection illustrate this (both at the default 15 Hz processing rate, UrbanOmniDetect-2 X):

| Clip | Duration | Detections | Tracks kept | Tracks by class | Calibration |
|---|---|---|---|---|---|
| `intersection-day` | 31.0 s | 23,575 | 45 | 32 car, 2 bus, 2 bicycle, 9 person | `auto:fused` |
| `intersection-night` | 18.5 s | 3,412 | 8 | 8 car | `auto:lines` |

Traffic volumes differ between the recordings, so the comparison is indicative only. At night no pedestrian or cyclist track was kept, and the calibration candidates disagreed more: the `lines`, `fused` and `cuboids` cameras of the night clip had pitches of 21.25, 19.74 and 19.79 degrees (`camera.report.json`). Headlight glare can displace the bottom edges of vehicle boxes, and road markings are visible mainly where they are lit. For cameras that record in both conditions, calibrate on daytime footage and pass the camera file to the night runs.

## Moving, PTZ and re-zoomed cameras

Each video is calibrated once and the camera is assumed static for the whole video. A pan, tilt or zoom change within a video invalidates all positions after the change; split such recordings at the preset change. Wind sway and vibration move the whole image, which appears as small common motion of all objects; parked-vehicle freezing and the hold-still rule remove it from stationary objects, but not from moving ones.

When a run directory is reused, an automatic `camera.json` cached there is reused if its image size matches (`Pipeline.calibrate`). If the camera moved between recordings that share an output directory, use a new output directory or re-run with `--overwrite`, which ignores the cached detections, background and camera. For batches of different cameras, pass a directory of per-video cameras (`--camera cameras/`, files named `<video stem>.camera.json` or `<video stem>.json`).

The principal point is fixed at the image centre for all automatic methods. Streams that are cropped or digitally zoomed off-centre should be calibrated manually with `principal_point=`.

## Straight-down views

Automatic calibration is designed for oblique views, in which perspective changes the image shape of ground rectangles across the image. A camera that looks straight down images the ground plane at a uniform scale, so that signal is absent. Two biases then dominate.

- **Footprint bias.** The cuboid footprints that UrbanOmniDetect-2 regresses in nadir views are compressed along the image vertical. The cuboid solver explains the compression as perspective and returns a tilted camera with a long focal length at a large height.
- **Unreliable metric scale.** The scale derived from that camera is unreliable.

`Pipeline` warns when an automatic calibration places the camera more than `MAX_PLAUSIBLE_HEIGHT_M` = 250 m above the ground, and stores the note in `camera.meta["warnings"]` ([Camera calibration](guide/calibration.md#implausible-automatic-calibrations)). Process straight-down footage with a known camera instead: pitch 90°, the height from the flight log, and the focal length from the lens's field of view. Below 250 m a false tilt is not flagged, so inspect the calibration preview of each aerial clip. [Drone footage](examples/drone.md) works through one sample and quantifies both biases.

## Mixed traffic and class-dependent scale

The vehicle-size scale assumes that the `car` class is dominated by passenger cars with a median footprint of 4.6 m by 1.85 m ([Metric scale](guide/calibration.md#metric-scale)). Where a detector maps other vehicles to `car`, the scale inherits their size, and all distances and speeds are multiplied by the ratio of the prior to their dimensions. Examples are three-wheeled auto-rickshaws, which are about 2.6 m long, or microcars. On the straight-down sample the vehicle-size cue would multiply all distances by 1.83 ([Drone footage](examples/drone.md#why-automatic-calibration-is-not-used)). In such traffic, supply the scale from elsewhere:

- `scale="height=<metres>"` (`--set calibration.scale=height=<metres>`), when the mounting height is known;
- `scale="lane_width=<metres>"`, when lane lines are marked;
- a known camera or a manual calibration.

The prior itself can be changed with `calibration.car_length_m` and `calibration.car_width_m`; this helps only when the fleet's median car differs from it, not when other classes are mixed in.

## Near-level street-level cameras

A camera mounted 2 to 3 m above the road and pitched less than about 5° below the horizon sees most of the road at grazing angles. From the along-ray ground sampling distance above, the range within which a pixel spans at most `hygiene.max_gsd_m` = 0.30 m is

$$
d_\mathrm{max} \approx \sqrt{0.30\, f\, h} \quad \text{(metres, for } d \gg h\text{)},
$$

which is 27 m for $f$ = 1000 px and $h$ = 2.5 m, against 134 m for the elevated camera of the intersection samples ($f$ = 2212 px, $h$ = 27.8 m). The consequences:

- **Little metric output.** Detections beyond that range carry no metric position. The renderer draws them as corner brackets without a speed (`render.show_detections`).
- **Drift within the range.** A 2 px error of the box bottom at 25 m already moves the ground contact by about 0.5 m, so the positions of vehicles in a queue drift as they occlude each other.
- **Pitch sensitivity.** The horizon lies near the image centre, and small pitch errors change far distances by large factors.

Mount cameras higher where possible. Otherwise restrict the analysis to the near field (`hygiene.max_range_m`) and calibrate manually.

## Telephoto roadside cameras

A long focal length that observes traffic far from the camera approaches weak perspective. The depth range of the scene is small relative to its distance, so the image is close to an affine projection of the ground. Under weak perspective the focal length and the distance trade off against each other, and the evidence that the automatic methods use becomes weak:

- **Footprints.** They keep their right angles under nearly all candidate cameras.
- **Vanishing points.** Parallel markings converge slowly, so their vanishing points lie far outside the image and are poorly localised.

The along-ray ground sampling distance $d^2 / (f h)$ also grows with the square of the distance, so a long focal length alone does not keep distant traffic within `max_gsd_m`. For such cameras, calibrate manually from correspondences spread over the full depth of the observed road, or supply the focal length, which a zoom lens often reports, together with the mounting height.

## Traffic along the viewing ray

When all traffic moves towards or away from a level camera, as on a motorway seen from an overpass or a road seen along its axis, each speed is the time derivative of the range. The range is the coordinate measured least precisely: its error grows as $d^2 / (f h)$, while lateral positions are resolved at $d / f$ per pixel. Speed noise and bias are therefore larger than for crossing traffic at the same distance. Vehicles in a lane occlude each other along the ray, which shifts box bottoms and splits tracks. For calibration, lengths and widths of footprints seen head-on are poorly separated. The cuboid solver therefore weights the car aspect prior fully (`ASPECT_WEIGHT` = 1.0), and the metric scale relies on each footprint's dimension across the ray ([Automatic calibration](guide/auto-calibration.md#metric-scale-and-world-frame)). Prefer viewpoints that see the traffic obliquely, and check speeds against a known reference, such as a speed limit in free flow or a vehicle with a known speed, before using them.

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

- **Vehicle sizes** (default of automatic calibration) assume a median passenger-car footprint of 4.6 m by 1.85 m. The scale is a weighted median of per-footprint estimates in which widths carry 4 times the weight of lengths ([Metric scale](guide/calibration.md#metric-scale)), so a local fleet whose median car is 5 % narrower and 5 % shorter biases all distances and speeds by about +5 %. Other vehicle types labelled `car` bias it further ([Mixed traffic](#mixed-traffic-and-class-dependent-scale)). A systematic bias of the detector's cuboid corners (footprints regressed slightly too large or too small) propagates in the same way. The 4:1 weighting reflects an assumed ratio of the dispersions of car widths and lengths (`calib/from_cuboids.py`, `footprint_scale`), not a measurement.
- **Depth networks** provide a metric scale that can be wrong by a large factor for viewpoints unlike their training data; within `calibrate_auto` it is replaced by the vehicle-size scale by default.
- **Relative scale.** Without any cue (`lines` without `scale`, `geocalib` without `height`, or `calibrate_auto` when fewer than 10 footprints exist), the camera has height 1, `meta["metric"]` is `false`, and positions and speeds are in arbitrary units. Check `camera.json` before using the outputs.
- Latitude/longitude correspondences are converted to a local tangent plane with the WGS84 radii of curvature at the origin ([Manual calibration](guide/manual-calibration.md#correspondence-files)); over the few hundred metres a camera observes, the planar approximation is not a significant error source.

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

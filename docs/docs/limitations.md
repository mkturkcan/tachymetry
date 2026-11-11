# Technical notes and limitations

Tachymetry measures road users on a single calibrated ground plane from a single static camera. This page states the assumptions behind the metric outputs, quantifies where they break down, and lists what the track filters can and cannot detect. Notation follows [Camera calibration](guide/calibration.md#camera-model-and-conventions).

## Flat ground

Each position is the intersection of a pixel ray with the plane $z = 0$ (`Camera.image_to_ground`). A road surface point that lies $\Delta z$ above the calibrated plane, at horizontal distance $d$ from a camera at height $h$, is back-projected to a point displaced away from the camera by

$$
\delta = \frac{d\,\Delta z}{h - \Delta z} \approx \frac{d}{h}\,\Delta z .
$$

For a camera 8 m high, 0.2 m of unmodelled elevation at 40 m range displaces positions by 1.0 m. Sources of $\Delta z$ are grades that differ between the calibration region and the measurement region, road crowns (a 2 % cross slope raises the centre of a 3.5 m lane by 3.5 cm relative to its edge), kerbs and raised medians, bridges and ramps. Speeds are affected through the derivative of $\delta$ along the path: on a uniform grade that is not part of the calibrated plane, the error grows with range. Off-plane points also bias calibration: correspondences on kerb tops or sidewalks and vehicle footprints on a sloped approach pull the fitted plane.

Objects are reduced to a ground point. For vehicle tracks with instance masks (the default detectors), the ground point is the centre of a calibrated 3D box fitted to the masks, whose lowest edge stands at a fitted clearance of at most 0.42 m for vehicles up to 6 m long and 0.95 m beyond ([Vehicle footprints from masks](guide/footprints.md)). Without masks, the ground contact under the box bottom is pushed along the viewing ray by half the footprint extent, minus the parallax of class-prior clearances (car 0.08 m, bus and truck 0.25 m, two-wheelers 0.05 m; `tracking/measure.py`, `tracking/hygiene.py`). Persons and two-wheelers take the bottom-centre of their mask, and in steep views the mask centroid at a class height (`hygiene.steep_view_deg`). Errors of the clearance scale with $d / h$ as above.

## Far field and ground sampling distance

The ground extent of one pixel along the viewing ray grows quadratically with range. On the optical axis of a camera at height $h$ with focal length $f$ (pixels), the along-ray ground sampling distance at horizontal distance $d$ is

$$
\mathrm{GSD}(d) \approx \frac{d^2 + h^2}{f\,h},
$$

so a fixed error of one pixel in the mask or box bottom becomes a ground error proportional to $d^2 / (f h)$, and speed noise grows in the same way. `Camera.ground_sampling_distance` evaluates the worst-axis value from the image-to-ground Jacobian at a given pixel. Measurements are discarded where it exceeds `hygiene.max_gsd_m` = 0.30 m per pixel; the calibration preview and the bird's-eye-view renderer use the same bound.

| Camera | $f$ (px) | $h$ (m) | Range at 0.05 m/px | at 0.10 m/px | at 0.30 m/px |
|---|---|---|---|---|---|
| Synthetic pole camera (`default_camera((1280, 720))`) | 914 | 8.0 | 17.6 m | 26.9 m | 43.0 m |
| Shared camera of the intersection samples (`svbrd-intersection.json`) | 2182 | 27.7 | 48.8 m | 72.8 m | 132.7 m |

Ranges are along the central image column, computed with `Camera.ground_sampling_distance`. Within the usable region, speed accuracy still varies by an order of magnitude between near and far field. The [benchmarks](benchmarks.md) report errors aggregated over the usable region; per-range errors for a specific installation can be obtained with `tachymetry.synthetic` at the calibrated viewpoint.

Other far-field effects: boxes of distant vehicles collapse when the detector loses them, which the end trimming handles only at track ends (`hygiene.far_shrink` = 0.5); masks smaller than `hygiene.boxfit_min_mask_px` or seen below `hygiene.boxfit_min_depression_deg` are not fitted with the box silhouette; and the class vote down-weights small, peripheral boxes.

## Occlusion

Missing detections are bridged by the tracker (lost tracks coast for `tracker.max_lost_s` = 1.5 s), interpolated up to `hygiene.interp_gap_s` = 1.5 s, split at gaps above `hygiene.max_gap_s` = 5.0 s, and fragments are re-joined by stitching across gaps of up to `filters.stitch_max_gap_s` = 3.0 s. Boxes that touch the image border are flagged as truncated, and the ends of moving tracks whose box shrinks relative to the projection of the track's own 3D box are trimmed as partly visible.

Vehicle tracks with masks detect partial occlusion explicitly. A frame whose fitted box silhouette the mask fills markedly less than usual is treated as occluded: the contact path is recomputed without it, the heading and the hidden end are taken from the clean frames, and the pose is re-fitted with the track's shared box ([Vehicle footprints from masks](guide/footprints.md)). On the synthetic bus that passes behind a tree crown, the centre stays within 0.46 m and the heading within 1.1° of the truth with masks, against 1.41 m and 10.6° from the boxes of the same detections. The model assumes that the occluder hides part of a vehicle that remains otherwise consistent. A vehicle hidden for longer than the 2 s heading hold, or a mask merged with an adjacent vehicle, is not repaired. Persons, two-wheelers and box-only detectors have no occlusion model: a partly hidden box moves its ground contact, and only the robust smoothing and outlier repair (Hampel filter, 3 MAD and 0.8 m) limit the effect. Vehicles that are hidden for their whole passage are not counted; counts are lower bounds in dense queues seen from a low camera.

## Night and low light

Detection recall degrades at night. Runs of the default pipeline on two clips of the same intersection, at 15 Hz and with the same camera, illustrate this:

| Clip | Duration | Detections | Tracks kept | Tracks by class |
|---|---|---|---|---|
| `intersection-day` | 31.0 s | 23,491 | 50 | 30 car, 8 truck, 2 bus, 2 bicycle, 8 person |
| `intersection-night` | 18.5 s | 2,011 | 8 | 8 car |

Traffic volumes differ between the recordings, so the comparison is indicative only. On the infrared night clip, no pedestrian or cyclist track was kept. Calibrated on its own, the night clip gives a camera 34.4 m high at a pitch of 22.2°, against 27.4 m and 17.9° for the day clip ([Calibration verification](guide/calibration.md#calibration-verification)), because road markings are visible mainly where they are lit. For cameras that record in both conditions, calibrate on daytime footage and pass the camera file to the night runs, as the sample registry does.

## Moving, PTZ and re-zoomed cameras

Each video is calibrated once and the camera is assumed static for the whole video. A pan, tilt or zoom change within a video invalidates all positions after the change; split such recordings at the preset change. Wind sway and vibration move the whole image, which appears as small common motion of all objects; parked-vehicle freezing and the hold-still rule remove it from stationary objects, but not from moving ones.

When a run directory is reused, an automatic `camera.json` cached there is reused if its image size matches (`Pipeline.calibrate`). If the camera moved between recordings that share an output directory, use a new output directory or re-run with `--overwrite`, which ignores the cached detections, background and camera. For batches of different cameras, pass a directory of per-video cameras (`--camera cameras/`, files named `<video stem>.camera.json` or `<video stem>.json`).

The principal point is fixed at the image centre for all automatic methods. Streams that are cropped or digitally zoomed off-centre should be calibrated manually with `principal_point=`.

## Straight-down views

Automatic calibration is designed for oblique views. A camera that looks straight down images the ground plane at a uniform scale. Parallel ground lines stay parallel in the image, so their vanishing points lie at infinity, and no perspective change of vehicle silhouettes reveals the orientation.

On the straight-down sample, `calibrate_auto` with the default detector returns a near-level `lines` camera: a pitch of 2.8° and a focal length of 10,074 px. The car silhouettes do not fit it (cost 2.01), so the camera keeps relative scale, and the dashed markings change length along the road under it (spread 0.47). Both checks warn.

With UrbanOmniDetect-2, the cuboid footprints of nadir views are compressed along the image vertical, and the solver reads the compression as perspective. `Pipeline` warns when an automatic calibration places the camera more than `MAX_PLAUSIBLE_HEIGHT_M` = 250 m above the ground ([Camera calibration](guide/calibration.md#implausible-automatic-calibrations)).

Process straight-down footage with a known camera: pitch 90°, the height from the flight log, and the focal length from the lens's field of view ([Drone footage](examples/drone.md)).

## Mixed traffic and class-dependent scale

The class scheme has six road classes (person, bicycle, car, motorcycle, bus, truck), and regional vehicle types fall outside it. On the straight-down sample, cycle-rickshaws and rickshaw vans receive almost no evidence from any class prompt; their car fits stay at the smallest car, so the size rule makes them two-wheelers (18 bicycles, and motorcycles where the `motorcycle` prompt scored them higher than `bike`), and CNG auto-rickshaws with `car` or `van` evidence remain cars ([class vote](guide/tracking.md#class-vote)). The two carriages of a passing train are labelled `truck`. Counts by class and the class-dependent parameters (the person and two-wheeler speed gates, the dimension priors) apply to these vehicles as labelled.

The vehicle-size scale assumes that the `car` class is dominated by passenger cars ([Metric scale](guide/calibration.md#metric-scale)): the silhouette scale fits a rigid passenger-car model, and the cuboid footprint scale a median footprint of 4.6 m by 1.85 m. Where a detector maps other vehicles to `car`, such as three-wheelers of about 2.6 m or microcars, the scale inherits their size, and all distances and speeds are multiplied by the ratio of the model to their dimensions. In such traffic, supply the scale from elsewhere:

- `scale="height=<metres>"` (`--set calibration.scale=height=<metres>`), when the mounting height is known;
- `scale="lane_width=<metres>"`, when lane lines are marked;
- a known camera or a manual calibration.

The car width prior can be changed with `calibration.car_width_m` (and the footprint prior with `calibration.car_length_m`); this helps only when the fleet's median car differs from it, not when other classes are mixed in.

## Near-level street-level cameras

A camera mounted 2 to 3 m above the road and pitched less than about 5° below the horizon sees most of the road at grazing angles. From the along-ray ground sampling distance above, the range within which a pixel spans at most `hygiene.max_gsd_m` = 0.30 m is

$$
d_\mathrm{max} \approx \sqrt{0.30\, f\, h} \quad \text{(metres, for } d \gg h\text{)},
$$

which is 27 m for $f$ = 1000 px and $h$ = 2.5 m, against 133 m for the elevated camera of the intersection samples ($f$ = 2182 px, $h$ = 27.7 m). The consequences:

- **Little metric output.** Detections beyond that range carry no metric position. The renderer draws them as corner brackets without a speed (`render.show_detections`).
- **Drift within the range.** A 2 px error of the mask or box bottom at 25 m already moves the ground contact by about 0.5 m, so the positions of vehicles in a queue drift as they occlude each other.
- **Pitch sensitivity.** The horizon lies near the image centre, and small pitch errors change far distances by large factors.

Mount cameras higher where possible. Otherwise restrict the analysis to the near field (`hygiene.max_range_m`) and calibrate manually.

## Telephoto roadside cameras

A long focal length that observes traffic far from the camera approaches weak perspective. The depth range of the scene is small relative to its distance, so the image is close to an affine projection of the ground. Under weak perspective the focal length and the distance trade off against each other, and the evidence that the automatic methods use becomes weak:

- **Silhouettes and footprints.** Their shapes barely change between candidate cameras.
- **Vanishing points.** Parallel markings converge slowly, so their vanishing points lie far outside the image and are poorly localised.

The along-ray ground sampling distance $d^2 / (f h)$ also grows with the square of the distance, so a long focal length alone does not keep distant traffic within `max_gsd_m`. For such cameras, supply the field of view (`calibration.hfov_deg`), which a zoom lens often reports, or calibrate manually from correspondences spread over the full depth of the observed road.

## Traffic along the viewing ray

When all traffic moves towards or away from a level camera, as on a motorway seen from an overpass or a road seen along its axis, each speed is the time derivative of the range. The range is the coordinate measured least precisely: its error grows as $d^2 / (f h)$, while lateral positions are resolved at $d / f$ per pixel. Speed noise and bias are therefore larger than for crossing traffic at the same distance. Vehicles in a lane occlude each other along the ray, which shifts mask and box bottoms and splits tracks. For calibration, the vanishing point of the road gives the pitch only together with a focal length, so a known field of view is needed (the motorway sample uses 59°; [Calibration verification](guide/calibration.md#calibration-verification)), and the lengths of vehicles seen head-on are poorly constrained by their silhouettes. Prefer viewpoints that see the traffic obliquely, and check speeds against a known reference, such as a speed limit in free flow or a vehicle with a known speed, before using them.

## Orientation evidence for automatic calibration

The metric scale of automatic calibration comes from vehicle silhouettes, which constrain the scale but barely the focal length: among cameras that share a horizon, the silhouette cost prefers narrower fields of view ([Automatic calibration](guide/auto-calibration.md)). The orientation therefore has to come from one of:

- two ground directions in the road markings, or one with a vertical vanishing point (`lines`);
- a known horizontal field of view together with the main ground vanishing point (`calibration.hfov_deg`, `--hfov`);
- GeoCalib (optional), or depth and segmentation (optional);
- correspondences (manual calibration).

A scene with a single marked direction and no vertical structure, such as a straight road, needs the field of view. Curved roads and roundabouts provide few straight markings. On a hovering-drone clip of a roundabout that is not among the samples, the circular central island implies a pitch of about 40°, while GeoCalib returned 28° and the vanishing points gave no usable camera; such footage needs a known camera or manual calibration.

## Per-clip variance of automatic calibration

Automatic calibration of one clip is a single estimate with clip-dependent evidence: visible markings, traffic, lighting. The five intersection samples come from one camera pose; their backgrounds register within 4.2 px, except the infrared night clip at 12.4 px. Calibrated separately with the same 47.5° field of view, they give camera heights from 26.6 to 34.4 m, pitches from 17.6° to 22.2° and rolls from −2.5° to 3.6° ([Calibration verification](guide/calibration.md#calibration-verification)). A height error enters all distances and speeds in proportion. For a fixed camera, calibrate once on the clip with the clearest evidence and pass the camera file to its other clips, as the sample registry does with `assets/cameras/svbrd-intersection.json`; compare speeds between sites only after checking their calibrations.

## Runtime of the mask geometry

The box-silhouette fit of vehicle tracks runs on the CPU, at a median of 0.45 to 0.88 s per vehicle track on three sample clips (motorway-overpass, intersection-day, intersection-day-3), with means up to 0.90 s and maxima up to 2.8 s ([Vehicle footprints from masks](guide/footprints.md)). With masks, `build_tracks` took 43.4 s for the 40 masked vehicle tracks of `intersection-day`, against 3.8 s with `hygiene.mask_geometry = False`. Dense clips with hundreds of vehicle tracks therefore spend minutes in tracking. Lower `hygiene.boxfit_max_hz` to fit fewer poses, or set `hygiene.mask_geometry = False` to use the box path, which forgoes the occlusion handling.

## Calibration failure modes

| Method | Fails or degrades when | Signature |
|---|---|---|
| `manual` | points cluster in a small area (extrapolation), points are off the ground plane, the reference GSD is wrong, a point is mis-clicked | large `cv_rmse_m` relative to `rmse_m`; large individual `residuals_m`; wrong lane widths in the preview |
| `lines` | fewer than 8 marking segments (worn or no markings, snow, wet glare, night); only one marked direction and no vertical structure; non-orthogonal streets without a vertical vanishing point (the fallback assumes orthogonality); straight-down views | `CalibrationError`; `details["assumes_orthogonal_streets"]`; converging lane lines in the bird's-eye view; dash-regularity warning |
| `fov+vp` | a wrong field of view; the main vanishing point is not that of the road (mast arms, buildings) | pitch or height implausible; dash-regularity warning |
| silhouette scale | fewer car tracklets than needed (light traffic, short clips); the `car` class contains other vehicle types; the orientation is wrong | the warning that car silhouettes do not fit the camera, and relative scale |
| `cuboids`, `fused` (UrbanOmniDetect-2) | fewer than 40 trusted car footprints; viewpoints far from the detector's training distribution; straight-down views | `CalibrationError`; large footprint right-angle error in `details["metrics"]`; implausible-height warning |
| `depth` | the depth network misjudges the scale of high-mounted views; focal length error couples into pitch; segmentation includes non-road pixels | `details["warnings"]` (height outside 2.5 to 150 m); low `inlier_ratio` |
| `geocalib` | low-texture or unusual views; no metric scale without `height` | large posterior uncertainties, low `score`; `meta["metric"] = False` |

Accept a calibration only after inspecting its preview and its dash check ([Calibration verification](guide/calibration.md#calibration-verification)).

## Metric scale

Methods based on image geometry recover the camera up to scale; the scale then comes from one cue, and all distances and speeds inherit its error.

- **Vehicle silhouettes** (default of automatic calibration with a segmenting detector) fit a rigid passenger-car model to car masks; the scale factor is the one under which the model explains the masks best. A local fleet whose cars differ from the model, other vehicle types labelled `car` ([Mixed traffic](#mixed-traffic-and-class-dependent-scale)), or a wrong orientation bias it, and all distances and speeds inherit the bias. On the synthetic intersection, the silhouette scale gave a height of 7.71 m for a camera 8 m high ([Synthetic calibration](examples/synthetic-calibration.md)).
- **Cuboid footprints** (UrbanOmniDetect-2) assume a median passenger-car footprint of 4.6 m by 1.85 m. The scale is a weighted median of per-footprint estimates in which widths carry 4 times the weight of lengths; the 4:1 weighting reflects an assumed ratio of the dispersions of car widths and lengths (`calib/from_cuboids.py`, `footprint_scale`), not a measurement.
- **Depth networks** provide a metric scale that can be wrong by a large factor for viewpoints unlike their training data; within `calibrate_auto` it is replaced by the vehicle-size scale by default.
- **Relative scale.** Without a cue (`lines` without `scale`, `geocalib` without `height`, or `calibrate_auto` when neither silhouettes nor footprints give a scale), the camera has height 1, `meta["metric"]` is `false`, and positions and speeds are in arbitrary units. Check `camera.json` before using the outputs.
- Latitude/longitude correspondences are converted to a local tangent plane with the WGS84 radii of curvature at the origin ([Manual calibration](guide/manual-calibration.md#correspondence-files)); over the few hundred metres a camera observes, the planar approximation is not a significant error source.

## What the track filters can and cannot detect

The per-track referees and cross-track verdicts ([Association and track hygiene](guide/tracking.md), [Verdicts](guide/verdicts.md)) test physical plausibility and mutual consistency. They detect:

- sustained speeds above the class limit (vehicles 60 m/s, two-wheelers 25 m/s, persons 9 m/s) or accelerations above 12 m/s² for at least 0.6 s, and path reversals sharper than 150 degrees;
- tracks whose footprint centre projects outside their own box for more than 30 % of the observed samples;
- short-lived or low-confidence static tracks, static low-confidence vehicles (`filters.ghost_parked_score`), duplicate tracks of one object, passengers inside vehicles, riders on two-wheelers, and person tracks moving in lockstep with a vehicle.

They cannot detect:

- **A consistent calibration error.** A wrong scale or pitch changes all tracks coherently and leaves speeds plausible; only the calibration preview and independent ground truth reveal it.
- **Plausible identity swaps.** Two similar vehicles that exchange identities while moving side by side at similar speeds produce two plausible tracks.
- **Plausible false positives.** A persistent false detection with a high score, or a reflection that moves like a vehicle, passes all tests.
- **Missed objects.** Objects that are never detected produce no track; counts are lower bounds.
- **Class errors** that are consistent over a track (a van labelled as a truck); the class is a weighted vote over the track's detections.
- **Objects off the ground plane**, such as vehicles on an overpass, which are measured as if they were on the road below.

## Frame rate and timestamps

Frame times are derived from the frame index and the container frame rate (`io/video.py`): $t_k = k / \mathrm{fps}$, with fps read from the container and replaced by 30 when it is missing or outside 1 to 1000. Variable-frame-rate recordings, and recorders that drop frames without duplicating them, violate this assumption; a timing error of $\epsilon$ (relative) produces a speed error of $\epsilon$ (relative). Re-encode such files to a constant frame rate first. Folders of images are read at 10 frames per second unless `PipelineConfig.fps` (`--set fps=...`) gives the rate, which also overrides the container rate of videos.

Detection runs on each `stride`-th frame with $\mathrm{stride} = \mathrm{round}(\mathrm{fps} / \texttt{rate\_hz})$ (default `rate_hz` = 15), so the effective rate is $\mathrm{fps}/\mathrm{stride}$: 15 Hz for 30 fps video and 12.5 Hz for 25 fps video. Thresholds are expressed in seconds and metres (`config.py`), so the configuration does not change with the rate; the number of samples per filter window does. Rolling-shutter skew is not modelled.

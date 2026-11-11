# Changelog

## 0.1.0 (unreleased)

First public release, licensed under the Apache License 2.0. Detector weights are not distributed with the package: SAM 3 is downloaded from Hugging Face under Meta's terms after an access request, RF-DETR weights are Apache-2.0, and the optional UrbanOmniDetect-2 back-end (`tachymetry[uod2]`) depends on Ultralytics (AGPL-3.0).

### Detection

- SAM 3 (`sam3`, the default) through Hugging Face `transformers`, with a two-level prompt scheme.
    - The detection prompts `vehicle` and `person` give the tracked boxes and instance masks, at a score threshold of 0.40.
    - The class-evidence prompts `car`, `van`, `suv`, `bus`, `truck`, `bike` and `motorcycle` run in the same pass. Every detection stores, per prompt, the best score of the evidence detections it overlaps (IoU above 0.5, scores from 0.35 or 0.40), and two flags from confident `bike` detections: one lying within the detection (`bike_cover`) and one under a person (`bike_ride`).
    - The class of a track is the argmax of these scores weighted over its frames by box size and centrality, so that near-field views decide; car, van and suv fold into car, and the motorcycle score splits two-wheelers. Tracks without clear evidence fall back to the bike flag and then to the fitted size: a car fit at the car length bound is refitted as a truck, and one at the smallest car makes the track a two-wheeler. Vehicles and two-wheelers associate as one group.
    - Two-wheeler evidence that no vehicle detection covers is kept as a detection of its own.
    - The `facebook/sam3` checkpoint is gated. A missing approval raises a `PermissionError` that explains the access request and `hf auth login`.
- RF-DETR alternatives with Apache-2.0 weights: `rfdetr-seg-{n,s,m,l}` (boxes and instance masks) and `rfdetr-{n,s,m,l}` (boxes).
    - They run on a square input of twice the training resolution, at most 1008 pixels, with FP16 weights on CUDA.
    - `DETECTOR_SCORE_PROFILES` lowers the track-start and ghost score thresholds for their lower confidences.
- UrbanOmniDetect-2 (`uod2-*`, boxes and 3D cuboid corners) needs the `uod2` extra. The `yolo:<weights>` back-end is removed.
- Precomputed detections are loaded with `detector.name = "precomputed:<npz or folder>"`.
- Instance masks are stored bit-packed per detection in `detections.npz`.
- Post-processing:
    - duplicates whose masks overlap (IoU above 0.5) are suppressed within a super-class;
    - nested-vehicle suppression requires the enclosing detection to score at least as high and, with masks, the nested mask to lie on the enclosing one.
- `tachymetry download --detector <name>` fetches the weights; the default is `sam3`.

### Vehicle footprints from instance masks

- The ground contact of a masked detection is the bottom-centre of its mask.
- A truncation-aware ground-envelope anchor gives the centre along the vehicle axis from the vehicle's observed ends.
- A calibrated upright-box silhouette is fitted to the masks of each vehicle track. One length, width, height and clearance are shared by the whole track, with a pose and bounded heading correction per frame. The asymmetric silhouette cost lets an occluder or the frame border hide part of the box.
- Occluded frames are detected from the mask fill of the fitted silhouette. The heading and the hidden end of the vehicle are then taken from the clean frames and the motion of the box centres.
- Stopped vehicles take their heading from the street axes of the moving traffic.
- Poses of moving vehicles are fitted at up to 7.5 Hz (`hygiene.boxfit_max_hz`).
- In steep views, persons and two-wheelers are placed at their mask centroid projected at a class height (`hygiene.steep_view_deg`).

### Tracking

- Two-stage BYTE association with a continuous-time Kalman filter, super-class gating and a metric plausibility gate.
- Per-track filtering:
    - footprint centres and gap-aware zero-phase smoothing; gap bridges that contradict the motion on both sides are not differentiated;
    - presplit and identity-switch splitting, end trimming of truncated, far-field, low-confidence and partly visible ends, and Hampel repair;
    - parked-vehicle detection with hold-still, and physical-validity checks.
- Cross-track verdicts: fragment stitching, duplicates, riders, inferred cyclists, passengers, shadows and ghosts. A person is a rider when a confident bike lies under it, when it overlaps a tracked two-wheeler, when it stays within 1.2 m of a vehicle or two-wheeler, or when its median speed exceeds 12 km/h; riders are tested before passengers, whose boxes lie inside a vehicle detection. The score thresholds of the static parked-vehicle ghost rule are configurable (`filters.ghost_parked_score`, `filters.ghost_parked_cuboid_score`).

### Calibration

- Manual calibration from points and lines, with cross-validated lens selection and a browser-based picker.
- Automatic calibration:
    - Orientation comes from road-marking and vertical vanishing points, from a known horizontal field of view with the main ground vanishing point (`calibration.hfov_deg`, `tachymetry calibrate --hfov`), or from the optional depth-plus-segmentation and GeoCalib back-ends.
    - A given field of view is authoritative: candidates more than 10 % away from its focal length are dropped.
    - From road markings, the focal length comes from two orthogonal street directions unless the vertical segments reject right angles by a likelihood-ratio test (`vertical_consistency`), and from one street direction and the vertical otherwise. A far vertical vanishing point fixes the focal length poorly, so this solution no longer depends on the few short pole edges of a scene.
    - The metric scale comes from car silhouettes in the instance masks, or from cuboid footprints with UrbanOmniDetect-2. With SAM 3, only detections that car-type prompts matched and no truck, bus or two-wheeler prompt did enter the silhouettes.
- A dash-regularity check ranks the candidates and warns about irregular dashes. Dashed lane lines are searched on the road surface swept by vehicles, from the cross-ratio of their dash starts and their perspective order. Chains of five or more dashes compare their ground lengths and periods along the road.
- Automatic calibrations that place the camera more than 250 m above the ground raise a warning, which is also stored in `camera.meta["warnings"]`.
- Straight-down cameras are supported when given: `Camera.ground_up()` orients the bird's-eye view, and nadir views have no horizon.
- Lens models: identity, division, equidistant (`tan`), Brown–Conrady and Kannala–Brandt.

### Samples, input and output

- Seven public sample clips with their licences and credit lines (`tachymetry.data`):
    - five recordings of one signalised intersection, which share one packaged camera;
    - a motorway seen from an overpass, with its lens field of view;
    - a 30 s straight-down drone excerpt of mixed traffic with its known camera, fetched with HTTP range requests and a stream copy.
- `tachymetry demo --sample <name>` applies each sample's camera and overrides.
- Video input falls back to the FFmpeg binary of `imageio-ffmpeg` when OpenCV cannot decode a file or reports an implausible frame rate. `PipelineConfig.fps` overrides the input frame rate.
- Rendering in the side, dashboard, camera and bird's-eye layouts with H.264 output. Detections without a metric track are drawn as corner brackets without a speed (`render.show_detections`).
- Batch processing on multiple GPUs:
    - worker slots are assigned round-robin across devices, and SAM 3 workers budget 9 GB of GPU memory each;
    - logical GPU indices are mapped to physical ones under `CUDA_VISIBLE_DEVICES`;
    - finished videos are resumed, and a failure stops only its own video;
    - `--overwrite` recomputes cached detections and cameras.
- Exports: CSV, JSON, GeoJSON, MOTChallenge and Parquet.
- Synthetic scenes and traffic with ground truth, including instance masks with image-space occluders, and a benchmark with box-only and mask-geometry variants.

# Changelog

## Unreleased

- Detection: RF-DETR back-ends with Apache-2.0 weights, `rfdetr-seg-{n,s,m,l}` (boxes and instance masks) and `rfdetr-{n,s,m,l}` (boxes); `rfdetr-seg-l` is the new default. RF-DETR runs on a square input of twice its training resolution, at most 1008 pixels, in batches through its export graph with FP16 weights on CUDA.
- Detection: a SAM 3 back-end (`sam3`) through Hugging Face `transformers`, prompted with one text prompt per road class; a missing access approval for the gated `facebook/sam3` checkpoint raises a `PermissionError` that explains how to request access and log in.
- Detection: segmenting detectors store one instance mask per detection (`DetectionSet.masks`); masks are sampled from the low-resolution mask maps inside each box only (`detect.masks.mask_crops`).
- Detection: duplicates whose instance masks overlap (IoU above 0.5) are suppressed within each super-class; for RF-DETR and SAM 3, nested-vehicle suppression requires the enclosing box to score at least as high and, with masks, the nested detection to lie on the enclosing mask.
- Detection: `DetectorConfig.imgsz` and `DetectorConfig.conf` default to `None`, the back-end's own values (RF-DETR 1008 and 0.15 for `rfdetr-seg-l`, SAM 3 1008 and 0.30, UrbanOmniDetect-2 960 and 0.10). Detectors may return a fifth element per image, the mask crops.
- UrbanOmniDetect-2 (`uod2-*`) needs the optional `uod2` extra (`pip install "tachymetry[uod2]"`), which installs Ultralytics; `UltralyticsDetector` is now `UOD2Detector`. The `yolo:<weights>` back-end is removed. `transformers` and `rfdetr` are core dependencies.
- `tachymetry download --detector <name>` fetches RF-DETR, SAM 3 and UrbanOmniDetect-2 weights; the default is `rfdetr-seg-l`.
- Video decoding: a consumer that stops early (`max_frames`) now stops and joins the prefetch thread before the video is released.

## 0.1.0 (2026-11-09)

First public release.

- Detection with UrbanOmniDetect-2 (boxes and 3D cuboid corners) or an Ultralytics model; precomputed detections are loaded with `detector.name = "precomputed:<npz or folder>"`.
- Two-stage BYTE association with a continuous-time Kalman filter, super-class gating and a metric plausibility gate.
- Per-track filtering: footprint centres, gap-aware zero-phase smoothing, presplit and identity-switch splitting, end trimming (truncated, far-field and low-confidence ends), Hampel repair, parked-vehicle detection with hold-still, physical-validity checks.
- Cross-track verdicts: fragment stitching, duplicates, passengers, riders, shadows, ghosts and static low-confidence false positives.
- Calibration: manual (points and lines, cross-validated lens selection, browser-based picker), automatic from cuboid footprints, from road-marking and vertical vanishing points, and a fused estimate; optional depth-plus-segmentation and GeoCalib back-ends.
- Lens models: division, equidistant, Brown–Conrady, Kannala–Brandt.
- Rendering in side-by-side, dashboard, camera and bird's-eye-view layouts; H.264 output through FFmpeg.
- Batch processing over multiple GPUs with resume and per-video fault isolation.
- Exports: CSV, JSON, GeoJSON, MOTChallenge, Parquet.
- Synthetic scenes and traffic with ground truth, and an ablation benchmark.
- Sample clips: eight public clips with their licences and credit lines (`tachymetry.data.SAMPLES`, `attribution`). They comprise five recordings of a signalised intersection (CC BY 4.0), a roundabout from a hovering drone (CC0 1.0), a motorway from an overpass (MIT), and a 30 s straight-down drone excerpt of mixed traffic (CC BY 4.0), cut from a 15 min recording with HTTP range requests and a stream copy. Samples can carry a known camera and configuration overrides (`sample_camera`, `sample_overrides`), which `tachymetry demo --sample` applies.
- Video input falls back to the FFmpeg binary of `imageio-ffmpeg` when OpenCV cannot decode a file (for example AV1) or reports an implausible frame rate; `VideoSource.backend` reports the reader. Ogg video (`.ogv`, `.ogg`) is recognised in batch inputs.
- Rendering draws detections without a metric track (beyond the reliable range, or on tracks that failed kinematic validation) as lightened, class-coloured corner brackets without a speed (`render.show_detections`, on by default).
- Batch processing assigns worker slots round-robin across devices, so fewer videos than slots still spread over all selected GPUs.
- Straight-down cameras: `Camera.ground_up()` orients the bird's-eye view for pitches from level to straight down, and nadir views have no horizon. Automatic calibrations that place the camera more than 250 m above the ground raise a warning that is also stored in `camera.meta["warnings"]`.
- Cuboid calibration: the car aspect prior carries full weight, the metric scale uses each footprint's dimension across the viewing ray (widths weighted four times), and the pitch range extends to 10° above the horizon.

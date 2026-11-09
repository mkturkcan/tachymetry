# Changelog

## 0.1.0 (2026-11-09)

First public release.

- Detection with UrbanOmniDetect-2 (boxes and 3D cuboid corners) or any Ultralytics model.
- Two-stage BYTE association with a continuous-time Kalman filter, super-class gating and a metric plausibility gate.
- Per-track filtering: footprint centres, gap-aware zero-phase smoothing, presplit and identity-switch splitting, end trimming, Hampel repair, parked-vehicle detection with hold-still, physical-validity checks.
- Cross-track verdicts: fragment stitching, duplicates, passengers, riders, shadows, ghosts and static low-confidence false positives.
- Calibration: manual (points and lines, cross-validated lens selection, browser-based picker), automatic from cuboid footprints, from road-marking and vertical vanishing points, and a fused estimate; optional depth-plus-segmentation and GeoCalib back-ends.
- Lens models: division, equidistant, Brown–Conrady, Kannala–Brandt.
- Rendering in side-by-side, dashboard, camera and bird's-eye-view layouts; H.264 output through FFmpeg.
- Batch processing over multiple GPUs with resume and per-video fault isolation.
- Exports: CSV, JSON, GeoJSON, MOTChallenge, Parquet.
- Synthetic scenes and traffic with ground truth, and an ablation benchmark.

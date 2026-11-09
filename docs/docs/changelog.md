# Changelog

## 0.1.0

First public release.

- Detection with UrbanOmniDetect-2 (`uod2-{n,s,m,l,x}`), including 3D-cuboid corners, or any Ultralytics checkpoint (`yolo:<weights>`); class-specific confidence floors, near-duplicate suppression and nested-vehicle suppression.
- Two-stage BYTE association with a continuous-time Kalman filter, DIoU cost, shape and super-class gates and a metric speed gate.
- Track hygiene: footprint centres, gap and identity-switch splits, end trimming, Hampel repair, gap-aware zero-phase smoothing, end-fold trimming, hold-still and parked freezing, physical-validity referees.
- Cross-track verdicts: stitching, duplicates, passengers, riders, inferred cyclists, shadows and ghosts, including static low-confidence parked-vehicle hallucinations.
- Calibration: automatic (fused maximum-likelihood estimate over cuboid footprints and vanishing points, road markings, cuboids), optional depth and segmentation and GeoCalib back-ends, manual calibration from correspondences with cross-validated lens selection, and a browser picker for correspondences.
- Lens models: identity, division, equidistant-to-pinhole (`tan`), Brown–Conrady and Kannala–Brandt.
- Exports: CSV, JSON, Parquet, GeoJSON and MOTChallenge text; renders in the side, dashboard, camera and bird's-eye layouts.
- Batch processing over multiple GPUs with resume and failure isolation; command-line interface `tachymetry`.
- Synthetic scenes and traffic with ground truth (`tachymetry.synthetic`) and the ablation benchmark `tools/benchmark_synthetic.py`.

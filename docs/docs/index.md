# Tachymetry

Tachymetry converts video from a fixed traffic camera into road-user trajectories on the ground plane: positions in metres, speeds, headings, dimensions and counts per class. It runs detection, camera calibration, association, per-track repair and cross-track verdicts, and exports tables, GeoJSON and annotated videos. The same configuration applies at any frame rate and resolution because every threshold is expressed in seconds or metres.

<figure markdown>
![Tracked road users with metric 3D boxes and speeds on the public sample clip](../media/view_tracks.jpg)
<figcaption>Public sample clip (svbrd-llm-roadside-video-av, CC BY 4.0) with Tachymetry's output: metric 3D boxes, speeds and ground trails. Parked vehicles are drawn in grey without a speed.</figcaption>
</figure>

## Pipeline

| Stage | What it does | Page |
|---|---|---|
| Detection | UrbanOmniDetect-2 (YOLO26 detection with the eight projected corners of a 3D cuboid per road user) or any Ultralytics checkpoint, at a configurable analysis rate (15 Hz by default) | [Detection](guide/detection.md) |
| Calibration | Fused maximum-likelihood estimate from vehicle footprints and vanishing points, road markings, learned cuboids, optional depth and segmentation or GeoCalib, or manual correspondences; lens distortion included | [Calibration](guide/calibration.md) |
| Association | Two-stage BYTE association with a continuous-time Kalman filter, DIoU cost, shape and super-class gates and a metric speed gate | [Tracking](guide/tracking.md) |
| Track hygiene | Footprint centres, splits at gaps and identity switches, end trimming, Hampel repair, gap-aware zero-phase smoothing, hold-still and parked freezing, validity referees | [Tracking](guide/tracking.md) |
| Verdicts | Stitching of fragments, duplicates, passengers, riders, shadows, ghosts | [Verdicts](guide/verdicts.md) |
| Export and rendering | CSV, JSON, Parquet, GeoJSON, MOT; side, dashboard, camera and bird's-eye layouts | [Outputs](guide/outputs.md), [Rendering](guide/rendering.md) |
| Batch processing | Worker processes pinned to GPUs, longest-first dispatch, resume, failure isolation | [Batch processing](guide/batch.md) |

## Where to start

- [Installation](installation.md) for Linux, Windows, macOS and CPU-only machines.
- [Quick start](quickstart.md): the demo on the public sample clip, then your own videos.
- [Intersection clip, end to end](examples/intersection.md): the full pipeline in Python with the outputs read back.
- [Synthetic calibration](examples/synthetic-calibration.md) and [Lens rectification](examples/lens-rectification.md): calibration methods and lens models evaluated against known cameras.
- [Benchmarks](benchmarks.md): an ablation of the track-hygiene stages on simulated traffic with ground truth.
- [Limitations](limitations.md): the assumptions behind the metric output and what the filters cannot catch.

## Scope and assumptions

Tachymetry assumes a static camera, a locally planar road and a constant frame rate. Positions are footprint centres on the plane $z = 0$; their accuracy degrades with the ground sampling distance, which grows roughly with the square of the range, and ground points imaged coarser than 0.30 m per pixel are discarded by default. These and other limits are discussed in [Limitations](limitations.md).

## License and citation

Tachymetry is released under the GNU Affero General Public License v3.0, matching its default detector [UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2) and [Ultralytics](https://github.com/ultralytics/ultralytics). The demo footage is downloaded on demand from [svbrd-llm-roadside-video-av](https://huggingface.co/datasets/Ryan-xiangyu-ut/svbrd-llm-roadside-video-av) (Ryan-xiangyu-ut, CC BY 4.0) and is not redistributed. To cite Tachymetry, see [Citation](citation.md).

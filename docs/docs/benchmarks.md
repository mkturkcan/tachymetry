# Benchmarks

This page reports the synthetic ablation produced by `tools/benchmark_synthetic.py`. It measures what each stage of track hygiene and cross-track reasoning contributes to position, speed, identity and false-positive errors, against ground truth. Calibration is not part of this benchmark: all variants use the true camera. Calibration and lens accuracy on synthetic scenes are reported in [Synthetic calibration](examples/synthetic-calibration.md) and [Lens rectification](examples/lens-rectification.md).

```bash
python tools/benchmark_synthetic.py --out docs/media/benchmark.json   # writes benchmark.json and benchmark.md
```

The script runs on the CPU in a few minutes (`--seeds 3 --duration 60` by default).

## What is simulated

Traffic and detections come from `tachymetry.synthetic.traffic.simulate` with the default `TrafficSpec` (60 s at 15 Hz, seeds 0, 1 and 2). The road is the default `RoadLayout`: a four-way intersection with two 3.5 m lanes per direction.

**Agents.**

- *Vehicles*: 24 per minute. Each takes one of 12 paths: through, right turn or left turn from each of four approaches, in right-hand traffic.
  - Classes: 8 % trucks (6.5 to 9.5 m long), about 4 % buses (12 m) and cars otherwise. Car length, width and height are drawn from $\mathcal{N}(4.6, 0.35)$, $\mathcal{N}(1.85, 0.08)$ and $\mathcal{N}(1.5, 0.1)$ m.
  - Speeds: cruise uniform in 25 to 50 km/h; turning vehicles slow to 15 to 22 km/h.
  - Accelerations are limited to $[-3.5, 2.2]$ m/s².
  - 35 % of vehicles stop at the stop line for 3 to 8 s.
- *Parked cars*: 6 along the kerbs (4.5 x 1.85 x 1.5 m).
- *Pedestrians*: 10 per minute on the crosswalks at 1.1 to 1.6 m/s.

**Detections.** Every agent is an oriented 3D box. A detection is the image bounding box and the eight projected corners of that box.

- A box covered more than 60 % by nearer boxes is not detected.
- The following detector failures are injected:

| Failure | Model |
|---|---|
| Box jitter | Gaussian noise on each box edge with standard deviation 2 % of the box size plus 0.7 px; 3.5 % for parked cars (the source of phantom speeds) |
| Cuboid-corner noise | 3 % of the box size per corner |
| Misses | probability 0.06, plus 0.25 for boxes smaller than 18 px |
| Label flicker | car and truck labels swapped with probability 0.08 |
| Flickering false positives | Poisson, 0.25 per frame, car-labelled boxes on the road with score 0.3 to 0.75 and no cuboid |
| Persistent false positives | 2 fixed locations, present in 85 % of frames, score 0.35 to 0.7, no cuboid |

Scores of true detections are $\mathrm{clip}(0.55 + 0.4 \min(s/120, 1) + \mathcal{N}(0, 0.05),\ 0.15,\ 0.99)$ for a box of size $s$ px. Cuboid-corner confidences are 0.9 for boxes larger than 25 px, except persons, and 0.3 otherwise.

**Cameras.**

| Camera | Image | Focal length | Height | Pitch | Yaw | Roll | Position |
|---|---|---|---|---|---|---|---|
| pole | 1280 x 720 | 914 px (70° HFOV) | 8 m | 32° | 45° | 1.0° | (-14, -14) m |
| mast | 1920 x 1080 | 2060 px (50° HFOV) | 18 m | 22° | 40° | 0.5° | (-35, -35) m |

## Variants

All variants share the same detections and the same association (`tachymetry.tracking.builder.associate` with the default `TrackerConfig`, including the metric speed gate). They differ only after association.

| Variant | Configuration |
|---|---|
| Baseline: bottom-centre, frame differencing | Ground point under the box bottom-centre, velocity by `numpy.gradient` over time (`synthetic.traffic.naive_speeds`); no repair, no smoothing |
| + zero-phase smoothing (bottom-centre) | `build_tracks` with `hygiene.footprints = split = trim = repair = parked = validity = hold_still = False` and `filters.enabled = filters.stitch = False` |
| + footprint centres | as above with `hygiene.footprints = True` |
| + split, trim, repair, validity | as above with `hygiene.split = trim = repair = validity = True` |
| + parked and hold-still | as above with `hygiene.parked = hold_still = True` |
| Tachymetry (full) | the default `PipelineConfig`, which adds stitching and the cross-track verdicts |

## Metrics

Implemented in `tachymetry.synthetic.traffic.evaluate`. Every output sample of an `ok` track is matched to the nearest ground-truth agent at the same time within 2.5 m.

| Metric | Definition |
|---|---|
| Position error | Median distance between matched samples and their agents (m) |
| Speed MAE, p95 | Mean and 95th percentile of the absolute speed error over matched samples of non-pedestrian agents (km/h) |
| ID switches | Number of changes of track id along each agent's time-ordered matched samples, summed over agents |
| Fragments per object | Mean number of distinct track ids matched to an agent |
| Parked detected | Share of visible parked cars whose matched tracks are all in state `parked` |
| Parked phantom speed p90 | 90th percentile of the speed of all samples of tracks matched to parked cars (km/h) |
| False tracks | Tracks with fewer than 30 % of their samples matched |

Values are means over the three seeds.

## Results

--8<-- "docs/media/benchmark.md"

## Reading the results

- **Footprint centres** remove most of the position error. The box bottom-centre is the vehicle edge nearest the camera, not its centre. The median error drops from 1.10 m to 0.07 m on the pole camera and from 2.17 m to 0.08 m on the mast camera.
- **Zero-phase smoothing** removes most of the speed error. Frame differencing amplifies box jitter. The speed MAE falls from 11.93 km/h to 2.07 km/h on the pole camera and from 13.68 km/h to 1.48 km/h on the mast camera.
- **Splitting, trimming, repair and the validity referees** reduce identity switches from hundreds to single digits per run and remove almost all false tracks.
- **The parked and hold-still row alone raises ID switches and false tracks again.**
    - Parked tracks are exempt from the validity referees: `_validity` runs only for tracks in state `moving` (`tracking/hygiene.py`). Static false positives therefore survive as parked tracks.
    - The evaluation matches each sample to the nearest agent within 2.5 m. A static track on a traffic lane is matched to each vehicle that passes it, and each pass is counted as an identity switch.
    - The ghost verdict in the full pipeline removes these tracks: static, low-confidence vehicles without a trusted 3D footprint (`tracking/verdicts.py`, `mark_ghosts`). With them, the full configuration has the lowest value, or a value tied for lowest, in all columns for both cameras.
- **Parked phantom speed** is zero only with parked freezing. Parked tracks are frozen at their median position and get zero velocity. Without freezing, the 90th percentile of the speeds reported on parked cars stays between 1.1 and 42 km/h depending on the variant.

The simulation draws its boxes from the true camera geometry with Gaussian noise. Real detectors make correlated errors that it does not model: boxes clipped by occluders, cuboids biased by unusual vehicle shapes, systematic far-field shrinkage. The absolute numbers are therefore a lower bound on the errors to expect on real footage. The relative effect of each stage is the result this benchmark supports. See [Limitations](limitations.md).

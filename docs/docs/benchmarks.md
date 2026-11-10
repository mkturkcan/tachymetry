# Benchmarks

This page reports the synthetic benchmark produced by `tools/benchmark_synthetic.py`. Calibration is not part of it: all variants use the true camera. Calibration and lens accuracy on synthetic scenes are reported in [Synthetic calibration](examples/synthetic-calibration.md) and [Lens rectification](examples/lens-rectification.md). The benchmark has two parts:

1. an ablation of the track-hygiene stages and cross-track verdicts on box detections with injected detector failures;
2. a segmenting-detector simulation on the same traffic, which compares the box-only geometry with the mask geometry of [Vehicle footprints from masks](guide/footprints.md).

```bash
python tools/benchmark_synthetic.py --out docs/media/benchmark.json   # writes benchmark.json and benchmark.md
```

The script runs on the CPU (`--seeds 3 --duration 60` by default); `--no-masks` skips the mask part, whose cost is dominated by the mask geometry (track times in the table).

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

Implemented in `tachymetry.synthetic.traffic.evaluate`. In each frame, the output samples of `ok` tracks are matched one-to-one to the ground-truth agents present in that frame (Hungarian assignment on centre distance, gated at 2.5 m).

| Metric | Definition |
|---|---|
| Position error | Median distance between matched samples and their agents (m) |
| Speed MAE, p95 | Mean and 95th percentile of the absolute speed error over matched samples of non-pedestrian agents (km/h) |
| Heading error | Median and 95th percentile of the axial heading error over matched vehicle samples (degrees; a box at heading + 180° is the same box) |
| Length, width MAE | Mean absolute error of the length and width of matched vehicle tracks, each against its most often matched agent (m) |
| ID switches | Number of changes of the matched track of an agent between consecutive matched frames, summed over agents |
| Fragments per object | Mean number of distinct tracks matched to an agent |
| Parked detected | Share of visible parked cars whose matched tracks are all in state `parked` |
| Parked phantom speed p90 | 90th percentile of the speed of all samples of tracks matched to parked cars (km/h) |
| False tracks | Tracks matched on fewer than 30 % of their samples |
| Track time | CPU seconds of `build_tracks` per 60 s run (mask part) |

Values are means over the three seeds.

## Results

--8<-- "docs/media/benchmark.md"

## Reading the box-detection ablation

- **Footprint centres** remove most of the position error. The box bottom-centre is the vehicle edge nearest the camera, not its centre. The median error drops from 0.94 m to 0.07 m on the pole camera and from 2.19 m to 0.08 m on the mast camera.
- **Zero-phase smoothing** removes most of the speed error, which frame differencing amplifies from box jitter. The speed MAE falls from 11.89 to 1.93 km/h on the pole camera and from 13.64 to 1.32 km/h on the mast camera.
- **Splitting, trimming, repair and the validity referees** reduce identity switches from 47.67 to 7.00 (pole) and from 103 to 9.67 (mast), and remove the remaining false tracks.
- **Parked freezing** is the only stage that removes the speed reported on parked cars. Parked tracks are frozen at their median position with zero velocity, so the 90th percentile of their speeds drops from 1.13 to 0 km/h (pole) and from 1.29 to 0 km/h (mast).
- **The full configuration**, with stitching and verdicts, has the lowest value, or a value tied for lowest, in all columns for both cameras: a speed MAE of 0.78 and 0.47 km/h and no false tracks.

## Reading the mask simulation

The mask part simulates a segmenting detector on the same traffic model (`TrafficSpec(masks=True)`, `synthetic.masks`).

- **Silhouettes.** Each vehicle is a union of boxes (body above the clearance, a narrower cabin, wheels), painted far to near.
- **Masks and boxes.** Each mask is the visible part of the silhouette, with a one-pixel erosion or dilation with probability 0.5. The box is the bounding box of the visible mask.
- **Occluders.** Two tree crowns hide parts of the road; detections that keep less than 20 % of their silhouette are dropped.
- **Failures.** Misses, label flicker and flickering false positives follow the box simulation. Parked-car jitter and persistent false positives are not simulated in this part.

All three rows share these detections. "Boxes only" discards the masks, so vehicles take the box path with class-prior dimensions.

- **Heading.** Masks reduce the median heading error by an order of magnitude: from 8.38° to 0.65° on the pole camera and from 17.89° to 0.88° on the mast camera. The box path infers the heading from the trajectory alone. The 95th percentiles remain large in both paths (36.11° and 44.32° with masks), from vehicles seen nose-on and from short stretches around turns.
- **Position and speed.** The two paths are comparable, and the box path is ahead in part of the columns. With masks, the median position error is 0.15 m against 0.25 m on the pole camera, but 0.34 m against 0.15 m on the mast camera. The speed MAE is 1.27 against 1.04 km/h and 0.68 against 0.58 km/h. Two properties of the simulation favour the box path:
    - the simulated cars are drawn around the class-prior dimensions that the box path assumes, so its length and width errors (0.24 to 0.28 m and 0.06 m) are small by construction, while the mask path measures dimensions from the silhouettes (0.34 to 0.48 m and 0.28 to 0.31 m);
    - the box path also benefits from boxes that are bounding boxes of the visible masks, with no box jitter.
- **Identity.** ID switches are equal on the pole camera and higher with masks on the mast camera (25.33 against 15.00).
- **Runtime.** The mask geometry costs 18 and 36 CPU seconds per 60 s run, against 1 and 2 s for the box path.

The main measured benefit of the masks is therefore the heading and the extent of the vehicle box, which the renderer and the occlusion handling use. On real footage, the masks also replace boxes whose bottom edges are cut by occluders. The [synthetic bus behind a tree crown](guide/footprints.md) isolates that case: with masks, the centre stays within 0.46 m and the heading within 1.1° of the truth over all 64 samples; with the boxes of the same detections, the errors reach 1.41 m and 10.6°.

The simulations draw detections from the true camera geometry with noise models of their own. Real detectors make correlated errors that they do not model: boxes clipped by occluders other than the simulated crowns, masks merged across adjacent vehicles, systematic far-field shrinkage. The absolute numbers are therefore a lower bound on the errors to expect on real footage. The relative effect of each stage is the result this benchmark supports; see [Limitations](limitations.md).

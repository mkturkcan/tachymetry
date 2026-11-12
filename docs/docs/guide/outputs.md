# Outputs and file formats

A run directory (`tm.track(clip, out=...)`, one per video in batch mode) holds the exports, the camera, the caches and the run metadata. All positions are in metres in the world frame of the camera, speeds in m/s or km/h, angles in degrees, image coordinates in raw source pixels with the origin at the top-left pixel centre.

| File | Written by | Content |
|---|---|---|
| `tracks.csv` | export `csv` (default) | one row per sample of each `ok` track |
| `tracks_summary.csv` | export `csv` (default) | one row per track, rejected tracks included |
| `tracks.json` | export `json` (default) | summary, camera and per-track sample arrays |
| `tracks.geojson` | export `geojson` | trajectories as LineStrings |
| `tracks_mot.txt` | export `mot` | image boxes in MOTChallenge format |
| `tracks.parquet` | export `parquet` (needs `pyarrow`) | same columns as `tracks.csv` |
| `camera.json` | calibration | the camera model |
| `camera.report.json` | automatic or manual calibration | calibration evidence and candidates |
| `calibration_preview.jpg` | automatic calibration | ground grid on the camera view and bird's-eye view |
| `background.jpg` | background stage | traffic-free median image (JPEG quality 95) |
| `detections.npz`, `detections.json` | detection | cached detections and their cache key |
| `config.yaml` | each run | the complete configuration used |
| `summary.json` | each run | counts, speeds, report and timings |
| `overlay.mp4` | `render=True` | annotated video ([Rendering](rendering.md)) |

Select exports with `exports` in the configuration, `--formats csv,json,geojson,mot,parquet` on the command line, or `tm.track(..., exports=("csv", "geojson"))`.

## `tracks.csv`

One row per track and processed frame, `ok` tracks only (`results.SAMPLE_FIELDS`):

| Column | Unit | Meaning |
|---|---|---|
| `track_id` | | track id, numbered from 1 in order of start time |
| `class` | | `person`, `bicycle`, `car`, `motorcycle`, `bus` or `truck` (weighted vote over the track) |
| `state` | | `moving` or `parked` (vehicles, two-wheelers); `moving` or `stationary` (persons) |
| `verdict` | | `ok` in this file ([Verdicts](verdicts.md)) |
| `frame` | | source frame index (0-based) |
| `time_s` | s | `frame / fps` |
| `x_m`, `y_m` | m | smoothed footprint centre on the ground plane |
| `vx_mps`, `vy_mps` | m/s | velocity (analytic Savitzky-Golay derivative) |
| `speed_kmh` | km/h | $3.6\,\lVert v \rVert$; 0 for held samples |
| `heading_deg` | ° | direction of motion, counter-clockwise from world $+x$, in $[-180, 180)$; held through stops |
| `observed` | 0/1 | `OBSERVED` flag: the sample is backed by a detection |
| `interpolated` | 0/1 | `INTERPOLATED` flag: no usable measurement (truncated or big box, partly visible, or hidden behind an occluder) |
| `held` | 0/1 | `HELD` flag: position frozen (stop or parked), speed 0 |
| `footprint` | 0/1 | `FOOTPRINT` flag: centre from a footprint, i.e. an accepted silhouette box fit to the instance mask or a trusted cuboid |
| `x1`, `y1`, `x2`, `y2` | px | detection box |
| `score` | | detection score |

The full bit mask, including `REPAIRED` (4), `TRUNCATED` (8) and `SILHOUETTE` (64), is in `tracks.json` ([Sample flags](tracking.md#sample-flags)). First rows of the public sample clip (a standing pedestrian):

```text
track_id,class,state,verdict,frame,time_s,x_m,y_m,vx_mps,vy_mps,speed_kmh,heading_deg,observed,interpolated,held,footprint,x1,y1,x2,y2,score
1,person,stationary,ok,0,0.0,-63.948,33.05,0.0,0.0,0.0,152.7,1,0,1,0,264.2,634.2,278.7,689.2,0.831
1,person,stationary,ok,2,0.0667,-63.948,33.05,0.0,0.0,0.0,152.7,1,0,1,0,264.3,634.2,278.7,689.1,0.829
```

## `tracks_summary.csv`

One row per track including rejected ones (`results.SUMMARY_FIELDS`, from `Track.summary()`):

| Column | Unit | Meaning |
|---|---|---|
| `id`, `class`, `state` | | as in `tracks.csv` |
| `verdict`, `reason` | | `ok`, `ghost`, `passenger`, `rider`, `duplicate`, `shadow` or `invalid`, and the rule's reason |
| `t_start`, `t_end`, `duration_s` | s | first and last sample time, duration |
| `n_samples` | | number of samples |
| `observed_share` | | fraction of samples backed by a detection |
| `footprint_share` | | fraction of samples with the `FOOTPRINT` flag: centre from an accepted silhouette box fit or a trusted cuboid |
| `length_m`, `width_m`, `height_m` | m | object dimensions (footprint percentiles or class priors) |
| `path_m` | m | length of the smoothed path |
| `net_m` | m | distance between first and last position |
| `speed_mean_kmh`, `speed_median_kmh`, `speed_p85_kmh`, `speed_max_kmh` | km/h | over samples faster than 3 km/h (0 if none) |
| `moving_share` | | fraction of samples faster than 3 km/h |
| `mean_score` | | mean detection score |
| `class_rule` | | how the class was decided ([class vote](tracking.md#class-vote)): `evidence`, `bike_flag` or `size` with SAM 3 (and `size_small` with `hygiene.small_vehicle_rule`), `detections` with the other back-ends, `inferred` for cyclists inferred from person tracks |

## `tracks.json`

Format `tachymetry.tracks/1`: an object with `format`, `fps`, `meta` (source, fps, frame count, size, duration), `summary` (as `TrackSet.summary()`), `camera` (as `camera.json`) and `tracks`, a list with one object per `ok` track holding all `Track.summary()` fields (the columns above plus `start_xy` and `end_xy`) and the per-sample arrays `t`, `frame`, `xy`, `speed_kmh`, `heading_deg`, `flags` (bit mask) and `bbox`. With SAM 3, `class_shares` holds the weighted evidence share of each class prompt. A two-wheeler or vehicle that carries a rider has `rider_id`, and a cyclist inferred from a person track nothing more; rejected rider tracks (with `include_rejected`) name their two-wheeler or vehicle in `partner_id`.

## GeoJSON, MOT and Parquet

* `tracks.geojson`: a FeatureCollection with one LineString per `ok` track and its summary as properties. Coordinates are `[lon, lat]` (WGS84) when the camera was calibrated from latitude/longitude correspondences (`camera.meta["georef"]["type"] == "latlon"`); otherwise local metres, flagged by a `crs_note` member.
* `tracks_mot.txt`: MOTChallenge text, one line `frame, id, x, y, w, h, score, -1, -1, -1` per observed sample of an `ok` track, frames 1-based.
* `tracks.parquet`: the columns of `tracks.csv`.

## `camera.json`

Format `tachymetry.camera/1` (`Camera.to_dict`):

| Key | Meaning |
|---|---|
| `image_size` | `[width, height]` in pixels |
| `K`, `R`, `t` | intrinsics, world-to-camera rotation and translation ($X_c = R X_w + t$, OpenCV camera axes) |
| `lens` | lens model and parameters ([Lens models](lens-models.md)) |
| `summary` | `focal_px`, `height_m`, `yaw_deg`, `pitch_deg`, `roll_deg`, `hfov_deg` |
| `H_image_to_ground` | 3×3 homography from *rectified* pixels to ground $(x, y, 1)$, normalised to $H_{33} = 1$ |
| `meta` | `calibration_method` (for example `auto:fov+vp`, `manual`, or `known` for a camera given by its pose), `metric` (whether the scale is metric), calibration scores, `warnings` (for example an automatic calibration that places the camera more than 250 m above the ground), and for manual calibrations the lens model, point counts and `georef` |

`Camera.load(path)` reads it back; `camera.json` from one run can be passed to others (`camera="camera.json"`, resized automatically for other resolutions). For distorted lenses, apply `Camera.lens.rectify` to raw pixels before using `H_image_to_ground`, or use `Camera.image_to_ground`, which does both.

`camera.report.json` is `CalibrationResult.to_dict()`: method, residuals (`rmse_m`, `cv_rmse_m` for manual calibration), the consistency `score`, and `details`; for automatic calibration these include the selected candidate, the metric-scale evidence (for the vehicle-silhouette scale: the factor, the silhouette cost and the numbers of masks and tracklets used), each candidate with its consistency metrics (including the dash regularity when dashed lane lines were found), errors of methods that failed or disagreed with a given field of view, the number of cuboid footprints and the vanishing points ([Automatic calibration](auto-calibration.md)).

## `summary.json`

| Key | Content |
|---|---|
| `tachymetry`, `python`, `platform` | versions |
| `video` | source path, fps, frame count, size, duration |
| `detector` | detector description, also when detections were loaded from the cache (for SAM 3 including the detection and evidence prompts, input size and threshold) |
| `camera`, `calibration` | camera summary and calibration method |
| `summary` | `TrackSet.summary()`: number of `ok` tracks, by class, moving and parked by class, rejected by verdict, `vehicle_speed_kmh` (median, p85, max over the moving samples of car, bus and truck tracks) |
| `report` | hygiene counters, stitched fragments, verdict counts, `ok_by_class`, `ok_by_state`, number of detections and associated detections |
| `timings` | seconds per stage: `detect_s`, `calibrate_s`, `track_s`, `render_s`, `total_s` |

## `detections.npz`

NumPy archive of the `DetectionSet`: `frame` (M,), `time` (M,), `xyxy` (M, 4), `score` (M,), `cls` (M,) indices into `class_names`, `cuboid` (M, 8, 3) corner `x, y, confidence` with NaN for untrusted cuboids (cuboid detectors only), `processed_frames`, `image_size`, `fps` and `meta` (the detector description). With SAM 3, `evidence` (M, 9) float16 holds the class evidence of each detection, with the column names in `evidence_names` ([SAM 3 prompts](detection.md#sam-3-prompts)).

Instance masks of segmenting detectors are stored as a `MaskStore` (`detections.MaskStore`): each mask is cropped to its box (plus 2 pixels) and bit-packed with `np.packbits`. The archive then holds `mask_bits` (concatenated packed bytes), `mask_offset` (M,) byte offset of each mask, or −1 for a detection without one, `mask_origin` (M, 2) the crop's top-left pixel `(x0, y0)`, and `mask_shape` (M, 2) its `(h, w)`. `DetectionSet.masks.crop(i)` returns `(x0, y0, mask)` and `masks.full(i, image_size)` the full-frame mask. `detections.json` holds the cache key and the detector description. For the public sample clip with SAM 3: 23,491 detections in 465 processed frames, all with masks, in 10.1 MB of packed mask bits.

## Python access

`tm.track()` returns a `RunResult` with `source`, `tracks` (a `TrackSet`), `camera`, `detections`, `out_dir`, `report` and `background`.

```python
import tachymetry as tm

run = tm.load_run("runs/intersection-day")
cars = run.tracks.select("car", state="moving")
tr = cars[0]
print(len(run.tracks), len(cars), tr.id, tr.cls, len(tr), round(float(tr.speed_kmh.max()), 1))
print(run.tracks.by_id(tr.id).summary()["speed_median_kmh"])
df = run.tracks.to_dataframe()
print(df.shape)
```

```text
50 17 4 car 126 30.7
26.01
(11487, 21)
```

Iterating over a `TrackSet` or taking its `len` uses `ok` tracks only. `select(cls, state, include_rejected)` filters, `by_id` looks up one track, and `to_csv`, `summary_csv`, `to_dataframe`, `to_parquet`, `to_json`, `to_geojson`, `to_mot` and `export` write the formats above. A `Track` exposes the per-sample arrays (`t`, `frame`, `xy`, `vel`, `heading`, `raw_xy`, `contact`, `bbox`, `score`, `flags`, `cuboid`), the dimensions, `verdict`, `reason`, `state`, `meta` (for vehicle tracks on the mask path, `meta["geometry"]` with the geometry source, clearance, silhouette share, occluded and hidden samples and CPU time; [Vehicle footprints from masks](footprints.md#settings)), and the derived `speed`, `speed_kmh`, `duration`, `path_length` and `net_displacement` ([API reference](../reference/results.md)).

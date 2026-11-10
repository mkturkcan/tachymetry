
**pole (8 m, 32 deg, 70 deg HFOV)**, mean over 3 seeds of 60 s

| Variant | Position error (m, median) | Speed MAE (km/h) | Speed p95 (km/h) | ID switches | Fragments per object | Parked detected | Parked phantom speed p90 (km/h) | False tracks |
|---|---|---|---|---|---|---|---|---|
| Baseline: bottom-centre, frame differencing | 1.36 | 11.89 | 35.56 | 138 | 2.51 | 0.00 | 42.20 | 13.67 |
| + zero-phase smoothing (bottom-centre) | 0.94 | 1.93 | 7.59 | 53.00 | 1.71 | 0.00 | 2.20 | 6.00 |
| + footprint centres | 0.07 | 1.47 | 7.13 | 47.67 | 1.58 | 0.00 | 1.27 | 4.00 |
| + split, trim, repair, validity | 0.07 | 1.10 | 3.76 | 7.00 | 1.18 | 0.00 | 1.13 | 0.00 |
| + parked and hold-still | 0.04 | 0.83 | 3.56 | 7.00 | 1.18 | 1.00 | 0.00 | 0.00 |
| Tachymetry (full: + stitching and verdicts) | 0.04 | 0.78 | 3.37 | 6.00 | 1.16 | 1.00 | 0.00 | 0.00 |

**mast (18 m, 22 deg, 50 deg HFOV)**, mean over 3 seeds of 60 s

| Variant | Position error (m, median) | Speed MAE (km/h) | Speed p95 (km/h) | ID switches | Fragments per object | Parked detected | Parked phantom speed p90 (km/h) | False tracks |
|---|---|---|---|---|---|---|---|---|
| Baseline: bottom-centre, frame differencing | 1.90 | 13.64 | 38.64 | 258 | 2.25 | 0.00 | 40.10 | 8.67 |
| + zero-phase smoothing (bottom-centre) | 2.19 | 1.32 | 2.80 | 118 | 1.82 | 0.00 | 14.65 | 7.67 |
| + footprint centres | 0.08 | 0.82 | 2.40 | 103 | 1.34 | 0.00 | 1.40 | 2.33 |
| + split, trim, repair, validity | 0.08 | 0.82 | 2.28 | 9.67 | 1.22 | 0.00 | 1.29 | 0.00 |
| + parked and hold-still | 0.03 | 0.47 | 1.96 | 9.00 | 1.21 | 1.00 | 0.00 | 0.00 |
| Tachymetry (full: + stitching and verdicts) | 0.03 | 0.47 | 1.96 | 7.00 | 1.14 | 1.00 | 0.00 | 0.00 |

**pole (8 m, 32 deg, 70 deg HFOV), instance masks with two tree occluders**, mean over 3 seeds of 60 s

| Variant | Position error (m, median) | Speed MAE (km/h) | Speed p95 (km/h) | Heading error (deg, median) | Heading error (deg, p95) | Length MAE (m) | Width MAE (m) | ID switches | False tracks | Track time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: bottom-centre, frame differencing | 2.00 | 4.17 | 12.57 |  |  |  |  | 97.33 | 5.67 |  |
| Tachymetry, boxes only (masks discarded) | 0.25 | 1.04 | 4.83 | 8.38 | 17.91 | 0.28 | 0.06 | 15.33 | 0.00 | 1.02 |
| Tachymetry, instance masks (default) | 0.15 | 1.27 | 6.52 | 0.65 | 36.11 | 0.34 | 0.28 | 15.00 | 0.00 | 17.98 |

**mast (18 m, 22 deg, 50 deg HFOV), instance masks with two tree occluders**, mean over 3 seeds of 60 s

| Variant | Position error (m, median) | Speed MAE (km/h) | Speed p95 (km/h) | Heading error (deg, median) | Heading error (deg, p95) | Length MAE (m) | Width MAE (m) | ID switches | False tracks | Track time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline: bottom-centre, frame differencing | 2.03 | 3.89 | 13.42 |  |  |  |  | 133 | 7.00 |  |
| Tachymetry, boxes only (masks discarded) | 0.15 | 0.58 | 2.85 | 17.89 | 50.41 | 0.24 | 0.06 | 15.00 | 0.00 | 1.90 |
| Tachymetry, instance masks (default) | 0.34 | 0.68 | 3.79 | 0.88 | 44.32 | 0.48 | 0.31 | 25.33 | 0.00 | 36.49 |

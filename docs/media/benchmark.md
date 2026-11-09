
**pole (8 m, 32 deg, 70 deg HFOV)**, mean over 3 seeds of 60 s

| Variant | Position error (m, median) | Speed MAE (km/h) | Speed p95 (km/h) | ID switches | Fragments per object | Parked detected | Parked phantom speed p90 (km/h) | False tracks |
|---|---|---|---|---|---|---|---|---|
| Baseline: bottom-centre, frame differencing | 1.36 | 11.89 | 35.56 | 138 | 2.51 | 0.00 | 42.20 | 13.67 |
| + zero-phase smoothing (bottom-centre) | 0.94 | 1.96 | 7.59 | 53.00 | 1.71 | 0.00 | 2.06 | 6.00 |
| + footprint centres | 0.07 | 1.49 | 7.12 | 47.67 | 1.58 | 0.00 | 1.27 | 4.00 |
| + split, trim, repair, validity | 0.07 | 1.16 | 4.14 | 7.00 | 1.18 | 0.00 | 1.13 | 0.33 |
| + parked and hold-still | 0.05 | 1.29 | 7.07 | 46.00 | 1.55 | 1.00 | 0.00 | 4.00 |
| Tachymetry (full: + stitching and verdicts) | 0.04 | 0.87 | 3.98 | 6.33 | 1.17 | 1.00 | 0.00 | 0.33 |

**mast (18 m, 22 deg, 50 deg HFOV)**, mean over 3 seeds of 60 s

| Variant | Position error (m, median) | Speed MAE (km/h) | Speed p95 (km/h) | ID switches | Fragments per object | Parked detected | Parked phantom speed p90 (km/h) | False tracks |
|---|---|---|---|---|---|---|---|---|
| Baseline: bottom-centre, frame differencing | 1.90 | 13.64 | 38.64 | 258 | 2.25 | 0.00 | 40.10 | 8.67 |
| + zero-phase smoothing (bottom-centre) | 2.19 | 1.32 | 2.79 | 118 | 1.82 | 0.00 | 14.65 | 7.67 |
| + footprint centres | 0.08 | 0.81 | 2.36 | 103 | 1.34 | 0.00 | 1.40 | 2.33 |
| + split, trim, repair, validity | 0.08 | 0.80 | 2.24 | 9.67 | 1.22 | 0.00 | 1.28 | 0.00 |
| + parked and hold-still | 0.03 | 0.47 | 1.93 | 101 | 1.30 | 1.00 | 0.00 | 2.33 |
| Tachymetry (full: + stitching and verdicts) | 0.03 | 0.45 | 1.92 | 7.33 | 1.15 | 1.00 | 0.00 | 0.00 |

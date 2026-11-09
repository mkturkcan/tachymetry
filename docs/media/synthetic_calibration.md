| Method | Input | Focal error (%) | Pitch error (deg) | Roll error (deg) | Height error (m) | Scale error (%) | Ground RMS after rigid alignment (m) |
|---|---|---|---|---|---|---|---|
| Manual | 12 points, 1.0 px noise, lens auto | +0.14 | -0.02 | +0.04 | 0.00 | +0.06 | 0.019 |
| Road markings (`lines`) | background image, camera height 8 m | -2.67 | +0.43 | +0.33 | 0.00 | -1.84 | 0.286 |
| Cuboids (`cuboids`) | 60 s of simulated cuboid detections | -4.96 | +0.69 | +0.32 | -0.29 | -0.60 | 0.262 |
| Automatic (`calibrate_auto`, selected `auto:lines`) | background image + 60 s of detections | -2.67 | +0.43 | +0.33 | -0.07 | -0.96 | 0.210 |

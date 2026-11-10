| Method | Input | Focal error (%) | Pitch error (deg) | Roll error (deg) | Height error (m) | Scale error (%) | Ground RMS after rigid alignment (m) |
|---|---|---|---|---|---|---|---|
| Manual | 12 points, 1.0 px noise, lens auto | +0.14 | -0.02 | +0.04 | 0.00 | +0.06 | 0.019 |
| Road markings (`lines`) | background image, camera height 8 m | -2.67 | +0.43 | +0.33 | 0.00 | -1.84 | 0.286 |
| Automatic, masks (`auto:lines`) | background image + 60 s of instance masks | -2.67 | +0.43 | +0.33 | -0.39 | +3.18 | 0.414 |
| Automatic, masks, known field of view (`auto:fov+vp`) | as above, known 70 deg field of view | 0.00 | -0.11 | -0.09 | -0.25 | +2.37 | 0.286 |
| Automatic, cuboids (`auto:lines`) | background image + 60 s of cuboid detections | -2.67 | +0.43 | +0.33 | -0.07 | -0.96 | 0.210 |

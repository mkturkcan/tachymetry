# Manual calibration

Manual calibration fits a physical camera and lens to correspondences between raw image pixels and ground coordinates, optionally with polylines that are straight in the world. It is the most accurate method when four or more ground features can be located in an orthophoto, a site plan, a survey or with GPS, and it is the only method that produces a georeferenced camera. The estimator is `tachymetry.calib.calibrate_manual` (`calib/manual.py`).

```bash
tachymetry calibrate pick clip.mp4 --reference ortho.png --gsd 0.1 -o points.yaml   # click correspondences
tachymetry calibrate manual clip.mp4 --points points.yaml -o camera.json            # fit, validate, preview
tachymetry run clip.mp4 --camera camera.json
```

## Model

The unknowns are the physical camera and the free lens parameters,

$$
\theta = \left(\log f,\ \psi,\ \vartheta,\ \varphi,\ C_x,\ C_y,\ \log h,\ \kappa\right),
$$

with focal length $f$ (pixels), yaw $\psi$, pitch $\vartheta$, roll $\varphi$, ground position $(C_x, C_y)$ and height $h$ of the optical centre (metres), and lens parameters $\kappa$ (none for `identity`, one for `division` and `equidistant`, $k_1, k_2$ for `brown` and `fisheye`). The principal point is fixed at the image centre unless `principal_point` is given. A world point $X_i = (x_i, y_i, 0)$ projects to the raw pixel

$$
\hat u_i(\theta) = \mathcal{D}_\kappa\!\left(\pi\!\left(K (R X_i + t)\right)\right),
$$

where $\pi$ is the perspective division and $\mathcal{D}_\kappa$ the lens distortion (`Camera.ground_to_image`). The parametrisation has 7 pose and intrinsic degrees of freedom, against 8 for a free ground-to-image homography; it excludes non-physical shear and anisotropy, which is what keeps extrapolation away from the annotated area stable (module docstring of `calib/manual.py`).

## Estimator

**1. Initialisation.** For each starting value of the lens parameters in the model's `param_grid()` (e.g. $k \in \{0, -0.1, -0.2, -0.35, -0.5, 0.1\}$ for the division model; a single identity start without a free lens), the clicked pixels are rectified with that lens and a planar PnP problem (OpenCV IPPE for $\geq 4$ points) is solved for 49 focal lengths spanning a horizontal field of view of 20 to 140 degrees in 2.5 degree steps (`_init_pnp`). Cameras below the ground or with non-finite projections are discarded; the focal length with the lowest RMS reprojection error wins. Each start is then refined for at most 300 function evaluations with the lens fixed, and the two starts with the lowest cost proceed.

**2. Physical refinement.** Trust-region least squares (`scipy.optimize.least_squares`, method `trf`, `x_scale="jac"`, at most 4000 evaluations) minimises

$$
E(\theta) = \sum_{i=1}^{N} \sum_{c \in \{u, v\}} \rho\!\left(r_{ic}^2\right) + \sum_{\ell} \sum_{j=1}^{n_\ell} \rho\!\left(\lambda^2 \frac{2}{n_\ell}\, d_{\ell j}^2\right),
$$

where $r_i = \hat u_i(\theta) - u_i$ is the reprojection error in pixels, $\rho$ is the soft-$\ell_1$ loss with scale `robust_px` = 3 px, and $\lambda$ = `line_weight` = 1. For every polyline $\ell$ with $n_\ell \geq 3$ vertices, the vertices are rectified with the current lens, a total-least-squares line is fitted, and $d_{\ell j}$ are the signed vertex distances to it. The factor $\sqrt{2 / n_\ell}$ on each residual makes a densely clicked line count as much as one point (two residuals). Lines constrain only the lens; points constrain the camera and the lens. Lens parameters are box-constrained (`param_bounds()`: division $k \in [-0.9, 0.6]$; equidistant $1/s \in [0, 1.6 / r_\mathrm{half}]$ with $r_\mathrm{half}$ the half image diagonal; Brown–Conrady and Kannala–Brandt coefficients in $[-1, 1]$ and $[-0.5, 0.5]$). The lens is held fixed when the data cannot support it, i.e. when $N + 2 N_\mathrm{lines} < 6 + n_\kappa$.

**3. Model selection.** With `lens="auto"` (default) the candidate models `identity`, `equidistant` and `division` are each fitted and scored by $k$-fold cross-validation in ground metres (`cv_folds=10`; leave-one-out when the number of points is at most the number of folds, `cv_folds=None` forces it; skipped below 6 points). Each fold is refitted from the full-data camera. The held-out error is the distance between the back-projected held-out pixel and its world point. Models are visited from simple to complex, and a richer model replaces the current choice only if its cross-validated RMSE is below 97 % of the current one. A single model is fitted with `lens="equidistant"` (alias `"tan"`, the [recommended model](lens-models.md#recommended-model-equidistant-to-pinhole-tan) for wide-angle lenses), `"division"`, `"brown"` or `"fisheye"`; a `Lens` instance is kept fixed; `candidates=` changes the automatic set. On the command line: `tachymetry calibrate manual clip.mp4 --points points.yaml --lens tan`.

The result (`CalibrationResult`) reports `rmse_m` (in-sample ground RMSE), `cv_rmse_m` (cross-validated ground RMSE of the chosen model), `residuals_m` per point, and in `details` the per-model `lens_candidates`, the pixel RMSE `rmse_px` and `line_rmse_px`. `CalibrationResult.summary()` labels the cross-validated value "leave-one-out" regardless of the number of folds. Report `cv_rmse_m`, not `rmse_m`: with 7 or more free parameters, the in-sample residual of a small point set is optimistic.

!!! note "Handedness"
    The world frame must be right-handed ($x$ east or right, $y$ north or up). Map and image pixel coordinates have $y$ pointing down, which mirrors the scene; no camera above the ground produces a mirrored ground image. `calibrate_manual` fits a DLT homography to the correspondences, evaluates the sign of the image-to-world Jacobian determinant at the centroid of the image points (`Correspondences.handedness()`; negative for a right-handed world) and raises `CalibrationError` for a left-handed frame. Reference-image pixels should be given with `world_units: reference_px` (or `Correspondences.from_reference`), which flips $y$.

## Correspondence files

`Correspondences.load` reads YAML, JSON or CSV. Image coordinates are raw pixels of the video frame (origin at the top-left pixel centre).

=== "Metres"

    ```yaml
    image_size: [1920, 1080]        # optional; checked against the video when given
    world_units: metres             # metres | meters | m
    points:
      - {name: stop_line_w,  image: [412.0, 811.5],  world: [-6.10, -3.20]}
      - {name: stop_line_e,  image: [905.5, 802.0],  world: [ 0.90, -3.20]}
      - {name: kerb_ne,      image: [1102.5, 640.0], world: [12.40,  8.10]}
      - {name: crosswalk_nw, image: [233.0, 655.5],  world: [-9.80,  6.50]}
      - {name: pole_base,    image: [1510.0, 702.0], world: [18.20,  1.40]}
    lines:                          # optional polylines that are straight in the world (raw pixels)
      - [[100, 900], [400, 860], [700, 835], [1000, 822]]
    ```

=== "Reference image"

    ```yaml
    world_units: reference_px       # reference_px | pixels | px
    gsd: 0.1                        # metres per pixel of the north-up reference image
    points:
      - {name: kerb_ne, image: [1102.5, 640.0], world: [812.0, 455.0]}   # world = (column, row) in the reference
      - ...
    ```

    World coordinates become $(x, y) = (c \cdot \mathrm{gsd},\ -r \cdot \mathrm{gsd})$ for column $c$ and row $r$, which makes the frame right-handed with $x$ east and $y$ north for a north-up reference.

=== "Latitude and longitude"

    ```yaml
    world_units: latlon             # latlon | wgs84
    origin: [30.28642, -97.73651]   # optional; default: mean of the points
    points:
      - {name: kerb_ne, image: [1102.5, 640.0], world: [30.286501, -97.736420]}   # [lat, lon] in degrees
      - ...
    ```

=== "CSV"

    ```text
    name,u,v,x,y
    stop_line_w,412.0,811.5,-6.10,-3.20
    stop_line_e,905.5,802.0,0.90,-3.20
    kerb_ne,1102.5,640.0,12.40,8.10
    crosswalk_nw,233.0,655.5,-9.80,6.50
    ```

    Columns `u,v,x,y` (metres) or `u,v,lat,lon` (degrees); `name` is optional. CSV files carry no image size and no lines.

Latitude and longitude are converted to a local east-north frame about the origin $(\phi_0, \lambda_0)$ (`latlon_to_local`):

$$
x = R\,(\lambda - \lambda_0)\cos\phi_0, \qquad y = R\,(\phi - \phi_0), \qquad R = 6\,378\,137\ \mathrm{m},
$$

with angles in radians. The camera then carries `meta["georef"] = {"type": "latlon", "origin": [lat0, lon0]}`, and `tracks.geojson` is written in WGS84 (`local_to_latlon` inverts the mapping). The sphere of equatorial radius differs from the WGS84 ellipsoid by the ratio of $R$ to the meridional and prime-vertical radii of curvature: at $30°$ latitude, north-south distances in the local frame are about 0.42 % longer and east-west distances 0.08 % shorter than geodesic distances; the deviation is anisotropic and ranges from +0.67 % (north-south, equator) to about −0.34 % (poles).

In Python, `Correspondences(image, world, names=None, lines=[], image_size=None)` takes metric coordinates directly; `Correspondences.from_reference(image_pts, reference_px, gsd)` and `Correspondences.from_latlon(image_pts, latlon, origin=None)` convert the other two forms. `Correspondences.save` writes YAML or JSON in metres with a `georef` block describing the source frame; `load` does not read that block back, so a saved latitude/longitude file reloads as a local metric frame.

A correspondence file can also be passed directly as the camera of a run (`tachymetry run clip.mp4 --camera points.yaml`); it is then calibrated with the defaults above at the video's resolution.

## Browser picker

```bash
tachymetry calibrate pick clip.mp4 --reference ortho.png --gsd 0.1 -o points.yaml
tachymetry calibrate manual clip.mp4 --points points.yaml
```

`tachymetry calibrate pick` computes the calibration image of the source (the temporal median of 41 frames for a video, or the image itself) and calls `tachymetry.calib.picker.launch_picker`, which serves the self-contained page `tachymetry/assets/picker.html` from a standard-library HTTP server listening on `127.0.0.1` only (a free port by default), opens it in the default browser (the URL is printed as well), and blocks until the points are saved; Ctrl+C cancels without writing anything. The server answers only requests addressed to the loopback host, accepts a save only with the session's random token, and the page loads nothing from other origins (content security policy).

**Panes.** The left pane shows the camera frame. With a reference image (a north-up orthophoto, satellite tile or site plan, `--reference`), the right pane shows it and the world side of each pair is clicked there; the file then stores reference pixels (`world_units: reference_px`) with the ground sampling distance `gsd`, which can also be entered or corrected in the page and is required to save. Without a reference, the world $x$ (east) and $y$ (north) of each image point are typed in metres (`world_units: metres`). A line tool traces polylines along edges that are straight on the ground (at least three vertices); points can be dragged with a magnifier, nudged with the arrow keys, renamed and edited, with undo and redo.

**Live fit.** From four complete pairs the page fits a normalised DLT homography from the camera frame to the world plane. The correspondence table reports, for every pair, the *residual* (ground distance between the picked world position and its image point mapped through the homography fitted to all pairs; metres, or reference pixels while no GSD is set) and, from five pairs, the *leave-one-out* residual (homography refitted without that pair). The page warns about degenerate configurations (three points on a line), pairs on both sides of the horizon, and mirrored or left-handed axes. *Grid* (key G) projects a metric ground grid and the horizon onto the camera frame, *Overlay* (key O) warps the camera frame onto the right pane, and a bird's-eye preview shows the camera frame warped onto the ground plane. The picker's homography has eight degrees of freedom and no lens; `calibrate_manual` then fits the physical camera and the lens, so its residuals differ.

**Saving.** *Save* (Ctrl+S) posts the points to the server, which checks them (at least four pairs, finite coordinates, parsed with `Correspondences.from_dict`), writes the file atomically and ends the session; a `.json` output path writes JSON, any other suffix YAML (CSV is not supported). The YAML file starts with comment lines naming the source images and has the layout above:

```yaml
# Image-to-world correspondences from tachymetry.calib.picker
# camera frame: in-memory image (1280x720)
# reference image: in-memory image (800x800)
image_size: [1280, 720]
world_units: reference_px
gsd: 0.1
points:
- {name: p0, image: [100.0, 500.0], world: [300.0, 400.0]}
- {name: p1, image: [300.0, 530.0], world: [350.0, 380.0]}
- ...
lines: []
```

*Download YAML* saves the same content through the browser instead. The page also works without the server: open `picker.html` from the installed package directory (`tachymetry/assets/`) in a browser, load the camera frame and the reference image with its file buttons, and use *Download YAML* (Ctrl+S downloads in this mode).

From Python, `launch_picker(image, reference=None, gsd=None, out="points.yaml", port=0, open_browser=True, existing=None)` accepts image paths or BGR arrays and returns the path of the written file; `existing` pre-loads correspondences for editing (a file, a dictionary in the file layout or a `Correspondences`; a reference session needs `reference_px` correspondences picked on the same reference image, and the image size must match).

## Example

The script below calibrates a synthetic camera (8 m high, 32 degrees down, 70 degree horizontal field of view; `tachymetry.synthetic.default_camera`) from 12 ground points with 1 px Gaussian click noise.

```python
import numpy as np
from tachymetry.calib import Correspondences, calibrate_manual
from tachymetry.synthetic import default_camera

truth = default_camera((1280, 720))                       # 8 m high, 32 deg down, 70 deg HFOV
rng = np.random.default_rng(0)
uv = rng.uniform([0, 0], [1279, 719], size=(4000, 2))
uv = uv[truth.ground_sampling_distance(uv) <= 0.30][:12]  # usable ground only
world = truth.image_to_ground(uv)                         # surveyed ground points (m)
clicked = uv + rng.normal(0.0, 1.0, uv.shape)             # 1 px click noise
corr = Correspondences(clicked, world, image_size=(1280, 720))

res = calibrate_manual(corr)                              # lens="auto": identity / equidistant / division
print(res.summary())
for model, c in res.details["lens_candidates"].items():
    print(f"{model:12s} CV RMSE {c['cv_rmse_m']:.3f} m")
cam = res.camera
print(f"focal {cam.focal:.1f} px (true {truth.focal:.1f}), height {cam.height:.3f} m (true {truth.height:.3f}), "
      f"pitch {cam.pitch:.2f} deg (true {truth.pitch:.2f})")
```

Output (Linux, NumPy 2.4, OpenCV 4.13; runs in about one second on a CPU):

```text
[manual] f=913.7px (hfov 70.0 deg), height=7.99 m, pitch=32.03 deg, roll=1.45 deg, yaw=55.0 deg, lens=identity; fit RMSE 0.106 m; leave-one-out RMSE 0.174 m
identity     CV RMSE 0.174 m
equidistant  CV RMSE 0.174 m
division     CV RMSE 0.183 m
focal 913.7 px (true 914.0), height 7.992 m (true 8.000), pitch 32.03 deg (true 32.00)
```

The equidistant model contains the identity ($s \to \infty$) and reaches the same cross-validated error; the parsimony rule keeps the identity lens. The 10-fold cross-validated RMSE (0.174 m) exceeds the in-sample RMSE (0.106 m), as expected for 7 parameters fitted to 24 residuals. A full comparison with the automatic methods on the same scene is in the [synthetic calibration walkthrough](../examples/synthetic-calibration.md).

## Practical notes

- Spread the points over the whole area where measurements are needed, including the far field; extrapolation beyond the convex hull of the points is constrained only by the physical model.
- Points on the ground plane only: kerb tops, raised medians and pole bases on a sloped sidewalk are off-plane by 0.1 to 0.2 m and bias the fit.
- Add two or three long straight polylines (lane lines, kerbs) when the lens is visibly distorted; they constrain the lens without requiring world coordinates.
- An orthophoto's GSD must be its true ground resolution. Web-map tiles in Web Mercator have a GSD of $156543.03 \cos\phi / 2^z$ m per pixel at zoom $z$ and latitude $\phi$.

# Lens models

Wide-angle surveillance lenses bend straight lines, and an uncorrected lens biases every ground position away from the image centre. Tachymetry models distortion as a mapping between *raw* pixels (as recorded) and *rectified* pixels (the ideal pinhole image of the same size, with the same $K$). All models are radially symmetric about a distortion centre $c$ (default: the image centre $((w-1)/2, (h-1)/2)$), so a raw pixel $p_d$ at radius $r_d = \lVert p_d - c \rVert$ maps to

$$
p_u = c + \frac{r_u(r_d)}{r_d}\,(p_d - c).
$$

`Lens.rectify(pts)` maps raw to rectified pixels and `Lens.distort(pts)` the reverse (`geometry/lens.py`). `Camera.project` applies $K$ and then `distort`; `Camera.image_to_ground` applies `rectify` and then $K^{-1}$. Lenses serialise with `to_dict()` (`{"model": ..., "image_size": ..., "center": ..., <parameters>}`) and are rebuilt with `lens_from_dict` or `make_lens(model, image_size, **params)`; `LENS_MODELS` also accepts the aliases `none` (identity) and `tan` (equidistant).

## Recommended model: equidistant to pinhole (`tan`)

For wide-angle surveillance lenses, Tachymetry recommends the one-parameter equidistant-to-pinhole model (`EquidistantLens`, model name `equidistant`, alias `tan`). Its rectification about the image centre is

$$
r_u = s \tan\!\left(\frac{r_d}{s}\right), \qquad r_d = s \arctan\!\left(\frac{r_u}{s}\right).
$$

It is what results when a lens maps the angle $\theta$ between a ray and the optical axis linearly to image radius, $r_d = s\,\theta$ (equidistant projection), and the rectified image is the pinhole projection $r_u = s \tan\theta$ of the same rays. Its properties, as implemented in `geometry/lens.py`:

- **One parameter.** $s$ (pixels) is the fisheye focal length; $s \to \infty$ is the pinhole. Estimators optimise $q = 1/s$, so the pinhole is the finite point $q = 0$, where the cost is smooth, instead of $s = \infty$. The admissible range is $q \in [0, 1.6 / S]$ with $S$ the half image diagonal (negative $q$, i.e. pincushion distortion, is clamped to the identity).
- **Monotone over the whole image.** $r_u(r_d)$ increases strictly for $0 \le r_d < \pi s / 2$, and its inverse for all $r_u \ge 0$, so the model cannot fold back or inflate outside the region where it was fitted. Polynomial models such as Brown–Conrady can: with $k_2 = k_3 = 0$ and $k_1 < 0$, $r_d(r) = r(1 + k_1 r^2)$ peaks at $r = 1/\sqrt{-3 k_1}$ (normalised units). This matters when correspondences or line segments cover only part of the frame.
- **No separate focal length.** Near the centre $r_u \approx r_d$, so the rectified image keeps the scale of the raw image, and the camera's focal length remains a parameter of $K$ alone. For a ground plane, any remaining focal-length difference is absorbed by the camera pose and homography, so lens and camera do not compete for the same degree of freedom (Kannala–Brandt, in contrast, carries its own radial unit `focal`).

Where it is used:

| Context | How to select it | Estimation |
|---|---|---|
| Manual calibration | `lens="auto"` (default) or `lens="tan"` / `"equidistant"`; CLI `--lens tan` | Jointly with the camera; with `auto`, chosen by cross-validation among `identity`, `equidistant` and `division` (visited in this order) |
| Road-marking calibration | `calibrate_from_lines(..., lens="tan")`; CLI `tachymetry calibrate lines --lens tan` | Vanishing-point consistency; with `lens="auto"` both `tan` and `division` are estimated and the one with the higher support is kept |
| Automatic calibration of a run | configuration value `calibration.lens: tan` (`--set calibration.lens=tan`); default `auto` | As for road markings, inside `calibrate_auto` |
| A known lens | `EquidistantLens(size, s=...)` attached to a camera, or `"lens": {"model": "equidistant", "s": ...}` in the camera JSON | none |

### Worked example

The script renders the synthetic intersection through an equidistant lens with $s$ = 900 px at 1280 by 720 pixels and estimates $s$ back in three ways.

```python
import numpy as np
from tachymetry.calib import Correspondences, calibrate_from_lines, calibrate_manual, estimate_lens_vp_consistency
from tachymetry.calib.lines import detect_segments
from tachymetry.geometry import EquidistantLens
from tachymetry.synthetic import RoadLayout, default_camera, make_scene

size = (1280, 720)
true = EquidistantLens(size, s=900.0)                     # the lens to recover
camera = default_camera(size, lens=true)                  # 8 m high, 32 deg down, 70 deg HFOV
image = make_scene(camera, RoadLayout(), seed=0).image    # raw image seen through the lens

# 1. vanishing-point consistency on the LSD segments of the raw image
vp, info = estimate_lens_vp_consistency(detect_segments(image), size, model="equidistant")
# 2. road-marking calibration with the tan model
lines = calibrate_from_lines(image, scale="height=8", lens="tan")
# 3. manual calibration from 16 ground points with 0.5 px click noise
rng = np.random.default_rng(0)
uv = rng.uniform([0, 0], [1279, 719], size=(6000, 2))
uv = uv[camera.ground_sampling_distance(uv) <= 0.30][:16]
corr = Correspondences(uv + rng.normal(0, 0.5, uv.shape), camera.image_to_ground(uv), image_size=size)
manual = calibrate_manual(corr, lens="equidistant")

for name, lens in (("VP consistency", vp), ("lines, lens='tan'", lines.camera.lens), ("manual", manual.camera.lens)):
    s = getattr(lens, "s", float("inf"))
    print(f"{name:18s} {lens.kind:12s} s = {s:6.1f} px   error {100 * (s / true.s - 1):+5.2f} %")
```

Output (Linux, NumPy 2.4, OpenCV 4.13; about 8 s on a CPU):

```text
VP consistency     equidistant  s =  893.1 px   error -0.77 %
lines, lens='tan'  equidistant  s =  893.1 px   error -0.77 %
manual             equidistant  s =  897.2 px   error -0.31 %
```

The vanishing-point estimate is the arg-max over a grid whose step in $q = 1/s$ is $1.6 / (72 S) \approx 3.0 \times 10^{-5}$ px$^{-1}$, i.e. 2.7 % of $s$ at $s$ = 900 px; the −0.77 % error is within one step. `calibrate_from_lines` with `lens="tan"` uses the same estimator and accepted it (gain above 0.08 and support ratio above 1.2). The manual estimate is continuous and limited by the 0.5 px click noise.

## Models

**Identity** (`IdentityLens`, `identity`). $r_u = r_d$.

**Division** (`DivisionLens`, `division`; Fitzgibbon, 2001). With radii in units of `scale` $S$ (default: half the image diagonal, $S = \tfrac12\sqrt{w^2 + h^2}$), $\rho = r / S$:

$$
\rho_u = \frac{\rho_d}{1 + k \rho_d^2}.
$$

The inverse is the root of $k \rho_u \rho_d^2 - \rho_d + \rho_u = 0$ that is continuous at $k \to 0$,

$$
\rho_d = \frac{1 - \sqrt{1 - 4 k \rho_u^2}}{2 k \rho_u},
$$

which is undefined (`NaN`) where $1 - 4 k \rho_u^2 < 0$. $k < 0$ corrects barrel distortion. One parameter, closed form in both directions, and well suited to line-based estimation. Bounds $k \in [-0.9, 0.6]$.

**Equidistant** (`EquidistantLens`, `equidistant`, alias `tan`). $r_u = s \tan(r_d / s)$ with $\theta = r_d / s$ clipped below $\pi / 2$; see [the recommended model](#recommended-model-equidistant-to-pinhole-tan). Bounds $q = 1/s \in [0, 1.6 / S]$, i.e. $s \geq 0.625$ half-diagonals (the half-diagonal then subtends at most 1.6 rad); `param_grid()` starts at the identity and at $s \in \{3, 1.8, 1.2, 0.85\} \times S$. A default `EquidistantLens` has $s = 10^9$ and counts as the identity when $s$ exceeds 1000 image diagonals.

**Brown–Conrady** (`BrownConradyLens`, `brown`), with OpenCV's conventions. With normalised coordinates $x = (u - c_x)/F$, $y = (v - c_y)/F$ and $r^2 = x^2 + y^2$,

$$
\begin{aligned}
x_d &= x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2 x^2),\\
y_d &= y\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y^2) + 2 p_2 x y.
\end{aligned}
$$

`distort` evaluates this forward model; `rectify` inverts it with 30 fixed-point iterations $x \leftarrow (x_d - \Delta x(x, y)) / (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$, as `cv2.undistortPoints` does with fewer iterations. Estimators free $k_1, k_2$ (bounds $[-1, 1]$); $p_1, p_2, k_3$ are used when set. `to_opencv()` returns `(K, dist)` with $K = \mathrm{diag}(F, F, 1)$ plus the centre. The radial polynomial is not monotone for all coefficients (for $k_2 = k_3 = 0$ and $k_1 < 0$, $r_d$ peaks at $r = 1/\sqrt{-3 k_1}$), so the inverse is meaningful only within the radius where the forward model is monotone.

**Kannala–Brandt** (`FisheyeLens`, `fisheye`; OpenCV `fisheye`). With $\theta = \arctan(r_u / F)$,

$$
\theta_d = \theta\,(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8), \qquad r_d = F\,\theta_d, \qquad r_u = F \tan\theta .
$$

`rectify` solves $\theta_d(\theta) = r_d / F$ with 30 Newton iterations and clips $\theta$ to $[0, \pi/2)$. With all $k_i = 0$ the model is the equidistant projection, not the identity, so `is_identity` is always false; `FisheyeLens(focal=s)` with zero coefficients equals `EquidistantLens(s)`. Estimators free $k_1, k_2$ (bounds $[-0.5, 0.5]$).

### Radial units and defaults

| Model | Radial unit | Default | Free parameters (estimation) |
|---|---|---|---|
| `division` | `scale` | half the image diagonal | $k$ |
| `equidistant` | $s$ is the parameter | $s = 10^9$ (identity) | $1/s$ |
| `brown` | `focal` | half the image diagonal | $k_1, k_2$ |
| `fisheye` | `focal` | half the image diagonal | $k_1, k_2$ |

The defaults of `brown` and `fisheye` normalise by half the image diagonal, not by the camera's focal length as in OpenCV. To reuse OpenCV coefficients, construct the lens with `focal` equal to OpenCV's $f_x$ (square pixels assumed) and the principal point as `center`. Because every radius is measured in a unit tied to the image (or to the lens's own `focal`), the coefficients are resolution independent: `Camera.resized(new_size)` rescales the centre and the keys `scale`, `s` and `focal` by the mean of the horizontal and vertical scale factors and leaves the dimensionless coefficients unchanged.

## Estimation

### Plumb lines

`estimate_lens_plumbline(image, model="division")` (`calib/lines.py`; Devernay and Faugeras, 2001) estimates one parameter (division or equidistant) so that edge chains become straight:

1. **Chains** (`edge_chains`): Canny edges (thresholds 60 and 160) of the Gaussian-blurred image ($\sigma = 1$); contours are halved, because a thin edge is traced on both sides; chains are sub-sampled every 6 points and split where the direction turns by more than 25 degrees; pieces with at least 5 points and a chord of at least 6 % of the image diagonal are kept, at most the 400 longest.
2. **Straightness** of a chain under a candidate lens: rectify its points and take the smallest singular value of the centred point matrix divided by $\sqrt{n}$, i.e. the RMS distance to the total-least-squares line.
3. **Objective**: the mean squared straightness of the 50 % most straight chains (`trim=0.5`, at least 3), so curved world structure is ignored.
4. **Search**: 61 grid values over the parameter bounds, then a bounded scalar minimisation between the neighbours of the best grid value.

`info` reports `residual_px` and `residual_identity_px` (the objective without correction). With fewer than 5 chains the identity is returned. The trimmed objective is minimised by any parameter for which half of the chains are straight; chains that pass close to the distortion centre are straight for every radial model. On the synthetic intersection of the [lens example](../examples/lens-rectification.md) (true division lens, $k = -0.27$), the 43 extracted chains had a straightness of 0.28 px under the identity and the estimate was $k = -0.068$. Plumb-line estimation needs long, off-centre straight edges (building facades, long kerbs) to be informative.

### Vanishing-point consistency

`estimate_lens_vp_consistency(segments, image_size, model="division")` scores a candidate parameter by how well the rectified segments agree with common vanishing points: the segments are rectified, up to 3 vanishing points are found by sequential RANSAC (threshold 1.0 degree, 600 iterations, minimum support 2 %), and the score is the fraction of the total original segment length that supports them (original lengths, so rectification stretch near the border does not inflate the score). The parameter is the arg-max over a 19-value grid (division: $k \in [-0.9, 0.3]$; equidistant: $q = 1/s \in [0, 1.6/S]$) refined on a 9-value grid between the neighbours of the best value. There is no continuous refinement, so the estimate is quantised: steps of $1/60 \approx 0.017$ in $k$ for the division model and of $1.6 / (72 S)$ in $1/s$ for the equidistant model. `info["gain"]` is the score increase over the identity and `info["support_identity"]` the identity's score.

`calibrate_from_lines` runs this estimator on all LSD segments of the raw image. With `lens="auto"` it estimates both the equidistant (`tan`) and the division model and keeps the one with the higher support; with `"tan"`, `"equidistant"` or `"division"` it estimates only that model. In every case the estimate is accepted only if the gain exceeds 0.08 and the support is more than 1.2 times the identity's; otherwise the pinhole is kept. The estimator works on short segments whose curvature is invisible, because it uses the consistency of their directions rather than their straightness.

### Joint estimation in manual calibration

`calibrate_manual` estimates the lens parameters together with the camera from point correspondences and optional straight polylines, with multi-start initialisation over each model's `param_grid()`, and chooses between `identity`, `equidistant` and `division` (visited in this order) by cross-validation with a 3 % parsimony margin; `tan` is accepted as a name for `equidistant`, and `brown` and `fisheye` are fitted on request (`lens="brown"`). See [Manual calibration](manual-calibration.md#estimator).

### GeoCalib

GeoCalib's `simple_radial` and `simple_divisional` camera models map to `BrownConradyLens(k1=k, focal=f)` and `DivisionLens(k=k, scale=f)` (`calib.geocalib_backend.lens_from_geocalib`); see [Automatic calibration](auto-calibration.md#geocalib).

## Rectifying images

`Lens.rectify_image(image)` resamples a raw image into the rectified pinhole image of the same size (bilinear `cv2.remap` with a cached map; pixels outside the raw image are black), and `Lens.distort_image(image)` performs the reverse resampling (border pixels replicated), which the synthetic renderer uses. The following runs as written:

```python
from tachymetry.geometry import EquidistantLens
from tachymetry.synthetic import RoadLayout, default_camera, make_scene

lens = EquidistantLens((1280, 720), s=900.0)
raw = make_scene(default_camera((1280, 720), lens=lens), RoadLayout(), seed=0).image
rect = lens.rectify_image(raw)          # ideal pinhole image of the same size
back = lens.distort_image(rect)         # resampled back into the raw geometry
print(lens.to_dict())
print(lens.rectify([[0.0, 0.0]]), lens.distort(lens.rectify([[0.0, 0.0]])))
```

```text
{'model': 'equidistant', 'image_size': [1280, 720], 'center': [639.5, 359.5], 's': 900.0}
[[-193.13628849 -108.57309728]] [[0. 0.]]
```

The top-left raw pixel, 733.6 px from the centre, moves to 955.2 px from the centre after rectification, and `distort` inverts `rectify`. Rectified images with strong barrel distortion map the raw border outside the frame; the tracker never rectifies images, only points, so no field of view is lost in measurement.

Synthetic examples with each model, estimated parameters and residual maps are in [Lens rectification](../examples/lens-rectification.md).

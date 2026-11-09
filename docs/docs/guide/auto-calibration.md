# Automatic calibration

`tachymetry.calib.calibrate_auto(background, detections, cfg)` calibrates a camera without user input from two kinds of evidence: the ground footprints of vehicles predicted by a cuboid detector, and the vanishing points of road markings and vertical structures in a traffic-free background image. It is the default of `tachymetry run` (`camera="auto"`). This page gives the mathematics of each candidate method, the fused estimate, the ranking, the metric scale and the choice of world frame. Notation follows [Camera calibration](calibration.md#camera-model-and-conventions).

## Overview

1. **Background.** The temporal median of `calibration.background_frames` = 41 frames sampled uniformly over the video removes moving traffic (`calib.lines.background_image`).
2. **Candidates.** Each method in `calibration.methods` (default `("cuboids", "lines")`; `depth` and `geocalib` can be added) produces a camera of unit height, or of its own metric height.
3. **Fusion.** When both footprints and vanishing points exist, a `fused` candidate refines the cuboid camera by maximum likelihood over all evidence.
4. **Ranking.** Every candidate is scored on the same evidence with a method-independent consistency score; the highest score wins (`result.method = "auto:<winner>"`).
5. **Metric scale** from vehicle footprints (default) or a scale cue, then **world alignment**.

All candidates are listed in `result.details["candidates"]` with their scores and consistency terms; the candidate cameras are in `result.details["candidate_cameras"]` for side-by-side previews with `calib.compare`.

## Cuboid footprints

UrbanOmniDetect-2 regresses the eight image corners of a 3D box for every road user; corners 0 to 3 touch the ground (front-left, rear-left, rear-right, front-right), so every vehicle detection carries an image quadrilateral that is the projection of a rectangle on the ground.

**Trusted footprints.** At detection time a cuboid is kept only if all corners are finite, the weakest corner confidence is at least `detector.cuboid_conf` = 0.5, every corner lies inside the detection box expanded by 0.75 box sizes, and no corner collapsed to the image origin (`detect.postprocess.cuboid_trust`); otherwise its corners are set to `NaN`. `collect_footprints` then keeps `car` detections with a score of at least 0.4 whose box is at least $\max(3, 0.01\sqrt{w^2 + h^2})$ px inside the image border.

**Diversity weights.** The image is divided into a $24 \times 14$ grid of cells by footprint centre, and each footprint receives the weight $1 / n_\mathrm{cell}$, so that a parked car contributing one footprint per frame for the whole video casts one vote per cell in total. Above 2500 footprints, 2500 are drawn without replacement with probability proportional to the weight (seed 0) and then weighted equally. Weights are normalised to unit mean.

**Parametrisation.** The camera is searched on the physical manifold $\theta = (\log f, \vartheta, \varphi)$ (focal length, pitch, roll) with yaw $= 90°$, position $(0, 0)$ and height 1 fixed: yaw and position choose the world frame and the height is the scale, neither of which changes the shape of back-projected footprints. Bounds: $f \in [0.15\,w, 12\,w]$, $\vartheta \in [-10°, 89.5°]$, $\varphi \in [-25°, 25°]$; the negative pitch range admits cameras that look slightly above the horizon (`calib/from_cuboids.py`).

**Residuals.** Each footprint $q$ is back-projected to ground points $g_0, \dots, g_3$. With edges $e_k = g_{k+1} - g_k$ (indices modulo 4), unit edges $\hat e_k = e_k / \lVert e_k \rVert$, length $L = \tfrac12(\lVert e_0 \rVert + \lVert e_2 \rVert)$ and width $W = \tfrac12(\lVert e_1 \rVert + \lVert e_3 \rVert)$, the per-footprint residual vector is

$$
r_q = \sqrt{w_q}\,\Big[\ \underbrace{\hat e_k \cdot \hat e_{k+1}}_{k = 0..3}\ ,\ \underbrace{\tfrac12\, \hat e_0 \times \hat e_2\ ,\ \tfrac12\, \hat e_1 \times \hat e_3}_{\text{parallelism}}\ ,\ \lambda\left(\log\frac{L}{W} - \log\frac{4.6}{1.85}\right)\Big],
$$

i.e. the cosines of the four corner angles (zero for right angles), the sines between opposite edges (zero for parallel edges) and a prior on the length-to-width ratio of passenger cars with weight $\lambda$ = `ASPECT_WEIGHT` = 1.0. The prior carries full weight because when all vehicles travel along the viewing ray (a motorway seen from an overpass), lengths and widths are otherwise poorly constrained. Footprints that do not back-project (at or above the horizon) receive residual 3.0 in each component. One global residual penalises a systematic change of footprint size with range,

$$
r_\mathrm{slope} = 2.0\, \beta \sqrt{N}, \qquad \beta = \arg\min_{\beta, \alpha} \sum_q w_q \left(\log\sqrt{L_q W_q} - \alpha - \beta \log d_q\right)^2,
$$

where $d_q$ is the horizontal distance from the camera to the footprint centroid and $N$ the number of valid footprints: with a wrong pitch, distant vehicles shrink or grow systematically. The cost $\sum \rho(r^2)$ uses the soft-$\ell_1$ loss with scale 0.12 (`calib/from_cuboids.py`, `_residuals`).

**Search.** A grid of 32 starts (pitch $\in \{1, 8, 15, 25, 35, 50, 65, 80\}°$ by horizontal field of view $\in \{35, 55, 75, 95\}°$) is evaluated on a random subset of at most 600 footprints; the 8 best are refined on the subset (at most 200 evaluations each), and the best is refined on all footprints (at most 600 evaluations). At least 40 trusted footprints are required.

Compared with a free projective rectification (4 degrees of freedom up to similarity), the 3-parameter physical model has no shear or anisotropy freedom, which removes the degeneracy of scenes in which all vehicles share one heading. The cuboid candidate is made metric with the footprint scale (below) and its world frame has $+x$ along the dominant vehicle axis, the axial mean $\phi = \tfrac12 \arg \sum_q w_q e^{2 i \alpha_q}$ of the footprint headings $\alpha_q$, and its origin at the median footprint centroid. Its own `score` is $\mathrm{clip}(1 - a / 15°, 0, 1)\cdot \mathrm{clip}(1 - |\beta| / 0.5, 0, 1)$, with $a = \arcsin(\mathrm{median}\,|\cos|)$ of the corner angles.

## Road markings and vanishing points

`calibrate_from_lines(image, scale=None, lens="auto")` (`calib/from_lines.py`):

1. **Lens.** With `lens="auto"` (configuration `calibration.lens`), the equidistant (`tan`) and the division model are both estimated by vanishing-point consistency on all line segments of the raw image, and the one with the higher support is kept; `"tan"` (or `"equidistant"`) and `"division"` estimate one model only. The estimate is accepted only if it raises the supported segment-length fraction by more than 0.08 and to more than 1.2 times the pinhole's support (see [Lens models](lens-models.md#vanishing-point-consistency)). The image is rectified before the next steps.
2. **Marking mask.** A white top-hat with an elliptical kernel of $k = \max(5, \mathrm{round}(0.02 \max(w, h)))$ px (odd) isolates thin bright structures. Pixels above $\max(18, P_{94})$ of the non-zero top-hat values are kept if they have low chroma (HSV saturation below 70) or are yellow (hue 15 to 40 in OpenCV units, saturation above 80, value above 100, with a relaxed threshold of 0.6 times). An optional `region` mask (e.g. from segmentation) restricts the result; a $2 \times 2$ opening removes speckle (`marking_mask`).
3. **Segments.** OpenCV's LSD detector (probabilistic Hough on Canny edges as a fallback) with a minimum length of 1.2 % of the image diagonal. Marking segments are those with at least 60 % of 9 sample points inside the mask dilated by 3 px. At least 8 are required.
4. **Vanishing points.** Sequential length-weighted RANSAC (`estimate_vanishing_points`): hypotheses are intersections $v = l_a \times l_b$ of two segments drawn with probability proportional to length (3000 per vanishing point). For a segment with midpoint $m$ and direction $d$, the residual is the angle $\epsilon$ between $d$ and the ray from $m$ to $v$; a hypothesis scores $\sum \ell \max(0, 1 - (\epsilon / \epsilon_0)^2)$ with $\epsilon_0 = 1.5°$. The winner is refined three times by weighted homogeneous least squares,

    $$
    v = \arg\min_{\lVert v \rVert = 1} \sum_j w_j\,(l_j^\top v)^2, \qquad w_j = \ell_j \max\!\left(0.05,\ 1 - (\epsilon_j/\epsilon_0)^2\right),
    $$

    (the eigenvector of the smallest eigenvalue of $\sum_j w_j l_j l_j^\top$), its inliers are removed, and the search repeats for up to four horizontal vanishing points while the support exceeds 4 % of the total segment length. Hypotheses closer than $3\epsilon_0$ to a found point are rejected.
5. **Vertical.** Segments of the whole image within 25 degrees of vertical (at least 4) give one vertical vanishing point (support at least 20 % of their length). Otherwise a horizontal candidate is reclassified as vertical if it lies within 30 degrees of the image's vertical axis and more than $1.5\,h$ below the image centre (`classify_vertical`).

**Focal length and rotation.** For two vanishing points of orthogonal directions and the principal point $c$ (square pixels),

$$
f^2 = -(v_1 - c)\cdot(v_2 - c)
$$

(`focal_from_vps`). The preferred solution pairs each horizontal vanishing point with the vertical one, which are orthogonal for any road direction; candidates with a horizontal field of view outside 15 to 130 degrees are discarded, the best-supported one is used, and the two best focal lengths are averaged (weighted by support) when they agree within 10 %. This solution makes no assumption that streets meet at right angles. Only without a usable vertical vanishing point are two horizontal vanishing points assumed orthogonal (`select_orthogonal_pair`, scored by $\sqrt{s_i s_j}$ times $|\cos \angle(n_{ij}, d_z)|^8$ when a vertical direction exists), which is recorded as `details["assumes_orthogonal_streets"]`.

The rotation follows from the back-projected directions $d_x = K^{-1} v_x / \lVert K^{-1} v_x \rVert$ (the main road, world $+x$) and $d_z$ (vertical, oriented towards the top of the image), with $d_x$ orthogonalised against $d_z$ and oriented into the scene, $d_y = d_z \times d_x$ and $R = [d_x\ d_y\ d_z]$ projected onto $SO(3)$ (`camera_from_vps`). The camera is placed at height 1 and rejected unless $0° < \vartheta < 90°$. The `lines` score is the support of the main vanishing point as a fraction of the marking-segment length.

## Fused maximum-likelihood refinement

When at least 40 footprints, a cuboid candidate and at least one horizontal or vertical vanishing point are available, `fuse_evidence` refines $(\log f, \vartheta, \varphi)$ from the cuboid camera over both kinds of evidence. Vanishing points are first gated with the starting camera: a horizontal one is used only if its direction lies within 6 degrees of the ground plane, the vertical one only if it lies within 12 degrees of the ground normal (this drops, for example, elevated signal mast arms). Both groups of residuals are expressed in units of their standard deviation:

- **Footprints.** The cuboid residuals $r_q$ (above) are divided by $\sigma_c \sqrt{N / N_\mathrm{eff}}$, where $\sigma_c = 1.4826\,\mathrm{median}\,|r|$ is their robust spread at the starting camera and $N_\mathrm{eff} = \max(10, \text{number of occupied } 24 \times 14 \text{ image cells})$ is an effective sample size: a vehicle seen for several seconds yields dozens of correlated footprints, and one independent observation per occupied cell is assumed.
- **Vanishing points.** With $n = R e_z$ the world up vector in camera coordinates and $d$ the unit back-projection of a vanishing point,

    $$
    r_h = \frac{n \cdot d_h}{\sigma_\mathrm{vp}} \quad \text{(sine of the elevation of a ground direction)}, \qquad r_z = \frac{n \times d_z}{\sigma_\mathrm{vp}},
    $$

    with $\sigma_\mathrm{vp}$ = `calibration.vp_sigma_deg` = 0.5 degrees (in radians).

The soft-$\ell_1$ loss is applied with scale 2.0, i.e. at two standard deviations, which limits the influence of outliers in either group; bounds are those of the cuboid method (at most 800 evaluations). Horizontal vanishing points enter only through their elevation, so the result does not assume orthogonal streets either. `details` reports `effective_footprints` and `vp_sigma_deg`.

## Ranking

Each candidate camera is evaluated on the same raw footprints, back-projected through the candidate's own lens, and on the same vanishing points (`calib.auto.consistency`): the corner-angle error $a$ (degrees, from the median $|\cos|$ of the corner angles), the size-range slope $\beta$, and the median vanishing-point residual $c$ (degrees). The score is

$$
S = e^{-a / 10°}\; e^{-|\beta| / 0.15}\; e^{-c / 3°} \times \begin{cases} 1 & 10° \le \mathrm{HFOV} \le 125° \\ 0.1 & \text{otherwise} \end{cases},
$$

with factors omitted when their evidence is missing. The candidate with the highest $S$ is selected. The `lines` camera is computed from the same vanishing points that enter $c$, so its $c$ is close to zero by construction, and the comparison between `lines` and the footprint-based candidates is decided mainly by the footprint terms. In the [synthetic walkthrough](../examples/synthetic-calibration.md) (true camera: $f$ = 914 px, pitch 32°, roll 1.5°) the candidates scored `lines` 0.348 ($f$ = 890 px), `cuboids` 0.279 ($f$ = 869 px) and `fused` 0.270 ($f$ = 883 px), and `lines` was selected.

## Metric scale and world frame

Unless a non-vehicle scale cue is given (`scale="height=8"`, `"lane_width=3.5"`; see [Metric scale](calibration.md#metric-scale)), the selected camera is scaled from vehicle sizes with `footprint_scale` (at least 10 footprints). All raw footprints are back-projected with the camera, and each contributes the dimension that lies across its viewing ray, because distances along the ray are far more sensitive to pitch and corner errors: with $\cos\gamma$ the absolute cosine between the footprint axis and the ray to its centroid, a footprint with $\cos\gamma > 0.5$ (seen roughly head-on) contributes the estimate $1.85 / W$, and one with $\cos\gamma < 0.85$ (seen roughly from the side) contributes $4.6 / L$; oblique footprints contribute both. The scale is the weighted median of these estimates, with width estimates weighted 4 times the diversity weight and length estimates weighted once (car widths vary about half as much as car lengths). If no scale can be established, a warning is issued and the camera keeps relative scale (height 1, `meta["metric"] = False`, `details["scale"] = {"scale": "relative"}`).

The world frame is then aligned (`_align_world`): $+x$ points along the best-supported horizontal vanishing direction within 6 degrees of the ground (lane lines), or along the dominant vehicle axis when there is none (which can be a parking angle, hence the preference for markings); the origin is the median vehicle position.

## Monocular depth and road segmentation

`calibrate_from_depth(image, depth_model="depth-pro", segmentation="segformer", height=None, focal=None)` (`calib/depth.py`; `pip install "tachymetry[depth]"`) measures the road plane from one image:

1. A monocular metric depth network predicts the z-depth $Z(u, v)$ and, for Depth Pro (field-of-view head) and MoGe-2 (intrinsics), the focal length; Depth Anything V2 assumes a 60 degree horizontal field of view unless `focal` is given.
2. A semantic segmentation network with `road` and `sidewalk` labels (SegFormer-B2 by default) selects those pixels, with label ids resolved by name; the mask is eroded by 0.4 % of the image diagonal so that depth bleeding across kerbs, vehicles and poles is excluded.
3. Pixels are back-projected, $X = Z K^{-1} (u, v, 1)^\top$ (at most 300,000 points).
4. The plane is parametrised as $m^\top X + 1 = 0$ with $m = n / d$, which gives the residual

    $$
    r = m^\top X + 1 = \frac{n^\top X + d}{d} = 1 - \frac{Z}{Z_\mathrm{plane}},
    $$

    the relative disagreement between the measured depth and the plane's depth along the same ray. Monocular depth errors grow roughly in proportion to depth, so this residual is approximately homoscedastic and one threshold (0.03) serves all mounting heights. MSAC (truncated quadratic cost) scores minimal-sample hypotheses on at most 20,000 points with adaptive stopping (confidence 0.999; at most 2000 hypotheses, drawn in batches of 128); the winner is refined by iterated linear least squares on its inliers (at most 20 rounds; `fit_ground_plane`).
5. The unit normal $n$, oriented towards the camera, is the world up vector in camera coordinates: $\vartheta = \arcsin(-n_z)$, $\varphi = \operatorname{atan2}(-n_x, -n_y)$; $d$ is the camera height in metres, and yaw is set to 90 degrees.

The score is the inlier ratio times $e^{-\mathrm{rms}/0.03}$. Pitch and roll depend on the shape of the depth map but not on its scale; they are coupled to the focal length (a focal length $s$ times too large yields $\tan\vartheta$ about $1/s$ times the true value, per the module documentation). The height inherits the absolute scale of the network, which can be wrong by a large factor for viewpoints unlike the training data; a warning is recorded when it falls outside 2.5 to 150 m. Within `calibrate_auto`, the selected camera is rescaled from vehicle sizes by default, which replaces the network's scale.

| Back-end | Default checkpoint | Intrinsics |
|---|---|---|
| `depth-pro` | `apple/DepthPro-hf` | field-of-view head |
| `moge2` | `Ruicheng/moge-2-vitl-normal` (needs the `moge` package) | predicted |
| `depth-anything-v2` | `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf` | assumed 60° HFOV |
| `segformer` / `segformer-b5` | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` / `-b5-` | — |

Any checkpoint can be named as `"<backend>:<model id>"`, and `segmentation` also accepts a precomputed boolean road mask.

## GeoCalib

`calibrate_geocalib(image, height=None, camera_model="pinhole")` (`calib/geocalib_backend.py`; GeoCalib installed with `pip install git+https://github.com/cvg/GeoCalib.git`) uses GeoCalib (Veicht et al., ECCV 2024), which predicts dense perspective fields and fits gravity and focal length (and optionally one distortion coefficient) with a differentiable Levenberg–Marquardt optimiser. Gravity fixes pitch and roll (GeoCalib's pitch is positive looking up: $\vartheta_\mathrm{tachymetry} = -\vartheta_\mathrm{GeoCalib}$, roll is equal; the conversion is made from the gravity vector itself). Yaw is set to 90 degrees; the height is `height` when given and 1 otherwise (`meta["metric"] = False`). `camera_model="simple_radial"` maps to `BrownConradyLens(k1, focal=f)` and `"simple_divisional"` to `DivisionLens(k=k1, scale=f)`. The score is $\exp(-\sigma_g / 5°)\exp(-(\sigma_f / f) / 0.25)$ from GeoCalib's posterior standard deviations of gravity ($\sigma_g$) and focal length ($\sigma_f$); `n_runs` averages repeated forward passes, whose random decoder bases change the focal length by about 1 %.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `calibration.methods` | `("cuboids", "lines")` | candidates tried; add `depth` or `geocalib` |
| `calibration.scale` | `"vehicles"` | `vehicles`, `height=<m>`, `lane_width=<m>` or `none` |
| `calibration.lens` | `"auto"` | lens of the `lines` candidate: `auto` (tan or division, by vanishing-point consistency), `tan`, `division`, `none` |
| `calibration.background_frames` | 41 | frames in the temporal median |
| `calibration.car_length_m`, `car_width_m` | 4.6, 1.85 | vehicle-size prior |
| `calibration.vp_sigma_deg` | 0.5 | angular uncertainty of vanishing points in the fused estimate |

```bash
tachymetry run clip.mp4 --set calibration.methods=cuboids,lines,depth --set calibration.scale=height=9.5
```

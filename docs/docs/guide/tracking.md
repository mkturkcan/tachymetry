# Association and track hygiene

This page specifies how detections become metric trajectories: per-detection ground measurements (`track/measure.py`), online association (`track/associate.py`), and the offline per-track repair sequence (`track/hygiene.py`, `track/signal.py`). Cross-track reasoning (stitching and verdicts) follows in [Verdicts](verdicts.md). Every default below is the value in `config.py` (`TrackerConfig`, `HygieneConfig`); constants that are not configurable are marked *fixed*.

## Ground measurements

`measure_ground` computes, for every detection, quantities in world metres on the ground plane $z = 0$:

| Quantity | Definition |
|---|---|
| contact | ground point under the bottom-centre of the box, $\big(\tfrac{x_1 + x_2}{2}, y_2\big)$, i.e. the object edge nearest the camera |
| gsd | ground sampling distance at the contact: largest singular value of the image-to-ground Jacobian, metres per pixel ([Bird's-eye view](bev-speed.md#ground-sampling-distance)) |
| valid | contact at least 2 px (*fixed*) below the horizon, finite, `gsd <= max_gsd_m` (0.30) and, if set, within `max_range_m` (default `None`) of the camera |
| truncated | a box edge within $\max(3\,\text{px}, \texttt{edge\_margin\_frac} \cdot \text{diagonal})$ of the image border; `edge_margin_frac = 0.004` (8.8 px at 1920×1080) |
| big | box wider than `big_box_frac = 0.45` of the image width or taller than 0.45 of its height |
| footprint | from a trusted cuboid: the four ground corners, their centroid, heading, length and width |

A cuboid footprint is accepted only when its back-projected length $L$ and width $W$ lie in $[0.6\,L_{\min}, 1.4\,L_{\max}]$ and $[0.6\,W_{\min}, 1.4\,W_{\max}]$ of the class bounds below, the corner angle between edges 0-1 and 0-3 satisfies $|\cos| < 0.35$ (between 69.5° and 110.5°), and the contact is valid (*fixed* tolerances).

| Class | Prior $L \times W \times H$ (m), `DIM_PRIORS` | Bounds $L_{\min}..L_{\max}$, $W_{\min}..W_{\max}$ (m), `DIM_BOUNDS` | Lowest-edge height $h_c$ (m), `CLEARANCE_PRIORS` |
|---|---|---|---|
| car | 4.6 × 1.85 × 1.5 | 3.0..6.8, 1.45..2.4 | 0.08 |
| bus | 12.0 × 2.55 × 3.2 | 7.0..20.0, 2.2..2.9 | 0.25 |
| truck | 8.0 × 2.5 × 3.2 | 4.5..20.0, 1.8..2.9 | 0.25 |
| motorcycle | 2.0 × 0.8 × 1.4 | 1.3..2.8, 0.45..1.1 | 0.05 |
| bicycle | 1.75 × 0.6 × 1.6 | 1.2..2.4, 0.35..0.9 | 0.05 |
| person | 0.5 × 0.5 × 1.7 | 0.25..1.0, 0.25..1.0 | 0.0 |

## Online association

`ByteTracker` is a two-stage BYTE association (Zhang et al., ECCV 2022) with a constant-velocity Kalman filter in continuous time. `associate()` runs it over the processed frames in time order and returns a track id per detection row.

**State and noise.** The state is $\mathbf{x} = (c_x, c_y, w, h, \dot c_x, \dot c_y, \dot w, \dot h)$ in pixels and pixels per second, with transition $F(\Delta t) = \begin{pmatrix} I_4 & \Delta t\, I_4 \\ 0 & I_4 \end{pmatrix}$. With box height $h$ (at least 2 px) and the reference interval $\Delta t_0 = 1/30$ s:

$$
Q = \operatorname{diag}(\sigma_p^2 I_4, \sigma_v^2 I_4)\cdot\max\!\Big(\frac{\Delta t}{\Delta t_0}, 10^{-3}\Big),\qquad
\sigma_p = \frac{h}{20},\quad \sigma_v = \frac{h/160}{\Delta t_0},\qquad
R = \Big(\frac{h}{20}\Big)^2 I_4 .
$$

New tracks start with $P_0 = \operatorname{diag}\big((h/10)^2 I_4, (10\,h/160/\Delta t_0)^2 I_4\big)$ and zero velocity. Because $Q$ scales with $\Delta t$, the filter behaves the same at any `rate_hz`.

**Cost.** For predicted track boxes and detections, the cost is $c = (1 - \mathrm{DIoU})/2 \in [0, 1]$ with $\mathrm{DIoU} = \mathrm{IoU} - \rho^2 / d^2$, where $\rho$ is the distance between box centres and $d$ the diagonal of the smallest enclosing box. Unlike IoU, DIoU still ranks non-overlapping boxes, so fast objects at low frame rates associate.

**Gates.** A pair is forbidden (cost $10^6$) when any of the following holds:

1. the detection's super-class differs from the track's (`super_class_gating = True`);
2. IoU is below the stage minimum (0 in stage 1, `min_iou_low = 0.20` in stage 2);
3. $|\ln(A_\text{det}/A_\text{track})| > 0.92$ (area ratio outside 0.40 to 2.5) or $|\ln(a_\text{det}/a_\text{track})| > 0.79$ for the aspect ratios $a = w/h$ (ratio outside 0.45 to 2.2) (*fixed*);
4. with a camera, both ground contacts finite: $\lVert g_\text{det} - g_\text{track} \rVert > v_{\max}\,\Delta t + 1\,\text{m} + 5\,\text{gsd}$, where $\Delta t$ is the time since the track's last ground update and `max_speed_mps = {vehicle: 50, two_wheeler: 20, person: 7}` m/s. This metric plausibility gate removes the identity swap between a parked car and a car passing in front of it.

**Two stages.** Stage 1 matches all tracks against detections with score at least `high_conf = 0.50` by the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) and accepts pairs with cost at most `match_cost / 2 + 0.25` (0.65 with `match_cost = 0.80`). Stage 2 matches the remaining confirmed tracks against detections with `low_conf = 0.10 <= score < 0.50` and accepts cost at most 0.5 (*fixed*). Unmatched high-score detections start tentative tracks; low-score detections never do.

**Life cycle.** A tentative track is confirmed after at least two hits spanning `0.5 * min_confirm_s` (0.075 s with `min_confirm_s = 0.15`) and dies at its first miss. A confirmed track that misses becomes lost; after 0.3 s of coasting its velocity is damped by a factor $0.85^{0.1\,\Delta t/\Delta t_0}$ per prediction, and it dies after `max_lost_s = 1.5` s or when its predicted centre leaves the image by more than 5 % of the width or height (*fixed*).

## Per-track hygiene

`process_raw_track` turns the detection rows of one raw track into zero or more clean tracks. The steps run in the order below. Counters of every step are reported in `summary.json` under `report.hygiene` (`HygieneStats`).

### Validity and frame uniqueness

Rows with `valid = False` are removed, and when a track holds two detections in one frame only the higher-scoring one is kept.

### Class vote

The class of a track is the label with the largest summed weight $s \cdot \sqrt{A} \cdot \kappa$, with detection score $s$, box area $A$ and centrality
$\kappa = \operatorname{clip}\!\big(1 - 0.8\,|x_m - W/2| / (W/2), 0.05, 1\big)\cdot\big(0.4 + 0.6\,\operatorname{clip}(y_2 / H, 0, 1)\big)$,
so central, near-field views dominate (`class_vote`). The vote is repeated for every segment after splitting.

### Presplit at gaps and teleports

`split = True`. The contact path is median-filtered over `median_window_s = 0.4` s. A cut is placed between consecutive samples when the time gap exceeds `max_gap_s = 5.0` s or the step of the median path exceeds

$$
v_\text{tele}\,\Delta t + 3\sqrt{s_i^2 + s_{i+1}^2},\qquad s = \max(\texttt{position\_noise\_m}, 3\,\text{gsd}),
$$

with `position_noise_m = 0.25` m and `teleport_speed_mps = {vehicle: 55, two_wheeler: 25, person: 8}` m/s. Every resulting run is kept and processed separately.

### Identity-switch split

Within gaps of at most 1 s (*fixed*), a cut requires two independent signals:

* **box jump**: the next box centre deviates from the constant-velocity prediction (image velocity over the previous 0.5 s) by more than `switch_jump = 1.2` box widths (width at least $20 H/1080$ px), or, when neither box is truncated or big, the box area changes by a ratio outside `switch_area = (0.45, 2.2)`;
* **kinematic violation**: the next median ground position deviates from the prediction with the ground velocity over the previous 1.0 s by more than $\max(2\,\text{m}, 3\sigma)$ (*fixed*).

### End trimming

`trim = True`. Leading and trailing samples whose boxes are truncated or big are removed. Then, at either end, samples whose box height is below `far_shrink = 0.5` of the median box height of the samples 0.1 to 1.0 s further inside the track are removed (far-field collapse: boxes of objects leaving into the distance shrink faster than the object does). Truncated and big samples inside a track keep their time stamps but contribute no measurement. Segments shorter than 3 samples are dropped.

### Track building

A segment is dropped if it has fewer than 3 samples, lasts less than `min_track_s = 0.8` s, or has fewer than 3 usable (not truncated, not big) samples.

**Dimensions.** Length and width start from `DIM_PRIORS`. With at least 3 trusted footprints (not for persons), the samples with the lowest 75 % of ground sampling distances give $L$ = 70th percentile and $W$ = 60th percentile of the footprint lengths and widths, clipped to `DIM_BOUNDS`. Heights always come from the priors.

### Centre estimation

`footprints = True` (with `False`, the contact point itself is used, which is the "bottom-centre" baseline of the [benchmarks](../benchmarks.md)). The goal is the centre of the ground footprint, which the box bottom misses for two reasons: the bottom of the box images the footprint edge nearest the camera, and that edge (bumper or wheel bottom) sits above the ground, so its back-projection overshoots along the viewing ray.

For a sample with contact $g$, horizontal camera position $C$, ray direction $\hat r = (g - C)/\lVert g - C\rVert$, range $d = \lVert g - C \rVert$, heading $\psi$ and ray azimuth $\alpha$:

$$
e = |L\cos(\psi - \alpha)| + |W \sin(\psi - \alpha)|,\qquad
o = \frac{d\,h_c}{H_\text{cam}},\qquad
\hat x = g + \hat r\,\max\!\big(\tfrac{e}{2} - o, 0\big),
$$

where $e$ is the extent of the footprint along the ray, $h_c$ the class clearance from the table above and $H_\text{cam}$ the camera height (at least 0.5 m). The overshoot term is the similar-triangles correction: a point at height $h_c$ whose ray meets the ground at range $d$ lies $d\,h_c/H_\text{cam}$ in front of that ground point.

The heading $\psi$ for this model comes from a preliminary smoothing of the contact path (window $\max(1.5\,\text{s}, \texttt{smooth\_window\_s})$, order 1), used where the speed exceeds 1 m/s and held through stops; without motion, the axial mean of the footprint headings; without footprints, the ray direction.

Where trusted footprints exist, the footprint centroid replaces the model at those samples, and the residual between footprint centroid and model is interpolated in time to the other samples, fading linearly to zero between 1 and 3 s away from the nearest footprint sample (*fixed*). Samples measured from a footprint carry the `FOOTPRINT` flag.

### Outlier repair

`repair = True`, applied when at least 5 usable samples exist. A Hampel identifier in time (`signal.hampel`): over a centred window of `hampel_window_s = 1.0` s holding at least 5 samples, a sample is replaced by the window median when one coordinate deviates by more than `hampel_sigma = 3.0` times the scaled median absolute deviation ($1.4826\,\mathrm{MAD}$) and the Euclidean deviation exceeds `hampel_min_m = 0.8` m. Repaired samples carry the `REPAIRED` flag.

### Gap-aware zero-phase smoothing

`signal.smooth_kinematics` computes positions and velocities:

1. Samples without a measurement (truncated, big) are excluded; the remaining samples are split into runs at gaps longer than `interp_gap_s = 1.5` s.
2. Each run is resampled to a uniform grid with the median sample spacing $\Delta t$ (linear interpolation fills gaps up to 1.5 s).
3. A median filter over `median_window_s = 0.4` s (odd length, edges replicated) removes impulses.
4. A Savitzky-Golay filter over `smooth_window_s = 1.2` s of order `smooth_order = 2` gives the position, and the analytic first derivative of the same local polynomial gives the velocity. The filter fits polynomials at the run ends instead of padding (`mode="interp"`), and the centred window adds no phase lag.
5. Position and velocity are interpolated back to the original sample times.

At 15 Hz, $\Delta t = 1/15$ s, so the median filter spans 7 samples and the Savitzky-Golay window 19 samples (`signal.odd` rounds to the next odd count).

### End-fold trimming

Also controlled by `trim`. A track end is cut when it forms a hook: if the track is at least $3\times$`fold_dist_m` = 6 m long and still moving at its end (speed at least `move_speed_kmh = 3`), the last `fold_dist_m = 2.0` m of path is compared with the preceding 2 m; if the two chords (both at least 0.3 m long, *fixed*) turn by more than `fold_deg = 45°`, the samples of the final 2 m are removed. The same test runs on the reversed track for its start. The track is dropped if fewer than 3 samples or less than `min_track_s` remain.

### Kinematics and hold-still

Speed is the norm of the smoothed velocity. The heading is $\operatorname{atan2}(v_y, v_x)$ where the speed exceeds `move_speed_kmh = 3.0` km/h and is held (forward-filled, then back-filled) through slower samples. With `hold_still = True`, every run of samples slower than `stop_speed_kmh = 1.5` km/h that lasts at least 0.6 s and whose raw measurements have an RMS radius of at most 0.6 m about their median (both *fixed*) is frozen at that median with zero velocity and flagged `HELD`. This removes the phantom speed of jittering positions of stopped vehicles.

### Parked classification

`parked = True`, for vehicles and two-wheelers. With `moving_share` the fraction of samples faster than 3 km/h, `rog` the RMS radius of the raw measurements about their median, `net` the end-to-end displacement and `path` the smoothed path length, a track is parked when any of these holds:

1. `rog < parked_radius_m` (1.2 m) and `moving_share < parked_moving_share` (0.30);
2. `moving_share < 0.3` and `net < 3.0` m (*fixed*);
3. jitter, i.e. `net < 3.0` m and `path > 4 max(net, 0.5 m)`, with `moving_share < 0.6` (*fixed*);
4. image-space stillness with `net < 6.0` m (*fixed*): the median box centres of the first and last 20 % of samples are less than 0.35 median box widths apart, and the 90th percentile of the box-centre distance to the median centre is below 0.6 box widths. This clause is immune to the far-field amplification that makes ground positions of distant parked cars wander.

Parked tracks are frozen at the median position with zero velocity (all samples `HELD`) and take the axial mean of their footprint headings (or the circular mean of the motion headings). Persons are never parked; a person track with `moving_share < 0.25` and `net < 2.0` m gets the state `stationary` (*fixed*); it is not frozen as a whole, although its stops may be held by the hold-still rule.

### Physical validity referees

`validity = True`, applied to moving tracks only. The first failing referee sets the verdict `invalid` and the reason:

| Reason | Test |
|---|---|
| `impossible_speed` | speed above `impossible_speed_mps` = {vehicle: 60, two_wheeler: 25, person: 9} m/s for at least `accel_duration_s = 0.6` s |
| `impossible_acceleration` | (not persons) absolute derivative of the 0.4 s running-median speed above `max_accel_mps2 = 12` m/s² for at least 0.6 s |
| `path_reversal` | (tracks longer than 1.2 s) positions on a 0.4 s lattice; two consecutive steps, each longer than `reversal_min_m = 1.5` m, turn by more than `reversal_deg = 150°` |
| `centre_outside_box` | (not persons, at least 10 observed samples) the estimated centre lifted to 0.3 of the object height and projected into the image falls inside the detection box, expanded by `inside_box_margin = 0.15` of its width and height, in fewer than `inside_box_share = 0.70` of the observed samples |

Invalid tracks are excluded from exports and renders but remain candidates for [stitching](verdicts.md#stitching), after which the joined track is validated again.

## Sample flags

Each track sample carries bit flags (`track.model.Flags`), exported in `tracks.json` and partly in `tracks.csv` ([Outputs](outputs.md)):

| Flag | Value | Meaning |
|---|---|---|
| `OBSERVED` | 1 | the position is backed by a detection at this sample |
| `INTERPOLATED` | 2 | no usable measurement (truncated or big box); position from neighbours |
| `REPAIRED` | 4 | an outlier replaced by the local median |
| `TRUNCATED` | 8 | the box touched the image border |
| `HELD` | 16 | position frozen during a stop or for a parked vehicle; speed zero |
| `FOOTPRINT` | 32 | centre measured from a trusted cuboid footprint |

## Configuration

All parameters above are fields of `TrackerConfig` and `HygieneConfig` and can be overridden from Python (`tm.track(..., hygiene__smooth_window_s=0.8)`), YAML, or the command line (`--set hygiene.smooth_window_s=0.8`). The full list with defaults is in the [configuration reference](configuration.md); the API is documented in [`tachymetry.tracking`](../reference/tracking.md). The effect of the stages on synthetic traffic with ground truth is measured in the [benchmarks](../benchmarks.md).

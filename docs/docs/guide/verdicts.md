# Verdicts

After per-track hygiene ([Association and track hygiene](tracking.md)), `tracking.builder.build_tracks` reasons across tracks (`tracking/verdicts.py`). It first joins fragments of one object, then assigns a verdict to each track. Only tracks with verdict `ok` appear in `tracks.csv`, `tracks.json`, the GeoJSON and MOT exports and the rendered video; `tracks_summary.csv` lists all tracks with `verdict` and `reason`. Defaults below are from `FilterConfig` (`config.py`); constants that are not configurable are marked *fixed*.

## Order of application

1. **Stitching**, when `filters.stitch = True` (independent of `filters.enabled`).
2. Tracks are sorted by start time and renumbered from 1.
3. When `filters.enabled = True`: **duplicates**, then **riders, inferred cyclists and passengers**, then **shadows**, then **ghosts**.

Each rule considers only tracks that are still `ok` when it runs, so the order matters: a track rejected as a duplicate is not tested again as a ghost. A duplicate is resolved against the other track of the pair even if that track is rejected by a later rule, in which case both are excluded.

## Stitching

Occlusions and missed detections end a raw track early; the object then reappears under a new id. `stitch_candidates` considers tracks with verdict `ok` or `invalid` and tests each ordered pair $(A, B)$:

1. $A$ must not have left through the image border at its end: its last sample is not `TRUNCATED` and its last box is more than $\max(3\,\text{px}, 0.01 \cdot \text{diagonal})$ from the border (*fixed*);
2. same super-class;
3. gap $\tau = t^B_0 - t^A_\text{end}$ with $0 < \tau \le$ `stitch_max_gap_s` (3.0 s); if $B$ entered through the border, $\tau \le 0.5$ s (*fixed*);
4. the constant-velocity prediction $p = x^A_\text{end} + v^A_\text{end}\,\tau$ reaches the start of $B$: $\delta = \lVert x^B_0 - p\rVert \le r$ with $r =$ `stitch_radius_m` $+ 0.25\,\lVert v^A_\text{end}\rVert\,\tau$ (2.5 m plus a quarter of the distance travelled during the gap);
5. if both end speeds exceed 1.5 m/s: the heading change is at most `stitch_max_turn_deg` (50°) and the speeds differ by at most $\max(3\,\text{m/s}, 0.5\max(s_A, s_B))$ (*fixed*);
6. a parked track is only joined to another parked track closer than 1.5 m (*fixed*).

Admissible pairs get the cost $\delta / r + 0.2\,\tau/\texttt{stitch\_max\_gap\_s}$; a one-to-one assignment (Hungarian algorithm) keeps pairs with cost below 1.2 (*fixed*). Chains of pairs are concatenated at the detection level and processed again by the full hygiene sequence, so the joined track is re-classified, re-smoothed across the gap (interpolated up to `interp_gap_s`) and validated again. The number of joined fragments is reported as `report.stitched_fragments`.

**Catches:** fragments from occlusion by other vehicles, poles or signal heads, and dropouts of a few frames. **Does not catch:** gaps longer than 3 s, objects that left the frame and returned, or turns of more than 50° during the gap.

## Duplicates

Two concurrent tracks of one object arise from part-and-whole boxes (a trailer and its truck) or from a re-initialised track that coexists with its predecessor. For two `ok` tracks of the same super-class with at least 3 common frames spanning at least `duplicate_min_s` (1.0 s), a common frame is a hit when the box IoU exceeds 0.5 (*fixed*), or when the ground positions are closer than `duplicate_distance_m` (1.5 m) and one box covers more than 60 % of the other (*fixed*). If hits make up at least `duplicate_share` (0.5) of the common frames, the track with fewer observed samples is marked `duplicate` (reason `duplicate of <id>`), provided the common frames cover at least half of its samples.

**Catches:** overlapping duplicates. **Does not catch:** duplicates without temporal overlap (those are fragments; see stitching).

## Riders, inferred cyclists and passengers

A person track is a pedestrian only when it is none of the following. The rules are applied to each person track in this order, and the first that holds decides (`mark_passengers_and_riders`).

**Riders.** A person on a bicycle or motorcycle is one road user with it. The person track is marked `rider` when any of these holds:

1. **Bike under the person** (SAM 3): a `bike` detection scoring at least 0.5, enlarged by 15 % of its size on each side, covers more than 0.30 of the person's box (`bike_ride` evidence, [SAM 3 prompts](detection.md#sam-3-prompts)) in more than `rider_share` (0.50) of the person's detections, at least `rider_min_frames` (4);
2. **Tracked two-wheeler**: the person's box overlaps the box of a two-wheeler track, enlarged in the same way, with intersection over the person box above `rider_overlap` (0.30) in more than `rider_share` of their common frames (at least `rider_min_frames`);
3. **Co-location**: on every second detection of the person, the samples at which some vehicle or two-wheeler track is present count; when at least `rider_colocation_min` (5) of them exist and in more than half of them one of those tracks lies within `rider_distance_m` (1.2 m) on the ground, the person rides it;
4. **Cycling speed**: the median speed of the person track exceeds `rider_speed_kmh` (12 km/h).

The reason records the rule (`bike detection under the person`, `on two-wheeler <id>`, `co-located with <class> <id>`, `moves at cycling speed`). When rule 2 or 3 names a partner, the person's `meta["partner"]` holds its id and the partner's `meta["rider"]` the person's.

**Inferred cyclists.** A person who only satisfies rule 4, for at least `cyclist_min_duration_s` (2.0 s), is a two-wheeler that no prompt detected: the track is kept, re-labelled `bicycle` with bicycle dimensions (1.75 × 0.6 × 1.6 m) and `meta["inferred"] = "person moving at cycling speed"`. A shorter fast person track is a `rider` without a partner.

**Passengers and reflections.** A person track whose box lies inside a vehicle or two-wheeler detection box of the same frame (any detection, not only tracked ones), with intersection over the person box above `passenger_ioa` (0.70), in more than `passenger_share` (0.60) of its detections is marked `passenger` (reason `box contained in a vehicle box`). This removes drivers and passengers seen through windows and people reflected in bus windows. A pedestrian who walks behind a bus for part of their track keeps the `ok` verdict as long as less than 60 % of the detections are contained. Riders are tested first, so a motorcyclist whose box lies inside the `vehicle` detection of the motorcycle is a rider, not a passenger.

## Shadows

A person track that moves in lockstep with a moving vehicle is a false positive on the vehicle's shadow or on the vehicle itself. With at least 3 common frames spanning at least `shadow_min_s / 2` (1.5 s), both moving faster than 8 km/h in at least 80 % of the common frames, a mean offset below 4.0 m and an offset standard deviation (square root of the summed per-axis variances) below `2 * shadow_offset_std_m` (0.5 m), the person track is marked `shadow` (constants other than `shadow_*` *fixed*).

## Ghosts

`mark_ghosts` rejects three kinds of false tracks, in this order:

| Reason | Rule |
|---|---|
| `too short` | duration below `ghost_min_duration_s` (1.0 s) |
| `low confidence and static` | mean score of observed samples below `ghost_min_score` (0.35) and path length below 2.0 m (*fixed*) |
| `static low-confidence vehicle without a 3D footprint` | parked vehicle with a mean score below `ghost_parked_score` (0.40), or, when the detector provides cuboids, a cuboid-footprint share below 0.15 (*fixed*) and a mean score below `ghost_parked_cuboid_score` (0.60) |

The third rule targets parked-vehicle hallucinations: shadows, road markings, signs or dark patches detected as cars at a fixed position. With SAM 3 and RF-DETR, only the score clause applies. The score thresholds refer to the detector's score scale: the `rfdetr` [score profile](detection.md#score-profiles) lowers `ghost_min_score` and `ghost_parked_score` to 0.20, because RF-DETR scores real parked cars between 0.15 and 0.5. With UrbanOmniDetect-2 the cuboid clause adds evidence: a real parked vehicle seen by a cuboid detector receives trusted 3D footprints for a large share of its samples, while a hallucination rarely does and its score stays moderate. A confident persistent false positive (for example a vehicle printed on a billboard) is kept as a parked car.

## Counts on the sample clip

On the public sample clip (`intersection-day`, default configuration with SAM 3 and the sample's field of view), `summary.json` reports 168 raw tracks, 11 stitched fragments and the following verdicts:

| Verdict | Tracks |
|---|---|
| `ok` | 49 |
| `rider` | 2 |
| `ghost` | 1 |
| `duplicate` | 1 |

Both riders are cyclists with a `bike` detection under them. On the straight-down sample (`dhaka-roundabout`, run with its sample override), 34 of the person tracks are riders, 31 of them on a tracked two-wheeler and 3 by their speed, and 3 more become inferred cyclists.

The rejected tracks and their reasons are listed in the [end-to-end example](../examples/intersection.md). The contribution of stitching and verdicts on synthetic traffic with ground truth (fragments per object, false tracks) is the last row of the [benchmarks](../benchmarks.md).

## Disabling or tuning

```bash
tachymetry run clip.mp4 --set filters.enabled=false            # no verdicts (stitching still runs)
tachymetry run clip.mp4 --set filters.stitch_max_gap_s=5.0     # bridge longer occlusions
```

Rejected tracks remain available in Python: `run.tracks.select(include_rejected=True)` returns all tracks, and `TrackSet.to_csv(path, include_rejected=True)` writes them with their verdicts.

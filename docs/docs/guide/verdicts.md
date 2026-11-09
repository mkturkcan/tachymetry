# Verdicts

After per-track hygiene ([Association and track hygiene](tracking.md)), `track.builder.build_tracks` reasons across tracks (`track/verdicts.py`). It first joins fragments of one object, then assigns a verdict to every track. Only tracks with verdict `ok` appear in `tracks.csv`, `tracks.json`, the GeoJSON and MOT exports and the rendered video; `tracks_summary.csv` lists all tracks with `verdict` and `reason`. Defaults below are from `FilterConfig` (`config.py`); constants that are not configurable are marked *fixed*.

## Order of application

1. **Stitching**, when `filters.stitch = True` (independent of `filters.enabled`).
2. Tracks are sorted by start time and renumbered from 1.
3. When `filters.enabled = True`: **duplicates**, then **passengers, riders and inferred cyclists**, then **shadows**, then **ghosts**.

Each rule considers only tracks that are still `ok` when it runs, so the order matters: a track rejected as a duplicate is not tested again as a ghost. A duplicate is resolved against the other track of the pair even if that track is rejected by a later rule; on the public sample clip this happens once (track 48 is a duplicate of track 31, which is then rejected as a ghost; both are excluded).

## Stitching

Occlusions and missed detections end a raw track early; the object then reappears under a new id. `stitch_candidates` considers tracks with verdict `ok` or `invalid` and tests every ordered pair $(A, B)$:

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

## Passengers and reflections

A person track whose box lies inside an `ok` vehicle box, with intersection over the person box above `passenger_ioa` (0.70), in more than `passenger_share` (0.60) of its samples is marked `passenger` (reason `box contained in a vehicle box`). This removes drivers and passengers seen through windows and people reflected in bus windows. A pedestrian who walks behind a bus for part of their track keeps the `ok` verdict as long as less than 60 % of the samples are contained.

## Riders and inferred cyclists

A person and a two-wheeler with at least 4 common frames are one road user when the person box overlaps the two-wheeler box, expanded by 15 % of its size on each side (*fixed*), with intersection over the person box above `rider_overlap` (0.30) in more than `rider_share` (0.50) of the common frames, or, with at least 5 common frames, when the ground positions are closer than `rider_distance_m` (1.2 m) in more than half of them (*fixed*). The person track is marked `rider` (reason `riding two-wheeler <id>`), and the two-wheeler keeps the id of its rider in `meta["rider"]`.

A person track that remains `ok`, lasts at least 2.0 s and has a median speed above 15 km/h over its moving samples (both *fixed*) is re-labelled `bicycle` with bicycle dimensions (1.75 × 0.6 × 1.6 m) and `meta["inferred"] = "person moving at cycling speed"`. This recovers cyclists whose bicycle was not detected. A cyclist slower than 15 km/h without a detected bicycle remains a pedestrian.

## Shadows

A person track that moves in lockstep with a moving vehicle is a false positive on the vehicle's shadow or on the vehicle itself. With at least 3 common frames spanning at least `shadow_min_s / 2` (1.5 s), both moving faster than 8 km/h in at least 80 % of the common frames, a mean offset below 4.0 m and an offset standard deviation (square root of the summed per-axis variances) below `2 * shadow_offset_std_m` (0.5 m), the person track is marked `shadow` (constants other than `shadow_*` *fixed*).

## Ghosts

`mark_ghosts` rejects three kinds of false tracks, in this order:

| Reason | Rule |
|---|---|
| `too short` | duration below `ghost_min_duration_s` (1.0 s) |
| `low confidence and static` | mean score of observed samples below `ghost_min_score` (0.35) and path length below 2.0 m (*fixed*) |
| `static low-confidence vehicle without a 3D footprint` | parked vehicle with a mean score below 0.4, or, when the detector provides cuboids, a footprint share below 0.15 and a mean score below 0.6 (*fixed*) |

The third rule targets parked-vehicle hallucinations: shadows, road markings, signs or dark patches detected as cars at a fixed position. A real parked vehicle seen by a cuboid detector receives trusted 3D footprints for a large share of its samples; a hallucination rarely does, and its score stays moderate. A confident persistent false positive (mean score of at least 0.6 with footprints, for example a vehicle printed on a billboard) is kept as a parked car.

## Counts on the sample clip

On the public sample clip (`intersection-day`, default configuration), `summary.json` reports 110 raw tracks, 3 stitched fragments and the following verdicts:

| Verdict | Tracks |
|---|---|
| `ok` | 44 |
| `duplicate` | 2 |
| `rider` | 2 |
| `ghost` | 1 |

The rejected tracks and their reasons are listed in the [end-to-end example](../examples/intersection.md). The contribution of stitching and verdicts on synthetic traffic with ground truth (fragments per object, false tracks) is the last row of the [benchmarks](../benchmarks.md).

## Disabling or tuning

```bash
tachymetry run clip.mp4 --set filters.enabled=false            # no verdicts (stitching still runs)
tachymetry run clip.mp4 --set filters.stitch_max_gap_s=5.0     # bridge longer occlusions
```

Rejected tracks remain available in Python: `run.tracks.select(include_rejected=True)` returns all tracks, and `TrackSet.to_csv(path, include_rejected=True)` writes them with their verdicts.

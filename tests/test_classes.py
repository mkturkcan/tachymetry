"""Vehicle classes from SAM 3 class evidence, and the separation of riders and passengers from pedestrians."""

from dataclasses import replace

import numpy as np

from tachymetry.config import ROAD_CLASSES, FilterConfig, PipelineConfig
from tachymetry.detect.sam3 import EVIDENCE_COLUMNS, class_evidence, evidence_class
from tachymetry.detections import DetectionSet
from tachymetry.tracking.builder import build_tracks
from tachymetry.tracking.hygiene import SIZE_FALLBACK_SMALL_LW, class_vote
from tachymetry.tracking.model import Flags, Track
from tachymetry.tracking.verdicts import mark_passengers_and_riders

COL = {n: j for j, n in enumerate(EVIDENCE_COLUMNS)}


def test_class_evidence_matches_by_iou_with_per_prompt_floors():
    veh = np.array([[100, 100, 300, 200]], np.float32)  # a vehicle detection
    person = np.array([[400, 80, 440, 200]], np.float32)  # a person on a bicycle
    ev_xyxy = np.array(
        [
            [110, 100, 300, 200],  # truck, IoU 0.95
            [100, 100, 300, 360],  # bus, IoU 0.38: no vote
            [100, 100, 300, 200],  # car at 0.32: below the floor of 0.35
            [120, 150, 160, 200],  # bike inside the vehicle box: covers it, IoU too small to vote
            [395, 150, 445, 215],  # bike under the person
        ],
        np.float32,
    )
    ev_score = np.array([0.45, 0.9, 0.32, 0.55, 0.6])
    ev_prompt = ["truck", "bus", "car", "bike", "bike"]
    ev = class_evidence(np.concatenate([veh, person]), ev_xyxy, ev_score, ev_prompt)
    np.testing.assert_allclose(ev[0, COL["truck"]], 0.45)
    assert ev[0, COL["bus"]] == 0 and ev[0, COL["car"]] == 0 and ev[0, COL["bike"]] == 0
    np.testing.assert_allclose(ev[0, COL["bike_cover"]], 0.55)
    assert evidence_class(ev[0]) == "truck"
    np.testing.assert_allclose(ev[1, COL["bike_ride"]], 0.6)
    assert evidence_class(np.zeros(len(EVIDENCE_COLUMNS))) is None


def _evidence_track(boxes, ev, image_size=(1920, 1080)):
    n = len(boxes)
    dets = DetectionSet(
        frame=np.arange(n),
        time=np.arange(n) / 10.0,
        xyxy=np.asarray(boxes, np.float32),
        score=np.full(n, 0.8, np.float32),
        cls=np.full(n, ROAD_CLASSES.index("car"), np.int16),
        image_size=image_size,
        fps=10.0,
        processed_frames=np.arange(n),
        evidence=np.asarray(ev, np.float32),
        evidence_names=EVIDENCE_COLUMNS,
    )
    return dets, np.arange(n)


def test_track_class_is_decided_by_near_field_evidence():
    """Large, central, near-field views outvote many far-field ones."""
    far = [[940, 300, 980, 320]] * 8  # small, near the horizon
    near = [[700, 700, 1200, 1000]] * 3  # large, near the bottom centre
    ev = np.zeros((11, len(EVIDENCE_COLUMNS)))
    ev[:8, COL["truck"]] = 0.9  # the far field says truck
    ev[8:, COL["suv"]] = 0.8  # the near field says SUV
    cls, info = class_vote(*_evidence_track(far + near, ev), (1920, 1080))
    assert cls == "car" and info["rule"] == "evidence"
    assert info["shares"]["suv"] > info["shares"]["truck"]


def test_weak_evidence_falls_back_to_the_bike_flag_and_then_to_a_car():
    boxes = [[900, 600, 960, 700]] * 6
    ev = np.zeros((6, len(EVIDENCE_COLUMNS)))
    ev[:, COL["bus"]] = 0.0
    ev[:2, COL["truck"]] = 0.4  # share 0.13 < 0.20
    ev[:5, COL["bike_cover"]] = 0.6  # a confident bike under 5 of 6 detections
    cls, info = class_vote(*_evidence_track(boxes, ev), (1920, 1080))
    assert cls == "bicycle" and info["rule"] == "bike_flag"
    ev[:, COL["bike_cover"]] = 0.0
    cls, info = class_vote(*_evidence_track(boxes, ev), (1920, 1080))
    assert cls == "car" and info["rule"] == "size"


def test_motorcycle_evidence_fills_the_two_wheeler_slot():
    """A motorcycle that the ``bike`` prompt barely matches is a two-wheeler through its ``motorcycle`` evidence."""
    boxes = [[900, 600, 1000, 760]] * 6
    ev = np.zeros((6, len(EVIDENCE_COLUMNS)))
    ev[:, COL["motorcycle"]] = 0.7
    ev[:1, COL["bike"]] = 0.45  # share 0.075: alone it would lose to car
    ev[:3, COL["car"]] = 0.5  # share 0.25
    cls, info = class_vote(*_evidence_track(boxes, ev), (1920, 1080))
    assert cls == "motorcycle" and info["rule"] == "evidence"
    assert info["shares"]["two_wheeler"] > info["shares"]["car"] > info["shares"]["bike"]
    assert evidence_class(ev[0]) == "motorcycle"  # the same rule for a single frame


def _small_vehicle():
    """Detections with masks of a 2.2 m by 1.0 m vehicle (a cycle-rickshaw) that no class prompt matched."""
    from tachymetry.geometry import Camera
    from tachymetry.synthetic.traffic import Agent, TrafficSpec, render_detections

    cam = Camera.from_pose(
        (1280, 720), focal=1100.0, height=12.0, pitch=25.0, yaw=45.0, roll=0.5, position=(-14.0, -14.0)
    )
    t = np.arange(60) / 15.0
    xy = np.column_stack([-8.0 + 4.0 * t, np.full(len(t), -2.0)])
    agent = Agent(1, "car", 2.2, 1.0, 1.5, t, xy, np.zeros(len(t)), np.full(len(t), 4.0))
    spec = TrafficSpec(duration_s=4.0, rate_hz=15.0, masks=True, ghost_rate=0.0, miss_prob=0.0, seed=3)
    ds = render_detections(cam, [agent], spec)
    ev = np.zeros((len(ds), len(EVIDENCE_COLUMNS)), np.float32)
    return replace(ds, evidence=ev, evidence_names=EVIDENCE_COLUMNS), cam


def test_small_vehicle_rule_is_opt_in():
    """Without class evidence, a car fit at the smallest car stays a car unless ``hygiene.small_vehicle_rule``."""
    ds, cam = _small_vehicle()
    cfg = PipelineConfig()
    ok = [tr for tr in build_tracks(ds, cam, cfg)[0] if tr.is_ok]
    assert len(ok) == 1 and ok[0].cls == "car" and ok[0].meta["class_rule"] == "size"
    assert ok[0].length <= SIZE_FALLBACK_SMALL_LW[0] and ok[0].width <= SIZE_FALLBACK_SMALL_LW[1]
    cfg.hygiene.small_vehicle_rule = True
    ok = [tr for tr in build_tracks(ds, cam, cfg)[0] if tr.is_ok]
    assert len(ok) == 1 and ok[0].cls == "bicycle" and ok[0].meta["class_rule"] == "size_small"


def _track(tid, cls, sup, frames, xy, boxes, speed_mps=0.0):
    n = len(frames)
    xy = np.asarray(xy, float)
    return Track(
        id=tid,
        cls=cls,
        super_cls=sup,
        t=np.asarray(frames, float) / 10.0,
        frame=np.asarray(frames),
        rows=np.asarray(frames),
        xy=xy,
        vel=np.tile([speed_mps, 0.0], (n, 1)),
        heading=np.zeros(n),
        raw_xy=xy.copy(),
        contact=xy.copy(),
        bbox=np.asarray(boxes, float),
        score=np.full(n, 0.8),
        flags=np.full(n, Flags.OBSERVED, np.uint8),
        length=0.5,
        width=0.5,
        height=1.7,
    )


def _dets_with_ride(n, ride_frames, vehicle_box=None):
    """One person detection per frame (row = frame) with ``bike_ride`` evidence in ``ride_frames``,
    plus a vehicle detection per frame when ``vehicle_box`` is given."""
    boxes = [[500, 300, 540, 400]] * n
    cls = [ROAD_CLASSES.index("person")] * n
    frame = list(range(n))
    ev = np.zeros((n, len(EVIDENCE_COLUMNS)))
    ev[ride_frames, COL["bike_ride"]] = 0.7
    if vehicle_box is not None:
        boxes += [vehicle_box] * n
        cls += [ROAD_CLASSES.index("car")] * n
        frame += list(range(n))
        ev = np.concatenate([ev, np.zeros((n, len(EVIDENCE_COLUMNS)))])
    m = len(boxes)
    return DetectionSet(
        frame=np.asarray(frame),
        time=np.asarray(frame) / 10.0,
        xyxy=np.asarray(boxes, np.float32),
        score=np.full(m, 0.8, np.float32),
        cls=np.asarray(cls, np.int16),
        image_size=(1280, 720),
        fps=10.0,
        processed_frames=np.arange(n),
        evidence=ev.astype(np.float32),
        evidence_names=EVIDENCE_COLUMNS,
    )


def test_riders_are_found_before_passengers():
    """A person on a bike inside a vehicle box (a motorcyclist seen with the vehicle prompt) is a rider."""
    n = 10
    frames = np.arange(n)
    person = _track(1, "person", "person", frames, np.tile([0.0, 0.0], (n, 1)), [[500, 300, 540, 400]] * n, 2.0)
    dets = _dets_with_ride(n, np.arange(7), vehicle_box=[480, 280, 560, 420])
    counts = mark_passengers_and_riders([person], FilterConfig(), dets)
    assert person.verdict == "rider" and counts["rider"] == 1 and counts["passenger"] == 0
    # without the bike under it, the same person is a passenger
    person = _track(1, "person", "person", frames, np.tile([0.0, 0.0], (n, 1)), [[500, 300, 540, 400]] * n, 2.0)
    dets = _dets_with_ride(n, [], vehicle_box=[480, 280, 560, 420])
    mark_passengers_and_riders([person], FilterConfig(), dets)
    assert person.verdict == "passenger"


def test_persons_co_located_with_a_vehicle_are_riders_and_fast_ones_cyclists():
    n = 30  # 2.9 s at 10 Hz
    frames = np.arange(n)
    path = np.column_stack([np.arange(n) * 0.5, np.zeros(n)])
    moto = _track(2, "motorcycle", "two_wheeler", frames, path, [[0, 0, 10, 10]] * n, 5.0)
    rider = _track(1, "person", "person", frames, path + [0.3, 0.2], [[600, 0, 610, 10]] * n, 5.0)
    walker = _track(3, "person", "person", frames, path + [0.0, 4.0], [[700, 0, 710, 10]] * n, 1.2)
    fast = _track(4, "person", "person", frames, path + [0.0, 20.0], [[800, 0, 810, 10]] * n, 5.0)
    counts = mark_passengers_and_riders([moto, rider, walker, fast], FilterConfig())
    assert rider.verdict == "rider" and rider.meta["partner"] == 2 and moto.meta["rider"] == 1
    assert walker.verdict == "ok" and walker.cls == "person"
    assert fast.verdict == "ok" and fast.cls == "bicycle" and counts["cyclist_inferred"] == 1
    # the same speed for less than cyclist_min_duration_s: not a pedestrian, but no cyclist either
    short = _track(5, "person", "person", frames[:15], path[:15] + [0.0, 40.0], [[900, 0, 910, 10]] * 15, 5.0)
    mark_passengers_and_riders([short], FilterConfig())
    assert short.verdict == "rider" and short.cls == "person"


def test_silhouette_calibration_uses_only_unambiguous_cars():
    """Detections with truck evidence (pickups) do not enter the car-silhouette scale."""
    from tachymetry.calib.from_silhouettes import car_tracklets
    from tachymetry.detections import MaskStore, crop_mask

    n = 8
    boxes, ev, frame = [], [], []
    for f in range(n):
        for k, x in enumerate((200, 600)):  # a car and a pickup driving side by side
            boxes.append([x + 4 * f, 300, x + 4 * f + 120, 360])
            e = np.zeros(len(EVIDENCE_COLUMNS))
            e[COL["car"]] = 0.8
            if k == 1:
                e[COL["truck"]] = 0.6
            ev.append(e)
            frame.append(f)
    boxes = np.asarray(boxes, np.float32)
    full = np.zeros((720, 1280), bool)
    crops = []
    for b in boxes:
        m = full.copy()
        m[int(b[1]) : int(b[3]), int(b[0]) : int(b[2])] = True
        crops.append(crop_mask(m, b))
    m = len(boxes)
    dets = DetectionSet(
        frame=np.asarray(frame),
        time=np.asarray(frame) / 10.0,
        xyxy=boxes,
        score=np.full(m, 0.9, np.float32),
        cls=np.full(m, ROAD_CLASSES.index("car"), np.int16),
        image_size=(1280, 720),
        fps=10.0,
        processed_frames=np.arange(n),
        masks=MaskStore.from_crops(crops),
        evidence=np.asarray(ev, np.float32),
        evidence_names=EVIDENCE_COLUMNS,
    )
    tracklets = car_tracklets(dets)
    rows = {mk.row for t in tracklets for mk in t.masks}
    assert tracklets and all(r % 2 == 0 for r in rows)  # only the car (even rows)

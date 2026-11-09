import numpy as np

from tachymetry.config import PipelineConfig
from tachymetry.results import TrackSet
from tachymetry.synthetic.traffic import TrafficSpec, evaluate, simulate
from tachymetry.tracking.builder import build_tracks
from tachymetry.tracking.signal import smooth_kinematics


def test_constant_velocity_has_no_edge_artifacts():
    """Regression: zero-padded median filters bend track ends (phantom decelerations)."""
    t = np.arange(0, 6, 0.2)
    xy = np.column_stack([10.0 * t, 5.0 * t])
    pos, vel, _ = smooth_kinematics(t, xy, None, 0.4, 1.2, 2, 1.5)
    assert np.allclose(np.linalg.norm(vel, axis=1), np.hypot(10, 5), atol=1e-6)
    assert np.allclose(pos, xy, atol=1e-6)


def test_gaps_do_not_create_phantom_speed():
    """Regression: smoothing by sample index across gaps inflates speeds next to the gap."""
    t = np.arange(0, 8, 1 / 15)
    keep = (t < 3.0) | (t > 3.4)  # a 0.4 s detection gap
    t = t[keep]
    xy = np.column_stack([10.0 * t, np.zeros_like(t)])
    _, vel, _ = smooth_kinematics(t, xy, None, 0.4, 1.2, 2, 1.5)
    assert np.max(np.abs(np.linalg.norm(vel, axis=1) - 10.0)) < 1e-6


def test_pipeline_on_simulated_traffic(pole_camera):
    ds, agents = simulate(pole_camera, TrafficSpec(duration_s=30, seed=3))
    tracks, report = build_tracks(ds, pole_camera, PipelineConfig())
    rep = evaluate(tracks, agents, 15.0)
    assert rep.speed_error_kmh["mae"] < 2.0
    assert rep.position_error_m["median"] < 0.3
    assert rep.false_tracks == 0
    assert report["verdict_counts"].get("ok", 0) > 5


def test_exports(tmp_path, pole_camera):
    ds, _ = simulate(pole_camera, TrafficSpec(duration_s=15, seed=4))
    tracks, _ = build_tracks(ds, pole_camera, PipelineConfig())
    ts = TrackSet(tracks, pole_camera, 15.0, {})
    files = ts.export(tmp_path, ("csv", "json", "geojson", "mot"))
    for p in files.values():
        assert p.exists() and p.stat().st_size > 0
    header = (tmp_path / "tracks.csv").read_text().splitlines()[0]
    assert header.startswith("track_id,class,state,verdict,frame,time_s,x_m,y_m")


def test_low_confidence_track_ends_are_trimmed():
    """A track that drifts onto a neighbouring or phantom detection ends in low-score samples; they are trimmed."""
    from types import SimpleNamespace

    from tachymetry.config import HygieneConfig
    from tachymetry.tracking.hygiene import HygieneStats, _trim_ends

    n = 40
    score = np.full(n, 0.65)
    score[:2] = 0.2
    score[-8:] = 0.15
    dets = SimpleNamespace(score=score, time=np.arange(n) / 15.0, xyxy=np.tile([100.0, 100.0, 200.0, 150.0], (n, 1)))
    obs = SimpleNamespace(truncated=np.zeros(n, bool), big=np.zeros(n, bool))
    stats = HygieneStats()
    kept = _trim_ends(np.arange(n), SimpleNamespace(dets=dets, obs=obs, cfg=HygieneConfig()), stats)
    assert kept[0] == 2 and kept[-1] == n - 9 and stats.trimmed_samples == 10


def test_mask_store_round_trip(tmp_path):
    """Instance masks are stored cropped and bit-packed, and survive save/load and frame iteration."""
    from tachymetry.detections import DetectionSet, FrameDetections, crop_mask

    full = np.zeros((60, 80), bool)
    full[10:30, 20:50] = True
    full[12, 21] = False
    box = np.array([20.0, 10.0, 49.0, 29.0])
    c = crop_mask(full, box)
    fr = [
        FrameDetections(
            0, 0.0, box[None].astype(np.float32), np.array([0.9], np.float32), np.array([2], np.int16), masks=[c]
        ),
        FrameDetections(
            1,
            0.1,
            np.zeros((2, 4), np.float32),
            np.array([0.5, 0.4], np.float32),
            np.array([0, 2], np.int16),
            masks=[None, None],
        ),
    ]
    ds = DetectionSet.from_frames(fr, ("person", "bicycle", "car"), (80, 60), 10.0)
    assert ds.masks is not None and ds.masks.has(0) and not ds.masks.has(1)
    assert np.array_equal(ds.masks.full(0, (80, 60)), full)
    back = DetectionSet.load(ds.save(tmp_path / "d.npz"))
    assert np.array_equal(back.masks.full(0, (80, 60)), full)
    frames = list(back.frames())
    assert frames[0].masks[0] is not None and frames[1].masks == [None, None]
    sub = frames[0].subset(np.array([True]))
    assert np.array_equal(sub.masks[0][2], c[2])

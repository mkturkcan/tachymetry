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

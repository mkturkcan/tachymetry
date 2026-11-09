"""Simulate traffic with ground truth, run the tracker and score it (CPU only, no downloads)."""

from tachymetry.config import PipelineConfig
from tachymetry.geometry import Camera
from tachymetry.synthetic.traffic import TrafficSpec, evaluate, simulate
from tachymetry.track import build_tracks

if __name__ == "__main__":
    cam = Camera.from_pose((1280, 720), focal=914, height=8.0, pitch=32, yaw=45, position=(-14, -14))
    detections, agents = simulate(cam, TrafficSpec(duration_s=60, seed=0))
    tracks, report = build_tracks(detections, cam, PipelineConfig())
    print(report["verdict_counts"])
    print(evaluate(tracks, agents, rate_hz=15.0).row())

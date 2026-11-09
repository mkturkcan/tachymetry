"""Synthetic ablation benchmark with ground truth.

Simulates intersection traffic seen by known cameras, injects realistic
detector failures (box jitter, misses, label flicker, flickering and
persistent false positives, parked-car jitter), runs the tracker with
successively more of the Tachymetry pipeline enabled, and scores each variant
against ground truth. Runs on the CPU in a few minutes.

    python tools/benchmark_synthetic.py --out docs/media/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tachymetry.config import PipelineConfig
from tachymetry.geometry import Camera
from tachymetry.synthetic.traffic import TrafficSpec, evaluate, naive_speeds, simulate
from tachymetry.tracking.builder import associate, build_tracks
from tachymetry.tracking.measure import measure_ground


@dataclass
class _Naive:
    id: int
    t: np.ndarray
    xy: np.ndarray
    speed: np.ndarray
    state: str = "moving"
    super_cls: str = "vehicle"
    is_ok: bool = True


def variants() -> dict[str, PipelineConfig]:
    out = {}
    base = PipelineConfig()
    smooth = PipelineConfig()
    h = smooth.hygiene
    h.split = h.trim = h.repair = h.parked = h.validity = h.hold_still = False
    smooth.filters.enabled = False
    smooth.filters.stitch = False
    out["+ zero-phase smoothing (bottom-centre)"] = _copy(smooth, footprints=False)
    out["+ footprint centres"] = _copy(smooth, footprints=True)
    rep = _copy(smooth, footprints=True)
    rep.hygiene.split = rep.hygiene.trim = rep.hygiene.repair = rep.hygiene.validity = True
    out["+ split, trim, repair, validity"] = rep
    park = _copy(rep)
    park.hygiene.parked = park.hygiene.hold_still = True
    out["+ parked and hold-still"] = park
    out["Tachymetry (full: + stitching and verdicts)"] = base
    return out


def _copy(cfg: PipelineConfig, **hyg: bool) -> PipelineConfig:
    c = PipelineConfig.from_dict(cfg.to_dict())
    for k, v in hyg.items():
        setattr(c.hygiene, k, v)
    return c


def run(seeds: tuple[int, ...] = (0, 1, 2), duration: float = 60.0) -> dict:
    cams = {
        "pole (8 m, 32 deg, 70 deg HFOV)": Camera.from_pose(
            (1280, 720), focal=914, height=8.0, pitch=32, yaw=45, roll=1.0, position=(-14, -14)
        ),
        "mast (18 m, 22 deg, 50 deg HFOV)": Camera.from_pose(
            (1920, 1080), focal=2060, height=18.0, pitch=22, yaw=40, roll=0.5, position=(-35, -35)
        ),
    }
    results: dict = {"cameras": list(cams), "seeds": list(seeds), "duration_s": duration, "rows": {}}
    for cam_name, cam in cams.items():
        for seed in seeds:
            spec = TrafficSpec(duration_s=duration, seed=seed)
            ds, agents = simulate(cam, spec)
            cfg0 = PipelineConfig()
            obs = measure_ground(ds, cam, cfg0.hygiene)
            ids = associate(ds, obs, cfg0)
            naive = [_Naive(int(k), *v) for k, v in naive_speeds(ds, cam, ids).items()]
            rows = {"Baseline: bottom-centre, frame differencing": evaluate(naive, agents, spec.rate_hz).row()}
            for name, cfg in variants().items():
                t0 = time.time()
                tracks, _ = build_tracks(ds, cam, cfg)
                r = evaluate(tracks, agents, spec.rate_hz).row()
                r["seconds"] = round(time.time() - t0, 2)
                rows[name] = r
            for name, r in rows.items():
                results["rows"].setdefault(cam_name, {}).setdefault(name, []).append(r)
            print(f"{cam_name} seed {seed}: {len(ds)} detections, {len(agents)} agents")
    return results


def summarise(results: dict) -> str:
    keys = [
        "pos_med_m",
        "speed_mae_kmh",
        "speed_p95_kmh",
        "id_switches",
        "fragments_per_object",
        "parked_detected",
        "parked_phantom_kmh",
        "false_tracks",
    ]
    heads = [
        "Position error (m, median)",
        "Speed MAE (km/h)",
        "Speed p95 (km/h)",
        "ID switches",
        "Fragments per object",
        "Parked detected",
        "Parked phantom speed p90 (km/h)",
        "False tracks",
    ]
    lines = []
    for cam, rows in results["rows"].items():
        lines.append(f"\n**{cam}**, mean over {len(results['seeds'])} seeds of {results['duration_s']:.0f} s\n")
        lines.append("| Variant | " + " | ".join(heads) + " |")
        lines.append("|" + "---|" * (len(heads) + 1))
        for name, rs in rows.items():
            vals = []
            for k in keys:
                v = np.nanmean([r[k] if r[k] is not None else np.nan for r in rs])
                vals.append("" if np.isnan(v) else (f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"))
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=60.0)
    a = ap.parse_args()
    res = run(tuple(range(a.seeds)), a.duration)
    md = summarise(res)
    print(md)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path(a.out).with_suffix(".md").write_text(md + "\n", encoding="utf-8")

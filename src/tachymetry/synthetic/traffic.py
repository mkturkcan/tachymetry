"""Synthetic traffic with ground truth and a detector-noise model.

Agents move through a four-way intersection seen by a known camera: through
traffic, right and left turns, vehicles stopping at the stop line and pulling
away, parked cars along the kerb, and pedestrians on the crosswalks. Each agent
is an oriented 3D box; detections are the image bounding boxes and cuboid
corners of those boxes after occlusion, with the failure modes of real
detectors injected:

* box-edge jitter proportional to box size, and cuboid-corner noise,
* missed detections (more frequent for small and occluded objects),
* car/truck label flicker,
* short-lived false positives ("ghosts") on the road surface,
* persistent low-confidence false positives at a fixed location,
* frame-to-frame jitter on parked vehicles (the source of phantom speeds).

:func:`evaluate` matches tracks to ground truth and reports position and speed
errors, identity consistency, parked-vehicle handling and false tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ROAD_CLASSES
from ..detections import DetectionSet, FrameDetections
from ..geometry.camera import Camera
from .scene import Box3D, RoadLayout

__all__ = ["TrafficSpec", "Agent", "simulate", "evaluate", "naive_speeds"]


@dataclass
class TrafficSpec:
    duration_s: float = 60.0
    rate_hz: float = 15.0
    vehicles_per_min: float = 24.0
    pedestrians_per_min: float = 10.0
    parked: int = 6
    stop_prob: float = 0.35  #: share of vehicles that stop at the stop line
    jitter: float = 0.02  #: box-edge noise, fraction of box size
    corner_noise: float = 0.03  #: cuboid-corner noise, fraction of box size
    miss_prob: float = 0.06
    flicker_prob: float = 0.08
    ghost_rate: float = 0.25  #: short false positives per frame
    static_ghosts: int = 2
    parked_jitter: float = 0.035
    seed: int = 0


@dataclass
class Agent:
    aid: int
    cls: str
    L: float
    W: float
    H: float
    t: np.ndarray  # (n,)
    xy: np.ndarray  # (n, 2)
    heading: np.ndarray  # (n,)
    speed: np.ndarray  # (n,) m/s
    parked: bool = False


def _lane_paths(layout: RoadLayout) -> list[tuple[str, np.ndarray]]:
    """Polylines for through, right-turn and left-turn movements (right-hand traffic)."""
    E = layout.extent * 0.9
    lw = layout.lane_width
    hns, hew = layout.half_ns, layout.half_ew
    paths = []
    for rot in range(4):  # four approaches
        a = rot * math.pi / 2
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        # approach from -x towards +x in the right-hand lane (y = -lw/2 ... outer lane -1.5 lw)
        lane_y = -0.5 * lw
        start = np.array([-E, lane_y])
        entry = np.array([-hns - 1.0, lane_y])
        through = np.array([start, entry, [hns + 1.0, lane_y], [E, lane_y]])
        # right turn: into the road going -y (south), lane x = +... after rotation
        ctr = np.array([-hns - 1.0, -hew - 1.0])
        r = abs(lane_y - ctr[1])
        arc = [ctr + r * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(math.pi / 2, 0, 9)]
        right = np.vstack([start, entry, arc[1:], [ctr[0] + r, -E]])
        # left turn: wide arc into the road going +y
        ctr2 = np.array([-hns - 1.0, hew + 1.0])
        r2 = abs(lane_y - ctr2[1])
        arc2 = [ctr2 + r2 * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(-math.pi / 2, 0, 13)]
        left = np.vstack([start, entry, arc2[1:], [ctr2[0] + r2, E]])
        for name, p in (("through", through), ("right", right), ("left", left)):
            paths.append((f"{name}{rot}", p @ R.T))
    return paths


def _resample(poly: np.ndarray, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], step)
    pts = np.column_stack([np.interp(ss, s, poly[:, 0]), np.interp(ss, s, poly[:, 1])])
    return ss, pts


def _vehicle(
    aid: int, rng: np.random.Generator, path: np.ndarray, t0: float, spec: TrafficSpec, layout: RoadLayout, turn: bool
) -> Agent:
    cls = "truck" if rng.random() < 0.08 else ("bus" if rng.random() < 0.04 else "car")
    L, W, H = {
        "car": (rng.normal(4.6, 0.35), rng.normal(1.85, 0.08), rng.normal(1.5, 0.1)),
        "truck": (rng.uniform(6.5, 9.5), 2.45, 3.0),
        "bus": (12.0, 2.55, 3.2),
    }[cls]
    ss, pts = _resample(path)
    v_cruise = rng.uniform(25, 50) / 3.6
    v_turn = rng.uniform(15, 22) / 3.6
    dt = 1.0 / spec.rate_hz
    stop = rng.random() < spec.stop_prob
    # distance of the stop line along the path
    stop_s = float(ss[np.argmin(np.abs(np.linalg.norm(pts - path[1], axis=1)))]) - 2.0
    wait = rng.uniform(3, 8) if stop else 0.0
    s, v, t = 0.0, v_cruise, t0
    T, S, V = [], [], []
    waited = 0.0
    while s < ss[-1]:
        target = v_cruise
        if turn and stop_s - 5 < s < stop_s + 25:
            target = v_turn
        if stop and waited < wait:
            d_stop = stop_s - s
            if d_stop <= 0.05:
                target, v = 0.0, 0.0
                waited += dt
            else:
                target = min(target, math.sqrt(max(2 * 2.5 * d_stop, 0)))
        a = np.clip((target - v) / dt, -3.5, 2.2)
        v = max(0.0, v + a * dt)
        s += v * dt
        T.append(t)
        S.append(s)
        V.append(v)
        t += dt
    S = np.clip(np.asarray(S), 0, ss[-1])
    xy = np.column_stack([np.interp(S, ss, pts[:, 0]), np.interp(S, ss, pts[:, 1])])
    d = np.gradient(xy, axis=0)
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    moving = np.linalg.norm(d, axis=1) > 1e-6
    hd = np.interp(np.arange(len(hd)), np.flatnonzero(moving), hd[moving]) if moving.any() else hd
    return Agent(aid, cls, float(L), float(W), float(H), np.asarray(T), xy, hd, np.asarray(V))


def simulate(
    camera: Camera, spec: TrafficSpec | None = None, layout: RoadLayout | None = None
) -> tuple[DetectionSet, list[Agent]]:
    """Simulate traffic and the detections a cuboid detector would produce."""
    spec = spec or TrafficSpec()
    layout = layout or RoadLayout()
    rng = np.random.default_rng(spec.seed)
    paths = _lane_paths(layout)
    agents: list[Agent] = []
    aid = 0
    n_veh = int(spec.vehicles_per_min * spec.duration_s / 60)
    starts = np.sort(rng.uniform(-8, spec.duration_s - 5, n_veh))
    for t0 in starts:
        name, p = paths[rng.integers(len(paths))]
        agents.append(_vehicle(aid, rng, p, float(t0), spec, layout, not name.startswith("through")))
        aid += 1
    # parked cars along the kerbs
    for _ in range(spec.parked):
        side = rng.choice([-1, 1])
        x = rng.uniform(layout.half_ns + 10, layout.extent * 0.8) * rng.choice([-1, 1])
        y = side * (layout.half_ew - 1.1)
        n = int(spec.duration_s * spec.rate_hz)
        t = np.arange(n) / spec.rate_hz
        agents.append(
            Agent(
                aid,
                "car",
                4.5,
                1.85,
                1.5,
                t,
                np.tile([x, y], (n, 1)),
                np.full(n, 0.0 if rng.random() < 0.5 else math.pi),
                np.zeros(n),
                parked=True,
            )
        )
        aid += 1
    # pedestrians on crosswalks
    n_ped = int(spec.pedestrians_per_min * spec.duration_s / 60)
    for t0 in np.sort(rng.uniform(-5, spec.duration_s - 5, n_ped)):
        rot = rng.integers(4)
        a = rot * math.pi / 2
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        x = -(layout.half_ns + 0.5 + layout.crosswalk_length / 2)
        ya, yb = -(layout.half_ew + 1.5), layout.half_ew + 1.5
        if rng.random() < 0.5:
            ya, yb = yb, ya
        v = rng.uniform(1.1, 1.6)
        dist = abs(yb - ya)
        n = int(dist / v * spec.rate_hz)
        t = t0 + np.arange(n) / spec.rate_hz
        xy = np.column_stack([np.full(n, x + rng.normal(0, 0.4)), np.linspace(ya, yb, n)]) @ R.T
        hd = np.full(n, math.atan2(*(np.array([0, np.sign(yb - ya)]) @ R.T)[::-1]))
        agents.append(Agent(aid, "person", 0.5, 0.5, 1.7, t, xy, hd, np.full(n, v)))
        aid += 1

    # render detections frame by frame
    w, h = camera.image_size
    n_frames = int(spec.duration_s * spec.rate_hz)
    frames = []
    ghosts_static = [(rng.uniform(0.2 * w, 0.8 * w), rng.uniform(0.55 * h, 0.9 * h)) for _ in range(spec.static_ghosts)]
    for f in range(n_frames):
        t = f / spec.rate_hz
        boxes, scores, clss, cubs, depths = [], [], [], [], []
        for ag in agents:
            k = int(round((t - ag.t[0]) * spec.rate_hz))
            if k < 0 or k >= len(ag.t):
                continue
            b = Box3D(ag.xy[k, 0], ag.xy[k, 1], math.degrees(ag.heading[k]), ag.L, ag.W, ag.H)
            uv = camera.project(b.corners())
            if not np.all(np.isfinite(uv)):
                continue
            x1, y1 = uv.min(axis=0)
            x2, y2 = uv.max(axis=0)
            if x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
                continue
            boxes.append([x1, y1, x2, y2])
            cubs.append(uv)
            depths.append(float(np.linalg.norm(np.append(ag.xy[k], 0.0) - camera.center)))
            clss.append(ag.cls)
            scores.append(ag)
        if not boxes:
            frames.append(FrameDetections.empty(f, t, with_cuboid=True))
            continue
        boxes = np.array(boxes)
        order = np.argsort(depths)
        visible = np.ones(len(boxes), bool)
        # occlusion by nearer boxes
        for ii, i in enumerate(order):
            bi = boxes[i]
            area = (bi[2] - bi[0]) * (bi[3] - bi[1])
            cov = 0.0
            for j in order[:ii]:
                bj = boxes[j]
                ix = max(0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
                iy = max(0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
                cov += ix * iy
            if cov / max(area, 1) > 0.6:
                visible[i] = False
        out_b, out_s, out_c, out_q = [], [], [], []
        for i in range(len(boxes)):
            ag = scores[i]
            bi = boxes[i].copy()
            size = max(bi[2] - bi[0], bi[3] - bi[1])
            p_miss = spec.miss_prob + (0.25 if size < 18 else 0.0)
            if not visible[i] or rng.random() < p_miss:
                continue
            jit = spec.parked_jitter if ag.parked else spec.jitter
            bi += rng.normal(0, jit * size + 0.7, 4)
            bi = np.clip(bi, 0, [w - 1, h - 1, w - 1, h - 1])
            if bi[2] - bi[0] < 3 or bi[3] - bi[1] < 3:
                continue
            cls = ag.cls
            if cls in ("car", "truck") and rng.random() < spec.flicker_prob:
                cls = "truck" if cls == "car" else "car"
            sc = float(np.clip(0.55 + 0.4 * min(size / 120, 1) + rng.normal(0, 0.05), 0.15, 0.99))
            cub = cubs[i] + rng.normal(0, spec.corner_noise * size, (8, 2))
            conf = np.full((8, 1), 0.9 if size > 25 and cls != "person" else 0.3)
            out_b.append(bi)
            out_s.append(sc)
            out_c.append(ROAD_CLASSES.index(cls))
            out_q.append(np.concatenate([cub, conf], axis=1))
        n_g = rng.poisson(spec.ghost_rate)
        for _ in range(n_g):  # flickering false positives on the road
            cx, cy = rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.45 * h, 0.95 * h)
            s = rng.uniform(20, 70)
            out_b.append([cx - s, cy - 0.6 * s, cx + s, cy + 0.6 * s])
            out_s.append(float(rng.uniform(0.3, 0.75)))
            out_c.append(ROAD_CLASSES.index("car"))
            out_q.append(np.full((8, 3), np.nan))
        for gx, gy in ghosts_static:  # persistent false positive (sign, shadow, road marking)
            if rng.random() < 0.85:
                s = 30
                out_b.append([gx - s, gy - 0.5 * s, gx + s, gy + 0.5 * s] + rng.normal(0, 1.5, 4))
                out_s.append(float(rng.uniform(0.35, 0.7)))
                out_c.append(ROAD_CLASSES.index("car"))
                out_q.append(np.full((8, 3), np.nan))
        if out_b:
            fd = FrameDetections(
                f,
                t,
                np.array(out_b, np.float32),
                np.array(out_s, np.float32),
                np.array(out_c, np.int16),
                np.array(out_q, np.float32),
            )
        else:
            fd = FrameDetections.empty(f, t, with_cuboid=True)
        frames.append(fd)
    ds = DetectionSet.from_frames(
        frames,
        ROAD_CLASSES,
        camera.image_size,
        spec.rate_hz,
        {"detector": {"name": "synthetic"}, "rate_hz": spec.rate_hz},
    )
    return ds, agents


def naive_speeds(
    ds: DetectionSet, camera: Camera, ids: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Baseline: bottom-centre ground points differenced frame to frame (no repair, no smoothing)."""
    out = {}
    bottom = np.column_stack([(ds.xyxy[:, 0] + ds.xyxy[:, 2]) / 2, ds.xyxy[:, 3]])
    g = camera.image_to_ground(bottom)
    for tid in np.unique(ids[ids >= 0]):
        r = np.flatnonzero(ids == tid)
        r = r[np.argsort(ds.time[r])]
        if len(r) < 3:
            continue
        t = ds.time[r]
        xy = g[r]
        v = np.gradient(xy, t, axis=0)
        out[int(tid)] = (t, xy, np.linalg.norm(v, axis=1))
    return out


@dataclass
class EvalReport:
    position_error_m: dict[str, float] = field(default_factory=dict)
    speed_error_kmh: dict[str, float] = field(default_factory=dict)
    identity: dict[str, float] = field(default_factory=dict)
    parked: dict[str, float] = field(default_factory=dict)
    false_tracks: int = 0
    counts: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {
            "pos_med_m": self.position_error_m.get("median"),
            "pos_p95_m": self.position_error_m.get("p95"),
            "speed_mae_kmh": self.speed_error_kmh.get("mae"),
            "speed_p95_kmh": self.speed_error_kmh.get("p95"),
            "id_switches": self.identity.get("switches"),
            "coverage": self.identity.get("coverage"),
            "fragments_per_object": self.identity.get("fragments"),
            "parked_detected": self.parked.get("share_parked"),
            "parked_phantom_kmh": self.parked.get("phantom_speed_p90_kmh"),
            "false_tracks": self.false_tracks,
            "moving_vehicles_counted": self.counts.get("moving_vehicles"),
            "moving_vehicles_true": self.counts.get("moving_vehicles_true"),
        }


def evaluate(tracks: list[Any], agents: list[Agent], rate_hz: float, gate_m: float = 2.5) -> EvalReport:
    """Score tracks against ground truth with per-frame one-to-one (Hungarian) matching.

    Position and speed errors are taken over matched pairs; an identity switch is
    a change of the matched track of a ground-truth object between consecutive
    matched frames; a false track is one matched on fewer than 30 % of its samples.
    """
    from scipy.optimize import linear_sum_assignment

    rep = EvalReport()
    ok = [tr for tr in tracks if getattr(tr, "is_ok", True)]
    by_frame: dict[int, list[tuple[Any, int]]] = {}
    for tr in ok:
        for k, t in enumerate(tr.t):
            by_frame.setdefault(int(round(t * rate_hz)), []).append((tr, k))
    agent_at: dict[int, list[tuple[Agent, int]]] = {}
    for ag in agents:
        f0 = int(round(ag.t[0] * rate_hz))
        for j in range(len(ag.t)):
            agent_at.setdefault(f0 + j, []).append((ag, j))
    pos_err, sp_err = [], []
    timeline: dict[int, list[tuple[int, int]]] = {}
    matched_samples: dict[int, int] = {}
    for f, items in by_frame.items():
        gts = agent_at.get(f, [])
        if not gts:
            continue
        C = np.full((len(items), len(gts)), 1e6)
        for a, (tr, k) in enumerate(items):
            for b, (ag, j) in enumerate(gts):
                d = float(np.linalg.norm(tr.xy[k] - ag.xy[j]))
                if d <= gate_m:
                    C[a, b] = d
        ra, cb = linear_sum_assignment(C)
        for a, b in zip(ra, cb):
            if C[a, b] > gate_m:
                continue
            tr, k = items[a]
            ag, j = gts[b]
            pos_err.append(C[a, b])
            if ag.cls != "person":
                sp_err.append(abs(float(tr.speed[k]) - float(ag.speed[j])) * 3.6)
            timeline.setdefault(ag.aid, []).append((f, tr.id))
            matched_samples[tr.id] = matched_samples.get(tr.id, 0) + 1
    rep.false_tracks = sum(1 for tr in ok if matched_samples.get(tr.id, 0) < 0.3 * len(tr.t))
    pe, se = np.asarray(pos_err), np.asarray(sp_err)
    rep.position_error_m = {
        "median": float(np.median(pe)) if len(pe) else float("nan"),
        "p95": float(np.percentile(pe, 95)) if len(pe) else float("nan"),
    }
    rep.speed_error_kmh = {
        "mae": float(np.mean(se)) if len(se) else float("nan"),
        "p95": float(np.percentile(se, 95)) if len(se) else float("nan"),
    }
    switches, frags, covered = 0, [], []
    for ag in agents:
        seq = [tid for _, tid in sorted(timeline.get(ag.aid, []))]
        if not seq:
            continue
        switches += int(np.sum(np.diff(seq) != 0)) if len(seq) > 1 else 0
        frags.append(len(set(seq)))
        covered.append(len(seq) / max(len(ag.t), 1))
    rep.identity = {
        "switches": switches,
        "fragments": float(np.mean(frags)) if frags else float("nan"),
        "coverage": float(np.mean(covered)) if covered else float("nan"),
    }
    track_by_id = {tr.id: tr for tr in ok}
    parked = [ag for ag in agents if ag.parked and timeline.get(ag.aid)]  # visible parked cars only
    p_ok, phantom = 0, []
    for ag in parked:
        trs = [track_by_id[tid] for tid in {tid for _, tid in timeline[ag.aid]}]
        if trs and all(getattr(tr, "state", "") == "parked" for tr in trs):
            p_ok += 1
        for tr in trs:
            phantom.extend((tr.speed * 3.6).tolist())
    rep.parked = {
        "share_parked": (p_ok / len(parked)) if parked else float("nan"),
        "phantom_speed_p90_kmh": float(np.percentile(phantom, 90)) if phantom else 0.0,
    }
    moving_true = sum(1 for ag in agents if not ag.parked and ag.cls != "person" and timeline.get(ag.aid))
    moving_out = sum(
        1 for tr in ok if getattr(tr, "super_cls", "") == "vehicle" and getattr(tr, "state", "") == "moving"
    )
    rep.counts = {"moving_vehicles": moving_out, "moving_vehicles_true": moving_true}
    return rep

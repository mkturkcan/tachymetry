"""Regenerate every figure used by the project page (docs/index.html) and the documentation.

Real-footage figures use the public sample clips of :mod:`tachymetry.data` (downloaded on demand from
their hosts; licences and credits in ``tachymetry.data.attribution``): ``intersection-day`` for the
landing-page and guide figures, and all eight samples for the scene gallery (``docs/media/gallery``).
Samples that carry a known camera (``tachymetry.data.sample_camera``, the straight-down drone clip) use
it, with the configuration overrides they need; the others are calibrated automatically. Synthetic figures use
:mod:`tachymetry.synthetic`, whose cameras and lenses are known, and also write the measured tables
that the documentation includes (``synthetic_calibration.md``, ``lens_rectification.md``,
``figures.json``).

    pip install -e ".[depth]"
    python tools/make_docs_media.py                      # all figures into docs/media
    python tools/make_docs_media.py --only lens,synthetic --device cpu
    python tools/make_docs_media.py --only gallery          # stills of the eight sample clips
    CUDA_VISIBLE_DEVICES=1 python tools/make_docs_media.py --device cuda:0

Pipeline runs on the sample clips are cached in ``--work`` (default ``~/.cache/tachymetry/docs_media``),
so re-running only redraws. JPEG quality is 85 (80 for the gallery); images are at most 1920 px wide
(1280 px for the gallery).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUALITY = 85
GROUPS = ("clip", "calib", "synthetic", "lens", "social", "gallery", "occlusion")
INK = (13, 11, 10)
TEXT = (232, 235, 236)
MUTED = (163, 156, 155)
AMBER = (71, 181, 255)
CYAN = (255, 227, 111)


# ----------------------------------------------------------------------------- helpers
def save(img: np.ndarray, path: Path, max_width: int = 1920, quality: int = QUALITY) -> Path:
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"  {shown}  {w}x{h}  {path.stat().st_size / 1e3:.0f} kB")
    return path


def header(width: int, title: str, subtitle: str = "", height: int | None = None) -> np.ndarray:
    from tachymetry.render.text import draw_text, text_size

    height = height or max(56, int(round(width * 0.045)) // 2 * 2)
    bar = np.full((height, width, 3), INK, np.uint8)
    fs = int(height * 0.34)
    draw_text(bar, title, (22, int((height - fs) / 2) - 2), size=fs, color=TEXT, weight="semibold")
    if subtitle:
        tw, _ = text_size(title, fs, "semibold")
        draw_text(bar, subtitle, (22 + tw + 18, int((height - fs * 0.8) / 2)), size=int(fs * 0.8), color=MUTED)
    return bar


def titled(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    return np.concatenate([header(img.shape[1], title, subtitle), img], axis=0)


def hstack(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    parts = []
    for k, im in enumerate(images):
        if im.shape[0] != h:
            im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=INK)
        parts.append(im)
        if k < len(images) - 1:
            parts.append(np.full((h, gap, 3), INK, np.uint8))
    return np.concatenate(parts, axis=1)


def fmt(v: float | None, nd: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    if abs(v) < 0.5 * 10**-nd:
        v = 0.0
    return f"{v:+.{nd}f}" if v else f"{0:.{nd}f}"


# ----------------------------------------------------------------------------- sample clip
def run_sample(work: Path, device: str) -> Any:
    """The default pipeline on ``intersection-day``, with the overrides the sample registry gives it."""
    return run_named_sample("intersection-day", work, device)


def read_frame(run: Any, t: float) -> tuple[np.ndarray, float]:
    from tachymetry.io.video import VideoSource

    with VideoSource(run.source) as src:
        idx = int(round(t * src.fps))
        img = src.read(idx)
        return img, idx / src.fps


def pick_times(run: Any, n: int = 2, min_sep: float = 6.0) -> list[float]:
    """Times with the most moving road users in view (and some history for trails)."""
    ok = [tr for tr in run.tracks.ok if tr.state == "moving"]
    if not ok:
        return [0.5 * float(run.tracks.meta.get("duration_s") or 1.0)] * n
    t_all = np.unique(np.concatenate([tr.t for tr in ok]))
    inner = t_all[(t_all > t_all[0] + 3.5) & (t_all < t_all[-1] - 0.5)]
    t_all = inner if len(inner) else t_all
    score = np.zeros(len(t_all))
    for tr in ok:
        live = (t_all >= tr.t[0]) & (t_all <= tr.t[-1])
        sp = np.interp(t_all, tr.t, tr.speed_kmh)
        score += live * (sp > 8.0) * (1.0 + 0.5 * (tr.super_cls != "vehicle"))
    order = np.argsort(-score, kind="stable")
    picked: list[float] = []
    for i in order:
        if all(abs(t_all[i] - p) >= min_sep for p in picked):
            picked.append(float(t_all[i]))
        if len(picked) == n:
            break
    return picked


def clip_figures(run: Any, out: Path, stats: dict) -> None:
    from tachymetry.render.bev import BEVCanvas
    from tachymetry.render.overlay import draw_camera_overlay
    from tachymetry.render.state import StateIndex
    from tachymetry.render.video import render_frame

    cam = run.camera
    index = StateIndex(list(run.tracks.tracks), trail_s=3.0)
    t_hero, t_view = pick_times(run, 2)
    stats["clip"] = {"hero_time_s": t_hero, "viewer_time_s": t_view, "summary": run.summary(), "camera": cam.describe()}
    print(f"Hero frame at {t_hero:.2f} s, viewer frame at {t_view:.2f} s")

    img, t = read_frame(run, t_hero)
    save(img, out / "hero_frame.jpg")
    save(draw_camera_overlay(img.copy(), cam, index.at(t), "cuboid", "km/h", True, False), out / "hero_overlay.jpg")

    img, t = read_frame(run, t_view)
    states = index.at(t)
    save(img, out / "view_camera.jpg", 1600)
    save(draw_camera_overlay(img.copy(), cam, states, "cuboid", "km/h", True, False), out / "view_tracks.jpg", 1600)
    pts = np.vstack([tr.xy for tr in run.tracks.ok if tr.state == "moving"])
    bev = BEVCanvas(cam, run.background, (1600, 900), pts.mean(axis=0), 60.0, fit_points=pts, margin=0.04)
    save(bev.render(states), out / "view_bev.jpg", 1600)

    for layout in ("side", "dashboard", "camera", "bev"):
        save(render_frame(run, t_view, layout=layout), out / f"layout_{layout}.jpg", 1600)

    raw_vs_clean(run, t_view, out, stats)


def _id_color(k: int) -> tuple[int, int, int]:
    hue = (k * 47) % 180
    c = cv2.cvtColor(np.uint8([[[hue, 160, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def raw_vs_clean(run: Any, t_view: float, out: Path, stats: dict) -> None:
    """Same frame: detector + frame-to-frame association with naive speeds, and Tachymetry's output."""
    from tachymetry.config import PipelineConfig
    from tachymetry.render.bev import BEVCanvas
    from tachymetry.render.overlay import draw_camera_overlay
    from tachymetry.render.state import StateIndex
    from tachymetry.render.style import CLASS_COLORS
    from tachymetry.render.text import draw_chip
    from tachymetry.synthetic.traffic import naive_speeds
    from tachymetry.tracking.builder import associate
    from tachymetry.tracking.measure import measure_ground

    cfg = PipelineConfig()
    dets, cam = run.detections, run.camera
    obs = measure_ground(dets, cam, cfg.hygiene)
    ids = associate(dets, obs, cfg)
    naive = naive_speeds(dets, cam, ids)
    img, t = read_frame(run, t_view)
    f = int(dets.processed_frames[np.argmin(np.abs(dets.processed_frames - round(t * dets.fps)))])
    rows = np.flatnonzero(dets.frame == f)
    raw = img.copy()
    n_raw = 0
    for r in rows:
        tid = int(ids[r])
        x1, y1, x2, y2 = dets.xyxy[r]
        col = _id_color(tid) if tid >= 0 else (150, 150, 150)
        cv2.rectangle(raw, (int(x1), int(y1)), (int(x2), int(y2)), col, 2, cv2.LINE_AA)
        n_raw += 1
        if tid in naive:
            tt, _, sp = naive[tid]
            k = int(np.argmin(np.abs(tt - dets.time[r])))
            if dets.score[r] >= 0.3 and (y2 - y1) > 24:
                draw_chip(raw, f"#{tid} {sp[k] * 3.6:.0f} km/h", (int(x1), int(y1) - 4), col)
    index = StateIndex(list(run.tracks.tracks), trail_s=3.0)
    clean = draw_camera_overlay(img.copy(), cam, index.at(t), "cuboid", "km/h", True, False)
    a = titled(raw, "Detector + frame-to-frame association", "box bottom-centre, frame differencing")
    b = titled(clean, "Tachymetry", "footprint centres, repaired and smoothed tracks, verdicts")
    save(hstack([a, b], 12), out / "raw_vs_clean.jpg", 1920)

    # bird's-eye trajectories: raw bottom-centre points vs cleaned tracks
    pts = np.vstack([tr.xy for tr in run.tracks.ok if tr.state == "moving"])
    canvas = BEVCanvas(cam, run.background, (960, 960), pts.mean(axis=0), 60.0, fit_points=pts, margin=0.04)
    left, right = canvas.base.copy(), canvas.base.copy()
    names = dets.class_names
    for tid, (_t, xy, _speed) in naive.items():
        rr = np.flatnonzero(ids == tid)
        cls = names[int(np.bincount(dets.cls[rr]).argmax())]
        p = canvas.to_canvas(xy)
        ok = np.all(np.isfinite(p), axis=1)
        if ok.sum() >= 2:
            cv2.polylines(
                left,
                [np.round(p[ok] * 4).astype(np.int32)],
                False,
                CLASS_COLORS.get(cls, TEXT),
                1,
                cv2.LINE_AA,
                shift=2,
            )
    for tr in run.tracks.ok:
        p = canvas.to_canvas(tr.xy)
        col = CLASS_COLORS.get(tr.cls, TEXT)
        if tr.state == "parked":
            cv2.circle(right, tuple(np.round(p[0]).astype(int)), 4, (160, 152, 150), -1, cv2.LINE_AA)
        else:
            cv2.polylines(right, [np.round(p * 4).astype(np.int32)], False, col, 2, cv2.LINE_AA, shift=2)
    n_naive = len(naive)
    a = titled(left, f"Raw: {n_naive} associations", "bottom-centre ground points")
    b = titled(right, f"Tachymetry: {len(run.tracks.ok)} tracks", "parked vehicles as dots")
    save(hstack([a, b], 12), out / "raw_vs_clean_bev.jpg", 1920)
    stats["raw_vs_clean"] = {
        "frame": f,
        "raw_boxes_in_frame": n_raw,
        "raw_associations": n_naive,
        "tachymetry_tracks": len(run.tracks.ok),
    }


# ----------------------------------------------------------------------------- calibration on the clip
def calib_figures(run: Any, out: Path, stats: dict, device: str, work: Path | None = None) -> None:
    """Calibration previews of the intersection clip for the candidates of the default (segmenting) detector."""
    from tachymetry import data
    from tachymetry.calib.auto import calibrate_auto
    from tachymetry.calib.preview import preview
    from tachymetry.config import CalibrationConfig

    bg, dets = run.background, run.detections
    hfov = data.sample_overrides("intersection-day").get("calibration.hfov_deg")
    stats["calibration_clip"] = {}
    for name, cfg, title in (
        ("auto", CalibrationConfig(hfov_deg=hfov), f"Automatic: known {hfov:g} deg field of view + ground vanishing point"),
        ("lines", CalibrationConfig(methods=("lines",)), "Road markings and vertical vanishing points"),
        ("depth", CalibrationConfig(methods=("depth",)), "Monocular metric depth + road segmentation"),
    ):
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = calibrate_auto(bg, dets, cfg)
        except Exception as exc:  # optional back-ends may be missing
            print(f"  calib_{name}: skipped ({type(exc).__name__}: {exc})")
            continue
        cam = res.camera
        save(preview(bg, cam, title=f"{title} ({res.method}, silhouette scale)"), out / f"calib_{name}.jpg", 1920)
        stats["calibration_clip"][name] = {
            "method": res.method,
            "camera": cam.describe(),
            "focal_px": cam.focal,
            "height_m": cam.height,
            "pitch_deg": cam.pitch,
            "roll_deg": cam.roll,
            "scale": res.details.get("scale"),
            "seconds": round(time.time() - t0, 1),
        }
    if work is not None:
        dash_figure(run_named_sample("motorway-overpass", work, device), out, stats)


def dash_figure(run: Any, out: Path, stats: dict) -> None:
    """Dashed lane lines found without a camera, with their ground lengths and periods under the run's camera."""
    from tachymetry.calib.from_markings import dashed_lines, marking_regularity, road_region
    from tachymetry.render.text import draw_text

    cam, bg = run.camera, run.background
    rect = not cam.lens.is_identity  # dash ends are in rectified pixels when the camera has a lens
    # as in automatic calibration: chains on the road surface swept by vehicles, five dashes or more
    lines = [ln for ln in dashed_lines(bg, lens=cam.lens if rect else None, region=road_region(run.detections))
             if len(ln) >= 5]
    img = cam.lens.rectify_image(bg) if rect else bg.copy()
    scale = max(1, int(round(img.shape[0] / 720)))
    rows = []
    for k, ln in enumerate(lines):
        a, b = np.asarray(ln.a, float), np.asarray(ln.b, float)
        ga, gb = cam.image_to_ground(a, rectified=rect), cam.image_to_ground(b, rectified=rect)
        lengths = np.linalg.norm(gb - ga, axis=1)
        periods = np.linalg.norm(np.diff(ga, axis=0), axis=1)
        col = CYAN if k % 2 == 0 else AMBER
        for i in range(len(a)):
            p, q = tuple(np.round(a[i]).astype(int)), tuple(np.round(b[i]).astype(int))
            cv2.line(img, p, q, col, 3 * scale, cv2.LINE_AA)
            if np.isfinite(lengths[i]):
                m = (0.5 * (a[i] + b[i])).astype(int)
                draw_text(img, f"{lengths[i]:.1f}", (int(m[0]) + 8 * scale, int(m[1]) - 9 * scale), size=13 * scale,
                          color=TEXT)
        rows.append({"dashes": len(a), "length_m": [round(float(v), 2) for v in lengths],
                     "period_m": [round(float(v), 2) for v in periods]})
    reg = float(marking_regularity(cam, lines)) if len(lines) else float("nan")
    sub = f"{len(lines)} dash chains, spread {reg:.3f} (log units) under {cam.describe().split(', lens')[0]}"
    save(titled(img, "Dash check: ground lengths of lane dashes (m)", sub), out / "calib_dashes.jpg", 1920)
    stats["dash_check"] = {"sample": "motorway-overpass", "camera": cam.describe(), "regularity": reg, "lines": rows}


# ----------------------------------------------------------------------------- synthetic scenes
def _eval_grid(cam: Any, step: int = 20, max_gsd: float = 0.30) -> np.ndarray:
    w, h = cam.image_size
    uu, vv = np.meshgrid(np.arange(step / 2, w, step), np.arange(step / 2, h, step))
    uv = np.column_stack([uu.ravel(), vv.ravel()])
    g = cam.image_to_ground(uv)
    gsd = cam.ground_sampling_distance(uv)
    ok = np.all(np.isfinite(g), axis=1) & np.isfinite(gsd) & (gsd <= max_gsd)
    return uv[ok]


def _fit_2d(src: np.ndarray, dst: np.ndarray, scale: bool) -> tuple[np.ndarray, float]:
    """Least-squares rigid (or similarity) transform src -> dst. Returns residuals and the scale."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / np.sum(a * a)) if scale else 1.0
    pred = s * a @ R.T + md
    return np.linalg.norm(pred - dst, axis=1), s


def compare_cameras(est: Any, true: Any, uv: np.ndarray, frame_given: bool) -> dict[str, float]:
    g_t = true.image_to_ground(uv)
    g_e = est.image_to_ground(uv)
    ok = np.all(np.isfinite(g_t), axis=1) & np.all(np.isfinite(g_e), axis=1)
    g_t, g_e = g_t[ok], g_e[ok]
    if frame_given:
        rms = float(np.sqrt(np.mean(np.sum((g_e - g_t) ** 2, axis=1))))
    else:
        r, _ = _fit_2d(g_e, g_t, scale=False)
        rms = float(np.sqrt(np.mean(r**2)))
    _, s = _fit_2d(g_e, g_t, scale=True)
    return {
        "focal_pct": 100 * (est.focal - true.focal) / true.focal,
        "pitch_deg": est.pitch - true.pitch,
        "roll_deg": est.roll - true.roll,
        "height_m": est.height - true.height,
        "scale_pct": 100 * (s - 1),
        "ground_rms_m": rms,
        "n_eval": int(ok.sum()),
    }


def synthetic_figures(out: Path, stats: dict) -> None:
    from tachymetry.calib import calibrate_auto, calibrate_from_lines, calibrate_manual
    from tachymetry.calib.correspondences import Correspondences
    from tachymetry.calib.preview import preview
    from tachymetry.config import CalibrationConfig
    from tachymetry.synthetic import RoadLayout, default_camera, make_scene
    from tachymetry.synthetic.traffic import TrafficSpec, simulate

    size = (1280, 720)
    cam_true = default_camera(size)
    scene = make_scene(cam_true, RoadLayout(), seed=0)
    save(scene.image, out / "syn_scene.jpg")
    uv = _eval_grid(cam_true)
    g = cam_true.image_to_ground(uv)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(uv), 12, replace=False)
    img_pts = cam_true.ground_to_image(g[idx]) + rng.normal(0.0, 1.0, (12, 2))
    corr = Correspondences(img_pts, g[idx], image_size=size)
    ds_cub, _ = simulate(cam_true, TrafficSpec(duration_s=60, seed=0))  # boxes with cuboid corners
    ds_mask, _ = simulate(cam_true, TrafficSpec(duration_s=60, seed=0, masks=True))  # boxes with instance masks
    hfov = cam_true.fov[0]

    rows = []
    results = {}
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results["manual"] = (calibrate_manual(corr, size, lens="auto"), "12 points, 1.0 px noise, lens auto", True)
        results["lines"] = (
            calibrate_from_lines(scene.image, scale="height=8", lens="auto"),
            "background image, camera height 8 m",
            False,
        )
        results["auto"] = (calibrate_auto(scene.image, ds_mask), "background image + 60 s of instance masks", False)
        results["auto_fov"] = (
            calibrate_auto(scene.image, ds_mask, CalibrationConfig(hfov_deg=round(hfov))),
            f"as above, known {round(hfov):g} deg field of view",
            False,
        )
        results["auto_cuboids"] = (
            calibrate_auto(scene.image, ds_cub),
            "background image + 60 s of cuboid detections",
            False,
        )
    stats["synthetic_calibration"] = {
        "true_camera": cam_true.describe(),
        "seconds": round(time.time() - t0, 1),
        "rows": {},
    }
    for name, (res, inp, given) in results.items():
        m = compare_cameras(res.camera, cam_true, uv, given)
        label = {
            "manual": "Manual",
            "lines": "Road markings (`lines`)",
            "auto": f"Automatic, masks (`{res.method}`)",
            "auto_fov": f"Automatic, masks, known field of view (`{res.method}`)",
            "auto_cuboids": f"Automatic, cuboids (`{res.method}`)",
        }[name]
        rows.append(
            f"| {label} | {inp} | {fmt(m['focal_pct'])} | {fmt(m['pitch_deg'])} | {fmt(m['roll_deg'])} | "
            f"{fmt(m['height_m'])} | {fmt(m['scale_pct'])} | {m['ground_rms_m']:.3f} |"
        )
        stats["synthetic_calibration"]["rows"][name] = {
            "method": res.method,
            "camera": res.camera.describe(),
            **m,
            "cv_rmse_m": res.cv_rmse_m,
            "scale": res.details.get("scale") if name.startswith("auto") else None,
            "candidates": res.details.get("candidates") if name.startswith("auto") else None,
        }
    table = [
        "| Method | Input | Focal error (%) | Pitch error (deg) | Roll error (deg) | Height error (m) | "
        "Scale error (%) | Ground RMS after rigid alignment (m) |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    (out / "synthetic_calibration.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))

    res_m = results["manual"][0]
    pv = preview(scene.image, res_m.camera, title="Manual calibration: 12 correspondences with 1 px noise")
    off = pv.shape[0] - scene.image.shape[0]
    proj = res_m.camera.ground_to_image(g[idx])
    for p, q in zip(img_pts, proj):
        cv2.circle(pv, (int(round(p[0])), int(round(p[1])) + off), 7, AMBER, 2, cv2.LINE_AA)
        if np.all(np.isfinite(q)):
            cv2.circle(pv, (int(round(q[0])), int(round(q[1])) + off), 2, CYAN, -1, cv2.LINE_AA)
    save(pv, out / "syn_calib_manual.jpg")
    save(
        preview(scene.image, results["lines"][0].camera, title="Road markings, camera height 8 m"),
        out / "syn_calib_lines.jpg",
    )
    save(
        preview(
            scene.image,
            results["auto"][0].camera,
            title=f"Automatic calibration from instance masks (selected {results['auto'][0].method}, silhouette scale)",
        ),
        out / "syn_calib_auto.jpg",
    )


# ----------------------------------------------------------------------------- occlusion (mask geometry)
def _road_agent(aid: int, cls: str, dims: tuple, s0: float, s1: float, offset: float, v: float, t0: float,
                u: np.ndarray, n_hat: np.ndarray, c0: np.ndarray, rate: float) -> Any:
    import math

    from tachymetry.synthetic.traffic import Agent

    d = 1.0 if s1 > s0 else -1.0
    n = int(abs(s1 - s0) / v * rate)
    t = t0 + np.arange(n) / rate
    xy = c0 + (s0 + d * v * (t - t0))[:, None] * u + offset * n_hat
    return Agent(aid, cls, *dims, t, xy, np.full(n, math.atan2(d * u[1], d * u[0])), np.full(n, v))


def occlusion_figures(out: Path, stats: dict) -> None:
    """A bus passing behind a tree crown: footprints from instance masks and from boxes only, against ground truth.

    The scene of ``tests/test_mask_geometry.py::test_occluded_bus_keeps_its_footprint``: a 12 m bus at 25 km/h,
    40 m from a 12 m mast camera, and an oncoming car; segmenting-detector simulation with image-space occluders.
    """
    import math
    from dataclasses import replace

    from tachymetry.config import PipelineConfig
    from tachymetry.geometry import Camera
    from tachymetry.render.overlay import box_corners
    from tachymetry.render.style import CLASS_COLORS
    from tachymetry.render.text import draw_text
    from tachymetry.synthetic.masks import agent_parts, render_label_image, tree_occluder
    from tachymetry.synthetic.traffic import TrafficSpec, render_detections
    from tachymetry.tracking.builder import build_tracks

    rate = 15.0
    cam = Camera.from_pose((1280, 720), focal=1100.0, height=12.0, pitch=25.0, yaw=45.0, roll=0.5,
                           position=(-14.0, -14.0))
    u = np.array([math.cos(-math.pi / 4), math.sin(-math.pi / 4)])
    n_hat = np.array([-u[1], u[0]])
    c0 = np.array([-14.0, -14.0]) + 40.0 * np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    bus = _road_agent(0, "bus", (12.0, 2.55, 3.2), -30.0, 30.0, -1.75, 7.0, 0.0, u, n_hat, c0, rate)
    car = _road_agent(1, "car", (4.6, 1.8, 1.45), 30.0, -30.0, 1.75, 8.0, 0.5, u, n_hat, c0, rate)
    p = cam.ground_to_image((c0 + 4.0 * u - 1.75 * n_hat)[None])[0]
    tree = tree_occluder((p[0], p[1] + 35), 95, seed=1)
    spec = TrafficSpec(duration_s=8.5, rate_hz=rate, masks=True, occluders=(tree,), ghost_rate=0.0, miss_prob=0.03,
                       flicker_prob=0.0, seed=7)
    ds = render_detections(cam, [bus, car], spec)
    paths = {"masks": build_tracks(ds, cam, PipelineConfig())[0],
             "boxes": build_tracks(replace(ds, masks=None), cam, PipelineConfig())[0]}

    def bus_track(trs: list) -> tuple:
        for tr in trs:
            j = np.round((tr.t - bus.t[0]) * rate).astype(int)
            k = (j >= 0) & (j < len(bus.t))
            if tr.is_ok and k.sum() > 10 and np.median(np.linalg.norm(tr.xy[k] - bus.xy[j[k]], axis=1)) < 3:
                return tr, j[k], k
        raise RuntimeError("bus not tracked")

    def depth(xy: np.ndarray) -> float:
        return float(np.linalg.norm(np.append(xy, 0.0) - cam.center))

    # visible share of the bus silhouette per ground-truth sample
    vis = np.zeros(len(bus.t))
    area = np.zeros(len(bus.t))
    for i in range(len(bus.t)):
        parts = agent_parts("bus", bus.xy[i, 0], bus.xy[i, 1], bus.heading[i], bus.L, bus.W, bus.H)
        area[i] = (render_label_image(cam, [(0, parts, depth(bus.xy[i]))]) == 0).sum()
        seen = (render_label_image(cam, [(0, parts, depth(bus.xy[i]))], (tree,)) == 0).sum()
        vis[i] = seen / area[i] if area[i] else 0.0
    in_view = area >= 0.9 * area.max()  # whole silhouette inside the frame

    rows, entry, res = [], {}, {}
    for key, label in (("masks", "Instance masks (mask geometry)"), ("boxes", "Boxes only (same detections)")):
        tr, j, k = bus_track(paths[key])
        err = np.linalg.norm(tr.xy[k] - bus.xy[j], axis=1)
        hd = np.degrees(np.abs((tr.heading[k] - bus.heading[j] + np.pi / 2) % np.pi - np.pi / 2))
        res[key] = (tr, j, k, err, hd)
        entry[key] = {"samples": int(k.sum()), "centre_median_m": float(np.median(err)), "centre_max_m": float(err.max()),
                      "heading_max_deg": float(hd.max()), "length_m": float(tr.length), "width_m": float(tr.width),
                      "geometry": tr.meta.get("geometry", {}).get("source"),
                      "occluded_samples": tr.meta.get("geometry", {}).get("occluded_samples")}
        rows.append(f"| {label} | {int(k.sum())} | {np.median(err):.2f} | {err.max():.2f} | {hd.max():.1f} | "
                    f"{tr.length:.2f} | {tr.width:.2f} |")
    rows.append("| Ground truth | | | | | 12.00 | 2.55 |")
    table = ["| Path | Samples | Centre error, median (m) | Centre error, max (m) | Heading error, max (deg) | "
             "Length (m) | Width (m) |", "|---|---|---|---|---|---|---|", *rows]
    (out / "occlusion_bus.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))

    # three moments: fully visible, most hidden, visible again
    cand = np.flatnonzero(in_view)
    i_mid = int(cand[np.argmin(vis[cand])])
    clear = np.flatnonzero(in_view & (vis > 0.98))
    before = clear[clear < i_mid]
    after = clear[clear > i_mid]
    moments = [int(before[-1]) if len(before) else 0, i_mid, int(after[0]) if len(after) else len(bus.t) - 1]
    panels = []
    for i in moments:
        t = float(bus.t[i])
        f = int(np.argmin(np.abs(ds.processed_frames / rate - t)))
        parts_b = agent_parts("bus", bus.xy[i, 0], bus.xy[i, 1], bus.heading[i], bus.L, bus.W, bus.H)
        jc = int(round((t - car.t[0]) * rate))
        agents = [(0, parts_b, depth(bus.xy[i]))]
        if 0 <= jc < len(car.t):
            agents.append((1, agent_parts("car", car.xy[jc, 0], car.xy[jc, 1], car.heading[jc], car.L, car.W, car.H),
                           depth(car.xy[jc])))
        label = render_label_image(cam, agents, (tree,))
        img = np.full((720, 1280, 3), (58, 54, 52), np.uint8)
        from tachymetry.calib.preview import draw_ground_grid

        img = draw_ground_grid(img, cam, spacing=1.0, major=5.0, color=(92, 88, 86), alpha=0.9, horizon=False)
        img[label == 0] = (0.55 * np.array(CLASS_COLORS["bus"]) + 0.45 * 90).astype(np.uint8)
        img[label == 1] = (0.55 * np.array(CLASS_COLORS["car"]) + 0.45 * 90).astype(np.uint8)
        img[label == -2] = (38, 84, 46)

        def quad(xy: np.ndarray, heading: float, L: float, W: float) -> np.ndarray:
            return cam.ground_to_image(box_corners(np.asarray(xy, float), float(heading), L, W, 0.0)[:4, :2])

        cv2.polylines(img, [np.round(quad(bus.xy[i], bus.heading[i], bus.L, bus.W) * 4).astype(np.int32)], True,
                      (240, 240, 240), 3, cv2.LINE_AA, shift=2)
        notes = []
        for key, col, name in (("boxes", AMBER, "boxes only"), ("masks", CYAN, "masks")):
            tr, j, k, err, hd = res[key]
            m = np.flatnonzero(k)
            if not len(m):
                continue
            s_i = int(np.argmin(np.abs(tr.t[m] - t)))
            if abs(tr.t[m][s_i] - t) > 1.5 / rate:
                notes.append(f"{name}: no sample")
                continue
            q = quad(tr.xy[m][s_i], tr.heading[m][s_i], tr.length, tr.width)
            cv2.polylines(img, [np.round(q * 4).astype(np.int32)], True, col, 3, cv2.LINE_AA, shift=2)
            notes.append(f"{name}: {err[s_i]:.2f} m, {hd[s_i]:.1f} deg")
        # crop to the stretch of road around the tree (the full frame shows mostly empty ground)
        cu, cv_ = int(round(p[0])), int(round(p[1]))
        x0, y0 = max(0, cu - 560), max(0, cv_ - 250)
        img = np.ascontiguousarray(img[y0:y0 + 420, x0:x0 + 1120])
        draw_text(img, f"t = {t:.1f} s, {100 * vis[i]:.0f} % of the bus visible", (22, 18), size=36, color=TEXT)
        for n_i, line in enumerate(notes):
            draw_text(img, line, (22, 66 + 40 * n_i), size=32, color=AMBER if line.startswith("boxes") else CYAN)
        panels.append(img)
    legend = "white: ground truth   amber: boxes only   cyan: instance masks"
    save(titled(hstack(panels, 10), "Bus passing behind a tree crown (synthetic masks)", legend), out / "occlusion_bus.jpg",
         1920)
    stats["occlusion_bus"] = {"paths": entry, "moments_s": [float(bus.t[i]) for i in moments],
                              "visible_share": [float(vis[i]) for i in moments]}


# ----------------------------------------------------------------------------- lens rectification
def lens_cases(size: tuple[int, int]) -> dict[str, Any]:
    from tachymetry.geometry.lens import BrownConradyLens, DivisionLens, EquidistantLens, FisheyeLens

    return {
        "division": (DivisionLens(size, k=-0.27), "division, k = -0.27"),
        "equidistant": (EquidistantLens(size, s=900.0), "equidistant (tan), s = 900 px"),
        "brown": (BrownConradyLens(size, k1=-0.25, k2=0.06), "Brown-Conrady, k1 = -0.25, k2 = 0.06"),
        "fisheye": (FisheyeLens(size, k1=0.05, k2=-0.01, focal=914.0), "Kannala-Brandt, k1 = 0.05, k2 = -0.01"),
    }


def describe_lens(lens: Any) -> str:
    if lens.is_identity:
        return "identity"
    if lens.kind == "division":
        return f"division, k = {lens.k:.3f}"
    if lens.kind == "equidistant":
        return f"tan, s = {lens.s:.0f} px"
    if lens.kind == "brown":
        return f"Brown-Conrady, k1 = {lens.k1:.3f}, k2 = {lens.k2:.3f}"
    if lens.kind == "fisheye":
        return f"Kannala-Brandt, k1 = {lens.k1:.3f}, k2 = {lens.k2:.3f}"
    return lens.kind


def rectification_residual(est: Any, true: Any, size: tuple[int, int], step: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """|est - true| rectified positions over a raw-pixel grid after the best isotropic scale about the centre."""
    w, h = size
    uu, vv = np.meshgrid(np.arange(step / 2, w, step), np.arange(step / 2, h, step))
    uv = np.column_stack([uu.ravel(), vv.ravel()])
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    rt = true.rectify(uv) - c
    re = est.rectify(uv) - c
    ok = np.all(np.isfinite(rt), axis=1) & np.all(np.isfinite(re), axis=1)
    s = float(np.sum(re[ok] * rt[ok]) / max(np.sum(re[ok] ** 2), 1e-12))
    r = np.full(len(uv), np.nan)
    r[ok] = np.linalg.norm(s * re[ok] - rt[ok], axis=1)
    return r.reshape(uu.shape), uv


def residual_map(res: np.ndarray, size: tuple[int, int], vmax: float) -> np.ndarray:
    from tachymetry.render.text import draw_text

    w, h = size
    v = np.nan_to_num(res, nan=vmax)
    img = cv2.resize(v.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    norm = np.clip(img / vmax, 0, 1)
    lut = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    levels = [lv for lv in (0.5, 1, 2, 5, 10, 20) if lv < vmax]
    for lv in levels:
        m = (img >= lv).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(lut, cs, -1, (236, 235, 232), 1, cv2.LINE_AA)
    draw_text(lut, f"contours at {', '.join(f'{lv:g}' for lv in levels)} px", (16, h - 34), size=20, color=TEXT)
    return lut


def lens_figures(out: Path, stats: dict) -> None:
    from tachymetry.calib import calibrate_manual
    from tachymetry.calib.correspondences import Correspondences
    from tachymetry.calib.lines import detect_segments, estimate_lens_plumbline, estimate_lens_vp_consistency
    from tachymetry.geometry.lens import IdentityLens
    from tachymetry.render.text import draw_text
    from tachymetry.synthetic import RoadLayout, default_camera
    from tachymetry.synthetic.scene import make_ground_texture, render_view, scene_objects

    size = (1280, 720)
    layout = RoadLayout()
    tex = make_ground_texture(layout, seed=0)
    objs = scene_objects(layout, seed=0)
    table = ["| True lens | Estimator | Estimate | RMS residual (px) | Max residual (px) |", "|---|---|---|---|---|"]
    stats["lens"] = {}
    for key, (true_lens, label) in lens_cases(size).items():
        t0 = time.time()
        cam = default_camera(size, lens=true_lens)
        raw = render_view(cam, tex, objs, seed=0)
        ests: dict[str, Any] = {}
        ests["Plumb line, division"], _ = estimate_lens_plumbline(raw, model="division")
        ests["Plumb line, tan"], _ = estimate_lens_plumbline(raw, model="equidistant")
        segs = detect_segments(raw, None)
        best = None
        for m in ("equidistant", "division"):
            e, info = estimate_lens_vp_consistency(segs, size, model=m)
            if best is None or info["support"] > best[1]["support"]:
                best = (e, info)
        ests["Vanishing-point consistency (tan or division)"] = best[0]
        uv = _eval_grid(cam)
        g = cam.image_to_ground(uv)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(uv), 16, replace=False)
        pts = cam.ground_to_image(g[idx]) + rng.normal(0.0, 0.5, (16, 2))
        res_m = calibrate_manual(Correspondences(pts, g[idx], image_size=size), size, lens="auto")
        ests["Manual, 16 points, lens auto (cross-validated)"] = res_m.camera.lens
        ident, _ = rectification_residual(IdentityLens(size), true_lens, size)
        rows = {"No correction (identity)": (IdentityLens(size), ident)}
        for name, e in ests.items():
            rows[name] = (e, rectification_residual(e, true_lens, size)[0])
        entry = {}
        first = True
        for name, (e, r) in rows.items():
            rms, mx = float(np.sqrt(np.nanmean(r**2))), float(np.nanmax(r))
            table.append(f"| {label if first else ''} | {name} | {describe_lens(e)} | {rms:.2f} | {mx:.2f} |")
            first = False
            entry[name] = {"estimate": describe_lens(e), "rms_px": rms, "max_px": mx, "params": e.to_dict()}
        stats["lens"][key] = {"true": label, "rows": entry, "seconds": round(time.time() - t0, 1)}
        auto_names = [n for n in rows if n not in ("No correction (identity)",) and not n.startswith("Manual")]
        best_name = min(auto_names, key=lambda n: entry[n]["rms_px"])
        e_best, r_best = rows[best_name]
        rect = e_best.rectify_image(raw) if not e_best.is_identity else raw.copy()
        vmax = max(1.0, float(np.nanpercentile(r_best, 99.5)))
        heat = residual_map(r_best, size, vmax)
        draw_text(
            heat,
            f"RMS {entry[best_name]['rms_px']:.2f} px, max {entry[best_name]['max_px']:.2f} px "
            f"(no correction: RMS {entry['No correction (identity)']['rms_px']:.1f} px)",
            (16, 14),
            size=22,
            color=TEXT,
        )
        fig = hstack(
            [
                titled(raw, "Raw image", label),
                titled(rect, "Rectified", f"{best_name}: {describe_lens(e_best)}"),
                titled(heat, "Rectification residual", "after the best isotropic scale"),
            ],
            10,
        )
        save(fig, out / f"lens_{key}.jpg", 1920)
        print(f"  lens {key}: {time.time() - t0:.0f} s")
    (out / "lens_rectification.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


# ----------------------------------------------------------------------------- social card
def social_card(out: Path) -> None:
    from tachymetry.render.text import draw_text

    src = cv2.imread(str(out / "hero_overlay.jpg"))
    if src is None:
        return
    h, w = src.shape[:2]
    W, H = 1280, 640
    s = max(W / w, H / h)
    img = cv2.resize(src, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    y0 = (img.shape[0] - H) // 2
    x0 = (img.shape[1] - W) // 2
    img = img[y0 : y0 + H, x0 : x0 + W].astype(np.float32)
    ramp = np.clip(1.0 - np.linspace(0, 1, W) / 0.62, 0, 1)[None, :, None]
    img = img * (1 - 0.78 * ramp) + np.array(INK, np.float32) * 0.78 * ramp
    img = img.astype(np.uint8)
    draw_text(img, "Tachymetry", (64, 400), size=104, color=TEXT, weight="semibold")
    draw_text(img, "Metric trajectories, speeds and counts", (68, 528), size=34, color=TEXT)
    draw_text(img, "from traffic-camera video", (68, 572), size=34, color=MUTED)
    save(img, out / "social_preview.jpg", 1280)


# ----------------------------------------------------------------------------- scene gallery
GALLERY = ("intersection-day", "intersection-day-2", "intersection-day-3", "intersection-turn",
           "intersection-night", "motorway-overpass", "dhaka-roundabout")


_RUNS: dict[str, Any] = {}


def run_named_sample(name: str, work: Path, device: str) -> Any:
    """Run the pipeline on a registered sample with its known camera and overrides, if it has them (once per process)."""
    if name not in _RUNS:
        _RUNS[name] = _run_named_sample(name, work, device)
    return _RUNS[name]


def _run_named_sample(name: str, work: Path, device: str) -> Any:
    from tachymetry import data
    from tachymetry.config import PipelineConfig
    from tachymetry.pipeline import Pipeline

    cfg = PipelineConfig().override(data.sample_overrides(name))
    camera = data.sample_camera(name)
    path = data.sample_video(name)
    print(f"Pipeline on {name} ({'known camera' if camera is not None else 'automatic calibration'}) -> {work / name}")
    return Pipeline(cfg, camera=camera if camera is not None else "auto", device=device).run(path, work / name)


def gallery_figures(work: Path, out: Path, stats: dict, device: str) -> None:
    """One camera-view still per sample: the moment with the most moving road users, rendered as in overlay.mp4."""
    from tachymetry import data
    from tachymetry.config import RenderConfig
    from tachymetry.render.video import Renderer

    stats["gallery"] = {}
    for name in GALLERY:
        run = run_named_sample(name, work, device)
        t = pick_times(run, 1)[0]
        img, t = read_frame(run, t)
        r = Renderer(run, RenderConfig(), layout="camera")
        still = r.frame(img, t)[r.header_h :]  # drop the title bar; the caption on the page carries it
        save(still, out / "gallery" / f"{name}.jpg", 1280, quality=80)
        s = run.summary()
        stats["gallery"][name] = {
            "time_s": round(t, 2),
            "calibration": run.camera.meta.get("calibration_method"),
            "camera": run.camera.describe(),
            "size": list(run.camera.image_size),
            "tracks": s["tracks"],
            "by_class": s["by_class"],
            "vehicle_speed_kmh": s["vehicle_speed_kmh"],
            "attribution": data.attribution(name),
        }


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "media"), help="output directory (default docs/media)")
    ap.add_argument(
        "--work",
        default=str(Path.home() / ".cache" / "tachymetry" / "docs_media"),
        help="cache for the pipeline run on the sample clip",
    )
    ap.add_argument("--device", default="auto", help="detector and depth device, e.g. cuda:0 or cpu")
    ap.add_argument("--only", default=",".join(GROUPS), help=f"comma-separated subset of {','.join(GROUPS)}")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    stats_path = out / "figures.json"
    stats: dict[str, Any] = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    t0 = time.time()
    if only & {"clip", "calib"}:
        run = run_sample(Path(args.work), args.device)
        if "clip" in only:
            clip_figures(run, out, stats)
        if "calib" in only:
            calib_figures(run, out, stats, args.device, Path(args.work))
    if "synthetic" in only:
        synthetic_figures(out, stats)
    if "lens" in only:
        lens_figures(out, stats)
    if "social" in only:
        social_card(out)
    if "occlusion" in only:
        occlusion_figures(out, stats)
    if "gallery" in only:
        gallery_figures(Path(args.work), out, stats, args.device)
    stats_path.write_text(json.dumps(stats, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and not p.name.startswith("trailer"))
    print(f"Done in {time.time() - t0:.0f} s; {out} holds {total / 1e6:.1f} MB (without the trailer).")


if __name__ == "__main__":
    main()

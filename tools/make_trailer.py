"""Render the Tachymetry trailer, its poster frame and the README GIF.

Outputs, written to ``--out`` (``docs/media`` by default):

* ``trailer.mp4``: 1920x1080, 30 fps, H.264 (``yuv420p``, ``+faststart``), silent, 37 s;
* ``trailer_poster.jpg``: a 1920x1080 still used as the poster of the video;
* ``readme_hero.gif``: a 7 s looping camera and bird's-eye-view render of the
  daytime intersection, 1000 px wide, for the top of the README.

The footage is the set of public sample clips of :mod:`tachymetry.data`
(svbrd-llm-roadside-video-av, CC BY 4.0), downloaded into the Hugging Face cache
on first use. Each clip goes through :class:`tachymetry.pipeline.Pipeline`
(detection, automatic calibration, tracking). Detections are cached in the run
directories under ``--runs``, so only the first invocation needs a GPU; later
invocations take a few minutes on the CPU. Frames are drawn with the library's
own renderers (camera overlay, bird's-eye-view canvas, calibration grid, the
bundled Geist typefaces) and composed here into eight scenes:

1. title; 2. raw footage; 3. the metric ground grid of the automatic
calibration, revealed from near to far; 4. tracking with 3D boxes and speeds;
5. split view with the bird's-eye view and accumulating trajectories; 6. raw
tracker output (ByteTrack identities, frame-difference speeds) wiped away to the
filtered tracks, with rejected tracks (ghost, duplicate, rider) and parked vehicles
tagged; 7. a grid of
clips processed in parallel, labelled with the worker plan of
:func:`tachymetry.batch.devices.plan_workers`; 8. end card.

Usage::

    python tools/make_trailer.py
    python tools/make_trailer.py --device cuda:0 --runs runs/trailer --out docs/media
    python tools/make_trailer.py --only gif
    python tools/make_trailer.py --only stills --stills 1.5,5,9,14,19,24,29,35
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tachymetry.batch.devices import Device, plan_workers
from tachymetry.calib.preview import draw_ground_grid
from tachymetry.data import sample_video
from tachymetry.io.video import VideoSource
from tachymetry.io.writer import VideoWriter, ffmpeg_executable
from tachymetry.pipeline import Pipeline
from tachymetry.render.bev import BEVCanvas
from tachymetry.render.overlay import draw_camera_overlay
from tachymetry.render.state import StateIndex
from tachymetry.render.style import (
    AMBER,
    CLASS_COLORS,
    CYAN,
    FAINT,
    INK,
    MUTED,
    PARKED,
    PLURALS,
    TEXT,
    color_for,
    polyline_aa,
)
from tachymetry.render.text import draw_chip, font
from tachymetry.synthetic.traffic import naive_speeds
from tachymetry.track.builder import associate
from tachymetry.track.measure import measure_ground

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 30
DURATION = 37.0

# clips: run directory name, sample name
CLIPS = [("day1", "intersection-day"), ("day2", "intersection-day-2"), ("day3", "intersection-day-3"),
         ("turn", "intersection-turn"), ("night", "intersection-night")]
TAGLINE = "Metric trajectories and speeds from traffic video"
INSTALL = "pip install git+https://github.com/mkturkcan/tachymetry"
REPO = "github.com/mkturkcan/tachymetry"
AUTHOR = "Mehmet Kerem Turkcan"
AFFILIATION = "Associate Research Scientist, Department of Civil Engineering, Columbia University"
CREDIT = "Sample footage: svbrd-llm-roadside-video-av, CC BY 4.0"

# layout (pixels of the 1920x1080 frame); rectangles are (x, y, w, h)
STAGE = (160.0, 36.0, 1600.0, 900.0)
SPLIT_CAM = (160.0, 186.0, 1066.0, 600.0)
SPLIT_BEV = (1242.0, 186.0, 518.0, 600.0)
FULL_CROP = (0.0, 0.0, 1920.0, 1080.0)
ZOOM_CROP = (320.0, 0.0, 1600.0, 900.0)  # one source pixel per stage pixel
TILE = (522.0, 294.0)
TILE_RENDER = (960, 540)  # tiles draw their overlay at this size, so lines stay visible when small
CAPTION_X, CAPTION_Y, SUB_Y = 160.0, 994.0, 1030.0

# timeline (seconds)
T_DAY = 2.75  # the day clip starts (clip time 0) as the stage fades in
T_GRID = (6.95, 9.55)  # grid sweep, near to far
T_TRACK = (10.9, 11.7)  # tracking overlay fades in
T_SPLIT = (15.8, 17.0)
T_UNSPLIT = (21.2, 22.2)
T_RAW = (22.2, 23.0)  # zoom to 1:1 pixels and cross to the raw tracker output
T_WIPE = (23.2, 25.3)
T_TILES = (26.5, 27.6)
T_END = (31.8, 32.6)


# ------------------------------------------------------------------ helpers
def ease(x: float) -> float:
    """Cubic ease-in-out on [0, 1]."""
    x = min(max(float(x), 0.0), 1.0)
    return 4.0 * x ** 3 if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 3 / 2.0


def ramp(t: float, a: float, b: float) -> float:
    return ease((t - a) / (b - a))


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def lerp_rect(r0: tuple, r1: tuple, u: float) -> tuple:
    return tuple(lerp(p, q, u) for p, q in zip(r0, r1))


def scale_rect(r: tuple, s: float) -> tuple:
    x, y, w, h = r
    return (x + w * (1 - s) / 2, y + h * (1 - s) / 2, w * s, h * s)


def blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """``a`` towards ``b`` by ``w``."""
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b
    return cv2.addWeighted(b, float(w), a, float(1 - w), 0)


# --------------------------------------------------------------------- type
@lru_cache(maxsize=4096)
def _mask(text: str, size: int, weight: str) -> tuple[np.ndarray, int, int]:
    """Coverage mask of ``text`` (Geist) and its top-left offset from the baseline origin."""
    f = font(size, weight)
    l, t, r, b = (int(v) for v in f.getbbox(text, anchor="ls"))
    pad = 2
    img = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls")
    return np.asarray(img, np.float32) / 255.0, l - pad, t - pad


def text_width(text: str, size: int, weight: str = "regular") -> float:
    return float(font(size, weight).getlength(text))


def blend_mask(canvas: np.ndarray, mask: np.ndarray, x: float, y: float, color: tuple, alpha: float = 1.0) -> None:
    """Composite a coverage mask with its top-left corner at the sub-pixel position ``(x, y)``."""
    if alpha <= 0.002:
        return
    xi, yi = math.floor(x), math.floor(y)
    fx, fy = x - xi, y - yi
    if fx > 0.01 or fy > 0.01:
        mask = cv2.warpAffine(mask, np.float32([[1, 0, fx], [0, 1, fy]]), (mask.shape[1] + 1, mask.shape[0] + 1),
                              flags=cv2.INTER_LINEAR)
    h, w = mask.shape[:2]
    x0, y0 = max(xi, 0), max(yi, 0)
    x1, y1 = min(xi + w, canvas.shape[1]), min(yi + h, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - yi:y1 - yi, x0 - xi:x1 - xi, None] * float(alpha)
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(roi + (np.asarray(color, np.float32) - roi) * m + 0.5, 0, 255).astype(np.uint8)


def put_text(canvas: np.ndarray, text: str, x: float, y: float, size: int, weight: str = "regular",
             color: tuple = TEXT, alpha: float = 1.0, align: str = "left", blur: float = 0.0) -> None:
    """Draw ``text`` with its baseline at ``y``; ``x`` is its left edge, centre or right edge."""
    if alpha <= 0.002 or not text:
        return
    m, ox, oy = _mask(text, size, weight)
    if align != "left":  # whole-pixel alignment keeps static text crisp
        tw = text_width(text, size, weight)
        x -= round(tw / 2) if align == "center" else round(tw)
    if blur > 0.05:
        p = int(math.ceil(3 * blur)) + 1
        m = cv2.GaussianBlur(np.pad(m, p), (0, 0), blur)
        ox, oy = ox - p, oy - p
    blend_mask(canvas, m, x + ox, y + oy, color, alpha)


def fill_rect(canvas: np.ndarray, x: float, y: float, w: float, h: float, color: tuple, alpha: float = 1.0) -> None:
    """Anti-aliased rectangle with sub-pixel position and size."""
    if w <= 0 or h <= 0 or alpha <= 0.002:
        return
    xi0, xi1 = math.floor(x), math.ceil(x + w)
    yi0, yi1 = math.floor(y), math.ceil(y + h)
    cx = np.arange(xi0, xi1, dtype=np.float32)
    cy = np.arange(yi0, yi1, dtype=np.float32)
    cols = np.clip(np.minimum(cx + 1, x + w) - np.maximum(cx, x), 0, 1)
    rows = np.clip(np.minimum(cy + 1, y + h) - np.maximum(cy, y), 0, 1)
    blend_mask(canvas, rows[:, None] * cols[None, :], xi0, yi0, color, alpha)


@lru_cache(maxsize=1024)
def rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    """Anti-aliased rounded-rectangle coverage (4x supersampled)."""
    s = 4
    big = np.zeros((h * s, w * s), np.uint8)
    rr = max(1, r * s)
    cv2.rectangle(big, (rr, 0), (w * s - 1 - rr, h * s - 1), 255, -1)
    cv2.rectangle(big, (0, rr), (w * s - 1, h * s - 1 - rr), 255, -1)
    for cx, cy in ((rr, rr), (w * s - 1 - rr, rr), (rr, h * s - 1 - rr), (w * s - 1 - rr, h * s - 1 - rr)):
        cv2.circle(big, (cx, cy), rr, 255, -1, cv2.LINE_AA)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


@lru_cache(maxsize=1024)
def ring_mask(w: int, h: int, r: int) -> np.ndarray:
    """One-pixel outline of :func:`rounded_mask`."""
    inner = np.zeros((h, w), np.float32)
    inner[1:-1, 1:-1] = rounded_mask(w - 2, h - 2, max(r - 1, 1))
    return np.clip(rounded_mask(w, h, r) - inner, 0, 1)


def place(canvas: np.ndarray, img: np.ndarray, rect: tuple, alpha: float = 1.0, radius: int = 12,
          border: float = 0.07) -> None:
    """Composite ``img`` into ``rect`` as a rounded panel with a hairline border."""
    x, y, w, h = rect
    X, Y, Wd, Hd = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    if Wd < 8 or Hd < 8 or alpha <= 0.002:
        return
    if img.shape[1] != Wd or img.shape[0] != Hd:
        img = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA if Wd < img.shape[1] else cv2.INTER_CUBIC)
    x0, y0 = max(X, 0), max(Y, 0)
    x1, y1 = min(X + Wd, canvas.shape[1]), min(Y + Hd, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sl = (slice(y0 - Y, y1 - Y), slice(x0 - X, x1 - X))
    m = rounded_mask(Wd, Hd, radius)[sl][..., None] * float(alpha)
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    out = roi + (img[sl].astype(np.float32) - roi) * m
    if border > 0:
        rg = ring_mask(Wd, Hd, radius)[sl][..., None] * (border * float(alpha))
        out += (255.0 - out) * rg
    canvas[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


def panel(canvas: np.ndarray, x: float, y: float, w: float, h: float, fill: tuple, alpha: float, radius: int = 8,
          edge: tuple | None = None, edge_alpha: float = 0.0) -> None:
    """Flat rounded rectangle (label backgrounds, the install line)."""
    Wd, Hd = int(round(w)), int(round(h))
    if Wd < 4 or Hd < 4:
        return
    blend_mask(canvas, rounded_mask(Wd, Hd, radius), round(x), round(y), fill, alpha)
    if edge is not None and edge_alpha > 0:
        blend_mask(canvas, ring_mask(Wd, Hd, radius), round(x), round(y), edge, edge_alpha * alpha)


def view(full: np.ndarray, crop: tuple, size: tuple) -> np.ndarray:
    """Crop ``crop`` (source pixels) of ``full`` and resize it to ``size``."""
    x, y, w, h = crop
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + w)), full.shape[1]), min(int(round(y + h)), full.shape[0])
    sub = full[y0:y1, x0:x1]
    Wd, Hd = int(round(size[0])), int(round(size[1]))
    if sub.shape[1] == Wd and sub.shape[0] == Hd:
        return sub.copy()
    return cv2.resize(sub, (Wd, Hd), interpolation=cv2.INTER_AREA if Wd < sub.shape[1] else cv2.INTER_CUBIC)


class Wordmark:
    """A word set glyph by glyph (kerned layout) so that each letter can resolve on its own.

    The glyphs are accumulated into one coverage mask (no double blending where they meet) and, once they
    have settled, the mask hands over to the whole-word rendering so that the resting title is crisp.
    """

    def __init__(self, text: str, size: int, weight: str = "semibold"):
        f = font(size, weight)
        self.word, self.wox, self.woy = _mask(text, size, weight)
        self.width = f.getlength(text)
        self.glyphs = [(f.getlength(text[:i + 1]) - f.getlength(ch), *_mask(ch, size, weight))
                       for i, ch in enumerate(text)]

    def draw(self, canvas: np.ndarray, cx: float, baseline: float, t: float, t0: float, stagger: float = 0.055,
             dur: float = 0.9, alpha: float = 1.0, rise: float = 16.0, blur: float = 7.0, spread: float = 9.0,
             color: tuple = TEXT, settle: float = 0.35) -> None:
        if alpha <= 0.002 or t < t0:
            return
        n = len(self.glyphs)
        x0 = cx - round(self.width / 2)
        done = ramp(t, t0 + (n - 1) * stagger + dur, t0 + (n - 1) * stagger + dur + settle)
        pad = int(3 * blur + rise + spread * n / 2) + 8
        wh, ww = self.word.shape
        X0, Y0 = math.floor(x0 + self.wox) - pad, math.floor(baseline + self.woy) - pad
        size = (ww + 2 * pad, wh + 2 * pad)
        cov = np.zeros((size[1], size[0]), np.float32)
        if done < 1:
            keep = np.ones_like(cov)
            for i, (x, m, ox, oy) in enumerate(self.glyphs):
                e = ramp(t, t0 + i * stagger, t0 + i * stagger + dur)
                if e <= 0.0:
                    continue
                b, px, mm = blur * (1 - e), 0, m
                if b > 0.05:
                    px = int(math.ceil(3 * b)) + 1
                    mm = cv2.GaussianBlur(np.pad(m, px), (0, 0), b)
                gx = x0 + x + ox - px + (i - (n - 1) / 2) * spread * (1 - e) - X0
                gy = baseline + oy - px + rise * (1 - e) - Y0
                placed = cv2.warpAffine(mm, np.float32([[1, 0, gx], [0, 1, gy]]), size, flags=cv2.INTER_LINEAR)
                keep *= 1 - placed * e
            cov = 1 - keep
        if done > 0:
            word = np.zeros_like(cov)
            fx = x0 + self.wox - (X0 + pad)
            fy = baseline + self.woy - (Y0 + pad)
            word[pad:pad + wh, pad:pad + ww] = self.word
            if fx > 0.01 or fy > 0.01:
                word = cv2.warpAffine(word, np.float32([[1, 0, fx], [0, 1, fy]]), size, flags=cv2.INTER_LINEAR)
            cov = cov * (1 - done) + word * done
        blend_mask(canvas, cov, X0, Y0, color, alpha)


# -------------------------------------------------------------------- video
class Clip:
    """Sequential frame access with seeking only when needed."""

    def __init__(self, path: Path):
        self.src = VideoSource(path)
        self.fps = self.src.fps
        self.n = self.src.frame_count
        self._it = None
        self._idx = -10
        self._img = None

    def get(self, i: int) -> np.ndarray:
        i = int(min(max(i, 0), self.n - 1))
        if i == self._idx and self._img is not None:
            return self._img
        if self._it is None or i < self._idx or i > self._idx + 45:
            self._it = self.src.frames(start=i)
            self._idx = i - 1
        while self._idx < i:
            try:
                self._idx, _, self._img = next(self._it)
            except StopIteration:
                self._it = None
                break
        return self._img


def scaled_states(states: list, k: float) -> list:
    """Track states with image boxes rescaled for an overlay drawn at a smaller size."""
    return [replace(s, bbox=s.bbox * k) for s in states]


def bbox_at(tr, tau: float) -> np.ndarray | None:
    if tau < tr.t[0] or tau > tr.t[-1]:
        return None
    return np.array([np.interp(tau, tr.t, tr.bbox[:, j]) for j in range(4)])


def busiest_start(run, length: float, lo: float = 0.0) -> float:
    """Clip time at which a window of ``length`` seconds holds the most moving road users."""
    dur = float(run.tracks.meta.get("duration_s") or 0.0)
    moving = [tr for tr in run.tracks.ok if tr.state == "moving"]
    best, best_s = lo, -1.0
    for s in np.arange(lo, max(lo, dur - length - 0.2) + 1e-6, 0.5):
        ts = np.linspace(s, s + length, 9)
        score = sum(sum(1 for tr in moving if tr.t[0] <= t <= tr.t[-1]) for t in ts)
        if score > best_s:
            best, best_s = float(s), score
    return best


# ------------------------------------------------------------------- layers
class GridSweep:
    """The metric ground grid of the calibrated camera, revealed from near to far."""

    def __init__(self, cam, spacing: float = 5.0, major: float = 25.0):
        small = cam.resized((1280, 720))  # 1 px minor and 2 px major lines, upscaled: a light grid
        lines = draw_ground_grid(np.zeros((720, 1280, 3), np.uint8), small, spacing=spacing, major=major,
                                 color=(255, 255, 255), alpha=1.0, horizon=False, max_gsd=0.45)
        self.mask = cv2.resize(lines[..., 0], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        step = 4
        uu, vv = np.meshgrid(np.arange(0, W, step) + step / 2, np.arange(0, H, step) + step / 2)
        g = cam.image_to_ground(np.column_stack([uu.ravel(), vv.ravel()]))
        d = cam.distance_to(g).reshape(uu.shape)
        d = np.where(np.isfinite(d) & (d > 0.5), d, 1e5)
        self.logd = cv2.resize(np.log(d).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        on = self.mask > 0.2
        self.lo, self.hi = (float(v) for v in np.percentile(self.logd[on], [0.5, 99.5]))
        near = cv2.dilate((self.mask > 0.05).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32)
        self.support = cv2.GaussianBlur(near, (0, 0), 14)
        rows = np.flatnonzero(self.mask.max(axis=1) > 0)
        self.y0, self.y1 = int(rows.min()), int(rows.max()) + 1

    def apply(self, img: np.ndarray, s: float, alpha: float, glow: float) -> np.ndarray:
        """``s`` places the front (0 nearest, 1 farthest); ``alpha`` is the grid opacity."""
        if alpha <= 0.002:
            return img
        y0, y1 = self.y0, self.y1
        front = lerp(self.lo - 0.02, self.hi + 0.06, s)
        ld = self.logd[y0:y1]
        reveal = np.clip((front - ld) / 0.06, 0, 1)
        far = np.clip((self.hi - ld) / 0.35, 0, 1)  # the far end of the grid fades out instead of stopping
        band = np.clip(1 - np.abs(ld - front) / 0.06, 0, 1) ** 2 * glow * far
        m = (self.mask[y0:y1] * reveal * far * alpha)[..., None]
        amber = np.asarray(AMBER, np.float32)
        hot = np.asarray((214, 236, 255), np.float32)
        col = amber + (hot - amber) * (0.75 * band)[..., None]
        roi = img[y0:y1].astype(np.float32)
        out = roi + (col - roi) * m + amber * (0.16 * band * self.support[y0:y1])[..., None]
        img[y0:y1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        return img


class RawTracker:
    """The association stage alone (ByteTrack identities on the raw detections) with frame-difference speeds.

    This is the baseline of the synthetic benchmark (:func:`tachymetry.synthetic.traffic.naive_speeds`):
    what a detector plus tracker produces before hygiene, stitching, parked-vehicle handling and verdicts.
    """

    def __init__(self, run, cfg):
        dets, cam = run.detections, run.camera
        self.dets = dets
        self.ids = associate(dets, measure_ground(dets, cam, cfg.hygiene), cfg)
        self.speed = np.full(len(self.ids), np.nan)
        for tid, (_, _, sp) in naive_speeds(dets, cam, self.ids).items():
            r = np.flatnonzero(self.ids == tid)
            self.speed[r[np.argsort(dets.time[r])]] = sp * 3.6
        self.frames = dets.processed_frames
        self.rows: dict[int, np.ndarray] = {}
        assoc = np.flatnonzero(self.ids >= 0)
        order = assoc[np.argsort(dets.frame[assoc], kind="stable")]
        fr = dets.frame[order]
        for f in np.unique(fr):
            r = order[np.searchsorted(fr, f, "left"):np.searchsorted(fr, f, "right")]
            self.rows[int(f)] = r[np.argsort(dets.xyxy[r, 3])]  # far (high in the image) first
        self.n_ids = int(len(np.unique(self.ids[self.ids >= 0])))

    def draw(self, img: np.ndarray, frame: int) -> np.ndarray:
        k = int(np.searchsorted(self.frames, frame, side="right")) - 1
        if k < 0:
            return img
        rows = self.rows.get(int(self.frames[k]), np.zeros(0, np.int64))
        names = self.dets.class_names
        chips = []
        for r in rows:
            x1, y1, x2, y2 = (float(v) for v in self.dets.xyxy[r])
            col = CLASS_COLORS.get(names[self.dets.cls[r]], (200, 200, 200))
            polyline_aa(img, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), col, 2, closed=True)
            if np.isfinite(self.speed[r]) and y2 - y1 >= 16:
                chips.append((x1, y1, col, self.speed[r]))
        for x1, y1, col, v in chips:
            draw_chip(img, f"{v:.0f} km/h", (int(x1), int(y1) - 4), size=13, fg=(20, 16, 14), bg=col)
        return img


class Day:
    """The daytime intersection: footage, overlay, grid, raw tracker and bird's-eye view."""

    def __init__(self, run, cfg):
        self.run = run
        self.cam = run.camera
        self.clip = Clip(run.source)
        self.index = StateIndex(list(run.tracks.tracks))
        self.grid = GridSweep(self.cam)
        self.raw = RawTracker(run, cfg)
        self.moving = [tr for tr in run.tracks.ok if tr.state == "moving"]
        self.fit = np.vstack([tr.xy for tr in self.moving])
        self.bev = BEVCanvas(self.cam, run.background, (int(SPLIT_BEV[2]), int(SPLIT_BEV[3])), np.zeros(2), 50.0,
                             fit_points=self.fit)
        self.bev_base = self.bev.base.copy()
        self.rejected = [tr for tr in run.tracks.tracks if not tr.is_ok]
        parked = [tr for tr in run.tracks.ok if tr.state == "parked"]
        # one "parked" tag per curb: the largest parked vehicle left and right of the middle of the row
        self.parked_tags = []
        if parked:
            xs = np.array([np.median((tr.bbox[:, 0] + tr.bbox[:, 2]) / 2) for tr in parked])
            mid = (xs.min() + xs.max()) / 2
            for side in (xs <= mid, xs > mid):
                group = [tr for tr, s in zip(parked, side) if s]
                if group:
                    area = [np.median((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) for b in (tr.bbox for tr in group)]
                    self.parked_tags.append(group[int(np.argmax(area))])

    def frame_index(self, t: float) -> int:
        return int(round((t - T_DAY) * self.clip.fps))

    def overlay(self, img: np.ndarray, tau: float) -> np.ndarray:
        return draw_camera_overlay(img.copy(), self.cam, self.index.at(tau))

    def bev_frame(self, tau: float, since: float) -> np.ndarray:
        """Bird's-eye view at ``tau`` over the trajectories accumulated since ``since``."""
        base = self.bev_base
        hist = base.copy()
        for tr in self.moving:
            if tr.t[0] > tau or tr.t[-1] < since:
                continue
            a = int(np.searchsorted(tr.t, since, "left"))
            b = int(np.searchsorted(tr.t, tau, "right"))
            pts = tr.xy[a:b]
            if 1 <= b < len(tr.t):
                w = (tau - tr.t[b - 1]) / max(tr.t[b] - tr.t[b - 1], 1e-9)
                pts = np.vstack([pts, tr.xy[b - 1] * (1 - w) + tr.xy[b] * w])
            if len(pts) < 2:
                continue
            p = np.round(self.bev.to_canvas(pts) * 16).astype(np.int32)
            cv2.polylines(hist, [p], False, color_for(tr.cls), 2, cv2.LINE_AA, shift=4)
        self.bev.base = cv2.addWeighted(hist, 0.6, base, 0.4, 0)
        try:
            return self.bev.render(self.index.at(tau))
        finally:
            self.bev.base = base


class Tile:
    """One clip of the batch grid, drawn with the camera overlay at :data:`TILE_RENDER`."""

    def __init__(self, run, name: str, worker: str, offset: float, clip: Clip | None = None):
        self.clip = clip or Clip(run.source)
        self.cam = run.camera.resized(TILE_RENDER)
        self.k = TILE_RENDER[0] / run.camera.image_size[0]
        self.index = StateIndex(list(run.tracks.tracks))
        self.name, self.worker = name, worker
        self.offset = offset  # clip time = trailer time + offset
        self.duration = float(run.tracks.meta.get("duration_s") or 10.0)

    def content(self, tau: float) -> np.ndarray:
        f = int(round(tau * self.clip.fps))
        img = cv2.resize(self.clip.get(f), TILE_RENDER, interpolation=cv2.INTER_AREA)
        states = scaled_states(self.index.at(f / self.clip.fps), self.k)
        return draw_camera_overlay(img, self.cam, states, max_labels=0)


def tile_rects() -> list[tuple]:
    tw, th = TILE
    gap = (STAGE[2] - 3 * tw) / 2
    y0 = STAGE[1] + (STAGE[3] - (2 * th + gap)) / 2
    top = [(STAGE[0] + k * (tw + gap), y0, tw, th) for k in range(3)]
    xb = STAGE[0] + (STAGE[2] - (2 * tw + gap)) / 2
    return top + [(xb + k * (tw + gap), y0 + th + gap, tw, th) for k in range(2)]


# ------------------------------------------------------------------ trailer
class Trailer:
    def __init__(self, runs: dict, cfg):
        self.day = Day(runs["day1"], cfg)
        cam = self.day.cam
        s = runs["day1"].tracks.summary()
        mv = sorted(s["moving_by_class"].items(), key=lambda kv: -kv[1])
        counts = " · ".join(f"{n} {PLURALS.get(c, c) if n != 1 else c}" for c, n in mv[:4])
        n_ok = s["tracks"]
        n_parked = sum(s["parked_by_class"].values())
        self.captions = [
            (3.5, 6.7, "Roadside camera, 1080p", "Signalised intersection · public sample clip · 30 fps"),
            (7.1, 11.3, "Automatic camera calibration",
             f"focal {cam.focal:.0f} px · height {cam.height:.1f} m · pitch {cam.pitch:.1f}°"),
            (11.7, 15.6, "Tracking with metric 3D boxes and speeds", counts),
            (16.7, 21.0, "Bird’s-eye view and trajectories", "tracks.csv: positions in metres, speeds and headings"),
            (22.3, 26.3, "Track filtering: ghosts and parked vehicles",
             f"{self.day.raw.n_ids} raw tracker IDs → {n_ok} tracks ({n_parked} parked)"),
            (27.7, 31.7, "Batch processing on multiple GPUs", "tachymetry run videos/ --devices auto"),
        ]
        # worker labels: the library's worker plan for these jobs on a four-GPU machine (two workers per GPU);
        # jobs beyond the number of workers queue for the first free worker
        gpus = [Device(f"cuda:{i}", "cuda", "GPU", 96.0, 96.0) for i in range(4)]
        slots = plan_workers(gpus, n_jobs=len(CLIPS))
        workers, seen = [], {}
        for d in slots:
            seen[d] = seen.get(d, 0) + 1
            workers.append(f"{d} · worker {seen[d]}")
        labels = [workers[k % len(workers)] for k in range(len(CLIPS))]
        self.tiles = []
        for k, (key, name) in enumerate(CLIPS):
            if k == 0:  # the day tile continues the main clip
                offset = self.day.frame_index(T_TILES[0]) / self.day.clip.fps - T_TILES[0]
                self.tiles.append(Tile(runs[key], name, labels[k], offset, clip=self.day.clip))
            else:
                offset = busiest_start(runs[key], 6.0) - self.tile_in(k)
                self.tiles.append(Tile(runs[key], name, labels[k], offset))
        self.rects = tile_rects()
        self.dy = 0.0
        self.title = Wordmark("Tachymetry", 136)
        self.end_title = Wordmark("Tachymetry", 112)
        # wipe: when the divider passes a stage x position
        self._wipe_t = np.linspace(T_WIPE[0], T_WIPE[1], 400)
        self._wipe_x = np.array([self.wipe_x(t) for t in self._wipe_t])

    # ------------------------------------------------------------- timing
    @staticmethod
    def tile_in(k: int) -> float:
        return T_TILES[0] + 0.25 + 0.12 * k

    def wipe_x(self, t: float) -> float:
        return lerp(-4.0, STAGE[2] + 4.0, ramp(t, *T_WIPE))

    def passed(self, x: float) -> float | None:
        """Time at which the wipe divider passes stage column ``x``."""
        i = int(np.searchsorted(self._wipe_x, x))
        return float(self._wipe_t[i]) if i < len(self._wipe_t) else None

    # ------------------------------------------------------------- frames
    def frame(self, t: float, poster: bool = False) -> np.ndarray:
        canvas = np.full((H, W, 3), INK, np.uint8)
        self.dy = 36.0 if poster else 0.0  # the poster lowers the split view to make room for the title
        if t < 3.4:
            self.draw_title(canvas, t)
        if T_DAY <= t < T_TILES[0]:
            self.draw_day(canvas, t)
        if t >= T_TILES[0] and t < T_END[1] + 0.4:
            self.draw_tiles(canvas, t)
        if poster:
            self.draw_poster_title(canvas)
        else:
            for c in self.captions:
                self.draw_caption(canvas, t, *c)
        if t >= T_END[0] + 0.3:
            self.draw_end(canvas, t)
        return canvas

    def draw_title(self, canvas: np.ndarray, t: float) -> None:
        out = ramp(t, 2.55, 3.25)
        lift = -26 * out
        self.title.draw(canvas, W / 2, 548 + lift, t, 0.35, alpha=1 - out)
        a = ramp(t, 1.25, 2.05) * (1 - out)
        put_text(canvas, TAGLINE, W / 2, 632 + 10 * (1 - ramp(t, 1.25, 2.05)) + lift, 32, "regular", MUTED, a,
                 align="center")

    def draw_caption(self, canvas: np.ndarray, t: float, t_in: float, t_out: float, main: str, sub: str) -> None:
        if t < t_in or t > t_out:
            return
        out = ramp(t, t_out - 0.5, t_out)
        rule = ramp(t, t_in, t_in + 0.5)
        a_main = ramp(t, t_in + 0.12, t_in + 0.62)
        a_sub = ramp(t, t_in + 0.24, t_in + 0.74)
        fill_rect(canvas, CAPTION_X, CAPTION_Y - 9, 28 * rule, 2, AMBER, 1 - out)
        x = CAPTION_X + 42
        put_text(canvas, main, x + 12 * (1 - a_main) - 8 * out, CAPTION_Y, 26, "medium", TEXT, a_main * (1 - out))
        put_text(canvas, sub, x + 12 * (1 - a_sub) - 8 * out, SUB_Y, 17, "mono", MUTED, a_sub * (1 - out))

    def draw_poster_title(self, canvas: np.ndarray) -> None:
        """Poster text: the name and tagline above the split view, the install line and credit below it."""
        x, base = STAGE[0], 152
        put_text(canvas, "Tachymetry", x, base, 64, "semibold", TEXT)
        put_text(canvas, TAGLINE, x + round(text_width("Tachymetry", 64, "semibold")) + 30, base, 26, "regular", MUTED)
        fill_rect(canvas, CAPTION_X, CAPTION_Y - 9, 28, 2, AMBER)
        put_text(canvas, "$", CAPTION_X + 42, CAPTION_Y, 20, "mono", AMBER)
        put_text(canvas, INSTALL, CAPTION_X + 42 + round(text_width("$ ", 20, "mono")), CAPTION_Y, 20, "mono", TEXT)
        put_text(canvas, CREDIT, CAPTION_X + 42, SUB_Y, 15, "regular", FAINT)
        put_text(canvas, REPO, STAGE[0] + STAGE[2], CAPTION_Y, 20, "medium", CYAN, align="right")

    # ------------------------------------------------------------ day clip
    def draw_day(self, canvas: np.ndarray, t: float) -> None:
        d = self.day
        f = d.frame_index(t)
        tau = f / d.clip.fps
        img = d.clip.get(f)
        stage_in = ramp(t, T_DAY, T_DAY + 0.8)
        if t < T_UNSPLIT[0]:
            content = img
            grid_a = (1 - 0.12 * ramp(t, T_GRID[1], T_GRID[1] + 0.8)) * (1 - ramp(t, 10.6, 11.6))
            if t >= T_GRID[0] and grid_a > 0.002:
                glow = 1 - ramp(t, T_GRID[1] - 0.4, T_GRID[1] + 0.4)
                content = d.grid.apply(img.copy(), ramp(t, *T_GRID), grid_a, glow)
            ov = ramp(t, *T_TRACK)
            if ov > 0:
                content = blend(content, d.overlay(img, tau), ov)
            split = ramp(t, *T_SPLIT)
            cam_rect = (SPLIT_CAM[0], SPLIT_CAM[1] + self.dy, SPLIT_CAM[2], SPLIT_CAM[3])
            rect = lerp_rect(scale_rect(STAGE, lerp(0.965, 1.0, stage_in)), cam_rect, split)
            place(canvas, view(content, FULL_CROP, rect[2:]), rect, stage_in)
            if t >= T_SPLIT[0]:
                self.draw_bev(canvas, t, tau, ramp(t, T_SPLIT[0] + 0.2, T_SPLIT[1]))
            return
        # scene 6: back to the stage, zoom to 1:1, raw tracker output wiped away to the filtered tracks
        grow = ramp(t, *T_UNSPLIT)
        if t < T_UNSPLIT[1]:
            self.draw_bev(canvas, t, tau, 1 - ramp(t, T_UNSPLIT[0], T_UNSPLIT[1] - 0.2))
        rect = lerp_rect(SPLIT_CAM, STAGE, grow)
        zoom = ramp(t, *T_RAW)
        crop = lerp_rect(FULL_CROP, ZOOM_CROP, zoom)
        clean_full = d.overlay(img, tau)
        size = (int(round(rect[2])), int(round(rect[3])))
        clean = view(clean_full, crop, size)
        raw_a = ramp(t, T_RAW[0] + 0.1, T_RAW[1])
        x_div = self.wipe_x(t)
        if raw_a > 0 and x_div < size[0]:
            raw = view(d.raw.draw(img.copy(), f), crop, size)
            raw_label = raw_a * (1 - ramp(t, T_WIPE[1] - 0.2, T_WIPE[1]))
            self.stage_label(raw, "Raw tracker output", "bottom-right", raw_label)
            raw = blend(clean, raw, raw_a)
        else:
            raw = None
        if t >= T_WIPE[0]:
            self.draw_tags(clean, t, tau, crop, size)
            self.stage_label(clean, "Tachymetry", "top-left", 1 - ramp(t, 25.9, 26.4))
        if raw is not None:
            if x_div <= 0:
                content = raw
            else:
                cols = np.clip(np.arange(size[0], dtype=np.float32) + 0.5 - x_div, 0, 1)[None, :, None]
                content = (clean.astype(np.float32) * (1 - cols) + raw.astype(np.float32) * cols + 0.5).astype(np.uint8)
                if 0 < x_div < size[0]:
                    edge = ramp(t, T_WIPE[0], T_WIPE[0] + 0.25) * (1 - ramp(t, T_WIPE[1] - 0.25, T_WIPE[1]))
                    fill_rect(content, x_div - 1, 0, 2, size[1], AMBER, edge)
        else:
            content = clean
        place(canvas, content, rect, 1.0)

    def draw_bev(self, canvas: np.ndarray, t: float, tau: float, a: float) -> None:
        if a <= 0.002:
            return
        d = self.day
        img = d.bev_frame(tau, d.frame_index(T_TRACK[0]) / d.clip.fps)
        x, y, w, h = SPLIT_BEV
        place(canvas, img, (x + 60 * (1 - a), y + self.dy, w, h), a)

    def stage_label(self, img: np.ndarray, text: str, side: str, a: float) -> None:
        if a <= 0.002:
            return
        size, pad = 15, 12
        tw = text_width(text, size, "medium")
        w, h = round(tw) + 2 * pad, 30
        x = 20 if side.endswith("left") else img.shape[1] - 20 - w
        y = 20 if side.startswith("top") else img.shape[0] - 20 - h
        panel(img, x, y, w, h, INK, 0.78 * a, radius=7, edge=(255, 255, 255), edge_alpha=0.10)
        put_text(img, text, x + pad, y + 20, size, "medium", TEXT, a)

    def draw_tags(self, img: np.ndarray, t: float, tau: float, crop: tuple, size: tuple) -> None:
        """Verdict tags on the filtered side, shown as the divider passes them."""
        s = size[0] / crop[2]

        def to_stage(b: np.ndarray) -> np.ndarray:
            return np.array([(b[0] - crop[0]) * s, (b[1] - crop[1]) * s, (b[2] - crop[0]) * s, (b[3] - crop[1]) * s])

        items = [(tr, tr.verdict, AMBER) for tr in self.day.rejected] + \
                [(tr, "parked", PARKED) for tr in self.day.parked_tags]
        for tr, label, col in items:
            b = bbox_at(tr, tau)
            if b is None:
                continue
            b = to_stage(b)
            tp = self.passed((b[0] + b[2]) / 2)
            if tp is None or t < tp:
                continue
            hold = 1.6 if label == "parked" else 1.1
            a = ramp(t, tp, tp + 0.25) * (1 - ramp(t, tp + hold, tp + hold + 0.5)) * (1 - ramp(t, 25.9, 26.3))
            if a <= 0.002:
                continue
            layer = img.copy()
            box = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
            polyline_aa(layer, box, col, 2, closed=True)
            draw_chip(layer, label, (int(b[0]), int(b[1]) - 5), size=14, fg=(20, 16, 14), bg=col)
            img[...] = blend(img, layer, a)

    # --------------------------------------------------------------- tiles
    def draw_tiles(self, canvas: np.ndarray, t: float) -> None:
        d = self.day
        grow = ramp(t, *T_TILES)
        for k, tile in enumerate(self.tiles):
            rect = self.rects[k]
            out = ramp(t, T_END[0] + 0.06 * k, T_END[0] + 0.5 + 0.06 * k)
            if k == 0:
                rect = lerp_rect(STAGE, rect, grow)
                a = 1.0
                f = d.frame_index(t)
                tau = f / d.clip.fps
                w_small = ramp(t, T_TILES[0] + 0.5, T_TILES[1])
                crop = lerp_rect(ZOOM_CROP, FULL_CROP, grow)
                content = view(d.overlay(d.clip.get(f), tau), crop, rect[2:]) if w_small < 1 else None
                if w_small > 0:
                    small = cv2.resize(tile.content(tau), (int(round(rect[2])), int(round(rect[3]))),
                                       interpolation=cv2.INTER_AREA)
                    content = small if content is None else blend(content, small, w_small)
            else:
                t_in = self.tile_in(k)
                if t < t_in:
                    continue
                a = ramp(t, t_in, t_in + 0.7)
                rect = scale_rect(rect, lerp(0.94, 1.0, a))
                tau = tile.offset + t
                content = cv2.resize(tile.content(tau), (int(round(rect[2])), int(round(rect[3]))),
                                     interpolation=cv2.INTER_AREA)
            if out > 0:
                rect = scale_rect(rect, lerp(1.0, 0.97, out))
                a *= 1 - out
            label_a = ramp(t, max(self.tile_in(k), T_TILES[1] - 0.2 if k == 0 else 0) + 0.3,
                           max(self.tile_in(k), T_TILES[1] - 0.2 if k == 0 else 0) + 0.8)
            if label_a > 0:
                self.tile_labels(content, tile, k, t, label_a)
            place(canvas, content, rect, a, radius=10)

    def tile_labels(self, img: np.ndarray, tile: Tile, k: int, t: float, a: float) -> None:
        h, w = img.shape[:2]
        # soft dark band at the bottom for the name and status
        band = 46
        g = np.linspace(0, 0.62, band, dtype=np.float32)[:, None, None] * a
        roi = img[h - band:].astype(np.float32)
        img[h - band:] = (roi * (1 - g) + np.asarray(INK, np.float32) * g + 0.5).astype(np.uint8)
        lw = text_width(tile.worker, 13, "mono")
        panel(img, 10, 10, lw + 18, 24, INK, 0.75 * a, radius=6, edge=(255, 255, 255), edge_alpha=0.10)
        put_text(img, tile.worker, 19, 27, 13, "mono", TEXT, a)
        put_text(img, tile.name, 12, h - 14, 14, "medium", TEXT, a)
        # progress: longer clips take longer
        t0 = max(self.tile_in(k), T_TILES[1] - 0.2 if k == 0 else 0.0) + 0.4
        longest = max(x.duration for x in self.tiles)
        dur = 1.3 + 2.7 * math.sqrt(tile.duration / longest)
        p = ramp(t, t0, t0 + dur)
        status = "done" if p >= 1 else f"{int(p * 100):d}%"
        sw = text_width("100%", 13, "mono")
        panel(img, w - 10 - (sw + 18), 10, sw + 18, 24, INK, 0.75 * a, radius=6, edge=(255, 255, 255), edge_alpha=0.10)
        put_text(img, status, w - 19, 27, 13, "mono", CYAN if p >= 1 else TEXT, a, align="right")
        fill_rect(img, 0, h - 3, w * p, 3, CYAN, a)

    # ------------------------------------------------------------ end card
    def draw_end(self, canvas: np.ndarray, t: float) -> None:
        t0 = T_END[0] + 0.45
        self.end_title.draw(canvas, W / 2, 392, t, t0, stagger=0.045, dur=0.8)

        def item(k: int) -> tuple[float, float]:
            e = ramp(t, t0 + 0.55 + 0.12 * k, t0 + 1.25 + 0.12 * k)
            return e, 10 * (1 - e)

        a, dy = item(0)
        put_text(canvas, TAGLINE, W / 2, 452 + dy, 30, "regular", MUTED, a, align="center")
        a, dy = item(1)
        size = 23
        tw = text_width("$  " + INSTALL, size, "mono")
        pw, ph = tw + 56, 58
        panel(canvas, W / 2 - pw / 2, 516 + dy, pw, ph, (22, 19, 18), a, radius=10, edge=(255, 255, 255),
              edge_alpha=0.10)
        x = round(W / 2 - tw / 2)
        put_text(canvas, "$", x, 553 + dy, size, "mono", AMBER, a)
        put_text(canvas, INSTALL, x + round(text_width("$  ", size, "mono")), 553 + dy, size, "mono", TEXT, a)
        a, dy = item(2)
        put_text(canvas, REPO, W / 2, 636 + dy, 26, "medium", CYAN, a, align="center")
        a, dy = item(3)
        fill_rect(canvas, W / 2 - 20, 696 + dy, 40, 1, FAINT, a)
        put_text(canvas, AUTHOR, W / 2, 756 + dy, 26, "medium", TEXT, a, align="center")
        put_text(canvas, AFFILIATION, W / 2, 794 + dy, 21, "regular", MUTED, a, align="center")
        a, _ = item(5)
        put_text(canvas, CREDIT, W / 2, 1030, 16, "regular", FAINT, a, align="center")


# ---------------------------------------------------------------------- gif
def side_by_side(day: Day, width: int = 1000, height: int = 360):
    """The library's side layout (camera overlay next to the bird's-eye view), sized for the README."""
    cam_w = int(round(height * 16 / 9))
    bev_w = width - cam_w - 2
    draw_size = (800, 450)  # overlay drawn here, then reduced: labels stay legible at README size
    cam_small = day.cam.resized(draw_size)
    k = draw_size[0] / day.cam.image_size[0]
    bev = BEVCanvas(day.cam, day.run.background, (bev_w, height), np.zeros(2), 50.0, fit_points=day.fit)
    sep = np.full((height, 2, 3), INK, np.uint8)

    def render(tau: float) -> np.ndarray:
        f = int(round(tau * day.clip.fps))
        tau = f / day.clip.fps
        states = day.index.at(tau)
        img = cv2.resize(day.clip.get(f), draw_size, interpolation=cv2.INTER_AREA)
        img = draw_camera_overlay(img, cam_small, scaled_states(states, k))
        img = cv2.resize(img, (cam_w, height), interpolation=cv2.INTER_AREA)
        return np.hstack([img, sep, bev.render(states)])

    return render


def stabilise(frames: list[np.ndarray], threshold: int) -> list[np.ndarray]:
    """Hold pixels that barely change (sensor noise, compression shimmer) so the GIF stores only motion."""
    ref = frames[0].copy()
    out = [ref.copy()]
    kernel = np.ones((3, 3), np.uint8)
    for f in frames[1:]:
        diff = cv2.absdiff(f, ref).max(axis=2)
        changed = cv2.dilate((diff > threshold).astype(np.uint8), kernel) > 0
        ref[changed] = f[changed]
        out.append(ref.copy())
    return out


def write_gif(frames: list[np.ndarray], path: Path, delay_cs: int = 7, max_mb: float = 8.0) -> Path:
    """Palette-optimised GIF (ffmpeg palettegen/paletteuse) with a uniform frame delay, looping forever."""
    exe = ffmpeg_executable()
    if exe is None:
        raise RuntimeError("FFmpeg is required for the GIF (pip install imageio-ffmpeg)")
    h, w = frames[0].shape[:2]
    for threshold, dither in ((7, "bayer:bayer_scale=4"), (10, "bayer:bayer_scale=3"), (14, "bayer:bayer_scale=2"),
                              (20, "none")):
        data = b"".join(f.tobytes() for f in stabilise(frames, threshold))
        graph = ("[0:v]split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
                 f"[b][p]paletteuse=dither={dither}:diff_mode=rectangle")
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"100/{delay_cs}", "-i", "-", "-filter_complex", graph, "-loop", "0", str(path)]
        subprocess.run(cmd, input=data, check=True)
        mb = path.stat().st_size / 2 ** 20
        print(f"  gif: threshold {threshold}, dither {dither}: {mb:.2f} MB")
        if mb <= max_mb:
            break
    return path


def make_gif(day: Day, path: Path, seconds: float = 7.0, fps: float = 15.0, xfade: float = 0.7) -> Path:
    """A looping side-by-side render: the last ``xfade`` seconds dissolve into the moments before the first frame.

    The clip is sampled at ``fps`` (an even divisor of the source rate) and played with a uniform delay of
    ``round(100 / fps)`` centiseconds, the timing unit of GIF.
    """
    render = side_by_side(day)
    start = busiest_start(day.run, seconds + xfade, lo=xfade) + xfade
    n, m = int(round(seconds * fps)), int(round(xfade * fps))
    head = [render(start - (m - j) / fps) for j in range(m)]
    frames = []
    for i in range(n):
        f = render(start + i / fps)
        j = i - (n - m)
        if j >= 0:
            f = blend(f, head[j], ease((j + 1) / (m + 1)))
        frames.append(f)
    return write_gif(frames, path, delay_cs=int(round(100 / fps)))


# --------------------------------------------------------------------- main
def load_runs(runs_dir: Path, device: str) -> tuple[dict, object]:
    pipe = Pipeline(device=device)
    runs = {}
    for key, name in CLIPS:
        t0 = time.time()
        runs[key] = pipe.run(sample_video(name), runs_dir / key)
        print(f"  {name}: {runs[key].summary()['tracks']} tracks ({time.time() - t0:.1f} s)")
    return runs, pipe.config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs" / "trailer",
                    help="pipeline run directories (detections are cached here)")
    ap.add_argument("--device", default="auto", help="device for detection when no cached detections exist")
    ap.add_argument("--crf", type=int, default=25)
    ap.add_argument("--only", default="video,poster,gif", help="comma-separated: video, poster, gif, stills")
    ap.add_argument("--stills", default="1.6,5,9,13.5,19,24.2,29.5,35", help="times (s) for --only stills")
    ap.add_argument("--stills-dir", type=Path, default=None)
    ap.add_argument("--poster-time", type=float, default=19.6)
    args = ap.parse_args()
    jobs = {s.strip() for s in args.only.split(",") if s.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("pipeline runs")
    runs, cfg = load_runs(args.runs, args.device)
    trailer = Trailer(runs, cfg)
    if "stills" in jobs:
        sd = args.stills_dir or args.runs / "stills"
        sd.mkdir(parents=True, exist_ok=True)
        for s in args.stills.split(","):
            t = float(s)
            cv2.imwrite(str(sd / f"still_{t:05.2f}.png"), trailer.frame(t))
        print(f"stills in {sd}")
    if "video" in jobs:
        path = args.out / "trailer.mp4"
        n = int(round(DURATION * FPS))
        t0 = time.time()
        with VideoWriter(path, FPS, crf=args.crf, preset="slow") as wr:
            for i in range(n):
                wr.write(trailer.frame(i / FPS))
                if i % 150 == 0:
                    print(f"  frame {i}/{n} ({time.time() - t0:.0f} s)")
        print(f"{path}: {path.stat().st_size / 2 ** 20:.1f} MB, {n / FPS:.1f} s")
    if "poster" in jobs:
        path = args.out / "trailer_poster.jpg"
        cv2.imwrite(str(path), trailer.frame(args.poster_time, poster=True), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{path}: {path.stat().st_size / 2 ** 10:.0f} KB")
    if "gif" in jobs:
        path = make_gif(trailer.day, args.out / "readme_hero.gif")
        print(f"{path}: {path.stat().st_size / 2 ** 20:.2f} MB")
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()

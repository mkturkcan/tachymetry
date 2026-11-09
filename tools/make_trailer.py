"""Render the Tachymetry trailer, its poster frame, the README GIF and their credits.

Outputs, written to ``--out`` (``docs/media`` by default):

* ``trailer.mp4``: 1920x1080, 30 fps, H.264 (``yuv420p``, ``+faststart``), silent, 42 s;
* ``trailer_poster.jpg``: a 1920x1080 still used as the poster of the video;
* ``readme_hero.gif``: a 7 s looping camera and bird's-eye-view render, 1000 px wide;
* ``CREDITS.md``: the footage sources and licences of the three files above.

The footage is the public sample clips of :mod:`tachymetry.data`: a signalised intersection
(svbrd-llm-roadside-video-av, CC BY 4.0), a straight-down drone view of the Doyel Chattor
roundabout in Dhaka (DRINF, CC BY 4.0), a roundabout seen from a hovering drone (Wikimedia
Commons, CC0 1.0) and a motorway seen from an overpass (Hugging Face, MIT). Each clip is
processed by :class:`tachymetry.pipeline.Pipeline` with the configuration overrides of
:func:`tachymetry.data.sample_overrides` and either the known camera of
:func:`tachymetry.data.sample_camera` (the straight-down drone clip) or automatic calibration.
Detections are cached in the run directories under ``--runs``, so only the first invocation needs
a GPU; later invocations take a few minutes on the CPU. Camera frames come from the library's
renderer (:class:`tachymetry.render.video.Renderer`: 3D boxes, speeds, corner brackets for traffic
beyond the metric range), bird's-eye views from :class:`tachymetry.render.bev.BEVCanvas`, the ground
grid from :func:`tachymetry.calib.preview.draw_ground_grid`, and text uses the bundled Geist typefaces.

Scenes:

1. title;
2. montage of the raw sample clips;
3. the straight-down drone clip with 3D boxes and speeds, pulling back from a 2x view;
4. its bird's-eye view next to the camera view, on the same ground area, with accumulating trajectories;
5. the metric ground grid of the automatic calibration of the intersection, revealed from near to far;
6. 3D boxes and speeds on the motorway and at the roundabout;
7. raw tracker output (ByteTrack identities, frame-difference speeds) wiped away to the filtered
   tracks, with rejected and parked tracks tagged;
8. the eight sample clips processed in parallel, labelled with the worker plan of
   :func:`tachymetry.batch.devices.plan_workers`;
9. end card with the footage credits.

Usage::

    python tools/make_trailer.py
    python tools/make_trailer.py --device cuda:0 --runs runs/trailer --out docs/media
    python tools/make_trailer.py --only gif
    python tools/make_trailer.py --only stills --stills 2,6,11,16,20,24,30,35,40
"""

from __future__ import annotations

import argparse
import json
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
from tachymetry.config import PipelineConfig
from tachymetry.data import SAMPLES, attribution, sample_camera, sample_overrides, sample_video
from tachymetry.geometry.camera import Camera
from tachymetry.geometry.lens import lens_from_dict
from tachymetry.io.video import VideoSource
from tachymetry.io.writer import VideoWriter, ffmpeg_executable
from tachymetry.pipeline import Pipeline
from tachymetry.render.bev import BEVCanvas
from tachymetry.render.overlay import draw_camera_overlay
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
from tachymetry.render.video import Renderer
from tachymetry.synthetic.traffic import naive_speeds
from tachymetry.tracking.builder import associate
from tachymetry.tracking.measure import measure_ground

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 30
DURATION = 42.0

DAY, DHAKA, ROUND, MOTOR, NIGHT = (
    "intersection-day",
    "dhaka-roundabout",
    "roundabout-aerial",
    "motorway-overpass",
    "intersection-night",
)
CLIPS = [DAY, DHAKA, ROUND, MOTOR, "intersection-day-2", "intersection-day-3", "intersection-turn", NIGHT]
GIF_SCENE = ROUND
GIF_CROP = {ROUND: (180.0, 60.0, 1280.0, 720.0)}  # camera-view crop of the GIF (source pixels), per scene
POSTER_TIME = 17.2

TAGLINE = "Metric trajectories and speeds from traffic video"
INSTALL = "pip install git+https://github.com/mkturkcan/tachymetry"
REPO = "github.com/mkturkcan/tachymetry"
AUTHOR = "Mehmet Kerem Turkcan"
AFFILIATION = "Associate Research Scientist, Department of Civil Engineering, Columbia University"

# layout (pixels of the 1920x1080 frame); rectangles are (x, y, w, h)
STAGE = (160.0, 36.0, 1600.0, 900.0)
HALF_L = (160.0, 36.0, 792.0, 900.0)
HALF_R = (968.0, 36.0, 792.0, 900.0)
TILE = (388.0, 218.0)
TILE_RENDER = (640, 360)  # tiles draw their overlay at this size, so lines stay visible when small
CAPTION_X, CAPTION_Y, SUB_Y = 160.0, 994.0, 1030.0

# source-pixel regions of the 3840x2160 drone clip
DHAKA_CLOSE = (700.0, 150.0, 1920.0, 1080.0)  # 2x view of the north-west approach
DHAKA_HALF = (729.6, 0.0, 1900.8, 2160.0)  # the roundabout at the scale of the full view, half the width

# timeline (seconds)
T_TITLE_OUT = (2.55, 3.25)
T_STAGE_IN = (2.75, 3.45)
XF = 0.3  # dissolve between shots
MONTAGE = [  # sample, first second on screen, label
    (DHAKA, 2.75, "Doyel Chattor, Dhaka · straight-down drone · 3840×2160"),
    (ROUND, 3.95, "Roundabout · hovering drone · 1920×1080"),
    (MOTOR, 5.05, "Motorway · overpass camera · 1280×720"),
    (DAY, 6.15, "Signalised intersection · elevated camera · 1920×1080"),
    (NIGHT, 7.25, "The same intersection at night · infrared"),
]
T_DHAKA = 8.35  # the drone clip with overlays dissolves in
T_PULL = (9.0, 12.8)  # pull back from DHAKA_CLOSE to the full view
T_SPLIT = (13.0, 14.0)  # the camera view narrows to the left half
T_BEV_IN = (13.8, 14.5)  # then the bird's-eye view slides in beside it
T_CALIB = 18.3
T_SWEEP = (18.9, 21.0)
T_MOTOR = 22.6
T_ROUND = 25.0
T_FILTER = 27.4
T_RAW = (27.9, 28.7)  # zoom to 1:1 pixels and cross to the raw tracker output
T_WIPE = (28.9, 30.9)
T_TILES = (32.4, 33.5)
T_END = (37.0, 37.8)

CAPTIONS = [  # (in, out, key); the text is filled in from the runs
    (3.4, 8.2, "montage"),
    (8.9, 13.0, "drone"),
    (13.8, 18.2, "bev"),
    (18.8, 22.4, "calibration"),
    (23.0, 27.2, "speeds"),
    (28.0, 32.2, "filtering"),
    (33.4, 36.9, "batch"),
]


# ------------------------------------------------------------------ helpers
def ease(x: float) -> float:
    """Cubic ease-in-out on [0, 1]."""
    x = min(max(float(x), 0.0), 1.0)
    return 4.0 * x**3 if x < 0.5 else 1.0 - (-2.0 * x + 2.0) ** 3 / 2.0


def ramp(t: float, a: float, b: float) -> float:
    return ease((t - a) / (b - a))


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def lerp_rect(r0: tuple, r1: tuple, u: float) -> tuple:
    return tuple(lerp(p, q, u) for p, q in zip(r0, r1))


def scale_rect(r: tuple, s: float) -> tuple:
    x, y, w, h = r
    return (x + w * (1 - s) / 2, y + h * (1 - s) / 2, w * s, h * s)


def zoom_rect(r0: tuple, r1: tuple, u: float) -> tuple:
    """Crop window between ``r0`` and ``r1`` for a zoom of uniform speed about their common fixed point."""
    x0, y0, w0, h0 = r0
    x1, y1, w1, h1 = r1
    if abs(w1 - w0) < 1e-6:
        return lerp_rect(r0, r1, u)
    w = w0 * (w1 / w0) ** u
    k = w / w0
    px = (x0 * w1 - x1 * w0) / (w1 - w0)
    py = (y0 * h1 - y1 * h0) / (h1 - h0)
    return (px - (px - x0) * k, py - (py - y0) * k, w, h0 * k)


def zoom_about(full: tuple, cx: float, cy: float, z: float) -> tuple:
    """Crop window ``z`` times smaller than ``full`` that keeps the point ``(cx, cy)`` fixed on screen."""
    x, y, w, h = full
    return (cx - (cx - x) / z, cy - (cy - y) / z, w / z, h / z)


def blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """``a`` towards ``b`` by ``w``."""
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b
    return cv2.addWeighted(b, float(w), a, float(1 - w), 0)


def plural(n: int, cls: str) -> str:
    return f"{n} {PLURALS.get(cls, cls) if n != 1 else cls}"


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
        mask = cv2.warpAffine(
            mask, np.float32([[1, 0, fx], [0, 1, fy]]), (mask.shape[1] + 1, mask.shape[0] + 1), flags=cv2.INTER_LINEAR
        )
    h, w = mask.shape[:2]
    x0, y0 = max(xi, 0), max(yi, 0)
    x1, y1 = min(xi + w, canvas.shape[1]), min(yi + h, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - yi : y1 - yi, x0 - xi : x1 - xi, None] * float(alpha)
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(roi + (np.asarray(color, np.float32) - roi) * m + 0.5, 0, 255).astype(np.uint8)


def put_text(
    canvas: np.ndarray,
    text: str,
    x: float,
    y: float,
    size: int,
    weight: str = "regular",
    color: tuple = TEXT,
    alpha: float = 1.0,
    align: str = "left",
) -> None:
    """Draw ``text`` with its baseline at ``y``; ``x`` is its left edge, centre or right edge."""
    if alpha <= 0.002 or not text:
        return
    m, ox, oy = _mask(text, size, weight)
    if align != "left":  # whole-pixel alignment keeps static text crisp
        tw = text_width(text, size, weight)
        x -= round(tw / 2) if align == "center" else round(tw)
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


def place(
    canvas: np.ndarray, img: np.ndarray, rect: tuple, alpha: float = 1.0, radius: int = 12, border: float = 0.07
) -> None:
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


def panel(
    canvas: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: tuple,
    alpha: float,
    radius: int = 8,
    edge: tuple | None = None,
    edge_alpha: float = 0.0,
) -> None:
    """Flat rounded rectangle (label backgrounds, the install line)."""
    Wd, Hd = int(round(w)), int(round(h))
    if Wd < 4 or Hd < 4:
        return
    blend_mask(canvas, rounded_mask(Wd, Hd, radius), round(x), round(y), fill, alpha)
    if edge is not None and edge_alpha > 0:
        blend_mask(canvas, ring_mask(Wd, Hd, radius), round(x), round(y), edge, edge_alpha * alpha)


def label(img: np.ndarray, text: str, a: float, corner: str = "top-left", size: int = 15) -> None:
    """A dark rounded chip with ``text`` in a corner of ``img``."""
    if a <= 0.002:
        return
    pad, h = 12, size + 15
    w = round(text_width(text, size, "medium")) + 2 * pad
    x = 20 if corner.endswith("left") else img.shape[1] - 20 - w
    y = 20 if corner.startswith("top") else img.shape[0] - 20 - h
    panel(img, x, y, w, h, INK, 0.78 * a, radius=7, edge=(255, 255, 255), edge_alpha=0.10)
    put_text(img, text, x + pad, y + round(h * 0.68), size, "medium", TEXT, a)


def view(full: np.ndarray, crop: tuple, size: tuple) -> np.ndarray:
    """The region ``crop = (x, y, w, h)`` of ``full`` (sub-pixel, source pixels) resampled to ``size``.

    Shrinking averages areas first and applies the fractional offset afterwards, so slow zooms and pans move
    smoothly instead of stepping by whole source pixels.
    """
    x, y, w, h = crop
    Wd, Hd = int(round(size[0])), int(round(size[1]))
    s = Wd / w
    if s < 1.0:
        x0, y0 = max(math.floor(x) - 2, 0), max(math.floor(y) - 2, 0)
        x1, y1 = min(math.ceil(x + w) + 2, full.shape[1]), min(math.ceil(y + h) + 2, full.shape[0])
        sw, sh = max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))
        src = cv2.resize(full[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_AREA)
        kx, ky = sw / (x1 - x0), sh / (y1 - y0)
        m = np.float32([[s / kx, 0, -(x - x0) * s], [0, s / ky, -(y - y0) * s]])
    else:
        src = full
        m = np.float32([[s, 0, -x * s], [0, s, -y * s]])
    if np.allclose(m[:, :2], np.eye(2), atol=1e-6) and np.allclose(m[:, 2], np.round(m[:, 2]), atol=1e-3):
        dx, dy = -int(round(m[0, 2])), -int(round(m[1, 2]))
        if 0 <= dx and 0 <= dy and dx + Wd <= src.shape[1] and dy + Hd <= src.shape[0]:
            return src[dy : dy + Hd, dx : dx + Wd].copy()
    flags = cv2.INTER_LINEAR if s < 1.0 else cv2.INTER_CUBIC
    return cv2.warpAffine(src, m, (Wd, Hd), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class Wordmark:
    """A word set glyph by glyph (kerned layout) so that each letter can resolve on its own.

    The glyphs are accumulated into one coverage mask (no double blending where they meet) and, once they
    have settled, the mask hands over to the whole-word rendering so that the resting title is crisp.
    """

    def __init__(self, text: str, size: int, weight: str = "semibold"):
        f = font(size, weight)
        self.word, self.wox, self.woy = _mask(text, size, weight)
        self.width = f.getlength(text)
        self.glyphs = [
            (f.getlength(text[: i + 1]) - f.getlength(ch), *_mask(ch, size, weight)) for i, ch in enumerate(text)
        ]

    def draw(
        self,
        canvas: np.ndarray,
        cx: float,
        baseline: float,
        t: float,
        t0: float,
        stagger: float = 0.055,
        dur: float = 0.9,
        alpha: float = 1.0,
        rise: float = 16.0,
        blur: float = 7.0,
        spread: float = 9.0,
        color: tuple = TEXT,
        settle: float = 0.35,
    ) -> None:
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
            word[pad : pad + wh, pad : pad + ww] = self.word
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
    """Start (clip time) of the window of ``length`` seconds that maximises the count of moving road users."""
    dur = float(run.tracks.meta.get("duration_s") or 0.0)
    moving = [tr for tr in run.tracks.ok if tr.state == "moving"]
    pick, pick_score = lo, -1.0
    for s in np.arange(lo, max(lo, dur - length - 0.2) + 1e-6, 0.5):
        ts = np.linspace(s, s + length, 9)
        score = sum(sum(1 for tr in moving if tr.t[0] <= t <= tr.t[-1]) for t in ts)
        if score > pick_score:
            pick, pick_score = float(s), score
    return pick


class Footage:
    """A sample clip and its run: frames, the library's camera renderer and a reduced overlay for tiles."""

    def __init__(self, name: str, run, cfg: PipelineConfig):
        self.name, self.run, self.cfg = name, run, cfg
        self.cam = run.camera
        self.size = tuple(int(v) for v in self.cam.image_size)
        self.full = (0.0, 0.0, float(self.size[0]), float(self.size[1]))
        self.clip = Clip(run.source)
        self.fps = self.clip.fps
        self.duration = self.clip.n / self.fps
        # camera layout at the source resolution; the header bar is cut off
        self.renderer = Renderer(run, replace(cfg.render, max_width=self.size[0]), layout="camera")
        self.index = self.renderer.index
        self.moving = [tr for tr in run.tracks.ok if tr.state == "moving"]
        self._tile_cam = self.cam.resized(TILE_RENDER)

    def frame_index(self, tau: float) -> int:
        return int(min(max(round(tau * self.fps), 0), self.clip.n - 1))

    def raw(self, tau: float) -> np.ndarray:
        return self.clip.get(self.frame_index(tau))

    def overlay(self, tau: float) -> np.ndarray:
        """The library's annotated camera frame at clip time ``tau`` (source resolution)."""
        f = self.frame_index(tau)
        return self.renderer.frame(self.clip.get(f), f / self.fps)[self.renderer.header_h :]

    def tile(self, tau: float) -> np.ndarray:
        f = self.frame_index(tau)
        img = cv2.resize(self.clip.get(f), TILE_RENDER, interpolation=cv2.INTER_AREA)
        states = scaled_states(self.index.at(f / self.fps), TILE_RENDER[0] / self.size[0])
        return draw_camera_overlay(img, self._tile_cam, states, max_labels=0)

    def median_speed(self, cls: str) -> float:
        """Median over the moving tracks of a class of their median moving speed (km/h)."""
        sp = [tr.summary()["speed_median_kmh"] for tr in self.moving if tr.cls == cls]
        return float(np.median(sp)) if sp else float("nan")

    def vehicle_speed(self) -> float:
        """Median moving speed of vehicles, as in the run summary (km/h)."""
        return float(self.run.tracks.summary()["vehicle_speed_kmh"]["median"] or float("nan"))


# ------------------------------------------------------------------- layers
class GridSweep:
    """The metric ground grid of the calibrated camera, revealed from near to far."""

    def __init__(self, cam, spacing: float = 5.0, major: float = 25.0):
        w, h = cam.image_size
        small = cam.resized((w * 2 // 3, h * 2 // 3))  # 1 px minor and 2 px major lines, upscaled: a light grid
        lines = draw_ground_grid(
            np.zeros((small.image_size[1], small.image_size[0], 3), np.uint8),
            small,
            spacing=spacing,
            major=major,
            color=(255, 255, 255),
            alpha=1.0,
            horizon=False,
            max_gsd=0.45,
        )
        self.mask = cv2.resize(lines[..., 0], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        step = 4
        uu, vv = np.meshgrid(np.arange(0, w, step) + step / 2, np.arange(0, h, step) + step / 2)
        g = cam.image_to_ground(np.column_stack([uu.ravel(), vv.ravel()]))
        d = cam.distance_to(g).reshape(uu.shape)
        d = np.where(np.isfinite(d) & (d > 0.5), d, 1e5)
        self.logd = cv2.resize(np.log(d).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
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

    def __init__(self, run, cfg: PipelineConfig):
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
            r = order[np.searchsorted(fr, f, "left") : np.searchsorted(fr, f, "right")]
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


class HalfBEV:
    """Bird's-eye view of the ground under a crop of a straight-down camera, at the crop's scale and orientation."""

    def __init__(self, fo: Footage, crop: tuple, size: tuple[int, int]):
        cam = fo.cam
        x, y, w, h = crop
        c = np.array([[x + w / 2, y + h / 2]])
        e = cam.image_to_ground(np.array([[x + w / 2 - 50, y + h / 2], [x + w / 2 + 50, y + h / 2]]))
        mpp = float(np.linalg.norm(e[1] - e[0])) / 100 * w / size[0]
        self.fo = fo
        self.bev = BEVCanvas(cam, fo.run.background, size, cam.image_to_ground(c)[0], mpp * min(size))
        self.base = self.bev.base.copy()

    def grid_step(self) -> float:
        """Spacing of the canvas grid in metres (the rule of :meth:`BEVCanvas._draw_grid`)."""
        span = self.bev.mpp * min(self.bev.W, self.bev.H)
        return 5.0 if span <= 60 else 10.0 if span <= 150 else 25.0

    def render(self, tau: float, since: float) -> np.ndarray:
        """Track footprints at ``tau`` over the trajectories accumulated since ``since``."""
        hist = self.base.copy()
        for tr in self.fo.moving:
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
        self.bev.base = cv2.addWeighted(hist, 0.6, self.base, 0.4, 0)
        try:
            return self.bev.render(self.fo.index.at(tau))
        finally:
            self.bev.base = self.base


def tile_rects() -> list[tuple]:
    tw, th = TILE
    gap = (STAGE[2] - 4 * tw) / 3
    y0 = STAGE[1] + (STAGE[3] - (2 * th + gap)) / 2
    return [(STAGE[0] + c * (tw + gap), y0 + r * (th + gap), tw, th) for r in range(2) for c in range(4)]


def credit_line(name: str) -> str:
    """Attribution without the URL, for the end card."""
    return attribution(name).rsplit(", ", 1)[0]


def sources(names: list[str]) -> list[str]:
    """One sample per distinct footage source (the intersection clips share one), in order."""
    out, seen = [], set()
    for n in names:
        key = (SAMPLES[n].credit, SAMPLES[n].license)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


# ------------------------------------------------------------------ trailer
class Trailer:
    def __init__(self, footage: dict[str, Footage]):
        self.fo = footage
        day, dhaka, motor, rnd = footage[DAY], footage[DHAKA], footage[MOTOR], footage[ROUND]
        # clip time at the first second of each shot
        self.montage = []
        for k, (name, t_in, text) in enumerate(MONTAGE):
            t_out = MONTAGE[k + 1][1] + XF if k + 1 < len(MONTAGE) else T_DHAKA + XF
            self.montage.append((name, t_in, t_out, busiest_start(footage[name].run, t_out - t_in), text))
        self.dhaka_start = busiest_start(dhaka.run, T_CALIB + XF - T_DHAKA)
        self.calib_start = busiest_start(day.run, T_MOTOR + XF - T_CALIB)
        self.motor_start = busiest_start(motor.run, T_ROUND + XF - T_MOTOR)
        self.round_start = busiest_start(rnd.run, T_FILTER + XF - T_ROUND)
        self.filter_start = 18.6  # the wipe passes over the clip around 20 s, where a rider and a ghost are visible
        self.grid = GridSweep(day.cam)
        self.raw = RawTracker(day.run, day.cfg)
        self.half_bev = HalfBEV(dhaka, DHAKA_HALF, (int(HALF_R[2]), int(HALF_R[3])))
        self.rejected = [tr for tr in day.run.tracks.tracks if not tr.is_ok]
        self.parked_tags = self._parked_tags(day)
        self.captions = self._captions()
        # worker labels: the library's worker plan for these jobs on a four-GPU machine (two workers per GPU);
        # jobs beyond the number of workers queue for the first free worker
        gpus = [Device(f"cuda:{i}", "cuda", "GPU", 96.0, 96.0) for i in range(4)]
        slots = plan_workers(gpus, n_jobs=len(CLIPS))
        workers, seen = [], {}
        for d in slots:
            seen[d] = seen.get(d, 0) + 1
            workers.append(f"{d} · worker {seen[d]}")
        self.workers = [workers[k % len(workers)] for k in range(len(CLIPS))]
        self.rects = tile_rects()
        cost = [footage[n].duration * footage[n].size[0] * footage[n].size[1] for n in CLIPS]
        self.tile_dur = [1.2 + 2.4 * math.sqrt(c / max(cost)) for c in cost]
        self.tile_start = [0.0] + [
            busiest_start(footage[n].run, 6.0) - self.tile_in(k) for k, n in enumerate(CLIPS) if k
        ]
        self.title = Wordmark("Tachymetry", 136)
        self.end_title = Wordmark("Tachymetry", 112)
        self.credits = [credit_line(n) for n in sources([DHAKA, ROUND, MOTOR, DAY])]
        self._wipe_t = np.linspace(T_WIPE[0], T_WIPE[1], 400)
        self._wipe_x = np.array([self.wipe_x(t) for t in self._wipe_t])

    @staticmethod
    def _parked_tags(fo: Footage) -> list:
        """One "parked" tag per curb: the largest parked vehicle on each side of the middle of the row."""
        parked = [tr for tr in fo.run.tracks.ok if tr.state == "parked"]
        tags = []
        if parked:
            xs = np.array([np.median((tr.bbox[:, 0] + tr.bbox[:, 2]) / 2) for tr in parked])
            mid = (xs.min() + xs.max()) / 2
            for side in (xs <= mid, xs > mid):
                group = [tr for tr, s in zip(parked, side) if s]
                if group:
                    area = [np.median((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) for b in (tr.bbox for tr in group)]
                    tags.append(group[int(np.argmax(area))])
        return tags

    def _captions(self) -> list[tuple]:
        day, dhaka, motor, rnd = self.fo[DAY], self.fo[DHAKA], self.fo[MOTOR], self.fo[ROUND]
        sizes = sorted({fo.size for fo in self.fo.values()}, key=lambda s: s[0] * s[1])
        rates = sorted(fo.fps for fo in self.fo.values())
        report = day.run.out_dir / "camera.report.json" if day.run.out_dir is not None else None
        n_fp = None
        if report is not None and report.exists():
            n_fp = json.loads(report.read_text(encoding="utf-8")).get("details", {}).get("n_footprints")
        cam = day.cam
        found = f"{n_fp} sampled vehicle footprints · " if n_fp else ""
        s = day.run.tracks.summary()
        dc = dhaka.cam
        text = {
            "montage": (
                "Roadside, overpass and drone video",
                f"public sample clips · {sizes[0][0]}×{sizes[0][1]} to {sizes[-1][0]}×{sizes[-1][1]}"
                f" · {rates[0]:.0f} to {rates[-1]:.0f} fps",
            ),
            "drone": (
                "Straight-down drone view; camera from flight altitude",
                f"altitude {dc.height:.1f} m · field of view {dc.fov[0]:.1f}° · median speed: cars"
                f" {dhaka.median_speed('car'):.0f} km/h, pedestrians {dhaka.median_speed('person'):.1f} km/h",
            ),
            "bev": (
                "Bird’s-eye view in metres, with trajectories",
                f"the same ground area as the camera view · {self.half_bev.grid_step():g} m grid"
                " · positions exported to tracks.csv",
            ),
            "calibration": (
                "Automatic calibration from vehicle footprints",
                f"{found}focal {cam.focal:.0f} px · height {cam.height:.1f} m · pitch {cam.pitch:.1f}°",
            ),
            "speeds": (
                "Speeds from an overpass and a hovering drone",
                f"median {motor.vehicle_speed():.0f} km/h and"
                f" {rnd.vehicle_speed():.0f} km/h · traffic beyond the metric range"
                " is bracketed, without a speed",
            ),
            "filtering": (
                "Track filtering: ghosts and parked vehicles",
                f"{self.raw.n_ids} raw tracker IDs → {s['tracks']} tracks ({sum(s['parked_by_class'].values())}"
                " parked)",
            ),
            "batch": ("Batch processing on multiple GPUs", "tachymetry run videos/ --devices auto"),
        }
        return [(a, b, *text[key]) for a, b, key in CAPTIONS]

    # ------------------------------------------------------------- timing
    @staticmethod
    def tile_in(k: int) -> float:
        return T_TILES[0] + 0.25 + 0.09 * k

    def wipe_x(self, t: float) -> float:
        return lerp(-4.0, STAGE[2] + 4.0, ramp(t, *T_WIPE))

    def passed(self, x: float) -> float | None:
        """Time at which the wipe divider passes stage column ``x``."""
        i = int(np.searchsorted(self._wipe_x, x))
        return float(self._wipe_t[i]) if i < len(self._wipe_t) else None

    # ------------------------------------------------------------- frames
    def frame(self, t: float, poster: bool = False) -> np.ndarray:
        canvas = np.full((H, W, 3), INK, np.uint8)
        if t < 3.4:
            self.draw_title(canvas, t)
        if T_STAGE_IN[0] <= t < T_DHAKA + XF:
            self.draw_montage(canvas, t)
        if T_DHAKA <= t < T_CALIB + XF:
            self.draw_dhaka(canvas, t)
        if T_CALIB <= t < T_MOTOR + XF:
            self.draw_calibration(canvas, t)
        if T_MOTOR <= t < T_FILTER + XF:
            self.draw_speeds(canvas, t)
        if T_FILTER <= t < T_TILES[0]:
            self.draw_filter(canvas, t)
        if T_TILES[0] <= t < T_END[1] + 0.4:
            self.draw_tiles(canvas, t)
        if poster:
            self.draw_poster_text(canvas)
        else:
            for c in self.captions:
                self.draw_caption(canvas, t, *c)
        if t >= T_END[0] + 0.3:
            self.draw_end(canvas, t)
        return canvas

    def draw_title(self, canvas: np.ndarray, t: float) -> None:
        out = ramp(t, *T_TITLE_OUT)
        lift = -26 * out
        self.title.draw(canvas, W / 2, 548 + lift, t, 0.35, alpha=1 - out)
        a = ramp(t, 1.25, 2.05) * (1 - out)
        y = 632 + 10 * (1 - ramp(t, 1.25, 2.05)) + lift
        put_text(canvas, TAGLINE, W / 2, y, 32, "regular", MUTED, a, align="center")

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

    def draw_poster_text(self, canvas: np.ndarray) -> None:
        """Poster text in the caption band: name and tagline, the repository, the footage credit."""
        fill_rect(canvas, CAPTION_X, CAPTION_Y - 10, 28, 2, AMBER)
        put_text(canvas, "Tachymetry", CAPTION_X + 42, CAPTION_Y, 30, "semibold", TEXT)
        x = CAPTION_X + 42 + round(text_width("Tachymetry", 30, "semibold")) + 22
        put_text(canvas, TAGLINE, x, CAPTION_Y, 22, "regular", MUTED)
        put_text(canvas, "Footage: " + credit_line(DHAKA), CAPTION_X + 42, SUB_Y + 2, 15, "regular", FAINT)
        put_text(canvas, REPO, STAGE[0] + STAGE[2], CAPTION_Y, 20, "medium", CYAN, align="right")

    # ------------------------------------------------------------- shots
    def draw_montage(self, canvas: np.ndarray, t: float) -> None:
        stage_in = ramp(t, *T_STAGE_IN)
        for k, (name, t_in, t_out, start, text) in enumerate(self.montage):
            if not t_in <= t < t_out:
                continue
            fo = self.fo[name]
            a = stage_in if k == 0 else ramp(t, t_in, t_in + XF)
            u = (t - t_in) / (t_out - t_in)
            crop = zoom_about(fo.full, fo.size[0] / 2, fo.size[1] / 2, 1.0 + 0.05 * u)  # a slow push-in
            rect = scale_rect(STAGE, lerp(0.965, 1.0, stage_in)) if k == 0 else STAGE
            content = view(fo.raw(start + t - t_in), crop, rect[2:])
            label(content, text, ramp(t, t_in + 0.1, t_in + 0.45))
            place(canvas, content, rect, a)

    def draw_dhaka(self, canvas: np.ndarray, t: float) -> None:
        fo = self.fo[DHAKA]
        tau = self.dhaka_start + t - T_DHAKA
        img = fo.overlay(tau)
        a = ramp(t, T_DHAKA, T_DHAKA + XF)
        split = ramp(t, *T_SPLIT)
        if split <= 0.0:
            crop = zoom_rect(DHAKA_CLOSE, fo.full, ramp(t, *T_PULL))
            place(canvas, view(img, crop, STAGE[2:]), STAGE, a)
            return
        # the left panel keeps the scale of the full view while it narrows to the roundabout
        rect = lerp_rect(STAGE, HALF_L, split)
        place(canvas, view(img, lerp_rect(fo.full, DHAKA_HALF, split), rect[2:]), rect, 1.0)
        b = ramp(t, *T_BEV_IN)
        if b > 0.002:
            bev = self.half_bev.render(fo.frame_index(tau) / fo.fps, self.dhaka_start)
            x, y, w, h = HALF_R
            x = max(x + 40 * (1 - b), rect[0] + rect[2] + 16)  # never over the narrowing camera view
            place(canvas, bev, (x, y, w, h), b)

    def draw_calibration(self, canvas: np.ndarray, t: float) -> None:
        fo = self.fo[DAY]
        tau = self.calib_start + t - T_CALIB
        img = fo.raw(tau)
        grid_a = (1 - 0.12 * ramp(t, T_SWEEP[1], T_SWEEP[1] + 0.8)) * (1 - ramp(t, T_MOTOR - 0.9, T_MOTOR - 0.1))
        content = img
        if t >= T_SWEEP[0] and grid_a > 0.002:
            glow = 1 - ramp(t, T_SWEEP[1] - 0.4, T_SWEEP[1] + 0.4)
            content = self.grid.apply(img.copy(), ramp(t, *T_SWEEP), grid_a, glow)
        place(canvas, view(content, fo.full, STAGE[2:]), STAGE, ramp(t, T_CALIB, T_CALIB + XF))

    def draw_speeds(self, canvas: np.ndarray, t: float) -> None:
        shots = [
            (MOTOR, T_MOTOR, T_ROUND + XF, self.motor_start, (0.62, 0.55), 1.06, "Motorway · overpass camera"),
            (ROUND, T_ROUND, T_FILTER + XF, self.round_start, (0.33, 0.33), 1.30, "Roundabout · hovering drone"),
        ]
        for name, t_in, t_out, start, (fx, fy), z, text in shots:
            if not t_in <= t < t_out:
                continue
            fo = self.fo[name]
            u = ease((t - t_in) / (t_out - t_in))
            crop = zoom_about(fo.full, fx * fo.size[0], fy * fo.size[1], 1.0 + (z - 1.0) * u)
            content = view(fo.overlay(start + t - t_in), crop, STAGE[2:])
            label(content, text, ramp(t, t_in + 0.1, t_in + 0.45))
            place(canvas, content, STAGE, ramp(t, t_in, t_in + XF))

    def draw_filter(self, canvas: np.ndarray, t: float) -> None:
        """Raw tracker output wiped away to the filtered tracks, at one source pixel per stage pixel."""
        fo = self.fo[DAY]
        tau = self.filter_start + t - T_FILTER
        f = fo.frame_index(tau)
        tau = f / fo.fps
        size = (int(STAGE[2]), int(STAGE[3]))
        zoom = ramp(t, *T_RAW)
        crop = lerp_rect(fo.full, (320.0, 0.0, 1600.0, 900.0), zoom)
        clean = view(fo.overlay(tau), crop, size)
        raw_a = ramp(t, T_RAW[0] + 0.1, T_RAW[1])
        x_div = self.wipe_x(t)
        raw = None
        if raw_a > 0 and x_div < size[0]:
            raw = view(self.raw.draw(fo.raw(tau).copy(), f), crop, size)
            label(raw, "Raw tracker output", raw_a * (1 - ramp(t, T_WIPE[1] - 0.2, T_WIPE[1])), "bottom-right")
            raw = blend(clean, raw, raw_a)
        if t >= T_WIPE[0]:
            self.draw_tags(clean, t, tau, crop, size)
            label(clean, "Tachymetry", 1 - ramp(t, T_TILES[0] - 0.5, T_TILES[0] - 0.05))
        if raw is None:
            content = clean
        elif x_div <= 0:
            content = raw
        else:
            cols = np.clip(np.arange(size[0], dtype=np.float32) + 0.5 - x_div, 0, 1)[None, :, None]
            content = (clean.astype(np.float32) * (1 - cols) + raw.astype(np.float32) * cols + 0.5).astype(np.uint8)
            if 0 < x_div < size[0]:
                edge = ramp(t, T_WIPE[0], T_WIPE[0] + 0.25) * (1 - ramp(t, T_WIPE[1] - 0.25, T_WIPE[1]))
                fill_rect(content, x_div - 1, 0, 2, size[1], AMBER, edge)
        place(canvas, content, STAGE, ramp(t, T_FILTER, T_FILTER + XF))

    def draw_tags(self, img: np.ndarray, t: float, tau: float, crop: tuple, size: tuple) -> None:
        """Verdict tags on the filtered side, shown as the divider passes them."""
        s = size[0] / crop[2]
        items = [(tr, tr.verdict, AMBER) for tr in self.rejected] + [(tr, "parked", PARKED) for tr in self.parked_tags]
        for tr, text, col in items:
            b = bbox_at(tr, tau)
            if b is None:
                continue
            b = np.array([(b[0] - crop[0]) * s, (b[1] - crop[1]) * s, (b[2] - crop[0]) * s, (b[3] - crop[1]) * s])
            tp = self.passed((b[0] + b[2]) / 2)
            if tp is None or t < tp:
                continue
            hold = 1.6 if text == "parked" else 1.1
            a = ramp(t, tp, tp + 0.25) * (1 - ramp(t, tp + hold, tp + hold + 0.5))
            a *= 1 - ramp(t, T_TILES[0] - 0.6, T_TILES[0] - 0.2)
            if a <= 0.002:
                continue
            layer = img.copy()
            polyline_aa(layer, np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]), col, 2, closed=True)
            draw_chip(layer, text, (int(b[0]), int(b[1]) - 5), size=14, fg=(20, 16, 14), bg=col)
            img[...] = blend(img, layer, a)

    # --------------------------------------------------------------- tiles
    def draw_tiles(self, canvas: np.ndarray, t: float) -> None:
        grow = ramp(t, *T_TILES)
        for k, name in enumerate(CLIPS):
            fo = self.fo[name]
            rect = self.rects[k]
            out = ramp(t, T_END[0] + 0.04 * k, T_END[0] + 0.5 + 0.04 * k)
            size = (int(round(rect[2])), int(round(rect[3])))
            if k == 0:  # the filtered intersection shrinks into the first tile and keeps playing
                rect = lerp_rect(STAGE, rect, grow)
                size = (int(round(rect[2])), int(round(rect[3])))
                a = 1.0
                tau = self.filter_start + t - T_FILTER
                small = ramp(t, T_TILES[0] + 0.5, T_TILES[1])
                content = None
                if small < 1:
                    crop = lerp_rect((320.0, 0.0, 1600.0, 900.0), fo.full, grow)
                    content = view(fo.overlay(tau), crop, size)
                if small > 0:
                    tile = cv2.resize(fo.tile(tau), size, interpolation=cv2.INTER_AREA)
                    content = tile if content is None else blend(content, tile, small)
            else:
                t_in = self.tile_in(k)
                if t < t_in:
                    continue
                a = ramp(t, t_in, t_in + 0.6)
                rect = scale_rect(rect, lerp(0.94, 1.0, a))
                size = (int(round(rect[2])), int(round(rect[3])))
                content = cv2.resize(fo.tile(self.tile_start[k] + t), size, interpolation=cv2.INTER_AREA)
            if out > 0:
                rect = scale_rect(rect, lerp(1.0, 0.97, out))
                a *= 1 - out
            shown = max(self.tile_in(k), T_TILES[1] - 0.2 if k == 0 else 0.0)
            label_a = ramp(t, shown + 0.25, shown + 0.7)
            if label_a > 0:
                self.tile_labels(content, k, t, shown + 0.35, label_a)
            place(canvas, content, rect, a, radius=9)

    def tile_labels(self, img: np.ndarray, k: int, t: float, t0: float, a: float) -> None:
        h, w = img.shape[:2]
        band = 36
        g = np.linspace(0, 0.72, band, dtype=np.float32)[:, None, None] * a
        roi = img[h - band :].astype(np.float32)
        img[h - band :] = (roi * (1 - g) + np.asarray(INK, np.float32) * g + 0.5).astype(np.uint8)
        worker = self.workers[k]
        panel(img, 8, 8, text_width(worker, 12, "mono") + 14, 21, INK, 0.75 * a, 5, (255, 255, 255), 0.10)
        put_text(img, worker, 15, 23, 12, "mono", TEXT, a)
        put_text(img, CLIPS[k], 10, h - 11, 12, "medium", TEXT, a)
        p = ramp(t, t0, t0 + self.tile_dur[k])
        status = "done" if p >= 1 else f"{int(p * 100):d}%"
        sw = text_width("100%", 12, "mono")
        panel(img, w - 8 - (sw + 14), 8, sw + 14, 21, INK, 0.75 * a, 5, (255, 255, 255), 0.10)
        put_text(img, status, w - 15, 23, 12, "mono", CYAN if p >= 1 else TEXT, a, align="right")
        fill_rect(img, 0, h - 3, w * p, 3, CYAN, a)

    # ------------------------------------------------------------ end card
    def draw_end(self, canvas: np.ndarray, t: float) -> None:
        t0 = T_END[0] + 0.45
        self.end_title.draw(canvas, W / 2, 330, t, t0, stagger=0.045, dur=0.8)

        def item(k: int) -> tuple[float, float]:
            e = ramp(t, t0 + 0.55 + 0.12 * k, t0 + 1.25 + 0.12 * k)
            return e, 10 * (1 - e)

        a, dy = item(0)
        put_text(canvas, TAGLINE, W / 2, 390 + dy, 30, "regular", MUTED, a, align="center")
        a, dy = item(1)
        size = 23
        tw = text_width("$  " + INSTALL, size, "mono")
        pw, ph = tw + 56, 58
        panel(canvas, W / 2 - pw / 2, 452 + dy, pw, ph, (22, 19, 18), a, 10, (255, 255, 255), 0.10)
        x = round(W / 2 - tw / 2)
        put_text(canvas, "$", x, 489 + dy, size, "mono", AMBER, a)
        put_text(canvas, INSTALL, x + round(text_width("$  ", size, "mono")), 489 + dy, size, "mono", TEXT, a)
        a, dy = item(2)
        put_text(canvas, REPO, W / 2, 572 + dy, 26, "medium", CYAN, a, align="center")
        a, dy = item(3)
        fill_rect(canvas, W / 2 - 20, 626 + dy, 40, 1, FAINT, a)
        put_text(canvas, AUTHOR, W / 2, 680 + dy, 26, "medium", TEXT, a, align="center")
        put_text(canvas, AFFILIATION, W / 2, 716 + dy, 21, "regular", MUTED, a, align="center")
        a, dy = item(5)
        y = 836 + dy
        put_text(canvas, "Footage", W / 2, y, 14, "medium", MUTED, a, align="center")
        for line in self.credits:
            y += 26
            put_text(canvas, line, W / 2, y, 15, "regular", FAINT, a, align="center")
        put_text(canvas, "This video is licensed CC BY 4.0.", W / 2, y + 36, 14, "regular", FAINT, a, align="center")


# ---------------------------------------------------------------------- gif
def crop_camera(cam: Camera, x0: int, y0: int, w: int, h: int) -> Camera:
    """The camera of the image region ``[x0, x0 + w) x [y0, y0 + h)``."""
    K = cam.K.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    lens = cam.lens.to_dict()
    lens["image_size"] = [w, h]
    lens["center"] = [lens["center"][0] - x0, lens["center"][1] - y0]
    return Camera((w, h), K, cam.R.copy(), cam.t.copy(), lens=lens_from_dict(lens), meta=dict(cam.meta))


def side_by_side(fo: Footage, width: int = 1000, height: int = 360, crop: tuple | None = None):
    """The library's side layout (camera overlay next to the bird's-eye view), sized for the README.

    ``crop`` (source pixels, 16:9) narrows the camera view so that vehicles and their speed labels stay legible.
    """
    cam_w = int(round(height * 16 / 9))
    bev_w = width - cam_w - 2
    x0, y0, cw, ch = (int(v) for v in (crop or fo.full))
    draw_size = (800, 450)  # overlay drawn here, then reduced: labels stay legible at README size
    cam_small = crop_camera(fo.cam, x0, y0, cw, ch).resized(draw_size)
    k = draw_size[0] / cw
    fit = np.vstack([tr.xy for tr in fo.moving])
    bev = BEVCanvas(fo.cam, fo.run.background, (bev_w, height), np.zeros(2), 50.0, fit_points=fit)
    sep = np.full((height, 2, 3), INK, np.uint8)
    shift = np.array([x0, y0, x0, y0], np.float64)

    def render(tau: float) -> np.ndarray:
        f = fo.frame_index(tau)
        tau = f / fo.fps
        states = fo.index.at(tau)
        img = cv2.resize(fo.clip.get(f)[y0 : y0 + ch, x0 : x0 + cw], draw_size, interpolation=cv2.INTER_AREA)
        img = draw_camera_overlay(img, cam_small, [replace(s, bbox=(s.bbox - shift) * k) for s in states])
        img = cv2.resize(img, (cam_w, height), interpolation=cv2.INTER_AREA)
        return np.hstack([img, sep, bev.render(states)])

    return render


def stabilise(frames: list[np.ndarray], threshold: int) -> list[np.ndarray]:
    """Hold pixels that barely change (sensor noise, compression shimmer) so the GIF stores only motion.

    The held reference is carried once around the loop first, so the last frame and the first agree.
    """
    kernel = np.ones((3, 3), np.uint8)
    ref = frames[0].copy()
    out = []
    for k in range(2 * len(frames)):
        f = frames[k % len(frames)]
        diff = cv2.absdiff(f, ref).max(axis=2)
        changed = cv2.dilate((diff > threshold).astype(np.uint8), kernel) > 0
        ref[changed] = f[changed]
        if k >= len(frames):
            out.append(ref.copy())
    return out


def write_gif(frames: list[np.ndarray], path: Path, delay_cs: int = 7, max_mb: float = 8.0) -> Path:
    """Palette-optimised GIF (ffmpeg palettegen/paletteuse) with a uniform frame delay, looping forever."""
    exe = ffmpeg_executable()
    if exe is None:
        raise RuntimeError("FFmpeg is required for the GIF (pip install imageio-ffmpeg)")
    h, w = frames[0].shape[:2]
    for threshold, dither in (
        (7, "bayer:bayer_scale=4"),
        (10, "bayer:bayer_scale=3"),
        (14, "bayer:bayer_scale=2"),
        (20, "none"),
    ):
        data = b"".join(f.tobytes() for f in stabilise(frames, threshold))
        graph = (
            "[0:v]split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
            f"[b][p]paletteuse=dither={dither}:diff_mode=rectangle"
        )
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}"]
        cmd += ["-r", f"100/{delay_cs}", "-i", "-", "-filter_complex", graph, "-loop", "0", str(path)]
        subprocess.run(cmd, input=data, check=True)
        mb = path.stat().st_size / 2**20
        print(f"  gif: threshold {threshold}, dither {dither}: {mb:.2f} MB")
        if mb <= max_mb:
            break
    return path


def make_gif(fo: Footage, path: Path, seconds: float = 7.0, xfade: float = 0.7) -> Path:
    """A looping side-by-side render: the last ``xfade`` seconds dissolve into the moments before the first frame.

    The clip is sampled at half the source frame rate (12 to 15 frames per second) and played with a uniform delay of
    whole centiseconds, the timing unit of GIF.
    """
    render = side_by_side(fo, crop=GIF_CROP.get(fo.name))
    fps = fo.fps / 2
    delay = int(round(100 / fps))
    start = busiest_start(fo.run, seconds + xfade, lo=xfade) + xfade
    n, m = int(round(seconds * fps)), int(round(xfade * fps))
    head = [render(start - (m - j) / fps) for j in range(m)]
    frames = []
    for i in range(n):
        f = render(start + i / fps)
        j = i - (n - m)
        if j >= 0:
            f = blend(f, head[j], ease((j + 1) / (m + 1)))
        frames.append(f)
    return write_gif(frames, path, delay_cs=delay)


def write_credits(path: Path, gif_scene: str) -> Path:
    """``CREDITS.md``: the footage sources and licences of the trailer, its poster and the README GIF."""

    def block(names: list[str]) -> str:
        return "\n".join(f"- {attribution(n)} (sample `{n}`)" for n in names)

    trailer = sources([DHAKA, ROUND, MOTOR, DAY])
    same = ", ".join(f"`{n}`" for n in CLIPS if n.startswith("intersection"))
    text = f"""# Credits for the trailer, poster and README GIF

`trailer.mp4`, `trailer_poster.jpg` and `readme_hero.gif` are rendered by `tools/make_trailer.py` from the
public sample clips of `tachymetry.data`. The clips were excerpted, resized and annotated with Tachymetry's
output (3D boxes, speeds, bird's-eye views and calibration grids). The footage is licensed CC BY 4.0, CC0 1.0
and MIT as listed below; the three files are released under CC BY 4.0 with these attributions.

## trailer.mp4

{block(trailer)}

The intersection appears in five clips of the same dataset: {same}.

## trailer_poster.jpg

{block([DHAKA])}

## readme_hero.gif

{block([gif_scene])}

The MIT licence of Small-Crowd-and-Traffic-Video-Footages requires its copyright and permission notice to be kept
with copies of the footage; the notice is on the dataset page linked above.
"""
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------- main
def load_footage(runs_dir: Path, device: str) -> dict[str, Footage]:
    """Run the pipeline on each sample (cached detections and cameras in ``runs_dir`` are reused)."""
    footage = {}
    for name in CLIPS:
        t0 = time.time()
        cfg = PipelineConfig().override(sample_overrides(name))
        run = Pipeline(cfg, camera=sample_camera(name) or "auto", device=device).run(
            sample_video(name), runs_dir / name
        )
        footage[name] = Footage(name, run, cfg)
        print(f"  {name}: {run.summary()['tracks']} tracks ({time.time() - t0:.1f} s)")
    return footage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media")
    ap.add_argument(
        "--runs", type=Path, default=ROOT / "runs" / "trailer", help="pipeline run directories (detections are cached)"
    )
    ap.add_argument("--device", default="auto", help="device for detection when no cached detections exist")
    ap.add_argument("--crf", type=int, default=25)
    ap.add_argument(
        "--only", default="video,poster,gif,credits", help="comma-separated: video, poster, gif, credits, stills"
    )
    ap.add_argument("--stills", default="2,6,11,16,20,24,30,35,40", help="times (s) for --only stills")
    ap.add_argument("--stills-dir", type=Path, default=None)
    ap.add_argument("--poster-time", type=float, default=POSTER_TIME)
    ap.add_argument("--gif-scene", default=GIF_SCENE, choices=CLIPS)
    args = ap.parse_args()
    jobs = {s.strip() for s in args.only.split(",") if s.strip()}
    args.out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("pipeline runs")
    footage = load_footage(args.runs, args.device)
    trailer = Trailer(footage)
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
        print(f"{path}: {path.stat().st_size / 2**20:.1f} MB, {n / FPS:.1f} s")
    if "poster" in jobs:
        path = args.out / "trailer_poster.jpg"
        cv2.imwrite(str(path), trailer.frame(args.poster_time, poster=True), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{path}: {path.stat().st_size / 2**10:.0f} KB")
    if "gif" in jobs:
        path = make_gif(footage[args.gif_scene], args.out / "readme_hero.gif")
        print(f"{path}: {path.stat().st_size / 2**20:.2f} MB")
    if "credits" in jobs:
        print(write_credits(args.out / "CREDITS.md", args.gif_scene))
    print(f"done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()

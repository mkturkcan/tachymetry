"""Annotated video and still rendering."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RenderConfig
from ..io.video import VideoSource, prefetch
from ..io.writer import VideoWriter
from .bev import BEVCanvas
from .overlay import draw_camera_overlay
from .state import StateIndex
from .style import CLASS_COLORS, INK, MUTED, PLURALS, TEXT
from .text import draw_text, text_size

__all__ = ["Renderer", "render_video", "render_frame"]

LAYOUTS = ("side", "dashboard", "camera", "bev")


def _bev_points(result: Any) -> np.ndarray | None:
    pts = [tr.xy for tr in result.tracks.ok if tr.state == "moving"] or [tr.xy for tr in result.tracks.ok]
    return np.vstack(pts) if pts else None


def _bev_window(result: Any, min_span: float = 30.0) -> tuple[np.ndarray, float]:
    pts = [tr.xy for tr in result.tracks.ok if tr.state == "moving"]
    if not pts:
        pts = [tr.xy for tr in result.tracks.ok]
    cam = result.camera
    if pts:
        p = np.vstack(pts)
        lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
        center = (lo + hi) / 2
        span = float(max(hi - lo)) + 12.0
    else:
        from ..calib.preview import bev_extent

        x0, x1, y0, y1 = bev_extent(cam)
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        span = max(x1 - x0, y1 - y0)
    return center, max(span, min_span)


class Renderer:
    """Renders frames of a :class:`~tachymetry.results.RunResult` in a chosen layout."""

    def __init__(self, result: Any, cfg: RenderConfig | None = None, layout: str | None = None):
        self.result = result
        self.cfg = cfg or RenderConfig()
        self.layout = layout or self.cfg.layout
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        self.camera = result.camera
        w, h = self.camera.image_size
        self.src_size = (w, h)
        self.index = StateIndex(list(result.tracks.tracks), trail_s=self.cfg.trail_s)
        self.title = self.cfg.title or Path(result.source).stem
        self.draw = self.cfg.draw
        if self.draw == "auto":
            self.draw = "cuboid"
        center, span = _bev_window(result)
        if self.cfg.bev_range_m:
            span = float(self.cfg.bev_range_m)
        mw = self.cfg.max_width
        self.header_h = 0
        if self.layout == "side":
            s = min(1.0, (mw - 2) / (w + h))
            self.cam_disp = (int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2)
            bsz = self.cam_disp[1]
            self.bev = BEVCanvas(
                self.camera,
                result.background,
                (bsz, bsz),
                center,
                span,
                fit_points=None if self.cfg.bev_range_m else _bev_points(result),
            )
            self.header_h = max(40, int(round(0.06 * self.cam_disp[1])) // 2 * 2)
            self.size = (self.cam_disp[0] + 2 + bsz, self.cam_disp[1] + self.header_h)
        elif self.layout == "dashboard":
            W, H = 1920, 1080
            cw = 1440
            ch = int(round(cw * h / w)) // 2 * 2
            if ch > 860:
                ch = 860
                cw = int(round(ch * w / h)) // 2 * 2
            self.cam_disp = (cw, ch)
            self.bev = BEVCanvas(
                self.camera,
                result.background,
                (W - cw - 2, H),
                center,
                span * 1.15,
                fit_points=None if self.cfg.bev_range_m else _bev_points(result),
            )
            self.size = (W, H)
        elif self.layout == "camera":
            s = min(1.0, mw / w)
            self.cam_disp = (int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2)
            self.header_h = max(40, int(round(0.06 * self.cam_disp[1])) // 2 * 2)
            self.bev = None
            self.size = (self.cam_disp[0], self.cam_disp[1] + self.header_h)
        else:
            side = min(mw, 1080)
            self.cam_disp = (0, 0)
            self.bev = BEVCanvas(
                self.camera,
                result.background,
                (side, side),
                center,
                span,
                fit_points=None if self.cfg.bev_range_m else _bev_points(result),
            )
            self.size = (side, side)
        self.counted: dict[int, str] = {}
        self._starts = sorted((float(tr.t[0]), tr.id, tr.cls) for tr in result.tracks.ok if tr.state == "moving")

    # ------------------------------------------------------------------ parts
    def _counts(self, t: float) -> Counter:
        c: Counter = Counter()
        for t0, _, cls in self._starts:
            if t0 > t:
                break
            c[cls] += 1
        return c

    def _header(self, width: int, t: float) -> np.ndarray:
        h = self.header_h
        bar = np.full((h, width, 3), INK, np.uint8)
        fs = max(14, int(round(h * 0.42)))
        draw_text(bar, "Tachymetry", (int(h * 0.45), int((h - fs) / 2) - 2), size=fs, color=TEXT, weight="semibold")
        tw, _ = text_size("Tachymetry", fs, "semibold")
        draw_text(
            bar,
            self.title,
            (int(h * 0.45) + tw + int(h * 0.5), int((h - fs * 0.8) / 2) - 1),
            size=int(fs * 0.8),
            color=MUTED,
        )
        counts = self._counts(t)
        parts = [f"{n} {PLURALS.get(k, k) if n != 1 else k}" for k, n in counts.most_common(4)]
        right = "   ".join(parts) + (f"      {int(t // 60):02d}:{t % 60:04.1f}")
        rw, _ = text_size(right, int(fs * 0.8))
        draw_text(
            bar, right, (width - rw - int(h * 0.45), int((h - fs * 0.8) / 2) - 1), size=int(fs * 0.8), color=MUTED
        )
        bar[-1:, :] = (34, 30, 28)
        return bar

    def _dashboard_strip(self, width: int, height: int, t: float, states: list) -> np.ndarray:
        import cv2

        strip = np.full((height, width, 3), INK, np.uint8)
        pad = int(height * 0.12)
        draw_text(
            strip, "Tachymetry", (pad, int(height * 0.10)), size=int(height * 0.16), color=TEXT, weight="semibold"
        )
        draw_text(strip, self.title, (pad, int(height * 0.34)), size=int(height * 0.085), color=MUTED)
        counts = self._counts(t)
        x = pad
        y = int(height * 0.58)
        for k, n in counts.most_common(5):
            col = CLASS_COLORS.get(k, (200, 200, 200))
            cv2.circle(strip, (x + 7, y + int(height * 0.05)), 7, col, -1, cv2.LINE_AA)
            label = f"{n} {PLURALS.get(k, k) if n != 1 else k}"
            draw_text(strip, label, (x + 22, y), size=int(height * 0.085), color=TEXT)
            x += 22 + text_size(label, int(height * 0.085))[0] + 34
        # live speed distribution of moving vehicles
        sp = [s.speed_kmh for s in states if s.track.super_cls == "vehicle" and s.speed_kmh > 3]
        hx0, hx1 = int(width * 0.58), width - pad
        hy0, hy1 = int(height * 0.18), int(height * 0.80)
        draw_text(
            strip,
            f"Vehicle speeds now ({self.cfg.units})",
            (hx0, int(height * 0.06)),
            size=int(height * 0.075),
            color=MUTED,
        )
        bins = np.arange(0, 85, 5)
        hist = np.histogram(np.clip(sp, 0, 79.9), bins=bins)[0] if sp else np.zeros(len(bins) - 1)
        bw = (hx1 - hx0) / len(hist)
        mx = max(1, hist.max())
        for i, v in enumerate(hist):
            bh = (hy1 - hy0) * v / mx
            cv2.rectangle(
                strip,
                (int(hx0 + i * bw + 2), int(hy1 - bh)),
                (int(hx0 + (i + 1) * bw - 2), hy1),
                (246, 150, 96),
                -1,
                cv2.LINE_AA,
            )
        cv2.line(strip, (hx0, hy1), (hx1, hy1), (70, 64, 62), 1, cv2.LINE_AA)
        for v in (0, 20, 40, 60, 80):
            xx = int(hx0 + (v / 5) * bw)
            draw_text(strip, f"{v}", (xx - 6, hy1 + 6), size=int(height * 0.06), color=MUTED)
        return strip

    # ----------------------------------------------------------------- frames
    def frame(self, image: np.ndarray, t: float) -> np.ndarray:
        import cv2

        states = self.index.at(t)
        cfg = self.cfg
        parts = []
        if self.layout != "bev":
            cam_img = draw_camera_overlay(
                image.copy(), self.camera, states, self.draw, cfg.units, cfg.show_parked, cfg.show_ids
            )
            cam_img = cv2.resize(cam_img, self.cam_disp, interpolation=cv2.INTER_AREA)
            parts.append(cam_img)
        bev_img = self.bev.render(states, cfg.show_parked) if self.bev is not None else None
        if self.layout == "side":
            body = np.concatenate([parts[0], np.full((self.cam_disp[1], 2, 3), INK, np.uint8), bev_img], axis=1)
            return np.concatenate([self._header(body.shape[1], t), body], axis=0)
        if self.layout == "camera":
            return np.concatenate([self._header(parts[0].shape[1], t), parts[0]], axis=0)
        if self.layout == "bev":
            return bev_img
        W, H = self.size
        canvas = np.full((H, W, 3), INK, np.uint8)
        cw, ch = self.cam_disp
        canvas[:ch, :cw] = parts[0]
        canvas[:, cw + 2 :] = bev_img
        canvas[ch : ch + 2, :cw] = (34, 30, 28)
        canvas[ch + 2 :, :cw] = self._dashboard_strip(cw, H - ch - 2, t, states)
        return canvas


def render_video(
    result: Any,
    path: str | Path,
    layout: str | None = None,
    cfg: RenderConfig | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
    fps: float | None = None,
    crf: int | None = None,
) -> Path:
    """Render an annotated MP4 of ``result`` (H.264, ``yuv420p``)."""
    cfg = cfg or RenderConfig()
    r = Renderer(result, cfg, layout)
    path = Path(path)
    with VideoSource(result.source) as src:
        out_fps = fps or cfg.fps or src.fps
        stride = max(1, int(round(src.fps / out_fps))) if out_fps < src.fps else 1
        first = int(round(start_s * src.fps))
        last = int(round(end_s * src.fps)) if end_s is not None else None
        total = ((last if last is not None else max(src.frame_count, 0)) - first) // stride
        with VideoWriter(path, src.fps / stride, crf if crf is not None else cfg.crf) as wr:
            for k, (_, t, img) in enumerate(prefetch(src.frames(stride=stride, start=first, end=last), depth=8)):
                wr.write(r.frame(img, t))
                if progress and k % 10 == 0:
                    progress("render", k, max(total, k + 1))
    return path


def render_frame(
    result: Any, time_s: float, layout: str | None = "side", cfg: RenderConfig | None = None
) -> np.ndarray:
    """A single annotated frame (for figures and previews)."""
    r = Renderer(result, cfg, layout)
    with VideoSource(result.source) as src:
        idx = int(round(time_s * src.fps))
        img = src.read(idx)
        if img is None:
            raise ValueError(f"cannot read frame at {time_s} s")
        return r.frame(img, idx / src.fps)

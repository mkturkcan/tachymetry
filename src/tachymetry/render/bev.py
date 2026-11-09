"""Bird's-eye-view canvas: orthorectified background, metric grid, footprints and trails."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..geometry.camera import Camera
from .state import TrackState
from .style import AMBER, CYAN, FAINT, HALO, INK, MUTED, blend_poly, color_for, polyline_aa
from .text import draw_text

__all__ = ["BEVCanvas"]


class BEVCanvas:
    """A fixed metric view of the ground, oriented with the camera's forward direction up.

    ``center``/``span`` select the ground window (metres); ``size`` the canvas
    in pixels. The background is the orthorectified traffic-free image.
    """

    def __init__(
        self,
        camera: Camera,
        background: np.ndarray | None,
        size: tuple[int, int],
        center: np.ndarray,
        span: float,
        darken: float = 0.62,
        grid: float | None = None,
        fit_points: np.ndarray | None = None,
        margin: float = 0.08,
    ):
        import cv2

        self.camera = camera
        self.W, self.H = size
        self.center = np.asarray(center, dtype=np.float64)
        self.mpp = span / min(self.W, self.H)
        fwd = camera.ground_up()
        # canvas up (-v) = the image's up direction on the ground; canvas right (+u) = up rotated clockwise
        self.up = fwd
        self.right = np.array([fwd[1], -fwd[0]])
        if fit_points is not None and len(fit_points) >= 2:
            # fit the window to the data in the rotated canvas frame
            p = np.asarray(fit_points, dtype=np.float64)
            a, b = p @ self.right, p @ self.up
            lo_a, hi_a = np.percentile(a, [1, 99])
            lo_b, hi_b = np.percentile(b, [1, 99])
            ext_a, ext_b = max(hi_a - lo_a, 20.0), max(hi_b - lo_b, 20.0)
            self.mpp = max(ext_a / self.W, ext_b / self.H) * (1 + 2 * margin)
            self.center = self.right * (lo_a + hi_a) / 2 + self.up * (lo_b + hi_b) / 2
        uu, vv = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        world = self.to_world(np.stack([uu.ravel(), vv.ravel()], 1))
        base = np.full((self.H, self.W, 3), INK, np.uint8)
        if background is not None:
            uv = camera.ground_to_image(world)
            mx = uv[:, 0].reshape(self.H, self.W).astype(np.float32)
            my = uv[:, 1].reshape(self.H, self.W).astype(np.float32)
            bad = ~np.isfinite(mx) | ~np.isfinite(my)
            mx[bad], my[bad] = -1, -1
            warped = cv2.remap(background, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=INK)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)[..., None].astype(np.float32)
            desat = 0.55 * warped.astype(np.float32) + 0.45 * gray
            base = np.clip(desat * darken + np.array(INK, np.float32) * (1 - darken), 0, 255).astype(np.uint8)
            # fade with ground sampling distance (far, smeared ground recedes)
            gsd = camera.ground_sampling_distance(uv)
            fade = np.clip(1.4 - gsd.reshape(self.H, self.W) / 0.35, 0, 1)[..., None]
            fade[bad] = 0
            base = (base * fade + np.array(INK, np.float32) * (1 - fade)).astype(np.uint8)
        self.base = base
        self._draw_grid(grid)

    # ------------------------------------------------------------ transforms
    def to_canvas(self, xy: np.ndarray) -> np.ndarray:
        d = np.asarray(xy, dtype=np.float64).reshape(-1, 2) - self.center
        u = d @ self.right / self.mpp + self.W / 2
        v = -(d @ self.up) / self.mpp + self.H / 2
        return np.column_stack([u, v])

    def to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        a = (uv[:, 0] - self.W / 2) * self.mpp
        b = -(uv[:, 1] - self.H / 2) * self.mpp
        return self.center + a[:, None] * self.right + b[:, None] * self.up

    # ----------------------------------------------------------------- static
    def _draw_grid(self, grid: float | None) -> None:
        import cv2

        span = self.mpp * min(self.W, self.H)
        step = grid or (5.0 if span <= 60 else 10.0 if span <= 150 else 25.0)
        corners = self.to_world(np.array([[0, 0], [self.W, 0], [0, self.H], [self.W, self.H]]))
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        over = self.base.copy()
        for x in np.arange(math.floor(lo[0] / step) * step, hi[0] + step, step):
            p = self.to_canvas(np.array([[x, lo[1]], [x, hi[1]]]))
            cv2.line(
                over, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), (58, 52, 50), 1, cv2.LINE_AA
            )
        for y in np.arange(math.floor(lo[1] / step) * step, hi[1] + step, step):
            p = self.to_canvas(np.array([[lo[0], y], [hi[0], y]]))
            cv2.line(
                over, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), (58, 52, 50), 1, cv2.LINE_AA
            )
        self.base = cv2.addWeighted(over, 0.55, self.base, 0.45, 0)
        # field of view outline
        w, h = self.camera.image_size
        border = np.concatenate(
            [
                np.column_stack([np.linspace(0, w - 1, 64), np.full(64, h - 1)]),
                np.column_stack([np.full(32, w - 1), np.linspace(h - 1, 0, 32)]),
                np.column_stack([np.linspace(w - 1, 0, 64), np.zeros(64)]),
                np.column_stack([np.zeros(32), np.linspace(0, h - 1, 32)]),
            ]
        )
        g = self.camera.image_to_ground(border)
        ok = np.all(np.isfinite(g), axis=1)
        if ok.sum() > 8:
            c = self.to_canvas(g[ok])
            c = c[(np.abs(c[:, 0]) < 5 * self.W) & (np.abs(c[:, 1]) < 5 * self.H)]
            if len(c) > 2:
                polyline_aa(self.base, c, (70, 64, 62), 1, closed=False, halo=False)
        # camera position (clamped to the canvas edge when it lies outside)
        cpx = self.to_canvas(self.camera.center[:2][None])[0]
        cpx = np.clip(cpx, [14, 14], [self.W - 14, self.H - 14])
        # scale bar: bottom left, else bottom right, else top left, whichever the camera chevron leaves clear
        bar_m = step * (2 if span > 40 else 1)
        px = bar_m / self.mpp
        group_w = px + 10 + 56  # bar, gap, label

        def clear(x: float, y: float) -> bool:
            return not (x - 16 < cpx[0] < x + group_w + 16 and y - 30 < cpx[1] < y + 22)

        spots = [(22, self.H - 26), (int(self.W - 22 - group_w), self.H - 26), (22, 30)]
        x0, y0 = next((s for s in spots if clear(*s)), spots[0])
        cv2.line(self.base, (x0, y0), (int(x0 + px), y0), (210, 210, 214), 2, cv2.LINE_AA)
        for xx in (x0, int(x0 + px)):
            cv2.line(self.base, (xx, y0 - 5), (xx, y0 + 5), (210, 210, 214), 2, cv2.LINE_AA)
        draw_text(self.base, f"{bar_m:g} m", (int(x0 + px + 10), y0 - 9), size=14, color=MUTED)
        # camera chevron
        chev = np.array([[0, -11], [8, 7], [0, 3], [-8, 7]], np.float64) + cpx
        cv2.fillPoly(self.base, [np.round(chev * 16).astype(np.int32)], CYAN, cv2.LINE_AA, shift=4)

    # ---------------------------------------------------------------- frames
    def render(self, states: Sequence[TrackState], show_parked: bool = True) -> np.ndarray:
        import cv2

        img = self.base.copy()
        for s in states:  # trails first
            tr = s.track
            if tr.state == "parked" or len(s.trail) < 3:
                continue
            p = self.to_canvas(s.trail)
            col = np.array(color_for(tr.cls), np.float64)
            n = len(p)
            for k in range(1, n):
                a = 0.15 + 0.85 * k / n
                c = tuple(int(v) for v in (col * a + np.array(INK) * (1 - a)))
                cv2.line(
                    img, tuple(np.round(p[k - 1]).astype(int)), tuple(np.round(p[k]).astype(int)), c, 2, cv2.LINE_AA
                )
        for s in states:
            tr = s.track
            parked = tr.state == "parked"
            if parked and not show_parked:
                continue
            col = color_for(tr.cls, parked)
            if tr.super_cls == "person":
                c = self.to_canvas(s.xy[None])[0]
                r = max(3, int(round(0.45 / self.mpp)))
                cv2.circle(img, tuple(np.round(c).astype(int)), r + 1, HALO, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(np.round(c).astype(int)), r, col, -1, cv2.LINE_AA)
                continue
            cs, sn = math.cos(s.heading), math.sin(s.heading)
            f = np.array([cs, sn]) * tr.length / 2
            l = np.array([-sn, cs]) * tr.width / 2
            quad = np.array([s.xy + f + l, s.xy - f + l, s.xy - f - l, s.xy + f - l])
            q = self.to_canvas(quad)
            if parked:
                blend_poly(img, q, col, 0.25)
                polyline_aa(img, q, col, 1, closed=True, halo=False)
            else:
                blend_poly(img, q, col, 0.85)
                polyline_aa(img, q, tuple(min(255, int(v * 1.15)) for v in col), 1, closed=True, halo=True)
                nose = self.to_canvas(np.array([s.xy + f * 0.98 + l * 0.7, s.xy + f * 0.98 - l * 0.7]))
                cv2.line(
                    img,
                    tuple(np.round(nose[0]).astype(int)),
                    tuple(np.round(nose[1]).astype(int)),
                    (240, 240, 244),
                    1,
                    cv2.LINE_AA,
                )
        _ = (AMBER, FAINT)
        return img

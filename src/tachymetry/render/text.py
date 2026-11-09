"""Anti-aliased text and label chips (PIL), with the bundled Geist typeface."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

__all__ = ["font", "draw_text", "text_size", "draw_chip"]

_WEIGHTS = {
    "regular": "Geist-Regular.ttf",
    "medium": "Geist-Medium.ttf",
    "semibold": "Geist-SemiBold.ttf",
    "mono": "GeistMono-Regular.ttf",
}


@lru_cache(maxsize=64)
def font(size: int, weight: str = "regular") -> Any:
    """A PIL font of the bundled Geist family (falls back to PIL's default font)."""
    from PIL import ImageFont

    name = _WEIGHTS.get(weight, _WEIGHTS["regular"])
    try:
        path = resources.files("tachymetry.assets").joinpath(name)
        with resources.as_file(path) as p:
            return ImageFont.truetype(str(p), size)
    except Exception:  # pragma: no cover - font missing from a broken install
        return ImageFont.load_default()


def text_size(text: str, size: int, weight: str = "regular") -> tuple[int, int]:
    f = font(size, weight)
    l, t, r, b = f.getbbox(text)
    return int(r - l), int(b - t)


def draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    size: int = 16,
    color: tuple[int, int, int] = (236, 235, 232),
    weight: str = "regular",
    anchor: str = "la",
) -> np.ndarray:
    """Draw ``text`` at ``org`` (top-left by default) on a BGR image; returns the image."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img[..., ::-1])
    d = ImageDraw.Draw(pil)
    d.text(org, text, font=font(size, weight), fill=tuple(int(c) for c in color[::-1]), anchor=anchor)
    img[...] = np.asarray(pil)[..., ::-1]
    return img


def draw_chip(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    size: int = 14,
    fg: tuple[int, int, int] = (13, 11, 10),
    bg: tuple[int, int, int] = (236, 235, 232),
    weight: str = "medium",
    pad: tuple[int, int] = (6, 3),
    radius: int | None = None,
    alpha: float = 0.92,
) -> tuple[int, int, int, int]:
    """Rounded label chip with its bottom-left corner at ``org``. Returns the chip box ``x0, y0, x1, y1``."""
    from PIL import Image, ImageDraw

    tw, th = text_size(text, size, weight)
    f = font(size, weight)
    asc, desc = f.getmetrics()
    box_h = asc + desc + 2 * pad[1] - 2
    x0, y1 = int(org[0]), int(org[1])
    x1, y0 = x0 + tw + 2 * pad[0], y1 - box_h
    h, w = img.shape[:2]
    if x1 > w:
        x0, x1 = w - (x1 - x0), w
    if y0 < 0:
        y0, y1 = 0, box_h
    x0, y0 = max(0, x0), max(0, y0)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return x0, y0, x1, y1
    pil = Image.fromarray(np.ascontiguousarray(roi[..., ::-1]))
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r = radius if radius is not None else box_h // 2
    d.rounded_rectangle([0, 0, pil.size[0] - 1, pil.size[1] - 1], r, fill=tuple(bg[::-1]) + (int(255 * alpha),))
    d.text((pad[0], pad[1] - 1), text, font=f, fill=tuple(fg[::-1]) + (255,))
    pil = Image.alpha_composite(pil.convert("RGBA"), layer).convert("RGB")
    img[y0:y1, x0:x1] = np.asarray(pil)[..., ::-1]
    return x0, y0, x1, y1

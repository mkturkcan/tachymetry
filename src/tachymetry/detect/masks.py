"""Instance-mask crops from low-resolution mask maps.

Segmenting detectors predict one mask map per detection at a fraction of the
image resolution (RF-DETR-Seg: a quarter of its input, SAM 3: 288 x 288).
Upsampling each map to the full frame and cropping it to its box afterwards
costs one full-frame plane per detection; :func:`mask_crops` samples only the
pixels inside each crop window and returns the same crops.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..detections import MaskCrop, crop_window

__all__ = ["crop_windows", "mask_crops"]


def crop_windows(xyxy: np.ndarray, image_size: tuple[int, int], pad: int = 2) -> np.ndarray:
    """``(N, 4)`` crop windows ``x0, y0, x1, y1`` (end exclusive) of :func:`~tachymetry.detections.crop_mask`.

    ``image_size`` is ``(width, height)``.
    """
    out = np.zeros((len(xyxy), 4), np.int64)
    for i, b in enumerate(np.asarray(xyxy, np.float64).reshape(-1, 4)):
        out[i] = crop_window(b, (image_size[1], image_size[0]), pad)
    return out


def mask_crops(
    maps: Any,
    xyxy: np.ndarray,
    image_size: tuple[int, int],
    threshold: float = 0.0,
    pad: int = 2,
    max_pixels: int = 1 << 23,
) -> list[MaskCrop | None]:
    """Binary full-resolution mask crops from ``(N, h, w)`` mask maps that cover the whole image.

    The result equals bilinear upsampling of each map to ``image_size``
    (``(width, height)``; ``align_corners=False``), thresholding with
    ``> threshold`` and :func:`~tachymetry.detections.crop_mask` with ``pad``,
    but only the pixels inside each crop window are sampled. ``maps`` is a
    ``torch`` tensor on the CPU or a GPU (logits with ``threshold=0`` or
    probabilities with ``threshold=0.5``). Detections are processed in chunks
    of at most ``max_pixels`` sampled pixels; empty masks give ``None``.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    n = len(xyxy)
    if n == 0:
        return []
    W, H = int(image_size[0]), int(image_size[1])
    win = crop_windows(xyxy, (W, H), pad)
    hw = np.stack([win[:, 3] - win[:, 1], win[:, 2] - win[:, 0]], axis=1)  # (h, w) per crop
    out: list[MaskCrop | None] = [None] * n
    valid = np.flatnonzero((hw[:, 0] > 0) & (hw[:, 1] > 0))
    if not len(valid):
        return out
    maps = maps.reshape(n, 1, maps.shape[-2], maps.shape[-1])
    dev = maps.device
    order = valid[np.argsort(-(hw[valid, 0] * hw[valid, 1]), kind="stable")]  # largest first
    start = 0
    while start < len(order):
        # grow the chunk while the padded sampling grid stays within max_pixels
        hmax, wmax = hw[order[start]]
        stop = start + 1
        while stop < len(order):
            h2, w2 = max(hmax, hw[order[stop], 0]), max(wmax, hw[order[stop], 1])
            if (stop - start + 1) * h2 * w2 > max_pixels:
                break
            hmax, wmax, stop = h2, w2, stop + 1
        idx = order[start:stop]
        k = len(idx)
        x0 = torch.as_tensor(win[idx, 0], dtype=torch.float32, device=dev)
        y0 = torch.as_tensor(win[idx, 1], dtype=torch.float32, device=dev)
        jj = torch.arange(int(wmax), dtype=torch.float32, device=dev)
        ii = torch.arange(int(hmax), dtype=torch.float32, device=dev)
        # normalised image coordinates of the pixel centres (align_corners=False)
        gx = (2.0 * (x0[:, None] + jj[None] + 0.5) / W - 1.0)[:, None, :].expand(k, int(hmax), int(wmax))
        gy = (2.0 * (y0[:, None] + ii[None] + 0.5) / H - 1.0)[:, :, None].expand(k, int(hmax), int(wmax))
        src = maps[torch.as_tensor(idx, device=dev)].float()
        sampled = F.grid_sample(
            src, torch.stack([gx, gy], dim=-1), mode="bilinear", padding_mode="border", align_corners=False
        )[:, 0]
        inside = (ii[None, :, None] < torch.as_tensor(hw[idx, 0], device=dev)[:, None, None]) & (
            jj[None, None, :] < torch.as_tensor(hw[idx, 1], device=dev)[:, None, None]
        )
        flat = (sampled > threshold)[inside].cpu().numpy()  # crops concatenated in row-major order
        pos = 0
        for i in idx:
            h, w = int(hw[i, 0]), int(hw[i, 1])
            m = flat[pos : pos + h * w].reshape(h, w)
            pos += h * w
            if m.any():
                out[i] = (int(win[i, 0]), int(win[i, 1]), m)
        start = stop
    return out

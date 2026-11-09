"""Lens distortion models.

Each model maps between *raw* pixel coordinates (as recorded by the camera) and
*rectified* pixel coordinates (an ideal pinhole image of the same size). All
models are radially symmetric about a distortion centre ``center`` (pixels) and
use a radial unit ``scale`` (pixels), so parameters are comparable across image
resolutions.

Models
------
``identity``      no distortion.
``division``      Fitzgibbon's one-parameter division model,
                  ``r_u = r_d / (1 + k r_d^2)``. Closed-form in both directions;
                  well suited to plumb-line estimation.
``equidistant``   Equidistant fisheye rectified to a pinhole of equal focal
                  length, ``r_u = s tan(r_d / s)``. One parameter (``s``, pixels).
                  The single most transferable model for wide surveillance lenses.
``brown``         Brown–Conrady radial-tangential (OpenCV ``k1 k2 p1 p2 k3``).
``fisheye``       Kannala–Brandt equidistant polynomial (OpenCV fisheye ``k1..k4``).

The radial unit for ``division`` and ``brown``/``fisheye`` is ``scale``, which
defaults to half the image diagonal (``division``) or the focal length
(``brown``, ``fisheye``, OpenCV convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "Lens",
    "IdentityLens",
    "DivisionLens",
    "EquidistantLens",
    "BrownConradyLens",
    "FisheyeLens",
    "lens_from_dict",
    "LENS_MODELS",
]


def _as_points(pts: Any) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    if arr.shape[-1] != 2:
        raise ValueError(f"expected (..., 2) points, got shape {arr.shape}")
    return arr


@dataclass
class Lens:
    """Base class. Subclasses implement :meth:`rectify` and :meth:`distort`."""

    image_size: tuple[int, int]  # (width, height)
    center: tuple[float, float] | None = None

    kind: ClassVar[str] = "base"
    n_params: ClassVar[int] = 0
    _maps: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.center is None:
            w, h = self.image_size
            self.center = ((w - 1) / 2.0, (h - 1) / 2.0)
        self.center = (float(self.center[0]), float(self.center[1]))
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))

    # -- point mapping -------------------------------------------------
    def rectify(self, pts: Any) -> np.ndarray:
        """Raw pixel coordinates -> rectified (pinhole) pixel coordinates."""
        raise NotImplementedError

    def distort(self, pts: Any) -> np.ndarray:
        """Rectified (pinhole) pixel coordinates -> raw pixel coordinates."""
        raise NotImplementedError

    @property
    def is_identity(self) -> bool:
        return False

    # -- parameters ----------------------------------------------------
    def params(self) -> np.ndarray:
        """Free parameters as a vector (used by estimators)."""
        return np.zeros(0)

    def with_params(self, p: np.ndarray) -> Lens:
        """Return a copy with free parameters ``p``."""
        return self

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box constraints for the free parameters (used by estimators)."""
        return np.zeros(0), np.zeros(0)

    def param_grid(self) -> list[np.ndarray]:
        """Initial values spanning the plausible range (multi-start estimation)."""
        return [self.params()]

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.kind, "image_size": list(self.image_size), "center": list(self.center)}

    # -- images ----------------------------------------------------------
    def rectify_image(self, image: np.ndarray, interpolation: int | None = None) -> np.ndarray:
        """Resample a raw image into the rectified pinhole image (same size)."""
        import cv2

        if self.is_identity:
            return image
        h, w = image.shape[:2]
        if (w, h) != self.image_size:
            raise ValueError(f"image is {w}x{h}, lens was defined for {self.image_size}")
        key = ("rectify", w, h)
        if key not in self._maps:
            uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            raw = self.distort(np.stack([uu.ravel(), vv.ravel()], axis=1))
            mx = raw[:, 0].reshape(h, w).astype(np.float32)
            my = raw[:, 1].reshape(h, w).astype(np.float32)
            self._maps[key] = (mx, my)
        mx, my = self._maps[key]
        return cv2.remap(
            image,
            mx,
            my,
            interpolation if interpolation is not None else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def distort_image(self, image: np.ndarray, interpolation: int | None = None) -> np.ndarray:
        """Resample a rectified image into raw (distorted) geometry (used by the synthetic renderer)."""
        import cv2

        if self.is_identity:
            return image
        h, w = image.shape[:2]
        key = ("distort", w, h)
        if key not in self._maps:
            uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            rect = self.rectify(np.stack([uu.ravel(), vv.ravel()], axis=1))
            mx = rect[:, 0].reshape(h, w).astype(np.float32)
            my = rect[:, 1].reshape(h, w).astype(np.float32)
            self._maps[key] = (mx, my)
        mx, my = self._maps[key]
        return cv2.remap(
            image,
            mx,
            my,
            interpolation if interpolation is not None else cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # -- helpers ---------------------------------------------------------
    def _radial(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center)
        d = pts - c
        r = np.linalg.norm(d, axis=-1)
        return d, r

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        args = ", ".join(
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in self.to_dict().items()
            if k not in ("model", "image_size", "center")
        )
        return f"{type(self).__name__}({args})"


@dataclass(repr=False)
class IdentityLens(Lens):
    """No distortion: raw and rectified pixels coincide."""

    kind: ClassVar[str] = "identity"

    def rectify(self, pts: Any) -> np.ndarray:
        return _as_points(pts).copy()

    def distort(self, pts: Any) -> np.ndarray:
        return _as_points(pts).copy()

    @property
    def is_identity(self) -> bool:
        return True


@dataclass(repr=False)
class DivisionLens(Lens):
    """Fitzgibbon (2001) single-parameter division model.

    ``r_u = r_d / (1 + k r_d^2)`` with radii in units of ``scale`` pixels
    (default: half image diagonal). ``k < 0`` corrects barrel distortion.
    """

    k: float = 0.0
    scale: float | None = None

    kind: ClassVar[str] = "division"
    n_params: ClassVar[int] = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scale is None:
            w, h = self.image_size
            self.scale = 0.5 * float(np.hypot(w, h))
        self.k = float(self.k)
        self.scale = float(self.scale)

    def rectify(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        d, r = self._radial(pts)
        rn = r / self.scale
        factor = 1.0 / (1.0 + self.k * rn**2)
        return np.asarray(self.center) + d * factor[..., None]

    def distort(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        d, r = self._radial(pts)
        ru = r / self.scale
        if abs(self.k) < 1e-12:
            return pts.copy()
        # Solve k ru rd^2 - rd + ru = 0 for the root continuous at k -> 0.
        disc = 1.0 - 4.0 * self.k * ru**2
        disc = np.where(disc < 0, np.nan, disc)
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = (1.0 - np.sqrt(disc)) / (2.0 * self.k * ru)
            rd = np.where(ru < 1e-12, ru, rd)
            ratio = np.where(ru < 1e-12, 1.0, rd / ru)
        return np.asarray(self.center) + d * ratio[..., None]

    @property
    def is_identity(self) -> bool:
        return abs(self.k) < 1e-12

    def params(self) -> np.ndarray:
        return np.array([self.k])

    def with_params(self, p: np.ndarray) -> DivisionLens:
        return DivisionLens(self.image_size, self.center, k=float(p[0]), scale=self.scale)

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-0.9]), np.array([0.6])

    def param_grid(self) -> list[np.ndarray]:
        return [np.array([k]) for k in (0.0, -0.1, -0.2, -0.35, -0.5, 0.1)]

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "k": self.k, "scale": self.scale}


@dataclass(repr=False)
class EquidistantLens(Lens):
    """Equidistant fisheye rectified to a pinhole: ``r_u = s tan(r_d / s)``.

    ``s`` (pixels) is the fisheye focal length; ``s -> inf`` is the identity.
    One parameter, monotone everywhere, and stable outside the annotated area,
    which makes it the recommended model for wide-angle surveillance cameras.
    """

    s: float = 1e9

    kind: ClassVar[str] = "equidistant"
    n_params: ClassVar[int] = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s = float(self.s)

    def rectify(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        d, r = self._radial(pts)
        theta = np.clip(r / self.s, 0.0, np.pi / 2 - 1e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(r < 1e-12, 1.0, self.s * np.tan(theta) / r)
        return np.asarray(self.center) + d * ratio[..., None]

    def distort(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        d, r = self._radial(pts)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(r < 1e-12, 1.0, self.s * np.arctan(r / self.s) / r)
        return np.asarray(self.center) + d * ratio[..., None]

    @property
    def is_identity(self) -> bool:
        w, h = self.image_size
        return self.s > 1e3 * np.hypot(w, h)

    def params(self) -> np.ndarray:
        # Optimise 1/s: identity is 0 and the cost is smooth there.
        return np.array([1.0 / self.s])

    def with_params(self, p: np.ndarray) -> EquidistantLens:
        inv = max(float(p[0]), 0.0)
        s = 1.0 / inv if inv > 1e-12 else 1e12
        return EquidistantLens(self.image_size, self.center, s=s)

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        half_diag = 0.5 * float(np.hypot(*self.image_size))
        return np.array([0.0]), np.array([1.6 / half_diag])  # s >= 0.63 half-diagonals (~ 180 deg fisheye)

    def param_grid(self) -> list[np.ndarray]:
        half_diag = 0.5 * float(np.hypot(*self.image_size))
        return [np.array([0.0])] + [np.array([1.0 / (m * half_diag)]) for m in (3.0, 1.8, 1.2, 0.85)]

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "s": self.s}


@dataclass(repr=False)
class BrownConradyLens(Lens):
    """Brown–Conrady radial-tangential model with OpenCV conventions.

    Radii are normalised by ``focal`` (pixels). Rectification inverts the
    forward model iteratively (as ``cv2.undistortPoints``).
    """

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    focal: float | None = None

    kind: ClassVar[str] = "brown"
    n_params: ClassVar[int] = 2  # k1, k2 are estimated by default; p1, p2, k3 on request

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.focal is None:
            self.focal = 0.5 * float(np.hypot(*self.image_size))
        self.focal = float(self.focal)

    def _forward_norm(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xd, yd

    def distort(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        c = np.asarray(self.center)
        x = (pts[..., 0] - c[0]) / self.focal
        y = (pts[..., 1] - c[1]) / self.focal
        xd, yd = self._forward_norm(x, y)
        return np.stack([xd * self.focal + c[0], yd * self.focal + c[1]], axis=-1)

    def rectify(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        c = np.asarray(self.center)
        xd = (pts[..., 0] - c[0]) / self.focal
        yd = (pts[..., 1] - c[1]) / self.focal
        x, y = xd.copy(), yd.copy()
        for _ in range(30):  # fixed-point iteration (OpenCV uses 5-10; we converge harder)
            r2 = x * x + y * y
            radial = 1 + self.k1 * r2 + self.k2 * r2**2 + self.k3 * r2**3
            dx = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            dy = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return np.stack([x * self.focal + c[0], y * self.focal + c[1]], axis=-1)

    @property
    def is_identity(self) -> bool:
        return all(abs(v) < 1e-12 for v in (self.k1, self.k2, self.p1, self.p2, self.k3))

    def params(self) -> np.ndarray:
        return np.array([self.k1, self.k2])

    def with_params(self, p: np.ndarray) -> BrownConradyLens:
        p = np.asarray(p, dtype=float)
        vals = dict(k1=self.k1, k2=self.k2, p1=self.p1, p2=self.p2, k3=self.k3)
        for name, v in zip(("k1", "k2", "p1", "p2", "k3"), p):
            vals[name] = float(v)
        return BrownConradyLens(self.image_size, self.center, focal=self.focal, **vals)

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-1.0, -1.0]), np.array([1.0, 1.0])

    def param_grid(self) -> list[np.ndarray]:
        return [np.array([k1, 0.0]) for k1 in (0.0, -0.1, -0.25, -0.4, 0.1)]

    def to_opencv(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(K, dist)`` usable with OpenCV functions."""
        K = np.array([[self.focal, 0, self.center[0]], [0, self.focal, self.center[1]], [0, 0, 1.0]])
        return K, np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
            "focal": self.focal,
        }


@dataclass(repr=False)
class FisheyeLens(Lens):
    """Kannala–Brandt model (OpenCV ``fisheye``), rectified to a pinhole of focal ``focal``.

    ``theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)``,
    ``r_d = focal * theta_d`` and ``r_u = focal * tan(theta)``.
    """

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    focal: float | None = None

    kind: ClassVar[str] = "fisheye"
    n_params: ClassVar[int] = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.focal is None:
            self.focal = 0.5 * float(np.hypot(*self.image_size))
        self.focal = float(self.focal)

    def _theta_d(self, theta: np.ndarray) -> np.ndarray:
        t2 = theta * theta
        return theta * (1 + self.k1 * t2 + self.k2 * t2**2 + self.k3 * t2**3 + self.k4 * t2**4)

    def distort(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        d, r = self._radial(pts)
        theta = np.arctan(r / self.focal)
        rd = self.focal * self._theta_d(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(r < 1e-12, 1.0, rd / r)
        return np.asarray(self.center) + d * ratio[..., None]

    def rectify(self, pts: Any) -> np.ndarray:
        pts = _as_points(pts)
        d, r = self._radial(pts)
        theta_d = r / self.focal
        theta = theta_d.copy()
        for _ in range(30):  # Newton iterations on theta_d(theta) = target
            t2 = theta * theta
            f = self._theta_d(theta) - theta_d
            df = 1 + 3 * self.k1 * t2 + 5 * self.k2 * t2**2 + 7 * self.k3 * t2**3 + 9 * self.k4 * t2**4
            theta = theta - f / np.where(np.abs(df) < 1e-9, 1e-9, df)
        theta = np.clip(theta, 0, np.pi / 2 - 1e-6)
        ru = self.focal * np.tan(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(r < 1e-12, 1.0, ru / r)
        return np.asarray(self.center) + d * ratio[..., None]

    @property
    def is_identity(self) -> bool:
        return False  # even with k = 0 the equidistant projection is not a pinhole

    def params(self) -> np.ndarray:
        return np.array([self.k1, self.k2])

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([-0.5, -0.5]), np.array([0.5, 0.5])

    def with_params(self, p: np.ndarray) -> FisheyeLens:
        p = np.asarray(p, dtype=float)
        vals = dict(k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4)
        for name, v in zip(("k1", "k2", "k3", "k4"), p):
            vals[name] = float(v)
        return FisheyeLens(self.image_size, self.center, focal=self.focal, **vals)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "k1": self.k1, "k2": self.k2, "k3": self.k3, "k4": self.k4, "focal": self.focal}


LENS_MODELS: dict[str, type[Lens]] = {
    cls.kind: cls for cls in (IdentityLens, DivisionLens, EquidistantLens, BrownConradyLens, FisheyeLens)
}
LENS_MODELS["none"] = IdentityLens
LENS_MODELS["tan"] = EquidistantLens


def lens_from_dict(d: dict[str, Any] | None, image_size: tuple[int, int] | None = None) -> Lens:
    """Build a lens from its :meth:`Lens.to_dict` representation."""
    if d is None:
        if image_size is None:
            raise ValueError("image_size is required for an identity lens")
        return IdentityLens(image_size)
    d = dict(d)
    model = d.pop("model", "identity")
    size = tuple(d.pop("image_size", image_size))  # type: ignore[arg-type]
    center = d.pop("center", None)
    try:
        cls = LENS_MODELS[model]
    except KeyError as exc:
        raise ValueError(f"unknown lens model {model!r}; choose from {sorted(LENS_MODELS)}") from exc
    return cls(size, tuple(center) if center is not None else None, **d)  # type: ignore[arg-type]


def make_lens(model: str, image_size: tuple[int, int], **kwargs: Any) -> Lens:
    """Construct a lens of the named model with default (identity-like) parameters."""
    cls = LENS_MODELS[model]
    if cls is EquidistantLens and "s" not in kwargs:
        kwargs["s"] = 1e9
    return cls(image_size, kwargs.pop("center", None), **kwargs)

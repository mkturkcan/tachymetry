"""Calibrated camera over a ground plane.

Conventions
-----------
* Image pixels ``(u, v)``: origin at the top-left pixel centre, ``u`` right,
  ``v`` down. *Raw* pixels are what the camera recorded; *rectified* pixels are
  the ideal pinhole image after removing lens distortion (same size).
* World frame: right-handed, metres, ``z`` up, ground plane ``z = 0``. When the
  camera is georeferenced ``x`` points east and ``y`` north; otherwise the
  calibration chooses a convenient yaw (typically ``x`` along the main road).
* Camera frame: OpenCV (``x`` right, ``y`` down, ``z`` forward).
* Extrinsics: ``X_cam = R @ X_world + t``; camera centre ``C = -R.T @ t``.
* Angles: ``yaw`` is the azimuth of the optical axis, counter-clockwise from
  world ``+x``; ``pitch`` is the depression below the horizon (positive = down);
  ``roll`` rotates the image clockwise about the optical axis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .lens import IdentityLens, Lens, lens_from_dict

__all__ = ["Camera", "rotation_from_angles", "angles_from_rotation", "CalibrationError"]

_UP = np.array([0.0, 0.0, 1.0])


class CalibrationError(RuntimeError):
    """Raised when a camera cannot be constructed from the given data."""


def rotation_from_angles(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """World-to-camera rotation for a camera with the given yaw, pitch and roll (degrees)."""
    psi, theta, phi = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    z = np.array([math.cos(theta) * math.cos(psi), math.cos(theta) * math.sin(psi), -math.sin(theta)])
    if abs(math.cos(theta)) < 1e-9:  # nadir view: yaw orients the image x axis
        x0 = np.array([math.sin(psi), -math.cos(psi), 0.0])
    else:
        x0 = np.cross(z, _UP)
        x0 /= np.linalg.norm(x0)
    y0 = np.cross(z, x0)
    x = math.cos(phi) * x0 + math.sin(phi) * y0
    y = -math.sin(phi) * x0 + math.cos(phi) * y0
    return np.stack([x, y, z], axis=0)


def angles_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rotation_from_angles`: ``(yaw, pitch, roll)`` in degrees."""
    z = R[2]
    theta = math.asin(float(np.clip(-z[2], -1.0, 1.0)))
    if abs(math.cos(theta)) < 1e-6:
        x = R[0]
        psi = math.atan2(x[0], -x[1])  # inverse of x0 = (sin psi, -cos psi, 0)
        return math.degrees(psi), math.degrees(theta), 0.0
    psi = math.atan2(z[1], z[0])
    x0 = np.cross(z, _UP)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(z, x0)
    phi = math.atan2(float(R[0] @ y0), float(R[0] @ x0))
    return math.degrees(psi), math.degrees(theta), math.degrees(phi)


@dataclass
class Camera:
    """A pinhole camera with lens distortion, posed above the ground plane ``z = 0``.

    Construct it with :meth:`from_pose`, :meth:`from_homography`,
    :meth:`load`, or through :mod:`tachymetry.calib`.
    """

    image_size: tuple[int, int]
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    lens: Lens = None  # type: ignore[assignment]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        if self.lens is None:
            self.lens = IdentityLens(self.image_size)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pose(
        cls,
        image_size: tuple[int, int],
        focal: float,
        height: float,
        pitch: float,
        yaw: float = 90.0,
        roll: float = 0.0,
        position: tuple[float, float] = (0.0, 0.0),
        principal_point: tuple[float, float] | None = None,
        lens: Lens | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Camera:
        """Camera at ``(x, y, height)`` with focal length ``focal`` (pixels) and angles in degrees."""
        w, h = image_size
        cx, cy = principal_point if principal_point is not None else ((w - 1) / 2.0, (h - 1) / 2.0)
        K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
        R = rotation_from_angles(yaw, pitch, roll)
        C = np.array([position[0], position[1], height], dtype=np.float64)
        return cls(image_size, K, R, -R @ C, lens=lens, meta=dict(meta or {}))

    @classmethod
    def from_homography(
        cls,
        H_ground_to_image: np.ndarray,
        image_size: tuple[int, int],
        lens: Lens | None = None,
        principal_point: tuple[float, float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Camera:
        """Recover ``K, R, t`` from a ground-to-(rectified)-image homography.

        Assumes square pixels and a known principal point (image centre by
        default); the focal length follows from the orthonormality of the first
        two rotation columns. Raises :class:`CalibrationError` if the homography
        is not induced by a real camera above the plane.
        """
        K, R, t = decompose_ground_homography(H_ground_to_image, image_size, principal_point)
        return cls(image_size, K, R, t, lens=lens, meta=dict(meta or {}))

    # ------------------------------------------------------------- properties
    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height_px(self) -> int:
        return self.image_size[1]

    @property
    def focal(self) -> float:
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.K[0, 2]), float(self.K[1, 2])

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates (metres)."""
        return -self.R.T @ self.t

    @property
    def height(self) -> float:
        """Height of the optical centre above the ground plane (metres)."""
        return float(self.center[2])

    @property
    def angles(self) -> tuple[float, float, float]:
        """``(yaw, pitch, roll)`` in degrees."""
        return angles_from_rotation(self.R)

    @property
    def yaw(self) -> float:
        return self.angles[0]

    @property
    def pitch(self) -> float:
        return self.angles[1]

    @property
    def roll(self) -> float:
        return self.angles[2]

    @property
    def fov(self) -> tuple[float, float]:
        """Horizontal and vertical field of view (degrees) of the rectified image."""
        w, h = self.image_size
        return (
            math.degrees(2 * math.atan(w / (2 * self.K[0, 0]))),
            math.degrees(2 * math.atan(h / (2 * self.K[1, 1]))),
        )

    @property
    def H(self) -> np.ndarray:
        """Homography from ground ``(x, y, 1)`` to rectified image pixels."""
        Hm = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], self.t])
        return Hm / Hm[2, 2] if abs(Hm[2, 2]) > 1e-12 else Hm

    @property
    def H_inv(self) -> np.ndarray:
        """Homography from rectified image pixels to ground ``(x, y, 1)``."""
        return np.linalg.inv(self.H)

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix (rectified pixels)."""
        return self.K @ np.column_stack([self.R, self.t])

    # ----------------------------------------------------------- projections
    def project(self, X: Any, distort: bool = True) -> np.ndarray:
        """Project world points ``(N, 3)`` to raw pixels. Points behind the camera are NaN."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        Xc = X @ self.R.T + self.t
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (Xc @ self.K.T)[:, :2] / Xc[:, 2:3]
        uv[Xc[:, 2] <= 1e-9] = np.nan
        return self.lens.distort(uv) if distort else uv

    def ground_to_image(self, xy: Any, z: Any = 0.0, distort: bool = True) -> np.ndarray:
        """World ``(N, 2)`` ground coordinates (at height ``z``) to raw pixels."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        zz = np.broadcast_to(np.asarray(z, dtype=np.float64), (len(xy),))
        return self.project(np.column_stack([xy, zz]), distort=distort)

    def rays(self, uv: Any, rectified: bool = False) -> np.ndarray:
        """World-frame ray directions ``(N, 3)`` through pixels (not normalised)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if not rectified:
            uv = self.lens.rectify(uv)
        homog = np.column_stack([uv, np.ones(len(uv))])
        return (homog @ np.linalg.inv(self.K).T) @ self.R  # R^T applied to row vectors

    def image_to_ground(self, uv: Any, z: Any = 0.0, rectified: bool = False) -> np.ndarray:
        """Intersect pixel rays with the horizontal plane at height ``z`` (default ground).

        Pixels whose ray does not reach the plane (above the horizon) return NaN.
        """
        d = self.rays(uv, rectified=rectified)
        C = self.center
        zz = np.broadcast_to(np.asarray(z, dtype=np.float64), (len(d),))
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (zz - C[2]) / d[:, 2]
        s = np.where(s > 0, s, np.nan)
        return C[:2] + s[:, None] * d[:, :2]

    def horizon(self) -> np.ndarray:
        """Horizon line ``(a, b, c)`` in rectified pixels (``a u + b v + c = 0``), normalised so the
        ground side is positive."""
        n_cam = self.R @ _UP
        line = np.linalg.inv(self.K).T @ n_cam
        norm = np.linalg.norm(line[:2])
        if norm < 1e-12:  # nadir view: the whole image is ground
            return np.array([0.0, 0.0, 1.0])
        line = line / norm
        # the image of a ground point must evaluate positive
        g = self.ground_to_image(self.center[:2] + 5.0 * self._forward_xy(), distort=False)[0]
        if np.all(np.isfinite(g)) and line @ np.array([g[0], g[1], 1.0]) < 0:
            line = -line
        return line

    def below_horizon(self, uv: Any, margin_px: float = 0.0, rectified: bool = False) -> np.ndarray:
        """Boolean mask: pixels that image the ground (strictly below the horizon by ``margin_px``)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if not rectified:
            uv = self.lens.rectify(uv)
        line = self.horizon()
        if not np.any(line[:2]):  # nadir view: no horizon in the image
            return np.all(np.isfinite(uv), axis=1)
        return (uv @ line[:2] + line[2]) > margin_px

    def ground_sampling_distance(self, uv: Any, mode: str = "max") -> np.ndarray:
        """Metres of ground per image pixel at the given raw pixels.

        ``mode="max"`` (default) returns the largest singular value of the
        image-to-ground Jacobian, i.e. the worst-axis (along-ray) footprint that
        bounds localisation error; ``mode="area"`` returns the square root of the
        pixel's ground area.
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        a = self.image_to_ground(uv)
        b = self.image_to_ground(uv + np.array([1.0, 0.0])) - a
        c = self.image_to_ground(uv + np.array([0.0, 1.0])) - a
        if mode == "area":
            return np.sqrt(np.abs(b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]))
        # singular values of J = [b c] (2x2): sqrt of eigenvalues of J^T J
        p = (b * b).sum(1)
        q = (c * c).sum(1)
        r = (b * c).sum(1)
        tr, det = p + q, p * q - r * r
        return np.sqrt(np.maximum(0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4 * det, 0))), 0))

    def distance_to(self, xy: Any) -> np.ndarray:
        """Horizontal distance (metres) from the camera's ground footprint to ground points."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return np.linalg.norm(xy - self.center[:2], axis=1)

    def _forward_xy(self) -> np.ndarray:
        return self.ground_up()

    def ground_up(self) -> np.ndarray:
        """Unit ground direction that appears as "up" in the image.

        Sum of the ground projections of the optical axis and of the image's up
        axis: the first vanishes for a nadir camera, the second for a level one,
        so the direction stays defined from horizontal to straight-down views.
        """
        f = self.R[2, :2] - self.R[1, :2]
        n = np.linalg.norm(f)
        return f / n if n > 1e-9 else np.array([1.0, 0.0])

    # ------------------------------------------------------------- transforms
    def with_lens(self, lens: Lens) -> Camera:
        return Camera(self.image_size, self.K.copy(), self.R.copy(), self.t.copy(), lens=lens, meta=dict(self.meta))

    def scaled(self, factor: float) -> Camera:
        """Same image geometry with the world scaled by ``factor`` (e.g. to fix metric scale)."""
        return Camera(
            self.image_size, self.K.copy(), self.R.copy(), self.t * factor, lens=self.lens, meta=dict(self.meta)
        )

    def transformed(self, rotation_deg: float = 0.0, translation: tuple[float, float] = (0.0, 0.0)) -> Camera:
        """Re-express the camera in a world frame rotated about ``z`` and translated on the ground.

        A world point ``X`` in the new frame relates to the old frame by
        ``X_new = Rz(rotation) @ X_old + [tx, ty, 0]``.
        """
        a = math.radians(rotation_deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        T = np.array([translation[0], translation[1], 0.0])
        R_new = self.R @ Rz.T
        t_new = self.t - R_new @ T
        return Camera(self.image_size, self.K.copy(), R_new, t_new, lens=self.lens, meta=dict(self.meta))

    def resized(self, new_size: tuple[int, int]) -> Camera:
        """Camera for the same view recorded at a different resolution."""
        sx = new_size[0] / self.image_size[0]
        sy = new_size[1] / self.image_size[1]
        S = np.diag([sx, sy, 1.0])
        K = S @ self.K
        lens_d = self.lens.to_dict()
        lens_d["image_size"] = list(new_size)
        lens_d["center"] = [lens_d["center"][0] * sx, lens_d["center"][1] * sy]
        for key in ("scale", "s", "focal"):
            if key in lens_d and lens_d[key] is not None:
                lens_d[key] = lens_d[key] * 0.5 * (sx + sy)
        return Camera(new_size, K, self.R.copy(), self.t.copy(), lens=lens_from_dict(lens_d), meta=dict(self.meta))

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        yaw, pitch, roll = self.angles
        return {
            "format": "tachymetry.camera/1",
            "image_size": list(self.image_size),
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "lens": self.lens.to_dict(),
            "summary": {
                "focal_px": round(self.focal, 3),
                "height_m": round(self.height, 4),
                "yaw_deg": round(yaw, 4),
                "pitch_deg": round(pitch, 4),
                "roll_deg": round(roll, 4),
                "hfov_deg": round(self.fov[0], 3),
            },
            "H_image_to_ground": (self.H_inv / self.H_inv[2, 2]).tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Camera:
        size = tuple(d["image_size"])
        lens = lens_from_dict(d.get("lens"), size)  # type: ignore[arg-type]
        return cls(size, np.array(d["K"]), np.array(d["R"]), np.array(d["t"]), lens=lens, meta=d.get("meta", {}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Camera:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def describe(self) -> str:
        yaw, pitch, roll = self.angles
        hfov = self.fov[0]
        return (
            f"f={self.focal:.1f}px (hfov {hfov:.1f} deg), height={self.height:.2f} m, "
            f"pitch={pitch:.2f} deg, roll={roll:.2f} deg, yaw={yaw:.1f} deg, lens={self.lens.kind}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Camera({self.image_size[0]}x{self.image_size[1]}, {self.describe()})"


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def decompose_ground_homography(
    H: np.ndarray,
    image_size: tuple[int, int],
    principal_point: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose ``H = K [r1 r2 t]`` (ground to rectified image) into ``K, R, t``.

    The focal length is the least-squares solution of the two orthonormality
    constraints on ``r1`` and ``r2`` (square pixels, known principal point).
    """
    H = np.asarray(H, dtype=np.float64)
    w, h = image_size
    cx, cy = principal_point if principal_point is not None else ((w - 1) / 2.0, (h - 1) / 2.0)
    T = np.array([[1.0, 0, cx], [0, 1.0, cy], [0, 0, 1.0]])
    A = np.linalg.inv(T) @ H
    a1, a2 = A[:, 0], A[:, 1]
    # Constraints linear in w = 1/f^2:  p * w + q = 0
    p1, q1 = a1[0] * a2[0] + a1[1] * a2[1], a1[2] * a2[2]
    p2, q2 = (a1[0] ** 2 + a1[1] ** 2) - (a2[0] ** 2 + a2[1] ** 2), a1[2] ** 2 - a2[2] ** 2
    den = p1 * p1 + p2 * p2
    if den < 1e-24:
        raise CalibrationError("degenerate homography: the focal length is unobservable (fronto-parallel view)")
    wsol = -(p1 * q1 + p2 * q2) / den
    if not np.isfinite(wsol) or wsol <= 0:
        raise CalibrationError(
            "homography is not induced by a pinhole camera above the plane "
            "(negative squared focal length); check correspondences and world-frame handedness"
        )
    f = 1.0 / math.sqrt(wsol)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    B = np.linalg.inv(K) @ H
    g = np.linalg.solve(H, np.array([cx, cy, 1.0]))  # ground point seen at the principal point
    g = g[:2] / g[2] if abs(g[2]) > 1e-12 else None
    best = None
    for sign in (1.0, -1.0):
        b1, b2, b3 = sign * B[:, 0], sign * B[:, 1], sign * B[:, 2]
        lam = 0.5 * (np.linalg.norm(b1) + np.linalg.norm(b2))
        r1, r2, tt = b1 / lam, b2 / lam, b3 / lam
        r3 = np.cross(r1, r2)
        U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
        R = U @ Vt
        if np.linalg.det(R) < 0:
            continue
        C = -R.T @ tt
        in_front = True if g is None else float(R[2, 0] * g[0] + R[2, 1] * g[1] + tt[2]) > 0
        if C[2] > 0 and in_front:  # camera above the ground and the ground in front of it
            best = (K, R, tt)
            break
    if best is None:
        raise CalibrationError(
            "homography places the camera below the ground plane; the world frame is probably "
            "left-handed (e.g. y pointing down/south as in image coordinates). Flip the sign of y."
        )
    return best

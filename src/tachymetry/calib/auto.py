"""Automatic calibration: candidate methods, evidence fusion and selection.

:func:`calibrate_auto` runs the applicable methods on a video's traffic-free
background image and its detections:

``cuboids``  vehicle-footprint rectification (needs a cuboid detector),
``lines``    vanishing points of road markings and vertical structures,
``depth``    monocular metric depth plus road segmentation (optional extras),
``geocalib`` single-image gravity and focal length (optional extra).

When both cuboid footprints and vanishing points are available, a **fused**
candidate refines the camera by maximum likelihood over all evidence at once:
footprint rectangularity and size consistency, horizontal vanishing points
constrained to the ground plane, and the vertical vanishing point constrained
to the ground normal. No assumption is made that streets meet at right angles.

Candidates are ranked by method-independent consistency (footprint right
angles, size-versus-range slope, vanishing-point residuals); all candidates are
kept in ``result.details["candidates"]`` and can be previewed side by side with
:func:`tachymetry.calib.preview.compare`.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import CalibrationConfig
from ..detections import DetectionSet
from ..geometry.camera import CalibrationError, Camera
from ..geometry.lens import IdentityLens, Lens
from .from_cuboids import ASPECT_WEIGHT, calibrate_from_cuboids, collect_footprints, footprint_metrics, footprint_scale
from .from_cuboids import _residuals as _cuboid_residuals
from .from_lines import ScaleCue, apply_scale_cue, calibrate_from_lines
from .lines import VanishingPoint, detect_segments, marking_mask
from .result import CalibrationResult
from .vanishing import vanishing_points_from_segments

__all__ = ["calibrate_auto", "fuse_evidence", "consistency"]


def _vp_residuals(
    cam: Camera, horiz: Sequence[VanishingPoint], vertical: VanishingPoint | None, weight: float
) -> np.ndarray:
    Ki = np.linalg.inv(cam.K)
    up = cam.R @ np.array([0.0, 0.0, 1.0])
    r = []
    for vp in horiz:
        d = Ki @ vp.point
        d /= np.linalg.norm(d)
        r.append(weight * float(up @ d))  # sine of the elevation of a ground direction: must be 0
    if vertical is not None:
        d = Ki @ vertical.point
        d /= np.linalg.norm(d)
        if d @ up < 0:
            d = -d
        r.extend(weight * np.cross(up, d))  # parallel to the ground normal
    return np.asarray(r)


def fuse_evidence(
    cam0: Camera,
    quads: np.ndarray,
    wts: np.ndarray,
    horiz: Sequence[VanishingPoint],
    vertical: VanishingPoint | None,
    car_length_m: float = 4.6,
    car_width_m: float = 1.85,
    vp_gate_deg: float = 6.0,
    vp_sigma_deg: float = 0.5,
) -> tuple[Camera, dict[str, Any]]:
    """Maximum-likelihood refinement of focal length, pitch and roll over footprints and vanishing points.

    Footprint residuals are normalised by their robust spread at the starting
    camera and by an effective sample size (one independent observation per
    occupied image cell, since a vehicle seen for several seconds yields dozens
    of correlated footprints). Vanishing-point residuals are angles normalised
    by ``vp_sigma_deg``. A soft-L1 loss at two standard deviations limits the
    influence of outliers in either group.
    """
    from scipy.optimize import least_squares

    size = cam0.image_size
    # gate the vanishing points with the starting camera (drops e.g. elevated signal mast arms)
    Ki = np.linalg.inv(cam0.K)
    up0 = cam0.R @ np.array([0.0, 0.0, 1.0])
    use_h = []
    for vp in horiz:
        d = Ki @ vp.point
        d /= np.linalg.norm(d)
        if abs(math.degrees(math.asin(np.clip(up0 @ d, -1, 1)))) < vp_gate_deg:
            use_h.append(vp)
    use_v = None
    if vertical is not None:
        d = Ki @ vertical.point
        d /= np.linalg.norm(d)
        if math.degrees(math.acos(min(1.0, abs(float(up0 @ d))))) < 2 * vp_gate_deg:
            use_v = vertical
    yaw0, pitch0, roll0 = cam0.angles
    p0 = np.array([math.log(cam0.focal), math.radians(pitch0), math.radians(roll0)])

    def make(p: np.ndarray) -> Camera:
        return Camera.from_pose(
            size,
            focal=math.exp(p[0]),
            height=1.0,
            pitch=math.degrees(p[1]),
            yaw=90.0,
            roll=math.degrees(p[2]),
            lens=IdentityLens(size),
        )

    asp = car_length_m / car_width_m
    scale_c = 1.0
    n_eff = 0
    if len(quads):
        r0 = _cuboid_residuals(make(p0), quads, wts, asp, ASPECT_WEIGHT, 2.0)
        sigma_c = max(1.4826 * float(np.median(np.abs(r0))), 1e-3)
        w, h = size
        ctr = quads.mean(axis=1)
        cells = np.unique(np.floor(ctr[:, 0] / (w / 24)) * 1000 + np.floor(ctr[:, 1] / (h / 14)))
        n_eff = max(10, len(cells))
        scale_c = sigma_c * math.sqrt(len(quads) / n_eff)
    sigma_vp = math.radians(vp_sigma_deg)

    def fun(p: np.ndarray) -> np.ndarray:
        c = make(p)
        parts = [_cuboid_residuals(c, quads, wts, asp, ASPECT_WEIGHT, 2.0) / scale_c] if len(quads) else []
        parts.append(_vp_residuals(c, use_h, use_v, 1.0 / sigma_vp))
        return np.concatenate(parts)

    w_img = size[0]
    lo = np.array([math.log(w_img * 0.15), math.radians(-10.0), math.radians(-25)])
    hi = np.array([math.log(w_img * 12), math.radians(89.5), math.radians(25)])
    res = least_squares(
        fun, np.clip(p0, lo, hi), bounds=(lo, hi), loss="soft_l1", f_scale=2.0, max_nfev=800, x_scale="jac"
    )
    cam = make(res.x).with_lens(cam0.lens)
    info = {
        "n_footprints": int(len(quads)),
        "effective_footprints": int(n_eff),
        "n_vp_horizontal": len(use_h),
        "vertical_vp": use_v is not None,
        "vp_sigma_deg": vp_sigma_deg,
        "cost": float(res.cost),
    }
    return cam, info


def consistency(
    cam: Camera, quads: np.ndarray, wts: np.ndarray, horiz: Sequence[VanishingPoint], vertical: VanishingPoint | None
) -> dict[str, float]:
    """Method-independent consistency metrics (lower is better except ``score``)."""
    out: dict[str, float] = {}
    if len(quads):
        m = footprint_metrics(cam, quads, wts)
        out["footprint_right_angle_deg"] = m["right_angle_deg"]
        out["size_range_slope"] = m["size_range_slope"]
    if horiz or vertical is not None:
        r = _vp_residuals(cam, horiz, vertical, 1.0)
        out["vp_residual_deg"] = float(np.degrees(np.arcsin(np.clip(np.median(np.abs(r)), 0, 1)))) if len(r) else 0.0
    s = 1.0
    if "footprint_right_angle_deg" in out and np.isfinite(out["footprint_right_angle_deg"]):
        s *= float(np.exp(-out["footprint_right_angle_deg"] / 10.0))
        s *= float(np.exp(-abs(out["size_range_slope"]) / 0.15))
    if "vp_residual_deg" in out:
        s *= float(np.exp(-out["vp_residual_deg"] / 3.0))
    hf = cam.fov[0]
    if not (10.0 <= hf <= 125.0):
        s *= 0.1
    out["score"] = s
    return out


def _metric_scale(
    cam: Camera,
    quads: np.ndarray,
    wts: np.ndarray,
    cfg: CalibrationConfig,
    cue: ScaleCue | None,
    dets: DetectionSet | None = None,
) -> tuple[Camera, dict[str, Any]]:
    if cue is not None and not cue.vehicles:
        return apply_scale_cue(cam, cue)
    if dets is not None and dets.masks is not None:
        from .from_silhouettes import silhouette_scale

        s, info = silhouette_scale(cam, dets, car_width_m=cfg.car_width_m)
        out = cam.scaled(s)
        out.meta["metric"] = True
        return out, info
    if len(quads) < 10:
        raise CalibrationError(
            "a vehicle-size scale needs instance masks (rfdetr-seg-*, sam3) or cuboid footprints (uod2-*); "
            "pass scale='height=<m>' or scale='lane_width=<m>' instead"
        )
    s, info = footprint_scale(cam, quads, wts, cfg.car_length_m, cfg.car_width_m)
    out = cam.scaled(s)
    out.meta["metric"] = True
    return out, {"scale": "vehicle footprints", **info}


def _align_world(cam: Camera, quads: np.ndarray, wts: np.ndarray, horiz: Sequence[VanishingPoint] = ()) -> Camera:
    """World +x along the main road, origin at the median vehicle position.

    The road direction is the best-supported ground vanishing direction (lane
    lines) when one exists; otherwise the dominant vehicle axis (which can be a
    parking angle, hence the preference for markings).
    """
    phi = None
    if horiz:
        Ki = np.linalg.inv(cam.K)
        up = cam.R @ np.array([0.0, 0.0, 1.0])
        best = None
        for vp in horiz:
            d = Ki @ vp.point
            d /= np.linalg.norm(d)
            if abs(float(up @ d)) > math.sin(math.radians(6.0)):
                continue
            if best is None or vp.support > best[0]:
                best = (vp.support, d)
        if best is not None:
            dw = cam.R.T @ best[1]
            phi = math.atan2(dw[1], dw[0])
    if len(quads) >= 5:
        g = cam.image_to_ground(quads.reshape(-1, 2)).reshape(-1, 4, 2)
        ok = np.all(np.isfinite(g), axis=(1, 2))
        g, w = g[ok], wts[ok]
        if phi is None and len(g):
            ax = (g[:, 0] + g[:, 3]) / 2 - (g[:, 1] + g[:, 2]) / 2
            phi = 0.5 * float(np.angle(np.sum(w * np.exp(2j * np.arctan2(ax[:, 1], ax[:, 0])))))
        if len(g):
            ctr = np.median(g.mean(axis=1), axis=0)
            cam = cam.transformed(rotation_deg=0.0, translation=(-float(ctr[0]), -float(ctr[1])))
    if phi is not None:
        cam = cam.transformed(rotation_deg=-math.degrees(phi))
    return cam


def calibrate_auto(
    background: np.ndarray,
    detections: DetectionSet | None = None,
    cfg: CalibrationConfig | None = None,
    lens: Lens | str | None = None,
    scale: Any = None,
) -> CalibrationResult:
    """Calibrate automatically from a background image and (optionally) detections.

    Parameters
    ----------
    background:
        Traffic-free image of the scene (:func:`~tachymetry.calib.lines.background_image`).
    detections:
        Detections from the same video; cuboid footprints enable the ``cuboids``
        method, the fused estimate and the vehicle-size metric scale.
    scale:
        Metric cue overriding ``cfg.scale`` (``"vehicles"``, ``"height=8"``, ...).
    """
    cfg = cfg or CalibrationConfig()
    size = (background.shape[1], background.shape[0])
    cue = ScaleCue.parse(scale if scale is not None else (None if cfg.scale in ("none", "") else cfg.scale))
    if detections is not None and tuple(detections.image_size) != size:
        import cv2

        background = cv2.resize(background, tuple(detections.image_size), interpolation=cv2.INTER_AREA)
        size = tuple(detections.image_size)
    quads, wts = collect_footprints(detections) if detections is not None else (np.zeros((0, 4, 2)), np.zeros(0))
    candidates: list[CalibrationResult] = []
    errors: dict[str, str] = {}
    horiz: list[VanishingPoint] = []
    vertical: VanishingPoint | None = None
    lens_obj: Lens = lens if isinstance(lens, Lens) else IdentityLens(size)

    if "lines" in cfg.methods:
        try:
            r = calibrate_from_lines(background, scale=None, lens=lens if lens is not None else cfg.lens)
            lens_obj = r.camera.lens
            candidates.append(r)
        except CalibrationError as exc:
            errors["lines"] = str(exc)
        try:  # vanishing-point evidence for fusion and scoring, in the rectified image
            rect = lens_obj.rectify_image(background) if not lens_obj.is_identity else background
            ms = detect_segments(rect, marking_mask(rect))
            al = detect_segments(rect, None)
            if len(ms) >= 8:
                horiz, vertical = vanishing_points_from_segments(ms, size, al)
        except Exception as exc:  # pragma: no cover - evidence is optional
            errors["vanishing_points"] = str(exc)

    if "cuboids" in cfg.methods and len(quads):
        try:
            candidates.append(
                calibrate_from_cuboids(
                    detections, lens=lens_obj, car_length_m=cfg.car_length_m, car_width_m=cfg.car_width_m
                )
            )
        except CalibrationError as exc:
            errors["cuboids"] = str(exc)

    if cfg.hfov_deg and horiz:
        # known lens: the best-supported ground vanishing direction fixes pitch and yaw, roll comes from a second
        # horizontal vanishing point or the vertical one when present, else 0
        try:
            candidates.append(_from_fov_and_vp(size, cfg.hfov_deg, horiz, vertical, lens_obj))
        except CalibrationError as exc:
            errors["fov+vp"] = str(exc)

    if "depth" in cfg.methods:
        try:
            from .depth import calibrate_from_depth

            candidates.append(calibrate_from_depth(background))
        except Exception as exc:
            errors["depth"] = str(exc)

    if "geocalib" in cfg.methods:
        try:
            from .geocalib_backend import calibrate_geocalib

            candidates.append(calibrate_geocalib(background))
        except Exception as exc:
            errors["geocalib"] = str(exc)

    quads_r = (
        lens_obj.rectify(quads.reshape(-1, 2)).reshape(-1, 4, 2) if len(quads) and not lens_obj.is_identity else quads
    )
    if len(quads_r) >= 40 and (horiz or vertical is not None):
        base = max((c for c in candidates if c.method == "cuboids"), default=None, key=lambda c: c.score or 0)
        if base is not None:
            try:
                cam_f, finfo = fuse_evidence(
                    base.camera,
                    quads_r,
                    wts,
                    horiz,
                    vertical,
                    cfg.car_length_m,
                    cfg.car_width_m,
                    vp_sigma_deg=cfg.vp_sigma_deg,
                )
                candidates.append(CalibrationResult(cam_f, "fused", details=finfo))
            except Exception as exc:
                errors["fused"] = str(exc)

    if cfg.hfov_deg and candidates:
        # a known field of view is authoritative: drop candidates whose focal length disagrees with it
        f_hint = (size[0] / 2) / math.tan(math.radians(cfg.hfov_deg / 2))
        kept = [c for c in candidates if abs(c.camera.focal / f_hint - 1.0) <= 0.10]
        kept_ids = {id(c) for c in kept}
        for c in candidates:
            if id(c) not in kept_ids:
                errors[c.method] = (
                    f"focal length {c.camera.focal:.0f} px disagrees with the {cfg.hfov_deg:g} deg field of view"
                )
        candidates = kept
    if not candidates:
        raise CalibrationError("automatic calibration failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))

    from .from_markings import dashed_lines, marking_regularity

    try:
        dashes = dashed_lines(background, lens=lens_obj)
    except Exception:  # pragma: no cover - the check is optional
        dashes = []
    ranked = []
    for c in candidates:
        cam = c.camera
        cons = consistency(cam, quads, wts, horiz, vertical)  # raw quads: the camera carries the lens
        if len(dashes) >= 2:
            # scale-free check: dash lengths and periods must not change along the road
            reg = marking_regularity(Camera(cam.image_size, cam.K, cam.R, cam.t), dashes)
            cons["dash_regularity"] = reg
            cons["score"] *= float(np.exp(-reg / 0.1)) if np.isfinite(reg) else 0.5
        c.score = cons["score"]
        c.details["consistency"] = cons
        ranked.append(c)
    ranked.sort(key=lambda c: -(c.score or 0))
    best = ranked[0]
    try:
        cam, sinfo = _metric_scale(best.camera, quads, wts, cfg, cue, detections)
    except CalibrationError as exc:
        cam, sinfo = best.camera, {"scale": "relative"}
        cam.meta["metric"] = False
        warnings.warn(f"{exc}; the camera has relative scale (height 1)", stacklevel=2)
    cam = _align_world(cam, quads, wts, horiz)
    cam.meta.update({"calibration_method": f"auto:{best.method}"})
    reg = best.details.get("consistency", {}).get("dash_regularity")
    if reg is not None and reg > 0.12:
        warnings.warn(
            f"dashed lane markings change length along the road under this camera (spread {reg:.2f} in log units): "
            "its pitch or focal length is likely wrong; give the lens field of view (calibration.hfov_deg), "
            "install GeoCalib, or calibrate manually",
            stacklevel=2,
        )
    details = {
        "selected": best.method,
        "scale": sinfo,
        "candidates": [
            {
                "method": c.method,
                "score": c.score,
                "camera": c.camera.describe(),
                "consistency": c.details.get("consistency"),
            }
            for c in ranked
        ],
        "errors": errors,
        "n_footprints": int(len(quads)),
        "vanishing_points": [vp.xy.tolist() for vp in horiz],
        "vertical_vp": None if vertical is None else vertical.xy.tolist(),
    }
    out = CalibrationResult(cam, f"auto:{best.method}", score=best.score, details=details)
    out.details["candidate_cameras"] = {c.method: c.camera for c in ranked}
    return out


def _from_fov_and_vp(
    size: tuple[int, int],
    hfov_deg: float,
    horiz: Sequence[VanishingPoint],
    vertical: VanishingPoint | None,
    lens: Lens,
) -> CalibrationResult:
    """Scale-free camera from a known field of view and the main ground vanishing point."""
    w, h = size
    f = (w / 2) / math.tan(math.radians(hfov_deg / 2))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    vps = sorted((v for v in horiz if v.finite), key=lambda v: -v.support)
    if not vps:
        raise CalibrationError("no finite ground vanishing point")
    roll = 0.0
    if len(vps) >= 2:
        a, b = vps[0].xy, vps[1].xy
        roll = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        roll = (roll + 90.0) % 180.0 - 90.0
    elif vertical is not None and vertical.finite:
        v = vertical.xy
        roll = -math.degrees(math.atan2(v[0] - cx, v[1] - cy))
    # rotate the main vanishing point by -roll about the principal point; its height then gives the pitch
    p = vps[0].xy - np.array([cx, cy])
    cr, sr = math.cos(math.radians(-roll)), math.sin(math.radians(-roll))
    q = np.array([cr * p[0] - sr * p[1], sr * p[0] + cr * p[1]])
    pitch = math.degrees(math.atan2(-q[1], f))
    if not 0.5 < pitch < 89.5:
        raise CalibrationError(f"the vanishing point implies a pitch of {pitch:.1f} deg")
    cam = Camera.from_pose(size, focal=f, height=1.0, pitch=pitch, yaw=90.0, roll=roll, lens=lens)
    cam.meta.update({"calibration_method": "fov+vp", "metric": False})
    return CalibrationResult(cam, "fov+vp", details={"hfov_deg": hfov_deg, "roll_deg": roll})

"""Automatic calibration from straight road markings.

Pipeline: plumb-line lens estimate on the traffic-free background, painted
marking extraction, LSD segments, vanishing points of the two ground
directions (lane lines, crosswalk stripes, kerbs, stop lines) and of the
vertical (poles, facades), then rotation and focal length: from the two ground
directions when the vertical does not contradict right-angled streets
(:func:`~tachymetry.calib.vanishing.vertical_consistency`), from one ground
direction and the vertical otherwise.
The metric scale comes from a :class:`ScaleCue`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry.camera import CalibrationError, Camera
from ..geometry.lens import IdentityLens, Lens
from .lines import Segments, detect_segments, estimate_lens_vp_consistency, marking_mask
from .result import CalibrationResult
from .scale import scale_from_distance, scale_from_lane_spacing, scale_to_height
from .vanishing import camera_from_vps, select_orthogonal_pair, vanishing_points_from_segments, vertical_consistency

__all__ = ["ScaleCue", "calibrate_from_lines", "apply_scale_cue"]


@dataclass
class ScaleCue:
    """One metric cue; set one field.

    ``height``      camera height above the road (m)
    ``distance``    ``((u1, v1), (u2, v2), metres)``: two ground pixels a known distance apart
    ``lane_width``  lane width (m) of the lines parallel to the main road
    ``vehicles``    use passenger-car footprint statistics (needs detections; see ``calibrate_auto``)
    """

    height: float | None = None
    distance: tuple[tuple[float, float], tuple[float, float], float] | None = None
    lane_width: float | None = None
    vehicles: bool = False

    @classmethod
    def parse(cls, spec: Any) -> ScaleCue | None:
        """Parse ``None``, a :class:`ScaleCue`, a dict, or strings like ``"height=8"`` / ``"vehicles"``."""
        if spec is None or isinstance(spec, ScaleCue):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        if isinstance(spec, (int, float)):
            return cls(height=float(spec))
        s = str(spec).strip().lower()
        if s in ("vehicles", "cars", "auto"):
            return cls(vehicles=True)
        if "=" in s:
            k, v = s.split("=", 1)
            if k in ("height", "h"):
                return cls(height=float(v))
            if k in ("lane", "lane_width"):
                return cls(lane_width=float(v))
        raise ValueError(f"cannot parse scale cue {spec!r} (try 'height=8', 'lane_width=3.5' or 'vehicles')")


def apply_scale_cue(
    cam: Camera, cue: ScaleCue | None, segments: Segments | None = None
) -> tuple[Camera, dict[str, Any]]:
    """Fix the metric scale of a unit-scale camera. Returns the camera and diagnostics."""
    if cue is None:
        cam.meta["metric"] = False
        return cam, {"scale": "relative (no metric cue: camera height set to 1)"}
    if cue.height is not None:
        out = scale_to_height(cam, cue.height)
        out.meta["metric"] = True
        return out, {"scale": f"camera height {cue.height:g} m"}
    if cue.distance is not None:
        a, b, d = cue.distance
        out = scale_from_distance(cam, a, b, d)
        out.meta["metric"] = True
        return out, {"scale": f"distance {d:g} m"}
    if cue.lane_width is not None:
        if segments is None or len(segments) < 3:
            raise CalibrationError("lane-width scale needs the main-direction marking segments")
        L = segments.lengths
        long = segments.subset(L >= np.percentile(L, 50))
        g = cam.image_to_ground(long.midpoints, rectified=True)
        ok = np.all(np.isfinite(g), axis=1)
        factor = scale_from_lane_spacing(g[ok, 1], expected_m=cue.lane_width)
        out = cam.scaled(factor)
        out.meta["metric"] = True
        return out, {"scale": f"lane width {cue.lane_width:g} m", "factor": factor}
    raise CalibrationError("vehicle-size scale requires detections; use tachymetry.calib.calibrate_auto")


def calibrate_from_lines(
    image: np.ndarray,
    scale: ScaleCue | str | float | dict | None = None,
    lens: str | Lens | None = "auto",
    region: np.ndarray | None = None,
    threshold_deg: float = 1.5,
    seed: int = 0,
) -> CalibrationResult:
    """Calibrate from a traffic-free image of the scene (see :func:`~tachymetry.calib.lines.background_image`).

    Parameters
    ----------
    image:
        BGR image, ideally a temporal median of the video.
    scale:
        Metric cue (:class:`ScaleCue` or ``"height=8"``, ``"lane_width=3.5"``). Without one
        the camera height is set to 1 and ``camera.meta["metric"]`` is ``False``.
    lens:
        ``"auto"`` estimates both the equidistant (``tan``) and the division
        model by vanishing-point consistency and keeps the better one, accepted
        only with a clear gain; ``"tan"``/``"equidistant"`` or ``"division"``
        estimate that model; ``None`` is a pinhole; a :class:`Lens` is used as given.
    region:
        Optional boolean mask of the road surface (e.g. from semantic segmentation).
    """
    cue = ScaleCue.parse(scale)
    h, w = image.shape[:2]
    size = (w, h)
    details: dict[str, Any] = {}
    if isinstance(lens, str) and lens in ("auto", "tan", "equidistant", "division"):
        raw_segs = detect_segments(image, None)
        if lens == "auto":
            models: tuple[str, ...] = ("equidistant", "division")
        else:
            models = ("equidistant",) if lens in ("tan", "equidistant") else ("division",)
        best: tuple[Lens, dict[str, Any]] | None = None
        for m in models:
            est, info = estimate_lens_vp_consistency(raw_segs, size, model=m, seed=seed)
            if best is None or info["support"] > best[1]["support"]:
                best = (est, info)
        assert best is not None
        est, info = best
        details["lens_search"] = info
        # accept only a clear gain in vanishing-point support (absolute fraction of segment length)
        ratio = info["support"] / max(info["support_identity"], 1e-9)
        lens_obj: Lens = est if (info["gain"] > 0.08 and ratio > 1.2) else IdentityLens(size)
    elif lens is None or lens in ("none", "identity"):
        lens_obj = IdentityLens(size)
    elif isinstance(lens, str):
        raise ValueError(
            f"lens {lens!r} cannot be estimated from straight lines; use 'auto', 'tan', 'division' or 'none', "
            "or calibrate manually for Brown-Conrady and Kannala-Brandt models"
        )
    else:
        lens_obj = lens
    rect = lens_obj.rectify_image(image) if not lens_obj.is_identity else image

    mask = marking_mask(rect, region=region)
    mark_segs = detect_segments(rect, mask)
    all_segs = detect_segments(rect, None)
    details["n_marking_segments"] = len(mark_segs)
    details["n_segments"] = len(all_segs)
    if len(mark_segs) < 8:
        raise CalibrationError(f"too few marking segments ({len(mark_segs)}); is the road visible and marked?")
    horiz, vertical = vanishing_points_from_segments(mark_segs, size, all_segs, threshold_deg, seed)
    details["vanishing_points"] = [{"xy": vp.xy.tolist(), "support_px": vp.support} for vp in horiz]
    details["vertical_vp"] = None if vertical is None else vertical.xy.tolist()
    w_img = size[0]
    f_lo, f_hi = (w_img / 2) / math.tan(math.radians(65)), (w_img / 2) / math.tan(math.radians(7.5))
    pp = np.array([(w_img - 1) / 2, (size[1] - 1) / 2])
    from .vanishing import focal_from_vps

    cam = None
    main = None
    pair = select_orthogonal_pair(horiz, size, vertical)
    if vertical is not None and horiz:
        # A ground direction and the vertical give the focal length without assuming that the
        # streets meet at right angles, but a far vertical vanishing point fixes it poorly; the
        # street pair is used instead when the vertical segments do not contradict it.
        cands = []
        for vp in horiz:
            f = focal_from_vps(vp.point, vertical.point, pp)
            if f is not None and f_lo <= f <= f_hi:
                cands.append((vp.support, vp, f))
        details["focal_candidates"] = [c[2] for c in cands]
        use_pair = False
        if cands and pair is not None:
            test = vertical_consistency(all_segs, vertical, pair[0], pair[1], pair[2], size, threshold_deg)
            details["orthogonality_test"] = test
            use_pair = not test["rejected"]
        if cands and not use_pair:
            cands.sort(key=lambda c: -c[0])
            _, main, f = cands[0]
            fs = [c[2] for c in cands[:2]]
            if len(fs) == 2 and abs(fs[0] - fs[1]) / fs[0] < 0.1:
                f = float(np.average(fs, weights=[c[0] for c in cands[:2]]))
            cam = camera_from_vps(main.point, None, size, v_z=vertical.point, focal=f, lens=lens_obj)
            details["solution"] = "ground direction + vertical (no orthogonality assumption)"
    if cam is None:
        if pair is None:
            raise CalibrationError("could not find two independent vanishing directions; use another method")
        vx, vy, f = pair
        main = vx if vx.support >= vy.support else vy
        other = vy if main is vx else vx
        cam = camera_from_vps(main.point, other.point, size, v_z=None, focal=f, lens=lens_obj)
        if "orthogonality_test" in details:
            details["solution"] = "two orthogonal ground directions, consistent with the vertical"
        else:
            details["solution"] = "two ground directions assumed orthogonal"
        details["assumes_orthogonal_streets"] = True
    vx = main
    cam, sinfo = apply_scale_cue(cam, cue, mark_segs.subset(main.inliers))
    details.update(sinfo)
    total = float(mark_segs.lengths.sum())
    support = vx.support / max(total, 1e-9)
    details["support_fraction"] = support
    cam.meta.update({"calibration_method": "lines"})
    yaw, pitch, roll = cam.angles
    details["angles"] = {"yaw": yaw, "pitch": pitch, "roll": roll}
    return CalibrationResult(cam, "lines", score=float(min(1.0, support)), details=details)

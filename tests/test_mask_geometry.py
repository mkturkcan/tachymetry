"""Vehicle geometry from instance masks: silhouette fit and occlusion handling on synthetic ground truth."""

import math
from dataclasses import replace

import numpy as np
import pytest

from tachymetry.config import PipelineConfig
from tachymetry.detections import crop_mask
from tachymetry.geometry import Camera
from tachymetry.geometry.silhouette import MaskRaster, fit_centre, project_box
from tachymetry.synthetic.masks import tree_occluder
from tachymetry.synthetic.traffic import Agent, TrafficSpec, render_detections
from tachymetry.tracking.builder import build_tracks

RATE = 15.0


@pytest.fixture
def mast_camera() -> Camera:
    return Camera.from_pose(
        (1280, 720), focal=1100.0, height=12.0, pitch=25.0, yaw=45.0, roll=0.5, position=(-14.0, -14.0)
    )


def test_silhouette_fit_recovers_box_pose(mast_camera):
    """The 3-DoF fit of a known box to its own rasterised silhouette returns the box's centre and heading."""
    import cv2

    c_true, th, (L, W, H) = np.array([3.0, 2.0]), math.radians(-40.0), (4.6, 1.85, 1.5)
    pts = project_box(mast_camera, c_true, th, L, W, H, 0.2)
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillConvexPoly(m, np.round(cv2.convexHull(pts.astype(np.float32)) * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    ys, xs = np.nonzero(m)
    r = MaskRaster(crop_mask(m.astype(bool), np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)), 4, (1280, 720))
    # from a reference heading 8 degrees off: the bounded refinement moves towards the silhouette under its prior,
    # and recovers the heading without the prior
    c, cost, th_fit = fit_centre(
        mast_camera, r, th + math.radians(8), L, W, H, 0.2, c_true + [0.7, -0.5], opt_heading=True
    )
    assert np.linalg.norm(c - c_true) < 0.15 and cost < 0.6
    assert abs(math.degrees(th_fit - th)) < 6.0
    c, cost, th_fit = fit_centre(
        mast_camera, r, th + math.radians(8), L, W, H, 0.2, c_true + [0.7, -0.5], opt_heading=True, prior_w=0.0
    )
    assert np.linalg.norm(c - c_true) < 0.1 and abs(math.degrees(th_fit - th)) < 1.5


def _road_agent(aid, cls, dims, s0, s1, offset, v, t0, u, n_hat, c0):
    d = 1.0 if s1 > s0 else -1.0
    n = int(abs(s1 - s0) / v * RATE)
    t = t0 + np.arange(n) / RATE
    xy = c0 + (s0 + d * v * (t - t0))[:, None] * u + offset * n_hat
    return Agent(aid, cls, *dims, t, xy, np.full(n, math.atan2(d * u[1], d * u[0])), np.full(n, v))


def test_occluded_bus_keeps_its_footprint(mast_camera):
    """A bus passing behind a tree keeps a full-size, road-aligned footprint (the failure of cuboid footprints).

    A 12 m bus drives at 25 km/h across the view, 40 m from a 12 m mast; a tree
    crown in front of the road hides its front and lower edge, then up to 40 %
    of its silhouette, and releases it again. With masks the footprint centre
    must stay within 0.6 m and the heading within 3 degrees of the truth on
    every tracked sample, occluded ones included, and the length within 1 m.
    The mask path stays near 0.5 m and 1 degree at worst (a steady 0.2 m
    across-axis offset that grows under occlusion); the box-only path on the
    same detections reaches about 1.4 m and 11 degrees, and the real failure
    was a 1.7 m drift with the box turned by up to 45 degrees.
    """
    u = np.array([math.cos(-math.pi / 4), math.sin(-math.pi / 4)])
    n_hat = np.array([-u[1], u[0]])
    c0 = np.array([-14.0, -14.0]) + 40.0 * np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    bus = _road_agent(0, "bus", (12.0, 2.55, 3.2), -30.0, 30.0, -1.75, 7.0, 0.0, u, n_hat, c0)
    car = _road_agent(1, "car", (4.6, 1.8, 1.45), 30.0, -30.0, 1.75, 8.0, 0.5, u, n_hat, c0)
    p = mast_camera.ground_to_image((c0 + 4.0 * u - 1.75 * n_hat)[None])[0]
    spec = TrafficSpec(
        duration_s=8.5,
        rate_hz=RATE,
        masks=True,
        occluders=(tree_occluder((p[0], p[1] + 35), 95, seed=1),),
        ghost_rate=0.0,
        miss_prob=0.03,
        flicker_prob=0.0,
        seed=7,
    )
    ds = render_detections(mast_camera, [bus, car], spec)
    tracks, _ = build_tracks(ds, mast_camera, PipelineConfig())

    def bus_track(trs):
        for tr in trs:
            j = np.round((tr.t - bus.t[0]) * RATE).astype(int)
            k = (j >= 0) & (j < len(bus.t))
            if tr.is_ok and k.sum() > 10 and np.median(np.linalg.norm(tr.xy[k] - bus.xy[j[k]], axis=1)) < 3:
                return tr, j[k], k
        raise AssertionError("bus not tracked")

    tr, j, k = bus_track(tracks)
    err = np.linalg.norm(tr.xy[k] - bus.xy[j], axis=1)
    hd = np.degrees(np.abs((tr.heading[k] - bus.heading[j] + np.pi / 2) % np.pi - np.pi / 2))
    assert tr.meta["geometry"]["source"] == "boxfit"
    assert tr.meta["geometry"]["occluded_samples"] >= 5  # the occlusion was seen and handled
    assert err.max() < 0.6, err.max()
    assert hd.max() < 3.0, hd.max()
    assert abs(tr.length - 12.0) < 1.0 and abs(tr.width - 2.55) < 0.4

    # the same detections without masks take the box-only path, which the tolerances above reject
    tr0, j0, k0 = bus_track(build_tracks(replace(ds, masks=None), mast_camera, PipelineConfig())[0])
    err0 = np.linalg.norm(tr0.xy[k0] - bus.xy[j0], axis=1)
    hd0 = np.degrees(np.abs((tr0.heading[k0] - bus.heading[j0] + np.pi / 2) % np.pi - np.pi / 2))
    assert err0.max() > 0.6 or hd0.max() > 3.0

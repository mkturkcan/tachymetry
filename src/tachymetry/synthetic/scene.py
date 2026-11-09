"""Procedural road scenes rendered through a known camera.

The renderer draws a metric ground texture (asphalt, lane markings, zebra
crosswalks, stop lines, kerbs and sidewalks) in a bird's-eye canvas, then
resamples it through a known camera model, including lens distortion, with
per-pixel mip-mapping so distant markings do not alias. Simple 3D objects
(vehicles as shaded cuboids, poles, building blocks) add vertical structure.

Because the camera is known, these scenes serve as ground truth for the
calibration and lens-rectification methods (see ``docs/examples``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..geometry.camera import Camera

__all__ = [
    "RoadLayout",
    "GroundTexture",
    "Box3D",
    "SyntheticScene",
    "make_ground_texture",
    "render_view",
    "make_scene",
    "default_camera",
]

ASPHALT = np.array([78, 80, 84], dtype=np.float32)
CONCRETE = np.array([150, 150, 146], dtype=np.float32)
WHITE = np.array([232, 232, 228], dtype=np.float32)
YELLOW = np.array([40, 190, 230], dtype=np.float32)  # BGR
GRASS = np.array([70, 112, 84], dtype=np.float32)


@dataclass
class RoadLayout:
    """Geometry of a synthetic road scene (metres)."""

    kind: str = "intersection"  # "intersection" | "straight"
    lanes_ew: int = 2  # lanes per direction on the east-west road
    lanes_ns: int = 2  # lanes per direction on the north-south road
    lane_width: float = 3.5
    sidewalk_width: float = 4.0
    crosswalk_length: float = 3.0  # stripe length along the road
    stripe_width: float = 0.6
    stripe_gap: float = 0.6
    line_width: float = 0.15
    dash_length: float = 3.0
    dash_gap: float = 6.0
    extent: float = 70.0  # half-size of the square canvas
    rotation_deg: float = 0.0  # rotate the whole layout about the origin
    grid_joints: float = 1.5  # sidewalk slab size (adds straight lines in both directions)

    @property
    def half_ew(self) -> float:
        return self.lanes_ew * self.lane_width

    @property
    def half_ns(self) -> float:
        return self.lanes_ns * self.lane_width


@dataclass
class GroundTexture:
    """A metric bird's-eye texture of the ground plane."""

    image: np.ndarray  # BGR uint8, rows run from +y (top) to -y
    x_min: float
    y_max: float
    resolution: float  # metres per pixel
    layout: RoadLayout | None = None
    pyramid: list[np.ndarray] = field(default_factory=list, repr=False)

    def world_to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return np.stack(
            [(xy[..., 0] - self.x_min) / self.resolution, (self.y_max - xy[..., 1]) / self.resolution], axis=-1
        )

    def build_pyramid(self, levels: int = 7) -> None:
        import cv2

        self.pyramid = [self.image.astype(np.float32)]
        for _ in range(levels - 1):
            self.pyramid.append(cv2.pyrDown(self.pyramid[-1]))


@dataclass
class Box3D:
    """An upright box on the ground: centre ``(x, y)``, heading (deg, CCW from +x), size ``(L, W, H)``."""

    x: float
    y: float
    heading: float
    length: float
    width: float
    height: float
    color: tuple[int, int, int] = (60, 60, 200)
    kind: str = "vehicle"  # "vehicle" | "pole" | "building"
    clearance: float = 0.0

    def corners(self) -> np.ndarray:
        """``(8, 3)`` corners: 0-3 at the bottom (front-left, rear-left, rear-right, front-right), 4-7 above."""
        a = math.radians(self.heading)
        f = np.array([math.cos(a), math.sin(a)])
        lft = np.array([-math.sin(a), math.cos(a)])
        hl, hw = self.length / 2, self.width / 2
        c = np.array([self.x, self.y])
        base = [c + hl * f + hw * lft, c - hl * f + hw * lft, c - hl * f - hw * lft, c + hl * f - hw * lft]
        z0, z1 = self.clearance, self.clearance + self.height
        return np.array([[p[0], p[1], z0] for p in base] + [[p[0], p[1], z1] for p in base])


@dataclass
class SyntheticScene:
    image: np.ndarray
    camera: Camera
    texture: GroundTexture
    objects: list[Box3D]


def default_camera(image_size: tuple[int, int] = (1280, 720), lens=None, **kw) -> Camera:
    """A typical pole-mounted intersection camera (8 m high, 32 deg down, 70 deg HFOV)."""
    w, _ = image_size
    hfov = kw.pop("hfov", 70.0)
    focal = kw.pop("focal", (w / 2) / math.tan(math.radians(hfov / 2)))
    params = dict(height=8.0, pitch=32.0, yaw=55.0, roll=1.5, position=(-17.0, -21.0))
    params.update(kw)
    return Camera.from_pose(image_size, focal=focal, lens=lens, **params)


# ----------------------------------------------------------------------------- texture
def _rect(cx: float, cy: float, length: float, width: float, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    f = np.array([math.cos(a), math.sin(a)]) * length / 2
    s = np.array([-math.sin(a), math.cos(a)]) * width / 2
    c = np.array([cx, cy])
    return np.array([c + f + s, c - f + s, c - f - s, c + f - s])


def make_ground_texture(
    layout: RoadLayout | None = None, resolution: float = 0.05, seed: int = 0, supersample: int = 2
) -> GroundTexture:
    """Draw the ground plane of ``layout`` at ``resolution`` metres per pixel."""
    import cv2

    layout = layout or RoadLayout()
    rng = np.random.default_rng(seed)
    E = layout.extent
    res = resolution / supersample
    n = int(round(2 * E / res))
    x_min, y_max = -E, E
    rot = math.radians(layout.rotation_deg)
    Rm = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])

    def to_px(poly: np.ndarray) -> np.ndarray:
        p = poly @ Rm.T
        return np.stack([(p[:, 0] - x_min) / res, (y_max - p[:, 1]) / res], axis=1)

    canvas = np.empty((n, n, 3), np.float32)
    canvas[:] = GRASS

    def fill(poly: np.ndarray, color: np.ndarray) -> None:
        pts = np.round(to_px(poly) * 16).astype(np.int32)
        cv2.fillPoly(canvas, [pts], color.tolist(), lineType=cv2.LINE_AA, shift=4)

    hew, hns, sw = layout.half_ew, layout.half_ns, layout.sidewalk_width
    big = 2 * E
    # sidewalks (concrete bands along both roads), then asphalt on top
    fill(_rect(0, 0, big, 2 * (hew + sw), 0), CONCRETE)
    if layout.kind == "intersection":
        fill(_rect(0, 0, 2 * (hns + sw), big, 0), CONCRETE)
    # sidewalk slab joints: straight lines in both ground directions
    joint = CONCRETE * 0.82
    g = layout.grid_joints
    for k in np.arange(-E, E, g):  # joints across the sidewalks
        for sgn in (-1, 1):
            fill(_rect(k, sgn * (hew + sw / 2), 0.04, sw, 0), joint)
            if layout.kind == "intersection":
                fill(_rect(sgn * (hns + sw / 2), k, sw, 0.04, 0), joint)
    for off in np.arange(hew, hew + sw + 1e-6, g):  # joints along the sidewalks
        for sgn in (-1, 1):
            fill(_rect(0, sgn * off, big, 0.04, 0), joint)
    if layout.kind == "intersection":
        for off in np.arange(hns, hns + sw + 1e-6, g):
            for sgn in (-1, 1):
                fill(_rect(sgn * off, 0, 0.04, big, 0), joint)
    # kerb edge lines (dark) and asphalt
    fill(_rect(0, 0, big, 2 * hew + 0.4, 0), CONCRETE * 0.55)
    fill(_rect(0, 0, big, 2 * hew, 0), ASPHALT)
    if layout.kind == "intersection":
        fill(_rect(0, 0, 2 * hns + 0.4, big, 0), CONCRETE * 0.55)
        fill(_rect(0, 0, 2 * hns, big, 0), ASPHALT)
        fill(_rect(0, 0, 2 * hns, 2 * hew, 0), ASPHALT)

    lw = layout.line_width
    box_x = hns + 0.5 if layout.kind == "intersection" else 0.0  # intersection half-extent along x
    box_y = hew + 0.5 if layout.kind == "intersection" else 0.0
    cw_len = layout.crosswalk_length

    def road_markings(axis: str) -> None:
        half = hew if axis == "x" else hns
        lanes = layout.lanes_ew if axis == "x" else layout.lanes_ns
        box = box_x if axis == "x" else box_y
        start = box + cw_len + 1.5 if layout.kind == "intersection" else 0.0
        for sgn in (-1, 1):
            seg_c = sgn * (start + (E - start) / 2)
            seg_l = E - start
            if layout.kind == "straight":
                seg_c, seg_l = 0.0, 2 * E
            ang = 0 if axis == "x" else 90

            def place(
                off: float, length: float, width: float, color: np.ndarray, along: float = seg_c, ang: float = ang
            ) -> None:
                if axis == "x":
                    fill(_rect(along, off, length, width, ang), color)
                else:
                    fill(_rect(off, along, length, width, ang), color)

            # double yellow centre line and solid white edge lines
            for o in (-0.15, 0.15):
                place(o, seg_l, lw, YELLOW)
            for o in (-half + 0.3, half - 0.3):
                place(o, seg_l, lw, WHITE)
            # dashed lane dividers
            period = layout.dash_length + layout.dash_gap
            for lane in range(1, lanes):
                for side in (-1, 1):
                    off = side * lane * layout.lane_width
                    lo = start if layout.kind == "intersection" else -E
                    pos = lo
                    while pos + layout.dash_length < E:
                        a0 = pos + layout.dash_length / 2
                        place(
                            off, layout.dash_length, lw, WHITE, along=sgn * a0 if layout.kind == "intersection" else a0
                        )
                        pos += period
                    if layout.kind == "straight":
                        break
            if layout.kind == "straight":
                break
            # stop line across the approach half (right-hand traffic: approach lanes on the right)
            if layout.kind == "intersection":
                stop_at = sgn * (box + cw_len + 1.0)
                off = -sgn * half / 2 if axis == "x" else sgn * half / 2
                if axis == "x":
                    fill(_rect(stop_at, off, 0.5, half - 0.3, 0), WHITE)
                else:
                    fill(_rect(off, stop_at, half - 0.3, 0.5, 0), WHITE)
                # zebra crosswalk: stripes parallel to traffic, spanning the road width
                cw_c = sgn * (box + cw_len / 2 + 0.2)
                pitch = layout.stripe_width + layout.stripe_gap
                k = -half + layout.stripe_gap
                while k + layout.stripe_width <= half - layout.stripe_gap + 1e-9:
                    o = k + layout.stripe_width / 2
                    if axis == "x":
                        fill(_rect(cw_c, o, cw_len, layout.stripe_width, 0), WHITE)
                    else:
                        fill(_rect(o, cw_c, layout.stripe_width, cw_len, 0), WHITE)
                    k += pitch

    road_markings("x")
    if layout.kind == "intersection":
        road_markings("y")

    # asphalt grain and low-frequency patches
    lo = cv2.resize(
        rng.normal(0, 1, (max(n // 160, 4), max(n // 160, 4))).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC
    )
    grain = rng.normal(0, 1, (n, n)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 1.2)
    canvas += (5.0 * lo + 4.0 * grain)[..., None]
    img = cv2.resize(np.clip(canvas, 0, 255), (n // supersample, n // supersample), interpolation=cv2.INTER_AREA)
    tex = GroundTexture(img.astype(np.uint8), x_min, y_max, resolution, layout)
    tex.build_pyramid()
    return tex


# ------------------------------------------------------------------------------ render
def _remap_points(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``img`` at float coordinates (any number of points)."""
    import cv2

    n = len(x)
    cols = 1024
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    mx = np.pad(x.astype(np.float32), (0, pad)).reshape(rows, cols)
    my = np.pad(y.astype(np.float32), (0, pad)).reshape(rows, cols)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out.reshape(rows * cols, -1)[:n]


def _sample_ground(camera: Camera, tex: GroundTexture) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel ground colour (mip-mapped) and a validity mask."""

    w, h = camera.image_size
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    xy = camera.image_to_ground(uv)
    valid = np.all(np.isfinite(xy), axis=1)
    # pixel footprint on the ground: distance * (1 / focal) / cos(incidence)
    C = camera.center
    d = camera.rays(uv)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    dist = np.where(valid, np.linalg.norm(np.column_stack([xy, np.zeros(len(xy))]) - C, axis=1), np.inf)
    cos_inc = np.clip(np.abs(dn[:, 2]), 1e-3, 1)
    gsd = dist / camera.focal / np.sqrt(cos_inc)
    level = np.clip(np.log2(np.maximum(gsd / tex.resolution, 1e-6)), 0, len(tex.pyramid) - 1.001)
    px = tex.world_to_px(np.where(valid[:, None], xy, 0))
    out = np.zeros((len(uv), 3), np.float32)
    lvl0 = np.floor(level).astype(int)
    frac = (level - lvl0).astype(np.float32)
    for L in range(len(tex.pyramid) - 1):
        sel = valid & (lvl0 == L)
        if not sel.any():
            continue
        acc = np.zeros((sel.sum(), 3), np.float32)
        for j, wgt in ((L, 1 - frac[sel]), (L + 1, frac[sel])):
            scale = 2.0**j
            samp = _remap_points(tex.pyramid[j], px[sel, 0] / scale, px[sel, 1] / scale)
            acc += samp.reshape(-1, 3) * wgt[:, None]
        out[sel] = acc
    return out.reshape(h, w, 3), valid.reshape(h, w)


def _project_poly(camera: Camera, pts3: np.ndarray, subdiv: int = 8) -> np.ndarray | None:
    """Project a planar 3D polygon with edge subdivision (so distorted edges curve correctly)."""
    dense = []
    for i in range(len(pts3)):
        a, b = pts3[i], pts3[(i + 1) % len(pts3)]
        for s in np.linspace(0, 1, subdiv, endpoint=False):
            dense.append(a + s * (b - a))
    uv = camera.project(np.asarray(dense))
    if not np.all(np.isfinite(uv)):
        return None
    return uv


def render_view(
    camera: Camera,
    texture: GroundTexture,
    objects: Sequence[Box3D] = (),
    seed: int = 0,
    noise: float = 2.0,
    vignette: float = 0.25,
) -> np.ndarray:
    """Render the raw (distorted) image seen by ``camera``."""
    import cv2

    rng = np.random.default_rng(seed)
    w, h = camera.image_size
    ground, valid = _sample_ground(camera, texture)
    # sky gradient where rays miss the ground
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky = (1 - t) * np.array([235, 205, 170], np.float32) + t * np.array([250, 238, 225], np.float32)
    img = np.where(valid[..., None], ground, np.broadcast_to(sky, (h, w, 3)))
    # haze with distance
    xy_c = camera.center
    img = img.astype(np.float32)

    # objects (painter's algorithm, far to near)
    def depth(o: Box3D) -> float:
        return -float(np.hypot(o.x - xy_c[0], o.y - xy_c[1]))

    for o in sorted(objects, key=depth):
        cr = o.corners()
        faces = [
            (cr[[4, 5, 6, 7]], 1.15),  # roof
            (cr[[0, 1, 5, 4]], 0.85),
            (cr[[1, 2, 6, 5]], 0.7),
            (cr[[2, 3, 7, 6]], 0.8),
            (cr[[3, 0, 4, 7]], 0.95),
        ]
        # back-face culling by viewing direction
        C = camera.center
        drawable = []
        for poly, shade in faces:
            nrm = np.cross(poly[1] - poly[0], poly[2] - poly[0])
            ctr = poly.mean(axis=0)
            if np.dot(nrm, C - ctr) <= 0:
                nrm = -nrm
            drawable.append((poly, shade, float(np.linalg.norm(C - ctr))))
        drawable.sort(key=lambda d: -d[2])
        if o.kind == "vehicle":
            # soft contact shadow
            base = cr[[0, 1, 2, 3]].copy()
            base[:, 2] = 0
            uv = _project_poly(camera, base)
            if uv is not None:
                mask = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mask, [np.round(uv * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
                mask = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32) / 255
                img *= (1 - 0.45 * mask)[..., None]
        for poly, shade, _ in drawable:
            uv = _project_poly(camera, poly)
            if uv is None:
                continue
            col = np.clip(np.asarray(o.color, np.float32) * shade, 0, 255)
            cv2.fillPoly(img, [np.round(uv * 16).astype(np.int32)], col.tolist(), cv2.LINE_AA, shift=4)
        if o.kind == "vehicle":  # windows: a darker band on the upper sides
            a, b = 0.55, 0.9
            for idx in ([0, 1], [1, 2], [2, 3], [3, 0]):
                p0, p1 = cr[idx[0]], cr[idx[1]]
                lo0, lo1 = p0 + [0, 0, a * o.height], p1 + [0, 0, a * o.height]
                hi0, hi1 = p0 + [0, 0, b * o.height], p1 + [0, 0, b * o.height]
                poly = np.array([lo0, lo1, hi1, hi0])
                ctr = poly.mean(axis=0)
                nrm = np.cross(lo1 - lo0, hi0 - lo0)
                outward = ctr[:2] - np.array([o.x, o.y])
                if np.dot(nrm[:2], outward) < 0:
                    nrm = -nrm
                if np.dot(nrm, camera.center - ctr) <= 0:
                    continue
                uv = _project_poly(camera, poly, subdiv=4)
                if uv is not None:
                    cv2.fillPoly(img, [np.round(uv * 16).astype(np.int32)], (52, 44, 40), cv2.LINE_AA, shift=4)

    # photometric realism: vignette + sensor noise
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    img *= (1 - vignette * 0.5 * r2)[..., None]
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def scene_objects(
    layout: RoadLayout, seed: int = 0, n_vehicles: int = 10, poles: bool = True, buildings: bool = True
) -> list[Box3D]:
    """Plausible static scene content: queued and parked vehicles, signal poles, building blocks."""
    rng = np.random.default_rng(seed)
    objs: list[Box3D] = []
    palette = [
        (200, 200, 205),
        (40, 40, 45),
        (60, 60, 180),
        (170, 120, 50),
        (90, 90, 95),
        (225, 225, 230),
        (40, 120, 200),
        (120, 60, 50),
    ]
    hew, hns, lane = layout.half_ew, layout.half_ns, layout.lane_width
    box_x = hns + 0.5 if layout.kind == "intersection" else 0.0
    for i in range(n_vehicles):
        side = rng.choice([-1, 1])
        lane_k = rng.integers(0, layout.lanes_ew)
        off = -side * (lane_k + 0.5) * lane
        x = side * (box_x + 6 + 7.0 * i / max(n_vehicles / 2, 1) * 3 + rng.uniform(0, 2))
        L, W, H = rng.uniform(4.2, 5.0), rng.uniform(1.75, 1.95), rng.uniform(1.4, 1.8)
        if rng.random() < 0.12:
            L, W, H = rng.uniform(9, 12), 2.5, rng.uniform(2.8, 3.4)
        heading = 180.0 if side > 0 else 0.0
        objs.append(
            Box3D(
                float(x),
                float(off),
                heading + rng.normal(0, 1.0),
                L,
                W,
                H,
                color=palette[i % len(palette)],
                clearance=0.2,
            )
        )
    if poles:
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            objs.append(Box3D(sx * (hns + 1.2), sy * (hew + 1.2), 0, 0.3, 0.3, 6.5, color=(70, 70, 72), kind="pole"))
    if buildings:
        d = layout.sidewalk_width + 1.0
        for sx, sy in ((-1, 1), (1, 1), (1, -1)):
            cx, cy = sx * (hns + d + 12), sy * (hew + d + 12)
            objs.append(Box3D(cx, cy, 0, 22, 22, rng.uniform(12, 25), color=(150, 160, 175), kind="building"))
    return objs


def make_scene(
    camera: Camera | None = None,
    layout: RoadLayout | None = None,
    seed: int = 0,
    objects: bool | Sequence[Box3D] = True,
    resolution: float = 0.05,
) -> SyntheticScene:
    """Render a complete synthetic scene. Returns image, camera, ground texture and objects."""
    camera = camera or default_camera()
    layout = layout or RoadLayout()
    tex = make_ground_texture(layout, resolution=resolution, seed=seed)
    if objects is True:
        objs = scene_objects(layout, seed=seed)
    elif objects is False:
        objs = []
    else:
        objs = list(objects)
    img = render_view(camera, tex, objs, seed=seed)
    return SyntheticScene(img, camera, tex, objs)

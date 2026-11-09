"""Pipeline configuration.

Thresholds are expressed in physical units (seconds, metres, metres per
second) or as fractions of the image, not in samples or pixels, so a
configuration does not depend on the frame rate or the image resolution.

Configurations round-trip through YAML::

    cfg = PipelineConfig.load("my.yaml")      # partial files override defaults
    cfg.detector.imgsz = 1280
    cfg.save("run/config.yaml")
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DetectorConfig",
    "TrackerConfig",
    "HygieneConfig",
    "FilterConfig",
    "RenderConfig",
    "CalibrationConfig",
    "PipelineConfig",
    "ROAD_CLASSES",
    "VEHICLE_CLASSES",
    "TWO_WHEELER_CLASSES",
    "SUPER_CLASS",
]

#: Road-user taxonomy shared by all detector back-ends.
ROAD_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
VEHICLE_CLASSES = ("car", "bus", "truck")
TWO_WHEELER_CLASSES = ("bicycle", "motorcycle")
#: Association is gated by super-class, so car/truck flicker never splits a track.
SUPER_CLASS = {
    "person": "person",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
}


@dataclass
class DetectorConfig:
    """Object detector and detection-level post-processing."""

    name: str = "uod2-x"  #: ``uod2-{n,s,m,l,x}``, ``yolo:<weights>``, ``precomputed:<dir>``
    weights: str | None = None  #: explicit checkpoint path (overrides the download)
    imgsz: int = 960  #: inference size; 960 recovers small, distant traffic
    conf: float = 0.10  #: model threshold; low-score detections only extend existing tracks
    class_conf: dict[str, float] = field(default_factory=lambda: {"person": 0.40, "bicycle": 0.50, "motorcycle": 0.50})
    classes: tuple[str, ...] = ROAD_CLASSES
    cuboid_conf: float = 0.5  #: minimum per-corner cuboid confidence (UrbanOmniDetect-2)
    nested_containment: float = 0.85  #: drop a vehicle box this much inside a larger vehicle box ...
    nested_ground_overlap: float = 0.30  #: ... when their ground footprints also overlap this much
    batch: int = 8
    half: bool = True  #: FP16 inference on CUDA
    max_box_width_frac: float = 1.01  #: drop boxes wider than this fraction of the frame (1.01 = off)


@dataclass
class TrackerConfig:
    """Online association (two-stage BYTE with a constant-velocity Kalman filter)."""

    high_conf: float = 0.50  #: detections above this start tracks and match first
    low_conf: float = 0.10  #: second-stage detections (extend tracks, never start them)
    match_cost: float = 0.80  #: maximum association cost (0 = perfect)
    min_iou_low: float = 0.20  #: IoU required in the low-score stage
    max_lost_s: float = 1.5  #: keep unmatched tracks alive this long
    min_confirm_s: float = 0.15  #: confirmation delay for new tracks
    super_class_gating: bool = True
    max_speed_mps: dict[str, float] = field(
        default_factory=lambda: {"vehicle": 50.0, "two_wheeler": 20.0, "person": 7.0}
    )  #: BEV plausibility gate (with a camera)


@dataclass
class HygieneConfig:
    """Offline track repair: splitting, trimming, smoothing and validity."""

    split: bool = True  #: presplit at teleports/gaps and identity switches
    trim: bool = True  #: trim truncated, frame-filling and far-field-collapsed track ends
    repair: bool = True  #: Hampel repair of isolated outliers
    parked: bool = True  #: classify and freeze parked vehicles
    validity: bool = True  #: reject physically impossible tracks
    footprints: bool = True  #: estimate footprint centres (else use the box bottom-centre)
    min_track_s: float = 0.8
    max_gap_s: float = 5.0  #: split tracks at longer detection gaps
    interp_gap_s: float = 1.5  #: interpolate gaps up to this long before smoothing
    teleport_speed_mps: dict[str, float] = field(
        default_factory=lambda: {"vehicle": 55.0, "two_wheeler": 25.0, "person": 8.0}
    )
    position_noise_m: float = 0.25  #: per-sample ground noise floor at 10 m (grows with range)
    switch_jump: float = 1.2  #: identity switch: box centre jump over the predicted box size
    switch_area: tuple[float, float] = (0.45, 2.2)  #: ... or an area ratio outside this band
    edge_margin_frac: float = 0.004  #: boxes this close to the border are truncated
    big_box_frac: float = 0.45  #: boxes larger than this fraction of the frame are degenerate
    far_shrink: float = 0.5  #: far-field collapse: box height below this fraction of its running median
    fold_deg: float = 45.0  #: end hooks: a turn this sharp ...
    fold_dist_m: float = 2.0  #: ... within this distance at either end
    hampel_window_s: float = 1.0
    hampel_sigma: float = 3.0
    hampel_min_m: float = 0.8
    median_window_s: float = 0.4  #: impulse rejection before smoothing
    smooth_window_s: float = 1.2  #: Savitzky-Golay window
    smooth_order: int = 2
    max_gsd_m: float = 0.30  #: discard ground points where one pixel spans more than this (worst axis)
    max_range_m: float | None = None  #: optional hard range limit
    stop_speed_kmh: float = 1.5
    move_speed_kmh: float = 3.0
    hold_still: bool = True  #: freeze positions while stopped (removes phantom speed)
    parked_radius_m: float = 1.2  #: radius of gyration of a parked vehicle
    parked_moving_share: float = 0.30
    impossible_speed_mps: dict[str, float] = field(
        default_factory=lambda: {"vehicle": 60.0, "two_wheeler": 25.0, "person": 9.0}
    )
    max_accel_mps2: float = 12.0
    accel_duration_s: float = 0.6
    reversal_deg: float = 150.0
    reversal_min_m: float = 1.5
    inside_box_margin: float = 0.15
    inside_box_share: float = 0.70


@dataclass
class FilterConfig:
    """Track-level false-positive verdicts and merging."""

    enabled: bool = True  #: run the cross-track verdicts (ghosts, duplicates, passengers, riders, shadows)
    ghost_min_duration_s: float = 1.0
    ghost_min_score: float = 0.35
    stitch: bool = True
    stitch_max_gap_s: float = 3.0
    stitch_radius_m: float = 2.5
    stitch_max_turn_deg: float = 50.0
    passenger_ioa: float = 0.70
    passenger_share: float = 0.60
    rider_overlap: float = 0.30
    rider_share: float = 0.50
    rider_distance_m: float = 1.2
    duplicate_distance_m: float = 1.5
    duplicate_share: float = 0.5
    duplicate_min_s: float = 1.0
    shadow_offset_std_m: float = 0.25
    shadow_min_s: float = 3.0


@dataclass
class CalibrationConfig:
    """Automatic calibration."""

    methods: tuple[str, ...] = ("cuboids", "lines")  #: tried in order; also ``depth``, ``geocalib``
    scale: str = "vehicles"  #: ``vehicles``, ``height=<m>``, ``lane_width=<m>`` or ``none``
    lens: str = "auto"  #: ``auto`` (tan or division, by vanishing-point consistency), ``tan``, ``division``, ``none``
    background_frames: int = 41
    car_length_m: float = 4.6
    car_width_m: float = 1.85
    vp_sigma_deg: float = 0.5  #: angular uncertainty of vanishing points in the fused estimate


@dataclass
class RenderConfig:
    """Visual output."""

    layout: str = "side"  #: ``side``, ``camera``, ``bev``, ``dashboard``
    draw: str = "auto"  #: ``auto`` (cuboids when available), ``box``, ``cuboid``, ``footprint``
    units: str = "km/h"  #: ``km/h``, ``mph`` or ``m/s``
    show_parked: bool = True  #: parked vehicles in a muted style, without speed
    show_ids: bool = False
    trail_s: float = 3.0
    max_width: int = 1920
    fps: float | None = None  #: output frame rate (source rate by default)
    crf: int = 18
    title: str | None = None
    bev_range_m: float | None = None


@dataclass
class PipelineConfig:
    """Complete configuration of a run."""

    rate_hz: float = 15.0  #: processing rate (frames per second analysed); 0 = every frame
    device: str = "auto"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    hygiene: HygieneConfig = field(default_factory=HygieneConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    exports: tuple[str, ...] = ("csv", "json")  #: ``csv``, ``parquet``, ``json``, ``geojson``, ``mot``
    keep_detections: bool = True

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PipelineConfig:
        cfg = cls()
        if d:
            _update(cfg, d)
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            import yaml

            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def override(self, dotted: dict[str, Any]) -> PipelineConfig:
        """Apply ``{"detector.imgsz": 1280, ...}`` style overrides in place and return self."""
        for key, value in dotted.items():
            obj: Any = self
            parts = key.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            if not hasattr(obj, parts[-1]):
                raise KeyError(f"unknown config key {key!r}")
            setattr(obj, parts[-1], _coerce(getattr(obj, parts[-1]), value))
        return self


def _to_plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_plain(v) for v in x]
    return x


def _coerce(current: Any, value: Any) -> Any:
    if isinstance(current, bool) and isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, tuple) and isinstance(value, (list, str)):
        return tuple(value.split(",")) if isinstance(value, str) else tuple(value)
    if isinstance(current, (int, float)) and not isinstance(current, bool) and isinstance(value, str):
        return type(current)(value)
    if current is None and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _update(obj: Any, d: dict[str, Any]) -> None:
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key {type(obj).__name__}.{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        elif isinstance(cur, dict) and isinstance(v, dict):
            merged = dict(cur)
            merged.update(v)
            setattr(obj, k, merged)
        else:
            setattr(obj, k, _coerce(cur, v))

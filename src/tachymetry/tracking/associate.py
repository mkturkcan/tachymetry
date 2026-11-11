"""Online multi-object association.

A two-stage BYTE association (Zhang et al., ECCV 2022) with a constant-velocity
Kalman filter in continuous time, so the tracker is correct at any processing
rate. Additions over the reference formulation:

* **DIoU cost** with shape gates: non-overlapping boxes of fast objects at low
  frame rates still associate, but boxes of a different size or aspect never do.
* **Super-class gating** (vehicle / two-wheeler / person): car-truck label
  flicker cannot split a track; the class is decided later by a weighted vote.
* **Metric plausibility gate** when a calibrated camera is available: an
  association that implies a ground speed above the class limit (plus the
  range-dependent localisation noise) is forbidden. This removes the classic
  identity swap between a parked car and a passing one.
* Tentative tracks need confirmation; lost tracks coast with damped velocity
  and die when their prediction leaves the image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import SUPER_CLASS, TrackerConfig

__all__ = ["ByteTracker", "diou_cost", "iou_matrix"]

_REF_DT = 1.0 / 30.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def diou_cost(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``((1 - DIoU) / 2, IoU)`` for all pairs; the cost lies in ``[0, 1]``."""
    iou = iou_matrix(a, b)
    if iou.size == 0:
        return iou, iou
    ca = 0.5 * (a[:, None, :2] + a[:, None, 2:])
    cb = 0.5 * (b[None, :, :2] + b[None, :, 2:])
    rho2 = ((ca - cb) ** 2).sum(-1)
    ex1 = np.minimum(a[:, None, 0], b[None, :, 0])
    ey1 = np.minimum(a[:, None, 1], b[None, :, 1])
    ex2 = np.maximum(a[:, None, 2], b[None, :, 2])
    ey2 = np.maximum(a[:, None, 3], b[None, :, 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    diou = iou - rho2 / np.maximum(c2, 1e-9)
    return (1.0 - diou) / 2.0, iou


@dataclass
class _Track:
    tid: int
    x: np.ndarray  # [cx, cy, w, h, vcx, vcy, vw, vh] (pixels, pixels/s)
    P: np.ndarray
    sup: str
    t_last: float
    t_start: float
    hits: int = 1
    confirmed: bool = False
    ground: np.ndarray | None = None
    t_ground: float = 0.0
    lost_since: float | None = None
    votes: dict[int, float] = field(default_factory=dict)

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _to_cxcywh(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=np.float64)


class ByteTracker:
    """Online tracker. Call :meth:`step` once per processed frame, in time order."""

    def __init__(
        self,
        cfg: TrackerConfig,
        class_names: tuple[str, ...],
        image_size: tuple[int, int],
        groups: dict[str, str] | None = None,
    ):
        """``groups`` overrides the association group of each class (default: its super-class)."""
        self.cfg = cfg
        self.class_names = class_names
        self.image_size = image_size
        self.tracks: list[_Track] = []
        self.next_id = 1
        if groups is not None and cfg.super_class_gating:
            self._sup = [groups.get(c, SUPER_CLASS.get(c, c)) for c in class_names]
        else:
            self._sup = [SUPER_CLASS.get(c, c) if cfg.super_class_gating else c for c in class_names]

    # ------------------------------------------------------------- Kalman
    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        return F

    def _Q(self, h: float, dt: float) -> np.ndarray:
        sp = h / 20.0
        sv = (h / 160.0) / _REF_DT
        k = max(dt / _REF_DT, 1e-3)
        return np.diag([sp, sp, sp, sp, sv, sv, sv, sv]) ** 2 * k

    @staticmethod
    def _R(h: float) -> np.ndarray:
        s = h / 20.0
        return np.diag([s, s, s, s]) ** 2

    def _predict(self, tr: _Track, t: float) -> None:
        dt = t - tr.t_last
        if dt <= 0:
            return
        x = self._F(dt) @ tr.x
        if tr.lost_since is not None and t - tr.lost_since > 0.3:
            x[4:] *= 0.85 ** (dt / _REF_DT * 0.1)  # gentle damping while coasting
        tr.x = x
        tr.P = self._F(dt) @ tr.P @ self._F(dt).T + self._Q(max(tr.x[3], 2.0), dt)
        tr.t_last = t

    def _update(self, tr: _Track, box: np.ndarray, t: float) -> None:
        z = _to_cxcywh(box)
        H = np.zeros((4, 8))
        H[:, :4] = np.eye(4)
        S = H @ tr.P @ H.T + self._R(max(z[3], 2.0))
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - H @ tr.x)
        tr.P = (np.eye(8) - K @ H) @ tr.P

    def _new(self, box: np.ndarray, score: float, cls: int, t: float, ground: np.ndarray | None) -> _Track:
        z = _to_cxcywh(box)
        h = max(z[3], 2.0)
        P = (
            np.diag(
                [
                    h / 10,
                    h / 10,
                    h / 10,
                    h / 10,
                    10 * h / 160 / _REF_DT,
                    10 * h / 160 / _REF_DT,
                    10 * h / 160 / _REF_DT,
                    10 * h / 160 / _REF_DT,
                ]
            )
            ** 2
        )
        tr = _Track(
            self.next_id,
            np.concatenate([z, np.zeros(4)]),
            P,
            self._sup[cls],
            t,
            t,
            ground=None if ground is None or not np.all(np.isfinite(ground)) else ground.copy(),
            t_ground=t,
        )
        tr.votes[cls] = score
        self.next_id += 1
        return tr

    # ------------------------------------------------------------- gating
    def _gate(
        self,
        cost: np.ndarray,
        iou: np.ndarray,
        tracks: list[_Track],
        boxes: np.ndarray,
        cls: np.ndarray,
        ground: np.ndarray | None,
        gsd: np.ndarray | None,
        t: float,
        min_iou: float,
    ) -> np.ndarray:
        big = 1e6
        cost = cost.copy()
        for i, tr in enumerate(tracks):
            tb = tr.box()
            ta = max((tb[2] - tb[0]) * (tb[3] - tb[1]), 1.0)
            tasp = max(tb[2] - tb[0], 1.0) / max(tb[3] - tb[1], 1.0)
            for j in range(len(boxes)):
                if self._sup[int(cls[j])] != tr.sup:
                    cost[i, j] = big
                    continue
                if iou[i, j] < min_iou:
                    cost[i, j] = big
                    continue
                b = boxes[j]
                da = abs(np.log(max((b[2] - b[0]) * (b[3] - b[1]), 1.0) / ta))
                dasp = abs(np.log((max(b[2] - b[0], 1.0) / max(b[3] - b[1], 1.0)) / tasp))
                if da > 0.92 or dasp > 0.79:
                    cost[i, j] = big
                    continue
                if ground is not None and tr.ground is not None and np.all(np.isfinite(ground[j])):
                    dt = max(t - tr.t_ground, 1e-3)
                    vmax = self.cfg.max_speed_mps.get(tr.sup, 50.0)
                    noise = 1.0 + (5.0 * gsd[j] if gsd is not None and np.isfinite(gsd[j]) else 0.0)
                    if np.linalg.norm(ground[j] - tr.ground) > vmax * dt + noise:
                        cost[i, j] = big
        return cost

    @staticmethod
    def _assign(cost: np.ndarray, thresh: float) -> list[tuple[int, int]]:
        from scipy.optimize import linear_sum_assignment

        if cost.size == 0:
            return []
        r, c = linear_sum_assignment(cost)
        return [(i, j) for i, j in zip(r, c) if cost[i, j] <= thresh]

    # --------------------------------------------------------------- step
    def step(
        self,
        t: float,
        xyxy: np.ndarray,
        score: np.ndarray,
        cls: np.ndarray,
        ground: np.ndarray | None = None,
        gsd: np.ndarray | None = None,
    ) -> np.ndarray:
        """Associate one frame. Returns the track id of every detection (``-1`` if unassigned).

        ``ground`` holds each detection's ground contact (metres) when a camera
        is known, enabling the plausibility gate; ``gsd`` its ground sampling distance.
        """
        cfg = self.cfg
        n = len(score)
        ids = np.full(n, -1, dtype=np.int64)
        for tr in self.tracks:
            self._predict(tr, t)
        high = np.flatnonzero(score >= cfg.high_conf)
        low = np.flatnonzero((score >= cfg.low_conf) & (score < cfg.high_conf))

        # stage 1: all tracks vs high-score detections
        tracks = list(self.tracks)
        boxes_t = np.array([tr.box() for tr in tracks]) if tracks else np.zeros((0, 4))
        matched_tracks: set[int] = set()
        if len(tracks) and len(high):
            cost, iou = diou_cost(boxes_t, xyxy[high])
            cost = self._gate(
                cost,
                iou,
                tracks,
                xyxy[high],
                cls[high],
                None if ground is None else ground[high],
                None if gsd is None else gsd[high],
                t,
                0.0,
            )
            for i, j in self._assign(cost, cfg.match_cost / 2 + 0.25):
                self._commit(tracks[i], high[j], xyxy, score, cls, ground, t, ids)
                matched_tracks.add(i)
        rem_high = [j for j in high if ids[j] < 0]
        # stage 2: remaining confirmed tracks vs low-score detections (strict IoU)
        rem_tracks = [i for i in range(len(tracks)) if i not in matched_tracks and tracks[i].confirmed]
        if rem_tracks and len(low):
            sub = [tracks[i] for i in rem_tracks]
            cost, iou = diou_cost(boxes_t[rem_tracks], xyxy[low])
            cost = self._gate(
                cost,
                iou,
                sub,
                xyxy[low],
                cls[low],
                None if ground is None else ground[low],
                None if gsd is None else gsd[low],
                t,
                cfg.min_iou_low,
            )
            for i, j in self._assign(cost, 0.5):
                self._commit(sub[i], low[j], xyxy, score, cls, ground, t, ids)
                matched_tracks.add(rem_tracks[i])
        # births
        for j in rem_high:
            if ids[j] >= 0:
                continue
            tr = self._new(xyxy[j], float(score[j]), int(cls[j]), t, None if ground is None else ground[j])
            self.tracks.append(tr)
            ids[j] = tr.tid
        # lost / dead
        w, h = self.image_size
        alive = []
        for k, tr in enumerate(self.tracks):
            if tr.t_last == t and k < len(tracks) and k not in matched_tracks and tr.lost_since is None:
                tr.lost_since = t
            if tr.lost_since is not None:
                age = t - tr.lost_since
                if not tr.confirmed and age > 0:
                    continue  # tentative tracks die on their first miss
                if age > cfg.max_lost_s:
                    continue
                cx, cy = tr.x[:2]
                if not (-0.05 * w <= cx <= 1.05 * w and -0.05 * h <= cy <= 1.05 * h):
                    continue
            alive.append(tr)
        self.tracks = alive
        return ids

    def _commit(
        self,
        tr: _Track,
        j: int,
        xyxy: np.ndarray,
        score: np.ndarray,
        cls: np.ndarray,
        ground: np.ndarray | None,
        t: float,
        ids: np.ndarray,
    ) -> None:
        self._update(tr, xyxy[j], t)
        tr.hits += 1
        tr.lost_since = None
        c = int(cls[j])
        b = xyxy[j]
        tr.votes[c] = tr.votes.get(c, 0.0) + float(score[j]) * float(np.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0)))
        if not tr.confirmed and (tr.hits >= 2 and t - tr.t_start >= self.cfg.min_confirm_s * 0.5):
            tr.confirmed = True
        if ground is not None and np.all(np.isfinite(ground[j])):
            tr.ground = ground[j].copy()
            tr.t_ground = t
        ids[j] = tr.tid

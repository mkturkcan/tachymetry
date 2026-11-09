"""Calibration result container."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.camera import Camera, _json_default

__all__ = ["CalibrationResult"]


@dataclass
class CalibrationResult:
    """A calibrated camera plus the evidence that supports it.

    Attributes
    ----------
    camera:
        The calibrated :class:`~tachymetry.geometry.Camera`.
    method:
        Name of the method that produced it (``manual``, ``lines``, ``cuboids``, ...).
    rmse_m:
        In-sample ground-plane residual (metres), when correspondences exist.
    cv_rmse_m:
        Leave-one-out ground-plane residual (metres): the honest accuracy figure.
    score:
        Method-independent self-consistency score in ``[0, 1]`` (higher is better),
        used by :func:`tachymetry.calib.auto` to rank candidates.
    details:
        Free-form diagnostics (vanishing points, inlier counts, fitted lens, ...).
    """

    camera: Camera
    method: str
    rmse_m: float | None = None
    cv_rmse_m: float | None = None
    residuals_m: np.ndarray | None = None
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"[{self.method}] {self.camera.describe()}"]
        if self.rmse_m is not None:
            parts.append(f"fit RMSE {self.rmse_m:.3f} m")
        if self.cv_rmse_m is not None:
            parts.append(f"leave-one-out RMSE {self.cv_rmse_m:.3f} m")
        if self.score is not None:
            parts.append(f"consistency {self.score:.3f}")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "rmse_m": self.rmse_m,
            "cv_rmse_m": self.cv_rmse_m,
            "score": self.score,
            "residuals_m": None if self.residuals_m is None else np.asarray(self.residuals_m).tolist(),
            "details": self.details,
            "camera": self.camera.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        """Write the camera (``path``) and a sidecar report (``<stem>.report.json``)."""
        path = Path(path)
        self.camera.meta.setdefault("calibration", {})
        self.camera.meta["calibration"].update(
            {"method": self.method, "rmse_m": self.rmse_m, "cv_rmse_m": self.cv_rmse_m, "score": self.score}
        )
        self.camera.save(path)
        report = path.with_name(path.stem + ".report.json")
        report.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        return path

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CalibrationResult({self.summary()})"

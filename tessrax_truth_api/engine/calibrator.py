"""Adaptive calibration scaffold for the Truth API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from tessrax_truth_api.utils import base_metrics_snapshot, load_config


@dataclass
class CalibrationSnapshot:
    """Represents the integrity and drift posture of the system."""

    integrity: float
    drift: float
    severity: float


class Calibrator:
    """Expose calibration derived thresholds and metrics."""

    def __init__(self) -> None:
        self._config = load_config()

    @property
    def thresholds(self) -> Dict[str, float]:
        return self._config.get("thresholds", {})

    def metrics(self) -> CalibrationSnapshot:
        metrics_dict = base_metrics_snapshot(self._config.get("calibration", {}))
        return CalibrationSnapshot(**metrics_dict)

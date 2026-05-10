"""Calibration splits, metrics, and public trade ingestion for empirical studies."""

from metaorder_signal.empirical.calibration import CalibratedParams, calibrate_from_trades
from metaorder_signal.empirical.metrics import StudyMetrics, baseline_random_signs, compute_metrics

__all__ = [
    "CalibratedParams",
    "calibrate_from_trades",
    "StudyMetrics",
    "compute_metrics",
    "baseline_random_signs",
]

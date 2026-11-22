# Copyright (c) Facebook, Inc. and its affiliates.
"""Score calibration for detector outputs.

Fits a strictly increasing, per-class probability map for raw detector
scores, with shrinkage toward global parameters for long-tail classes,
per-class operating points, domain-shift reporting, and deployment-safe
serialization. Pure NumPy/CPU; no new hard dependencies.
"""

from .api import (
    CALIBRATED_SCORE_FIELD,
    calibrate_instances,
    fit_calibrator,
    fit_calibrator_distributed,
    fit_calibrator_from_data,
    gather_calibration_data,
    merge_calibration_data,
)
from .calibrator import (
    MODE_FITTED,
    MODE_GLOBAL,
    MODE_SHRUNK,
    CalibrationReport,
    ClassCalibrationInfo,
    ScoreCalibrator,
    fit_platt,
)
from .errors import (
    CalibrationError,
    CalibrationFormatError,
    CalibrationMismatchError,
    CalibrationRangeError,
    CalibrationVersionError,
    DegenerateFitError,
)
from .io import load_calibration, save_calibration, try_load_calibration
from .metrics import (
    average_precision,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_table,
)
from .operating_points import (
    OperatingPoint,
    solve_thresholds_at_fpr,
    uniform_threshold_at_fpr,
)
from .shift import ClassShift, ShiftReport, detect_domain_shift, ks_statistic

__all__ = [
    "CALIBRATED_SCORE_FIELD",
    "MODE_FITTED",
    "MODE_GLOBAL",
    "MODE_SHRUNK",
    "CalibrationError",
    "CalibrationFormatError",
    "CalibrationMismatchError",
    "CalibrationRangeError",
    "CalibrationReport",
    "CalibrationVersionError",
    "ClassCalibrationInfo",
    "ClassShift",
    "DegenerateFitError",
    "OperatingPoint",
    "ScoreCalibrator",
    "ShiftReport",
    "average_precision",
    "brier_score",
    "calibrate_instances",
    "detect_domain_shift",
    "expected_calibration_error",
    "fit_calibrator",
    "fit_calibrator_distributed",
    "fit_calibrator_from_data",
    "fit_platt",
    "gather_calibration_data",
    "ks_statistic",
    "load_calibration",
    "log_loss",
    "merge_calibration_data",
    "reliability_table",
    "save_calibration",
    "solve_thresholds_at_fpr",
    "try_load_calibration",
    "uniform_threshold_at_fpr",
]

# Copyright (c) Facebook, Inc. and its affiliates.
"""Persistence for calibration parameters.

Calibration parameters and the class set they were fit for are stored in one
artifact and validated together at load time: a class-set mismatch is an
explicit refusal (:class:`CalibrationMismatchError`), never a silent
positional alignment. :func:`try_load_calibration` implements the fallback
path required for deployment: anything wrong with the artifact (missing,
unreadable, wrong version, class mismatch) yields ``None`` plus an observable
log record, and callers continue uncalibrated.

Log records emitted here contain counts and positions only — never class
names, image paths, or annotations.
"""

import json
import logging
import os
from typing import Optional, Sequence

from .calibrator import ScoreCalibrator
from .errors import (
    CalibrationFormatError,
    CalibrationMismatchError,
    CalibrationVersionError,
)

__all__ = ["save_calibration", "load_calibration", "try_load_calibration"]

logger = logging.getLogger(__name__)


def save_calibration(calibrator: ScoreCalibrator, path: str, class_ids: Sequence) -> None:
    """Serialize calibration parameters together with the class set.

    Args:
        calibrator: the fitted calibrator.
        path: destination JSON file.
        class_ids: ordered class identifiers (names or ids) the parameters
            belong to. Stored verbatim and checked at load time.
    """
    class_ids = list(class_ids)
    if len(class_ids) != calibrator.num_classes:
        raise ValueError(
            f"class_ids has {len(class_ids)} entries but the calibrator has "
            f"{calibrator.num_classes} classes"
        )
    payload = {
        "format_version": ScoreCalibrator.FORMAT_VERSION,
        "class_ids": class_ids,
        "num_classes": calibrator.num_classes,
        **calibrator.to_state_dict(),
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def load_calibration(
    path: str, expected_class_ids: Optional[Sequence] = None
) -> ScoreCalibrator:
    """Load a calibration artifact, validating version and class set.

    Args:
        path: artifact written by :func:`save_calibration`.
        expected_class_ids: the class set of the model being deployed. When
            given, it must equal the stored set exactly (same order); a
            mismatch raises :class:`CalibrationMismatchError`.

    Raises:
        CalibrationVersionError: unsupported artifact version.
        CalibrationMismatchError: stored and expected class sets differ.
        CalibrationFormatError: malformed artifact.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise CalibrationFormatError("calibration artifact is not a JSON object")
    version = payload.get("format_version")
    if version != ScoreCalibrator.FORMAT_VERSION:
        raise CalibrationVersionError(
            f"unsupported calibration format version: {version!r} "
            f"(expected {ScoreCalibrator.FORMAT_VERSION})"
        )
    stored = payload.get("class_ids")
    if stored is None:
        raise CalibrationFormatError("calibration artifact has no class set")
    if len(stored) != payload.get("num_classes"):
        raise CalibrationFormatError("class set and parameter count disagree")
    if expected_class_ids is not None and list(expected_class_ids) != list(stored):
        raise CalibrationMismatchError(stored, list(expected_class_ids))
    return ScoreCalibrator.from_state_dict(payload)


def try_load_calibration(
    path: str,
    expected_class_ids: Optional[Sequence] = None,
    log: Optional[logging.Logger] = None,
) -> Optional[ScoreCalibrator]:
    """Load calibration if possible; otherwise log and return ``None``.

    This is the deployment-safe path: a missing, stale, or mismatched
    artifact must not crash inference — the caller falls back to uncalibrated
    scores, and the reason is observable in the logs (counts and positions
    only, never class names).
    """
    log = log or logger
    try:
        return load_calibration(path, expected_class_ids)
    except FileNotFoundError:
        log.warning("calibration artifact not found; using uncalibrated scores")
    except CalibrationVersionError as e:
        log.warning("calibration version mismatch (%s); using uncalibrated scores", e)
    except CalibrationMismatchError as e:
        log.warning("calibration class-set mismatch (%s); using uncalibrated scores", e)
    except (CalibrationFormatError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        log.warning("calibration artifact unusable (%s); using uncalibrated scores", e)
    return None

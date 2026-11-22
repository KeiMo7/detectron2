# Copyright (c) Facebook, Inc. and its affiliates.
"""Per-class operating points: solve score thresholds for a target false-positive rate.

Given a target FPR on calibration data, find for each class the score
threshold that keeps the empirical false-positive rate at or below the target.
Classes with too few negative samples fall back to the pooled (global)
negative distribution, mirroring the shrinkage used for calibration itself.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["OperatingPoint", "solve_thresholds_at_fpr", "uniform_threshold_at_fpr"]


@dataclass
class OperatingPoint:
    """One class's solved threshold. Class is identified by index only."""

    class_index: int
    threshold: float
    target_fpr: float
    achieved_fpr: float
    detection_rate: float  # fraction of this class's positives kept at the threshold
    n_negatives: int
    used_global_pool: bool


def _threshold_for_fpr(neg_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose empirical FPR on ``neg_scores`` is <= target.

    A detection is kept when ``score >= threshold``. With ties, the achieved
    FPR can exceed the target slightly; callers see it reported exactly.
    """
    n = neg_scores.size
    if n == 0:
        return 1.0
    k = int(np.floor(target_fpr * n))  # allowed false positives above the bar
    if k >= n:
        return 0.0
    ordered = np.sort(neg_scores)[::-1]
    return float(ordered[k])


def solve_thresholds_at_fpr(
    scores,
    labels,
    classes,
    num_classes: Optional[int] = None,
    target_fpr: float = 0.1,
    min_negatives: int = 20,
) -> Tuple[np.ndarray, List[OperatingPoint]]:
    """Solve a per-class score threshold for a target false-positive rate.

    Args:
        scores/labels/classes: calibration data, shape (N,) each.
        num_classes: inferred if omitted.
        target_fpr: desired upper bound on the empirical FPR, in [0, 1].
        min_negatives: classes with fewer negatives use the pooled negatives.

    Returns:
        (thresholds, report): ``thresholds`` is a (num_classes,) array;
        ``report`` lists one :class:`OperatingPoint` per class.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must lie in [0, 1]")
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    c = np.asarray(classes, dtype=np.int64).reshape(-1)
    if not (s.shape == y.shape == c.shape):
        raise ValueError("scores, labels and classes must have the same shape")
    if num_classes is None:
        num_classes = int(c.max()) + 1 if c.size else 0

    pooled_neg = s[y < 0.5]
    thresholds = np.zeros(num_classes, dtype=np.float64)
    report: List[OperatingPoint] = []
    for k in range(num_classes):
        mask = c == k
        neg = s[mask & (y < 0.5)]
        pos = s[mask & (y > 0.5)]
        use_pool = neg.size < min_negatives
        ref = pooled_neg if use_pool else neg
        t = _threshold_for_fpr(ref, target_fpr)
        thresholds[k] = t
        achieved = float((neg >= t).mean()) if neg.size else 0.0
        det = float((pos >= t).mean()) if pos.size else 0.0
        report.append(
            OperatingPoint(
                class_index=k,
                threshold=t,
                target_fpr=target_fpr,
                achieved_fpr=achieved,
                detection_rate=det,
                n_negatives=int(neg.size),
                used_global_pool=use_pool,
            )
        )
    return thresholds, report


def uniform_threshold_at_fpr(scores, labels, target_fpr: float = 0.1) -> float:
    """Single threshold for all classes from the pooled negative scores."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    return _threshold_for_fpr(s[y < 0.5], target_fpr)

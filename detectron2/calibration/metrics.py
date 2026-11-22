# Copyright (c) Facebook, Inc. and its affiliates.
"""Calibration-quality metrics.

All functions are pure NumPy and run on CPU. They take probability-like
scores in [0, 1] and binary labels, and never see class names, image paths,
or annotations beyond the label vectors themselves.
"""

from typing import List, Tuple

import numpy as np

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "log_loss",
    "reliability_table",
    "average_precision",
]


def _as_arrays(probs, labels):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.shape != labels.shape:
        raise ValueError(f"shape mismatch: {probs.shape} vs {labels.shape}")
    return probs, labels


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins over [0, 1]."""
    probs, labels = _as_arrays(probs, labels)
    if probs.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Clip the top edge assignment so a prob of exactly 1.0 lands in the last bin.
    idx = np.minimum(np.digitize(probs, edges[1:-1]), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (n / probs.size) * abs(acc - conf)
    return float(ece)


def brier_score(probs, labels) -> float:
    """Mean squared error between predicted probabilities and labels."""
    probs, labels = _as_arrays(probs, labels)
    if probs.size == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def log_loss(probs, labels, eps: float = 1e-12) -> float:
    """Negative log-likelihood of the labels under the predicted probabilities."""
    probs, labels = _as_arrays(probs, labels)
    if probs.size == 0:
        return 0.0
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def reliability_table(probs, labels, n_bins: int = 15) -> List[Tuple[float, float, int, float, float]]:
    """Per-bin reliability data.

    Returns a list of ``(bin_lo, bin_hi, count, mean_confidence, mean_accuracy)``
    tuples, one per non-empty bin.
    """
    probs, labels = _as_arrays(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.minimum(np.digitize(probs, edges[1:-1]), n_bins - 1)
    table = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        table.append(
            (
                float(edges[b]),
                float(edges[b + 1]),
                n,
                float(probs[mask].mean()),
                float(labels[mask].mean()),
            )
        )
    return table


def average_precision(labels, scores) -> float:
    """Rank-based average precision (no interpolation), stable under ties.

    Ties are broken by input order via a stable sort, so any strictly
    order-preserving score transform yields the identical value.
    """
    labels, scores = _as_arrays(labels, scores)
    n_pos = labels.sum()
    if labels.size == 0 or n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    precision_at_tp = cum_tp[sorted_labels > 0.5] / (np.arange(labels.size) + 1)[
        sorted_labels > 0.5
    ]
    return float(precision_at_tp.sum() / n_pos)

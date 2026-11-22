# Copyright (c) Facebook, Inc. and its affiliates.
"""Domain-shift detection between calibration data and a new score batch.

Compares the per-class raw-score distribution seen at calibration time with
the distribution of a new (possibly unlabeled) prediction batch, and reports
*which* class shifted and *by how much* — not just a single global alarm.
Labels are not required on the new batch.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

__all__ = ["ks_statistic", "detect_domain_shift", "ShiftReport", "ClassShift"]


def ks_statistic(x, y) -> float:
    """Two-sample Kolmogorov–Smirnov statistic. Returns 0 for empty inputs."""
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size == 0 or y.size == 0:
        return 0.0
    grid = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


@dataclass
class ClassShift:
    """Shift measurement for one class, identified by index only."""

    class_index: int
    ks: float
    mean_reference: float
    mean_new: float
    n_reference: int
    n_new: int
    shifted: bool


@dataclass
class ShiftReport:
    """Per-class shift report. ``shifted`` is True if any class exceeds the threshold."""

    entries: List[ClassShift]
    ks_threshold: float

    @property
    def shifted(self) -> bool:
        return any(e.shifted for e in self.entries)

    @property
    def shifted_classes(self) -> List[int]:
        return [e.class_index for e in self.entries if e.shifted]

    def format(self) -> str:
        lines = [
            f"domain-shift report (KS threshold {self.ks_threshold:g}, "
            "classes identified by index)",
            "  idx | n_ref | n_new | KS | mean ref -> new | shifted",
        ]
        for e in self.entries:
            lines.append(
                f"  {e.class_index:>4d} | {e.n_reference:>6d} | {e.n_new:>6d} | "
                f"{e.ks:.4f} | {e.mean_reference:.4f} -> {e.mean_new:.4f} | "
                f"{'YES' if e.shifted else 'no'}"
            )
        return "\n".join(lines)


def detect_domain_shift(
    reference_scores,
    reference_classes,
    new_scores,
    new_classes,
    num_classes: Optional[int] = None,
    ks_threshold: float = 0.2,
    min_samples: int = 10,
) -> ShiftReport:
    """Compare a reference (calibration) score distribution with a new batch.

    Args:
        reference_scores/reference_classes: scores seen at calibration time.
        new_scores/new_classes: scores from the deployment batch (unlabeled).
        num_classes: inferred from both batches if omitted.
        ks_threshold: per-class KS statistic above which a class is flagged.
        min_samples: classes with fewer reference or new samples are skipped
            (their entry has ``shifted=False`` and ``ks=0``).

    Returns:
        A :class:`ShiftReport` with one entry per class.
    """
    rs = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
    rc = np.asarray(reference_classes, dtype=np.int64).reshape(-1)
    ns = np.asarray(new_scores, dtype=np.float64).reshape(-1)
    nc = np.asarray(new_classes, dtype=np.int64).reshape(-1)
    if rs.shape != rc.shape or ns.shape != nc.shape:
        raise ValueError("scores and classes must have matching shapes")
    if num_classes is None:
        top = -1
        if rc.size:
            top = int(rc.max())
        if nc.size:
            top = max(top, int(nc.max()))
        num_classes = top + 1

    entries: List[ClassShift] = []
    for k in range(num_classes):
        ref_k = rs[rc == k]
        new_k = ns[nc == k]
        if ref_k.size < min_samples or new_k.size < min_samples:
            entries.append(
                ClassShift(
                    class_index=k,
                    ks=0.0,
                    mean_reference=float(ref_k.mean()) if ref_k.size else 0.0,
                    mean_new=float(new_k.mean()) if new_k.size else 0.0,
                    n_reference=int(ref_k.size),
                    n_new=int(new_k.size),
                    shifted=False,
                )
            )
            continue
        ks = ks_statistic(ref_k, new_k)
        entries.append(
            ClassShift(
                class_index=k,
                ks=ks,
                mean_reference=float(ref_k.mean()),
                mean_new=float(new_k.mean()),
                n_reference=int(ref_k.size),
                n_new=int(new_k.size),
                shifted=ks > ks_threshold,
            )
        )
    return ShiftReport(entries=entries, ks_threshold=ks_threshold)

# Copyright (c) Facebook, Inc. and its affiliates.
"""Per-class Platt score calibration with shrinkage toward a global model.

Raw detector scores are ranking values, not probabilities. This module fits a
strictly increasing sigmoid map per class,

    p = sigmoid(a_c * logit(s) + b_c),   a_c > 0,

so downstream consumers can treat the output as a probability and pick
per-class operating points. Classes with few positives are shrunk toward the
global (pooled) parameters instead of being hard-fit on a handful of points;
classes with no positives at all use the global parameters verbatim and are
flagged as such in the fit report.

The transform is strictly increasing within every class, so any rank-based
per-class metric (AP, AR, ...) is exactly preserved. Note that per-class
calibration can reorder detections *across* classes; consumers that rank
globally across classes should be aware of this.

Everything here is NumPy-only and runs on CPU.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .errors import CalibrationRangeError, DegenerateFitError
from .metrics import brier_score, expected_calibration_error

__all__ = ["ScoreCalibrator", "CalibrationReport", "ClassCalibrationInfo", "fit_platt"]

logger = logging.getLogger(__name__)

_LOGIT_EPS = 1e-6

# Fit modes recorded in the report.
MODE_FITTED = "fitted"
MODE_SHRUNK = "shrunk"
MODE_GLOBAL = "global"


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(s / (1.0 - s))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _nll(x: np.ndarray, y: np.ndarray, alpha: float, b: float, l2: float) -> float:
    a = np.exp(alpha)
    z = a * x + b
    return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (alpha * alpha + b * b))


def fit_platt(
    scores,
    labels,
    l2: float = 1e-4,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """Fit ``p = sigmoid(a * logit(s) + b)`` by Newton's method on ``(log a, b)``.

    Parameterizing the slope as ``exp(alpha)`` guarantees ``a > 0`` and hence a
    strictly increasing transform.

    Returns:
        (a, b) with a > 0.

    Raises:
        DegenerateFitError: if there are no samples, or labels are all identical.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise DegenerateFitError("cannot fit Platt parameters on zero samples")
    n_pos = float(y.sum())
    if n_pos == 0.0 or n_pos == float(y.size):
        raise DegenerateFitError(
            "cannot fit Platt parameters: labels have a single unique value"
        )

    x = _logit(s)
    # Initialize intercept at the base-rate logit.
    base = min(max(n_pos / y.size, 1e-6), 1.0 - 1e-6)
    alpha, b = 0.0, float(np.log(base / (1.0 - base)))
    loss = _nll(x, y, alpha, b, l2)

    for _ in range(max_iter):
        a = float(np.exp(alpha))
        z = a * x + b
        p = _sigmoid(z)
        d = p - y
        ax = a * x
        g0 = float(np.dot(d, ax)) + l2 * alpha
        g1 = float(d.sum()) + l2 * b
        w = np.maximum(p * (1.0 - p), 1e-12)
        h00 = float(np.dot(w, ax * ax)) + l2
        h01 = float(np.dot(w, ax))
        h11 = float(w.sum()) + l2
        det = h00 * h11 - h01 * h01
        if det <= 0.0:
            break
        d0 = (g0 * h11 - g1 * h01) / det
        d1 = (h00 * g1 - h01 * g0) / det
        # Backtracking line search to keep the Newton step safe.
        step = 1.0
        accepted = False
        for _ in range(30):
            new_alpha, new_b = alpha - step * d0, b - step * d1
            new_loss = _nll(x, y, new_alpha, new_b, l2)
            if new_loss <= loss:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        alpha, b, loss = new_alpha, new_b, new_loss
        if max(abs(step * d0), abs(step * d1)) < tol:
            break
    return float(np.exp(alpha)), float(b)


@dataclass
class ClassCalibrationInfo:
    """Per-class summary of a calibration fit. Class is identified by index only."""

    class_index: int
    n_positives: int
    n_negatives: int
    mode: str  # one of MODE_FITTED / MODE_SHRUNK / MODE_GLOBAL
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float


@dataclass
class CalibrationReport:
    """Honest account of a calibration fit: per-class modes and before/after quality."""

    classes: List[ClassCalibrationInfo] = field(default_factory=list)
    ece_before: float = 0.0
    ece_after: float = 0.0
    brier_before: float = 0.0
    brier_after: float = 0.0

    @property
    def global_fallback_classes(self) -> List[int]:
        """Indices of classes that received the global parameters verbatim."""
        return [c.class_index for c in self.classes if c.mode == MODE_GLOBAL]

    @property
    def shrunk_classes(self) -> List[int]:
        return [c.class_index for c in self.classes if c.mode == MODE_SHRUNK]

    def format(self) -> str:
        lines = [
            "calibration report (classes identified by index)",
            f"  overall: ECE {self.ece_before:.4f} -> {self.ece_after:.4f}, "
            f"Brier {self.brier_before:.4f} -> {self.brier_after:.4f}",
            "  per-class: idx | n_pos | n_neg | mode | ECE before->after | Brier before->after",
        ]
        for c in self.classes:
            lines.append(
                f"    {c.class_index:>4d} | {c.n_positives:>6d} | {c.n_negatives:>6d} | "
                f"{c.mode:<6s} | {c.ece_before:.4f} -> {c.ece_after:.4f} | "
                f"{c.brier_before:.4f} -> {c.brier_after:.4f}"
            )
        return "\n".join(lines)


class ScoreCalibrator:
    """Per-class strictly-increasing score calibration.

    Attributes:
        global_params: ``(a, b)`` fitted on all classes pooled.
        class_params: ``(num_classes, 2)`` array of per-class ``(a, b)``.
        n_positives / n_negatives: per-class calibration-sample counts.
        modes: per-class fit mode (fitted / shrunk / global).
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        global_params,
        class_params,
        n_positives,
        n_negatives,
        modes,
        shrink_strength: float = 25.0,
        min_samples: int = 30,
        l2: float = 1e-4,
    ):
        self.global_params = np.asarray(global_params, dtype=np.float64).reshape(2)
        self.class_params = np.asarray(class_params, dtype=np.float64).reshape(-1, 2)
        self.n_positives = np.asarray(n_positives, dtype=np.int64).reshape(-1)
        self.n_negatives = np.asarray(n_negatives, dtype=np.int64).reshape(-1)
        self.modes = list(modes)
        if not (
            self.class_params.shape[0]
            == self.n_positives.shape[0]
            == self.n_negatives.shape[0]
            == len(self.modes)
        ):
            raise ValueError("per-class parameter arrays disagree in length")
        if np.any(self.class_params[:, 0] <= 0.0) or self.global_params[0] <= 0.0:
            raise ValueError("calibration slopes must be strictly positive")
        self.shrink_strength = float(shrink_strength)
        self.min_samples = int(min_samples)
        self.l2 = float(l2)

    @property
    def num_classes(self) -> int:
        return int(self.class_params.shape[0])

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        scores,
        labels,
        classes,
        num_classes: Optional[int] = None,
        shrink_strength: float = 25.0,
        min_samples: int = 30,
        l2: float = 1e-4,
    ) -> Tuple["ScoreCalibrator", CalibrationReport]:
        """Fit per-class calibration with shrinkage toward the global fit.

        Args:
            scores: raw detector scores in [0, 1], shape (N,).
            labels: binary match labels, shape (N,).
            classes: integer class indices, shape (N,).
            num_classes: total class count; inferred from data if omitted.
            shrink_strength: pseudo-count kappa of the global prior. Larger
                values pull sparse classes harder toward the global params.
            min_samples: classes with fewer positives than this always take
                the shrinkage path instead of a hard fit.
            l2: regularization for the Platt Newton fit.

        Returns:
            (calibrator, report).

        Raises:
            DegenerateFitError: if the pooled data cannot be fit at all
                (no samples, or no positives anywhere).
        """
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        c = np.asarray(classes, dtype=np.int64).reshape(-1)
        if not (s.shape == y.shape == c.shape):
            raise ValueError("scores, labels and classes must have the same shape")
        if s.size and ((s < 0.0) | (s > 1.0)).any():
            raise ValueError("calibration scores must lie in [0, 1]")
        if c.size and (c < 0).any():
            raise ValueError("class indices must be non-negative")
        if num_classes is None:
            num_classes = int(c.max()) + 1 if c.size else 0
        if c.size and c.max() >= num_classes:
            raise ValueError("class index out of range for num_classes")

        g_a, g_b = fit_platt(s, y, l2=l2)
        global_params = np.array([g_a, g_b], dtype=np.float64)

        class_params = np.zeros((num_classes, 2), dtype=np.float64)
        n_pos = np.zeros(num_classes, dtype=np.int64)
        n_neg = np.zeros(num_classes, dtype=np.int64)
        modes: List[str] = []
        infos: List[ClassCalibrationInfo] = []

        for k in range(num_classes):
            mask = c == k
            n_k_pos = int(y[mask].sum())
            n_k_neg = int(mask.sum()) - n_k_pos
            n_pos[k], n_neg[k] = n_k_pos, n_k_neg

            if n_k_pos == 0:
                # No positives at all: global params, and say so explicitly.
                params = global_params.copy()
                mode = MODE_GLOBAL
            elif n_k_pos < min_samples or n_k_neg == 0:
                # Too few points to trust a hard fit: shrink toward global.
                try:
                    hat = np.array(fit_platt(s[mask], y[mask], l2=l2))
                    w = n_k_pos / (n_k_pos + shrink_strength)
                    params = w * hat + (1.0 - w) * global_params
                except DegenerateFitError:
                    params = global_params.copy()
                mode = MODE_SHRUNK
            else:
                try:
                    params = np.array(fit_platt(s[mask], y[mask], l2=l2))
                    mode = MODE_FITTED
                except DegenerateFitError:
                    params = global_params.copy()
                    mode = MODE_GLOBAL
            params[0] = max(params[0], 1e-12)  # keep the slope strictly positive
            class_params[k] = params
            modes.append(mode)

            p_before = s[mask]
            p_after = _sigmoid(params[0] * _logit(s[mask]) + params[1])
            infos.append(
                ClassCalibrationInfo(
                    class_index=k,
                    n_positives=n_k_pos,
                    n_negatives=n_k_neg,
                    mode=mode,
                    ece_before=expected_calibration_error(p_before, y[mask]),
                    ece_after=expected_calibration_error(p_after, y[mask]),
                    brier_before=brier_score(p_before, y[mask]),
                    brier_after=brier_score(p_after, y[mask]),
                )
            )

        calibrator = cls(
            global_params=global_params,
            class_params=class_params,
            n_positives=n_pos,
            n_negatives=n_neg,
            modes=modes,
            shrink_strength=shrink_strength,
            min_samples=min_samples,
            l2=l2,
        )
        p_all_after = calibrator.predict(s, c)
        report = CalibrationReport(
            classes=infos,
            ece_before=expected_calibration_error(s, y),
            ece_after=expected_calibration_error(p_all_after, y),
            brier_before=brier_score(s, y),
            brier_after=brier_score(p_all_after, y),
        )
        for info in infos:
            if info.mode == MODE_GLOBAL and info.n_positives == 0:
                logger.info(
                    "class index %d has no positive calibration samples; "
                    "using global calibration parameters for it",
                    info.class_index,
                )
        return calibrator, report

    # -------------------------------------------------------------- predict
    def _apply(self, scores, params) -> np.ndarray:
        return _sigmoid(params[..., 0] * _logit(scores) + params[..., 1])

    def predict(self, scores, classes) -> np.ndarray:
        """Map raw scores to calibrated probabilities, per class.

        Raises:
            ValueError: on shape/range problems in the inputs.
            CalibrationRangeError: if any output falls outside [0, 1]; the
                offending class indices are reported, nothing is clamped.
        """
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        c = np.asarray(classes, dtype=np.int64).reshape(-1)
        if s.shape != c.shape:
            raise ValueError("scores and classes must have the same shape")
        if s.size == 0:
            return np.empty(0, dtype=np.float64)
        if ((s < 0.0) | (s > 1.0)).any():
            raise ValueError("scores must lie in [0, 1]")
        if (c < 0).any() or (c >= self.num_classes).any():
            raise ValueError("class index out of range for this calibrator")
        p = self._apply(s, self.class_params[c])
        bad = ~((p >= 0.0) & (p <= 1.0))
        if bad.any():
            raise CalibrationRangeError(sorted(set(c[bad].tolist())))
        return p

    def predict_global(self, scores) -> np.ndarray:
        """Map raw scores with the pooled (class-agnostic) parameters."""
        s = np.asarray(scores, dtype=np.float64).reshape(-1)
        if s.size == 0:
            return np.empty(0, dtype=np.float64)
        if ((s < 0.0) | (s > 1.0)).any():
            raise ValueError("scores must lie in [0, 1]")
        p = self._apply(s, self.global_params)
        bad = ~((p >= 0.0) & (p <= 1.0))
        if bad.any():
            raise CalibrationRangeError(range(self.num_classes))
        return p

    # ----------------------------------------------------------- state i/o
    def to_state_dict(self) -> dict:
        return {
            "global_params": self.global_params.tolist(),
            "class_params": self.class_params.tolist(),
            "n_positives": self.n_positives.tolist(),
            "n_negatives": self.n_negatives.tolist(),
            "modes": list(self.modes),
            "config": {
                "shrink_strength": self.shrink_strength,
                "min_samples": self.min_samples,
                "l2": self.l2,
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ScoreCalibrator":
        try:
            config = state.get("config", {})
            return cls(
                global_params=state["global_params"],
                class_params=state["class_params"],
                n_positives=state["n_positives"],
                n_negatives=state["n_negatives"],
                modes=state["modes"],
                shrink_strength=config.get("shrink_strength", 25.0),
                min_samples=config.get("min_samples", 30),
                l2=config.get("l2", 1e-4),
            )
        except KeyError as e:
            from .errors import CalibrationFormatError

            raise CalibrationFormatError(f"missing calibration field: {e}") from e

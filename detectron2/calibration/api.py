# Copyright (c) Facebook, Inc. and its affiliates.
"""Glue between the calibration core and detectron2 detection outputs.

The integration is additive only:

* :func:`calibrate_instances` attaches calibrated probabilities to an
  :class:`~detectron2.structures.Instances` under a *new, optional* field
  (``pred_probs`` by default). Existing fields and tensor structures are
  untouched; code that never reads the new field behaves exactly as before.
* With ``calibrator=None`` (no calibration data, or a failed load) the call
  is a no-op — the uncalibrated path, not an error.

Fitting uses IoU matching between predicted and ground-truth boxes to derive
binary labels, works for a single image (batch size 1) as well as lists, and
the intermediate ``(scores, labels, classes)`` triples can be merged across
shards for multi-GPU evaluation with a consistent aggregation rule: every
sample counts exactly once, regardless of how unbalanced the shards are.
"""

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from detectron2.structures import Instances, pairwise_iou

from .calibrator import CalibrationReport, ScoreCalibrator
from .errors import DegenerateFitError

__all__ = [
    "CALIBRATED_SCORE_FIELD",
    "calibrate_instances",
    "gather_calibration_data",
    "merge_calibration_data",
    "fit_calibrator",
    "fit_calibrator_from_data",
    "fit_calibrator_distributed",
]

logger = logging.getLogger(__name__)

#: Default name of the optional field carrying calibrated probabilities.
CALIBRATED_SCORE_FIELD = "pred_probs"


def calibrate_instances(
    instances: Instances,
    calibrator: Optional[ScoreCalibrator],
    out_field: str = CALIBRATED_SCORE_FIELD,
) -> Instances:
    """Attach calibrated probabilities to a detection result, in place.

    Args:
        instances: detector output with ``scores`` and ``pred_classes``.
        calibrator: fitted calibrator, or ``None`` for the uncalibrated path
            (no field is added and nothing else changes).
        out_field: name of the new field. Defaults to ``pred_probs``.

    Returns:
        The same ``instances`` object, with ``out_field`` set when a
        calibrator is given.
    """
    if calibrator is None:
        return instances
    scores = instances.get("scores")
    classes = instances.get("pred_classes")
    probs = calibrator.predict(
        scores.detach().cpu().numpy(), classes.detach().cpu().numpy()
    )
    instances.set(out_field, torch.as_tensor(probs).to(device=scores.device, dtype=scores.dtype))
    return instances


def _match_single_image(
    pred: Instances, gt: Instances, iou_threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy score-ordered matching for one image.

    A prediction is a positive iff it has IoU >= ``iou_threshold`` with an
    unmatched ground-truth box of the same class.
    """
    scores = pred.get("scores").detach().cpu().numpy().astype(np.float64)
    classes = pred.get("pred_classes").detach().cpu().numpy().astype(np.int64)
    n = scores.shape[0]
    labels = np.zeros(n, dtype=np.float64)
    if n == 0:
        return scores, labels, classes

    gt_boxes = gt.get("gt_boxes")
    gt_classes = gt.get("gt_classes").detach().cpu()
    if len(gt_boxes) == 0:
        return scores, labels, classes

    ious = pairwise_iou(pred.get("pred_boxes"), gt_boxes).cpu()
    pred_classes_t = pred.get("pred_classes").detach().cpu()
    matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    order = np.argsort(-scores, kind="stable")
    for i in order:
        same_class = (gt_classes == pred_classes_t[i]) & ~matched
        if not bool(same_class.any()):
            continue
        cand_ious = ious[i][same_class]
        best = int(torch.argmax(cand_ious))
        if float(cand_ious[best]) >= iou_threshold:
            gt_idx = int(torch.nonzero(same_class)[best].item())
            matched[gt_idx] = True
            labels[i] = 1.0
    return scores, labels, classes


def gather_calibration_data(
    predictions: Sequence[Instances],
    targets: Sequence[Instances],
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn matched predictions/targets into ``(scores, labels, classes)``.

    Works with any list length, including a single image. Images without
    ground truth contribute only negative labels for their predictions.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    chunks = [
        _match_single_image(p, t, iou_threshold) for p, t in zip(predictions, targets)
    ]
    return merge_calibration_data(chunks)


def merge_calibration_data(
    shards: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-shard calibration triples.

    This is the multi-GPU aggregation rule: the union of all samples, each
    counted once. Shards may be arbitrarily unbalanced; empty shards are
    dropped. The result is order-dependent only in floating-point rounding.
    """
    chunks = [tuple(np.asarray(a) for a in s) for s in shards if s is not None]
    chunks = [s for s in chunks if s[0].size > 0]
    if not chunks:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), np.empty(0, dtype=np.int64)
    return (
        np.concatenate([s[0] for s in chunks]).astype(np.float64),
        np.concatenate([s[1] for s in chunks]).astype(np.float64),
        np.concatenate([s[2] for s in chunks]).astype(np.int64),
    )


def fit_calibrator_from_data(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    num_classes: Optional[int] = None,
    **fit_kwargs,
) -> Tuple[Optional[ScoreCalibrator], Optional[CalibrationReport]]:
    """Fit from a ``(scores, labels, classes)`` triple.

    When there is nothing to learn from — no samples, or no positive label
    anywhere (e.g. an unlabeled calibration set) — this logs a notice and
    returns ``(None, None)`` so callers take the uncalibrated path instead of
    failing.
    """
    scores, labels, classes = data
    if scores.size == 0 or float(np.sum(labels)) == 0.0:
        logger.info(
            "no usable calibration annotations (%d samples); using uncalibrated scores",
            int(scores.size),
        )
        return None, None
    try:
        return ScoreCalibrator.fit(
            scores, labels, classes, num_classes=num_classes, **fit_kwargs
        )
    except DegenerateFitError as e:
        logger.info("calibration fit not possible (%s); using uncalibrated scores", e)
        return None, None


def fit_calibrator(
    predictions: Sequence[Instances],
    targets: Sequence[Instances],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None,
    **fit_kwargs,
) -> Tuple[Optional[ScoreCalibrator], Optional[CalibrationReport]]:
    """Match predictions to targets and fit a calibrator. See
    :func:`fit_calibrator_from_data` for the no-annotation behavior."""
    data = gather_calibration_data(predictions, targets, iou_threshold)
    return fit_calibrator_from_data(data, num_classes=num_classes, **fit_kwargs)


def fit_calibrator_distributed(
    predictions: Sequence[Instances],
    targets: Sequence[Instances],
    iou_threshold: float = 0.5,
    num_classes: Optional[int] = None,
    **fit_kwargs,
) -> Tuple[Optional[ScoreCalibrator], Optional[CalibrationReport]]:
    """Distributed variant of :func:`fit_calibrator`.

    Each rank matches its own shard; the triples are all-gathered and merged
    with :func:`merge_calibration_data`, so every sample counts exactly once
    no matter how unbalanced the per-rank shards are. Every rank fits on the
    same merged data and therefore holds the same parameters.
    """
    from detectron2.utils import comm

    local = gather_calibration_data(predictions, targets, iou_threshold)
    if comm.get_world_size() > 1:
        data = merge_calibration_data(comm.all_gather(local))
    else:
        data = local
    return fit_calibrator_from_data(data, num_classes=num_classes, **fit_kwargs)

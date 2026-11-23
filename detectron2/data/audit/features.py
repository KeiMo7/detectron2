# Copyright (c) Facebook, Inc. and its affiliates.
"""Local-feature extraction and pairwise verification.

Design notes (why this is not a checksum / whole-image comparison):

* Features are SIFT descriptors, which survive re-encoding, rescaling and
  watermarks, so identical content with a different file hash still matches.
* A pair is accepted only after a *similarity* transform is estimated by
  RANSAC from the matched keypoints.  Translation-only, similarity-only and
  full-affine hypotheses are all tried, which covers same-scene bursts,
  crops, and aspect-preserving rescales.
* Coverage is measured against the keypoints that actually fall inside the
  geometric overlap, and a photometric NCC check confirms that the shared
  pixels agree.  With a small repeated region, coverage is measured *within*
  the densest local cluster, which is how "same background, different
  subject" is detected.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import AuditConfig

__all__ = [
    "Features",
    "PairEvidence",
    "extract_features",
    "score_pair",
]


def _require_opencv() -> None:
    # ``import cv2`` is done at module import; this helper only documents the
    # optional dependency and gives a clearer error than a bare ImportError.
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "The duplicate audit needs OpenCV with SIFT (opencv-contrib or "
            "opencv >= 4.4).  Install opencv-python-headless; the audit is the "
            "only part of detectron2 that requires it."
        )


@dataclass
class Features:
    """Descriptors for one canonical image."""

    keypoints: np.ndarray  # (N, 2) float32, x/y in canonical frame
    descriptors: np.ndarray  # (N, 128) float32, root-SIFT normalised
    sizes: np.ndarray  # (N,) float32, keypoint scale (detection size)
    shape: Tuple[int, int]  # (H, W) canonical

    @property
    def count(self) -> int:
        return int(self.keypoints.shape[0])


def _root_sift(desc: np.ndarray) -> np.ndarray:
    desc = desc.astype(np.float32, copy=False)
    desc /= desc.sum(axis=1, keepdims=True) + 1e-7
    np.sqrt(desc, out=desc)
    return desc


def extract_features(gray: np.ndarray, cfg: AuditConfig) -> Features:
    """Extract root-SIFT features from a canonical grayscale image."""
    _require_opencv()
    # OpenCV's SIFT is deterministic for a fixed input and parameter set.
    cv2.setRNGSeed(cfg.deterministic_seed)
    h, w = gray.shape[:2]
    up = 1.0
    if cfg.upsample and cfg.upsample > 1.0 and max(h, w) <= cfg.upsample_max_side:
        up = float(cfg.upsample)
        gray = cv2.resize(
            gray, (int(round(w * up)), int(round(h * up))), interpolation=cv2.INTER_CUBIC
        )
    kwargs = {"nfeatures": int(cfg.max_features)}
    if cfg.sift_contrast_threshold is not None:
        kwargs["contrastThreshold"] = float(cfg.sift_contrast_threshold)
    if cfg.sift_edge_threshold is not None:
        kwargs["edgeThreshold"] = float(cfg.sift_edge_threshold)
    detector = cv2.SIFT_create(**kwargs)
    kps, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return Features(
            keypoints=np.zeros((0, 2), np.float32),
            descriptors=np.zeros((0, 128), np.float32),
            sizes=np.zeros((0,), np.float32),
            shape=(h, w),
        )
    # Keypoints are reported back in the *canonical* frame regardless of the
    # detection scale, so all thresholds and regions stay resolution-stable.
    pts = np.array([k.pt for k in kps], dtype=np.float32) / np.float32(up)
    sizes = np.array([k.size for k in kps], dtype=np.float32) / np.float32(up)
    # Deterministic ordering regardless of OpenCV's internal keypoint order.
    order = np.lexsort((pts[:, 1], pts[:, 0], -sizes))
    return Features(
        keypoints=pts[order],
        descriptors=_root_sift(desc[order]),
        sizes=sizes[order],
        shape=(h, w),
    )


def _descriptor_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between two L2-normalised descriptor sets."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    # (n, m, 128) is too big for large feature counts; use the expansion
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b with a single GEMM.
    dist_sq = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.sqrt(np.maximum(dist_sq, 0.0)).astype(np.float32)


def _ratio_test_matches(fa: Features, fb: Features, cfg: AuditConfig) -> np.ndarray:
    """Lowe ratio test with mutual (cross) consistency.

    Returns an (M, 2) int array of matched keypoint indices into fa / fb.
    """
    if fa.count == 0 or fb.count == 0:
        return np.zeros((0, 2), np.int32)
    dist = _descriptor_distances(fa.descriptors, fb.descriptors)
    # Two best per row (A -> B) and per column (B -> A).
    b_best = np.argpartition(dist, 1, axis=1)[:, :2]
    row = np.arange(fa.count)
    d0 = dist[row, b_best[:, 0]]
    d1 = dist[row, b_best[:, 1]]
    # Stable ordering when the two are close (argpartition order is arbitrary).
    swap = d1 < d0
    d0, d1 = np.where(swap, d1, d0), np.where(swap, d0, d1)
    b_idx = np.where(swap, b_best[:, 1], b_best[:, 0])
    ratio = d0 / np.maximum(d1, 1e-7)
    keep = ratio < cfg.ratio_test

    a_best = np.argpartition(dist, 1, axis=0)[:2, :]
    col = np.arange(fb.count)
    a_idx = a_best[0, :]
    other = dist[a_best[1, :], col] < dist[a_idx, col]
    a_idx = np.where(other, a_best[1, :], a_idx)
    mutual = a_idx[b_idx] == row

    ii = row[keep & mutual]
    jj = b_idx[keep & mutual]
    if ii.size == 0:
        return np.zeros((0, 2), np.int32)
    return np.stack([ii, jj], axis=1).astype(np.int32)


# --------------------------------------------------------------------------
# Geometric verification
# --------------------------------------------------------------------------


@dataclass
class PairEvidence:
    """All measurements for one image pair, before any thresholding."""

    score: float = 0.0
    inliers: int = 0
    matches: int = 0
    coverage: float = 0.0  # overlap-aware coverage of matched keypoints
    local_coverage: float = 0.0
    local_inliers: int = 0
    region: Optional[Tuple[int, int, int, int]] = None  # canonical xyxy
    scale: float = 1.0
    rotation_deg: float = 0.0
    ncc: float = 0.0
    ncc_overlap: float = 0.0
    local_ncc: float = 0.0
    # Canonical area of the larger image in the pair; divides the region area
    # to obtain the region's relative size for the local decision.
    region_area_ref: float = 0.0
    # Lattice mask of agreeing cells and the cell size, used to report the
    # repeated area precisely (a ring-shaped repeat is not a single rectangle).
    local_cell_mask: Optional[np.ndarray] = None
    local_cell_size: Optional[Tuple[int, int]] = None
    # Serializable rectangles covering the agreeing cells (canonical xyxy).
    region_cells: Optional[Tuple[Tuple[int, int, int, int], ...]] = None
    kind: str = "none"  # "global" | "local" | "none"
    reasons: List[str] = None  # human-readable basis
    borderline: bool = False
    match_ratio: float = 0.0  # inliers / min(kp_a, kp_b)
    # True when the measurement internally swapped the pair for symmetry, so
    # the region is expressed in the *second* caller image's frame.
    region_swapped: bool = False

    def to_dict(self) -> Dict:
        return {
            "score": round(float(self.score), 5),
            "inliers": int(self.inliers),
            "matches": int(self.matches),
            "coverage": round(float(self.coverage), 5),
            "local_coverage": round(float(self.local_coverage), 5),
            "local_inliers": int(self.local_inliers),
            "region": list(self.region) if self.region else None,
            "scale": round(float(self.scale), 5),
            "rotation_deg": round(float(self.rotation_deg), 3),
            "ncc": round(float(self.ncc), 5),
            "ncc_overlap": round(float(self.ncc_overlap), 5),
            "local_ncc": round(float(self.local_ncc), 5),
            "region_area_ref": round(float(self.region_area_ref), 3),
            "kind": self.kind,
            "reasons": list(self.reasons or []),
            "borderline": bool(self.borderline),
            "match_ratio": round(float(self.match_ratio), 5),
            "region_swapped": bool(self.region_swapped),
            "region_cells": [list(r) for r in self.region_cells] if self.region_cells else None,
        }


_MODEL_TRANSLATION = 0
_MODEL_SIMILARITY = 1
_MODEL_AFFINE = 2


def _estimate(p1: np.ndarray, p2: np.ndarray, cfg: AuditConfig) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Try translation / similarity / affine models, return candidates.

    Each entry is ``(model_kind, matrix_2x3, inlier_mask)``.  A translation
    model is tried first because same-camera bursts produce a near-identity
    transform, for which a scale-constrained model can be numerically awkward.
    """
    out = []
    if p1.shape[0] < 3:
        return out
    thr = float(cfg.ransac_reproj_threshold)

    # -- pure translation ---------------------------------------------------
    if p1.shape[0] >= 3:
        delta = p2 - p1
        # Deterministic, exhaustive-ish vote over translation offsets supplied
        # by pairs of correspondences.  A dense offset histogram is cheap and
        # robust for the burst case.
        med = np.median(delta, axis=0)
        inl = np.linalg.norm(delta - med, axis=1) < thr
        if inl.sum() >= 3:
            out.append((_MODEL_TRANSLATION, np.array([[1, 0, med[0]], [0, 1, med[1]]], np.float64), inl))

    # -- similarity ---------------------------------------------------------
    m, mask = cv2.estimateAffinePartial2D(
        p1,
        p2,
        method=cv2.RANSAC,
        ransacReprojThreshold=thr,
        maxIters=int(cfg.ransac_max_iters),
        confidence=float(cfg.ransac_confidence),
        refineIters=10,
    )
    if m is not None and mask is not None:
        out.append((_MODEL_SIMILARITY, m.astype(np.float64), mask.ravel().astype(bool)))

    # -- full affine (handles mild perspective / aspect changes) -----------
    m2, mask2 = cv2.estimateAffine2D(
        p1,
        p2,
        method=cv2.RANSAC,
        ransacReprojThreshold=thr,
        maxIters=int(cfg.ransac_max_iters),
        confidence=float(cfg.ransac_confidence),
        refineIters=10,
    )
    if m2 is not None and mask2 is not None:
        out.append((_MODEL_AFFINE, m2.astype(np.float64), mask2.ravel().astype(bool)))
    return out


def _transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def _inverse_transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a = m[:, :2]
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    if abs(det) < 1e-12:
        return np.full_like(pts, np.nan)
    inv = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]], dtype=np.float64) / det
    return (pts - m[:, 2]) @ inv.T


def _scale_rotation(m: np.ndarray) -> Tuple[float, float]:
    a = m[:, :2]
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    scale = float(math.sqrt(abs(det)) if det != 0 else 0.0)
    # Coordinate convention: x right, y down.
    rot = float(math.degrees(math.atan2(a[1, 0], a[0, 0])) % 360.0)
    return scale, rot


def _inside(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return (
        (points[:, 0] >= 0)
        & (points[:, 0] < w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < h)
    )


def _ncc_warp(g1: np.ndarray, g2: np.ndarray, m: np.ndarray, work_side: int = 256) -> Tuple[float, float]:
    """Zero-mean NCC of the shared region after warping g2 into g1's frame.

    Returns ``(ncc, overlap_fraction)``.  ``overlap_fraction`` is the share of
    the (reduced) g1 frame covered by valid warped g2 pixels.
    """
    h, w = g1.shape
    s = min(1.0, work_side / float(max(h, w)))
    a = cv2.resize(g1, (max(1, int(round(w * s))), max(1, int(round(h * s))),), interpolation=cv2.INTER_AREA)
    b = cv2.resize(g2, (max(1, int(round(g2.shape[1] * s))), max(1, int(round(g2.shape[0] * s))),), interpolation=cv2.INTER_AREA)
    m3 = np.vstack([m[:2], [0.0, 0.0, 1.0]])
    sm = np.diag([s, s, 1.0])
    warped_m = (sm @ m3 @ np.linalg.inv(sm))[:2].astype(np.float32)
    warped = cv2.warpAffine(
        b,
        warped_m,
        (a.shape[1], a.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        np.ones(b.shape, np.uint8) * 255,
        warped_m,
        (a.shape[1], a.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderValue=0,
    ) > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    overlap = float(mask.mean())
    if mask.sum() < 50:
        return 0.0, overlap
    x = a[mask].astype(np.float32)
    y = warped[mask].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum())) + 1e-9
    return float((x * y).sum() / denom), overlap


def _region_agreement_map(
    ga: np.ndarray, gb: np.ndarray, m: np.ndarray, cfg: AuditConfig, grid: int = 16
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Coarse map of where the two images agree after warping.

    The canonical frame of ``ga`` is divided into a ``grid x grid`` lattice.
    Each cell is mapped through ``m`` to ``gb``, sampled, and scored by the
    absolute difference of its (locally normalised) pixel statistics.  A cell
    that does not map fully inside ``gb`` is marked invalid.

    Returns ``(diff_map, valid_map, cell_w, cell_h)`` where ``diff_map`` is the
    normalised disagreement (0 = identical) and ``valid_map`` marks cells with
    usable samples.  This is the basis for local-region detection: connected
    agreeing cells form the repeated region, regardless of how the *unchanged*
    part wraps around an altered subject.
    """
    h, w = ga.shape
    cell_h = max(1, int(math.ceil(h / float(grid))))
    cell_w = max(1, int(math.ceil(w / float(grid))))
    diff = np.ones((grid, grid), np.float32)
    valid = np.zeros((grid, grid), bool)
    if gb.size == 0:
        return diff, valid, cell_w, cell_h
    # Sample the centre and a few offsets of each cell, transformed into gb.
    offs = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [0.5, 0.5]], np.float64)
    for gy in range(grid):
        y0 = gy * cell_h
        y1 = min(h, y0 + cell_h)
        if y1 <= y0:
            continue
        for gx in range(grid):
            x0 = gx * cell_w
            x1 = min(w, x0 + cell_w)
            if x1 <= x0:
                continue
            pts = np.stack(
                [x0 + offs[:, 0] * (x1 - x0), y0 + offs[:, 1] * (y1 - y0)], axis=1
            )
            q = _transform_points(m, pts)
            # Bilinear sampling reads x0+1/y0+1, so require strictly inside.
            if (
                np.any(q[:, 0] < 0)
                or np.any(q[:, 0] > gb.shape[1] - 2)
                or np.any(q[:, 1] < 0)
                or np.any(q[:, 1] > gb.shape[0] - 2)
            ):
                continue
            a_vals = _bilinear(ga, pts.astype(np.float32)).astype(np.float32)
            b_vals = _bilinear(gb, q.astype(np.float32)).astype(np.float32)
            a_std = a_vals.std()
            b_std = b_vals.std()
            if a_std < 3.0 and b_std < 3.0:
                # Both cells are flat: agreement is undefined, treat as matching.
                diff[gy, gx] = 0.0
                valid[gy, gx] = True
                continue
            a_n = (a_vals - a_vals.mean()) / (a_std + 1e-6)
            b_n = (b_vals - b_vals.mean()) / (b_std + 1e-6)
            diff[gy, gx] = float(np.mean(np.abs(a_n - b_n)))
            valid[gy, gx] = True
    return diff, valid, cell_w, cell_h


def _local_analysis(
    ga: np.ndarray,
    gb: np.ndarray,
    m: np.ndarray,
    cfg: AuditConfig,
    grid: int = 16,
) -> List[Tuple[int, float, Tuple[int, int, int, int], float]]:
    """Find a repeated region by pixel agreement and quantify it.

    Returns at most one entry ``(support_cells, coverage, region_xyxy,
    mean_agreement, cell_mask, (cell_w, cell_h), diff_map, valid_map)``:

    * ``support_cells`` - number of lattice cells whose content agrees;
    * ``coverage`` - agreeing cells / cells that map into the other image;
    * ``region_xyxy`` - bounding box of the agreeing cells (the area to show a
      human, in canonical pixels);
    * ``mean_agreement`` - mean per-cell agreement in ``[0, 1]`` over the
      agreeing cells, used as the photometric gate.

    The measure is over the agreeing *cells*, not over their bounding box, so a
    changed subject in the middle of an otherwise shared frame does not dilute
    the evidence.
    """
    diff, valid, cell_w, cell_h = _region_agreement_map(ga, gb, m, cfg, grid)
    if not valid.any():
        return []
    # "Agreeing" when the normalised disagreement is small: well below the
    # value produced by different content, and above resampling noise.
    agree = valid & (diff <= cfg.local_cell_agreement)
    if agree.sum() < cfg.min_local_inliers:
        return []
    # Locate the largest contiguous blob; then report every agreeing cell that
    # belongs to a blob of at least the minimum size, so a ring (thin shared
    # border around a changed subject) is not truncated to one side.
    from scipy import ndimage

    labels, count = ndimage.label(agree, structure=np.ones((3, 3)))
    keep = np.zeros_like(agree)
    for lb in range(1, count + 1):
        blob = labels == lb
        if int(blob.sum()) >= cfg.min_local_inliers:
            keep |= blob
    if keep.sum() < cfg.min_local_inliers:
        return []
    rows, cols = np.where(keep)
    region = (
        int(cols.min() * cell_w),
        int(rows.min() * cell_h),
        int(min((cols.max() + 1) * cell_w, ga.shape[1])),
        int(min((rows.max() + 1) * cell_h, ga.shape[0])),
    )
    coverage = float(keep.sum()) / float(max(1, int(valid.sum())))
    # Agreement in [0,1]; diff is in normalised units, capped for stability.
    mean_agreement = float(np.clip(1.0 - diff[keep].mean(), 0.0, 1.0))
    return [(int(keep.sum()), coverage, region, mean_agreement, keep, (cell_w, cell_h), diff, valid)]


def _bbox_area(b: Tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _cells_to_rects(mask, cell_size) -> Optional[Tuple[Tuple[int, int, int, int], ...]]:
    """Convert an agreeing-cell mask to explicit rectangles (canonical xyxy)."""
    if mask is None or cell_size is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return None
    cell_w, cell_h = int(cell_size[0]), int(cell_size[1])
    rects = []
    for gy in range(arr.shape[0]):
        gx = 0
        row = arr[gy]
        while gx < arr.shape[1]:
            if not row[gx]:
                gx += 1
                continue
            start = gx
            while gx < arr.shape[1] and row[gx]:
                gx += 1
            rects.append((start * cell_w, gy * cell_h, gx * cell_w, (gy + 1) * cell_h))
    return tuple(rects) if rects else None


def _bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample of ``img`` at float coordinates.

    Coordinates are clamped into the valid interior so a point exactly on the
    right/bottom border (or a hair outside due to float error) cannot index out
    of bounds.
    """
    h, w = img.shape[:2]
    x = np.clip(pts[:, 0].astype(np.float32), 0.0, float(max(0, w - 2)))
    y = np.clip(pts[:, 1].astype(np.float32), 0.0, float(max(0, h - 2)))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, max(0, w - 1))
    y1 = np.minimum(y0 + 1, max(0, h - 1))
    fx = x - x0
    fy = y - y0
    f = img.astype(np.float32)
    return (
        f[y0, x0] * (1 - fx) * (1 - fy)
        + f[y0, x1] * fx * (1 - fy)
        + f[y1, x0] * (1 - fx) * fy
        + f[y1, x1] * fx * fy
    )


def _orient_swap(fa: Features, fb: Features, ga: np.ndarray, gb: np.ndarray) -> bool:
    """Deterministic orientation for symmetric pair measurement.

    Two images with identical keypoint counts and shapes still need a stable
    tie-break, so the descriptor bytes are hashed: the digest is a pure
    function of the features, making the orientation independent of argument
    order.
    """
    ka = (fa.count, ga.shape[0], ga.shape[1])
    kb = (fb.count, gb.shape[0], gb.shape[1])
    if ka != kb:
        return ka < kb
    return _descriptor_digest(fa.descriptors) < _descriptor_digest(fb.descriptors)


def _descriptor_digest(desc: np.ndarray) -> bytes:
    import hashlib

    if desc.size == 0:
        return b"empty"
    return hashlib.sha256(desc.astype(np.float32, copy=False).tobytes()).digest()


def score_pair(
    fa: Features, fb: Features, ga: np.ndarray, gb: np.ndarray, cfg: AuditConfig
) -> PairEvidence:
    """Measure one pair; classification happens separately for each threshold.

    The geometric models are tried in order of increasing flexibility
    (translation, similarity, affine) and the inlier selection with the most
    inliers wins.  The local repeated region is evaluated for every model that
    produced a plausible scale, and the best local region is kept independently
    of which global model won, so a crop that leaves only a small shared area
    is still scored as a local duplicate.
    """
    ev = PairEvidence(reasons=[])
    # The measurement must not depend on argument order.  The local region
    # and its photometric check are evaluated in A's frame, so a pair measured
    # as (a, b) and as (b, a) would otherwise produce different regions.
    # Canonicalise by keypoint count (then shape) so both callers agree.
    if _orient_swap(fa, fb, ga, gb):
        fa, fb, ga, gb = fb, fa, gb, ga
        swapped = True
    else:
        swapped = False
    matches = _ratio_test_matches(fa, fb, cfg)
    ev.matches = int(matches.shape[0])
    if ev.matches < cfg.min_matches:
        ev.reasons.append(f"too few ratio-test matches ({ev.matches} < {cfg.min_matches})")
        return ev

    p1 = fa.keypoints[matches[:, 0]]
    p2 = fb.keypoints[matches[:, 1]]

    best_global = None
    local_candidates = []  # (count, coverage, region, agreement, transform)
    for kind, m, mask in _estimate(p1, p2, cfg):
        scale, rot = _scale_rotation(m)
        if not (cfg.min_scale <= scale <= cfg.max_scale):
            continue
        inliers = int(mask.sum())
        if inliers < 3:
            continue
        for cand in _local_analysis(ga, gb, m, cfg):
            local_candidates.append(cand + (m,))
        if best_global is None or inliers > best_global[0]:
            best_global = (inliers, m, mask, scale, rot)

    if best_global is None:
        ev.reasons.append("no geometrically consistent transform")
        return ev

    inliers, m, mask, scale, rot = best_global
    ev.inliers = inliers
    ev.scale = scale
    ev.rotation_deg = rot
    ev.match_ratio = inliers / max(1, min(fa.count, fb.count))

    q2 = _transform_points(m, fa.keypoints)
    q1 = _inverse_transform_points(m, fb.keypoints)
    vis_a = int(np.count_nonzero(_inside(q2, gb.shape)))
    vis_b = int(np.count_nonzero(_inside(q1, ga.shape)))
    ev.coverage = inliers / max(1, min(vis_a, vis_b))

    ncc, ncc_overlap = _ncc_warp(ga, gb, m)
    ev.ncc = ncc
    ev.ncc_overlap = ncc_overlap
    # Canonical area of the larger image; divides the region area to obtain the
    # region's relative size for the local decision.
    ev.region_area_ref = float(max(ga.shape[0] * ga.shape[1], gb.shape[0] * gb.shape[1]))

    if local_candidates:
        # Photometric agreement is a gate, not a tie-breaker: a denser set of
        # coincidental matches inside an altered subject must not outrank the
        # genuinely shared area.  Among regions that clear the gate, prefer the
        # largest genuinely repeated area.
        best = max(local_candidates, key=lambda c: (c[3], c[0], c[1]))
        ev.local_inliers = int(best[0])
        ev.local_coverage = float(best[1])
        ev.region = best[2]
        ev.local_ncc = float(best[3])
        ev.local_cell_mask = best[4]
        ev.local_cell_size = best[5]
        ev.region_cells = _cells_to_rects(best[4], best[5])

    # `region` is expressed in the canonical frame of the oriented first image.
    ev.region_swapped = swapped
    ev.reasons.append(
        f"{inliers} inliers over {ev.matches} matches "
        f"(scale={scale:.2f}, rot={rot:.1f} deg, coverage={ev.coverage:.2f}, "
        f"ncc={ncc:.2f})"
    )
    return ev


def classify_evidence(ev: PairEvidence, cfg: AuditConfig, threshold: Optional[float] = None) -> PairEvidence:
    """Turn a measurement into a duplicate decision at a given threshold.

    ``score`` blends how much of the compared keypoint field is explained by
    the estimated transform with the photometric agreement of the shared
    pixels:

    * global duplicates: ``coverage * (0.75 + 0.25 * clamp(ncc))``
    * local duplicates:  ``local_coverage * (0.75 + 0.25 * clamp(local_ncc))``

    The score is the larger of the two so that a whole-scene repeat and a
    shared local region can both rank highly.  Measurements are stored in the
    evidence and are re-classified for every threshold in a sweep, so a sweep
    never re-runs the expensive matching.
    """
    thr = cfg.dup_threshold if threshold is None else float(threshold)
    reasons = list(ev.reasons)

    global_ok = (
        ev.inliers >= cfg.min_inliers
        and ev.coverage >= cfg.min_coverage
        and (ev.ncc_overlap < cfg.ncc_overlap_gate or ev.ncc >= cfg.min_ncc)
    )

    # The repeated region's area relative to the image that contains it.  The
    # canonical image area is recorded on the evidence at scoring time so this
    # rule is well-defined.
    region_area_rel = None
    if ev.region is not None and ev.region_area_ref > 0:
        region_area_rel = _bbox_area(ev.region) / float(ev.region_area_ref)
    local_ok = (
        ev.local_inliers >= cfg.min_local_inliers
        and ev.local_coverage >= cfg.min_local_coverage
        and ev.region is not None
        and region_area_rel is not None
        and region_area_rel >= cfg.min_region_area
        and ev.local_ncc >= cfg.min_local_ncc
    )

    global_score = 0.0
    if global_ok:
        global_score = ev.coverage * (0.75 + 0.25 * max(0.0, min(1.0, ev.ncc)))
    local_score = 0.0
    if local_ok:
        # Local score uses the fraction of compared area that agrees and the
        # mean agreement of those cells; both are in [0, 1].
        local_score = ev.local_coverage * (0.75 + 0.25 * max(0.0, min(1.0, ev.local_ncc)))

    if global_ok:
        # A whole-frame match is a global duplicate; the local branch is only
        # the explanation for pairs that do *not* pass the global test.  This
        # keeps "burst" and "crop" families labelled consistently and avoids
        # reporting a small sub-region of a full-frame match as a local repeat.
        kind = "global"
        reasons.append("global: whole-frame transform, coverage and NCC pass")
    elif local_ok:
        kind = "local"
        reasons.append(
            f"local: repeated region (area={region_area_rel:.2f}) passes local coverage and NCC"
        )
    else:
        kind = "none"
        if ev.inliers < cfg.min_inliers:
            reasons.append(f"inliers {ev.inliers} < {cfg.min_inliers}")
        elif ev.inliers and ev.coverage < cfg.min_coverage:
            reasons.append(f"coverage {ev.coverage:.2f} < {cfg.min_coverage}")
        elif ev.inliers and ev.ncc_overlap >= cfg.ncc_overlap_gate and ev.ncc < cfg.min_ncc:
            reasons.append(f"photometric disagreement (ncc={ev.ncc:.2f})")
        elif ev.region is not None:
            reasons.append(
                f"local region insufficient (local_coverage={ev.local_coverage:.2f}, "
                f"local_ncc={ev.local_ncc:.2f})"
            )
        else:
            reasons.append("below duplicate criteria")

    ev.kind = kind
    ev.score = float(max(global_score, local_score))
    above = ev.score >= thr and kind != "none"
    ev.borderline = bool(
        above is False and ev.score >= thr - cfg.borderline_margin and kind != "none"
    ) or bool(above and ev.score < thr + cfg.borderline_margin)
    ev.reasons = reasons
    return ev

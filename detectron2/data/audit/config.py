# Copyright (c) Facebook, Inc. and its affiliates.
"""Configuration objects for the side-mounted duplicate/leakage audit.

The audit is deliberately *not* part of the dataset registration or loading
interface.  All knobs live here so that a caller can run the audit with an
explicit configuration object, while the regular dataloader keeps its existing
behaviour when the audit is disabled (the default).
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

__all__ = ["AuditConfig"]

# Bumped whenever the detection pipeline changes in a way that can change the
# produced clusters.  It is recorded in every report / store so that results
# from different algorithm versions are never silently mixed.
ALGORITHM_VERSION = "dupaudit-v1"


@dataclass
class AuditConfig:
    """Knobs of the near-duplicate / split-leakage audit.

    The defaults are the values calibrated in the accompanying verification
    report.  A threshold sweep is produced for every run, so callers can see
    how cluster and leakage counts move before they settle on a value.
    """

    # -- master switch -----------------------------------------------------
    enabled: bool = False

    # -- image canonicalisation -------------------------------------------
    # Images are never upsampled and their aspect ratio is preserved.  Only
    # very large images are downscaled, to bound per-image compute.
    max_side: int = 1024
    target_pixels: int = 600_000
    # Refuse (and count as skipped) pathological images above this size.
    max_pixels: int = 200_000_000

    # -- local feature extraction -----------------------------------------
    max_features: int = 600
    # Upsample small-to-medium images before SIFT.  Repeated texture in a crop
    # leaves few distinctive keypoints at the native scale, and the stable
    # top-N selection then keeps weak ones; a 2x upsample recovers the
    # medium-scale structure that survives resampling.  Only applies when the
    # canonical image is below ``upsample_max_side``.
    upsample: float = 2.0
    upsample_max_side: int = 900
    # SIFT contrast / edge thresholds.  ``None`` means OpenCV's default.
    sift_contrast_threshold: Optional[float] = 0.0035
    sift_edge_threshold: Optional[float] = None

    # -- descriptor matching ----------------------------------------------
    ratio_test: float = 0.85
    # Number of ratio-test matches below which a pair is not worth RANSAC.
    min_matches: int = 8

    # -- geometric verification -------------------------------------------
    ransac_reproj_threshold: float = 3.5
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.995
    # Valid similarity transform (scale, rotation, translation) range.
    min_scale: float = 0.15
    max_scale: float = 6.5

    # -- global duplicate criteria ----------------------------------------
    min_inliers: int = 18
    # A *global* duplicate must explain the larger part of the overlapping
    # keypoint field, not just a corner.  Partial matches (crops, a shared
    # background) are handled by the local branch instead.
    min_coverage: float = 0.55
    # Photometric check is only enforced when the warped overlap is large
    # enough to make it meaningful.
    min_ncc: float = 0.20
    ncc_overlap_gate: float = 0.50

    # -- local duplicate criteria -----------------------------------------
    # The shared area is located on a coarse pixel-agreement lattice (the
    # canonical image is divided into `grid` cells and each cell is compared
    # after warping).  A repeated region is the set of agreeing cells; it may
    # be a shared corner or a reuse of the background around a changed subject.
    min_local_inliers: int = 10
    # A *local* repeat is by definition partial: the shared area may be a small
    # fraction of the frame.  The floor only guards against single-cell
    # coincidences.
    min_local_coverage: float = 0.12
    min_region_area: float = 0.01
    # Normalised per-cell disagreement that still counts as "agreeing".
    local_cell_agreement: float = 0.35
    # Minimum mean agreement (1 - disagreement) over the agreeing cells.
    min_local_ncc: float = 0.55

    # -- clustering --------------------------------------------------------
    # An edge becomes a cluster edge when its similarity score reaches this
    # value.  Borderline edges (within `borderline_margin`) are reported for
    # human review but still participate in clustering.
    dup_threshold: float = 0.45
    borderline_margin: float = 0.03
    sweep_thresholds: Tuple[float, ...] = (0.35, 0.45, 0.55, 0.65)
    min_cluster_size: int = 2

    # -- candidate indexing (scales to 10k+ images) -----------------------
    # Codes shared by at most this many images are "rare" and are indexed
    # directly; this is what makes partial (local) repeats discoverable
    # without an O(n^2) comparison.
    rare_code_max_df: int = 8
    min_shared_rare: int = 6
    max_rare_codes_per_image: int = 2048
    # How many rare codes are persisted per image in the history store.  This
    # bounds store size; it only affects cross-batch *candidate proposals*,
    # never the verification of a pair.
    store_codes_per_image: int = 192
    # Banded min-hash signatures catch whole-scene repeats and crops cheaply.
    band_count: int = 16
    band_rows: int = 2
    min_shared_bands: int = 1
    max_band_df: int = 32
    # Deterministic safety valves.  Truncation is always reported, never
    # silent.
    max_candidates_per_sample: int = 64
    max_total_pairs: int = 200_000

    # -- resource bounds ---------------------------------------------------
    num_workers: int = 1
    verify_cache_size: int = 16
    deterministic_seed: int = 20241123

    # -- batch / history ---------------------------------------------------
    batch_id: str = "batch"
    store_dir: Optional[str] = None
    # "suggest": never rewrite historical clusters wholesale; record merge /
    # split suggestions for review.  "apply_safe": additionally apply
    # unambiguous 1:1 extensions (a historical cluster plus new members).
    history_policy: str = "suggest"
    auto_extend_history: bool = True

    # -- outputs -----------------------------------------------------------
    export_annotation_manifest: bool = True
    write_loader_overview: bool = True

    def __post_init__(self):
        if self.sweep_thresholds is None:
            self.sweep_thresholds = ()
        self.sweep_thresholds = tuple(float(t) for t in self.sweep_thresholds)
        if self.history_policy not in ("suggest", "apply_safe"):
            raise ValueError(f"unknown history_policy: {self.history_policy}")
        if self.ratio_test <= 0 or self.ratio_test > 1:
            raise ValueError("ratio_test must be in (0, 1]")
        if self.dup_threshold <= 0:
            raise ValueError("dup_threshold must be positive")
        if self.max_scale <= self.min_scale:
            raise ValueError("max_scale must exceed min_scale")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def digest_fields(self) -> Dict[str, Any]:
        """The subset of parameters that can change detection results."""
        keys = [
            "max_side",
            "target_pixels",
            "max_features",
            "ratio_test",
            "ransac_reproj_threshold",
            "min_inliers",
            "min_coverage",
            "min_ncc",
            "min_local_inliers",
            "min_local_coverage",
            "min_region_area",
            "min_local_ncc",
            "dup_threshold",
            "rare_code_max_df",
            "min_shared_rare",
            "band_count",
            "band_rows",
            "deterministic_seed",
        ]
        return {"algorithm": ALGORITHM_VERSION, **{k: getattr(self, k) for k in keys}}


# A convenience value used by tests and the CLI when the capability must stay
# completely dormant.
DISABLED = AuditConfig(enabled=False)

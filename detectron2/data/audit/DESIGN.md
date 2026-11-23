# Duplicate audit — design notes and calibration

This note records *why* the audit is built the way it is and the measurements
behind the default thresholds, so the numbers in the report can be traced to a
decision.

## Pipeline

```
DatasetCatalog (unchanged)
        │  collect_samples(split_datasets)  ->  AuditSample(key, file_name, split)
        ▼
load_canonical_image   (read-only; EXIF applied; aspect preserved; large images downscaled)
        ▼
extract_features       (SIFT, root-normalised; small images upsampled x2 *internally*)
        ▼
CandidateIndex         (rare-code index + banded min-hash signature; deterministic)
        ▼
score_pair             (ratio test -> translation/similarity/affine RANSAC ->
                        overlap-aware coverage, warped NCC, local cell-agreement map)
        ▼
classify_evidence      (global vs local vs none, at an arbitrary threshold)
        ▼
build_clusters         (connected components; stable ids from the smallest member)
        ▼
leakage_from_clusters  (cluster membership × registered split; never file names)
```

The measurement (inliers, coverage, NCC, scale, rotation, local region) is kept
separate from the classification, so a threshold sweep re-uses the expensive
measurements and only re-partitions.

## Calibration on real photographs

Five non-synthetic photographs (skimage samples) were used to pick the defaults
before the synthetic acceptance set was written. Representative figures
(overlap-aware coverage, warped NCC):

| pair | coverage | NCC |
|---|---|---|
| burst (tiny sensor noise) | 0.83 – 0.94 | 0.92 – 1.00 |
| crop 86% + rescale | 0.28 – 0.53 | 0.65 – 1.00 |
| rescale to 1/3 | 0.21 – 0.52 | 0.32 – 0.45 |
| JPEG re-encode (q30) | 0.42 – 0.78 | 0.71 |
| watermark | 0.86 – 0.99 | 0.99 |
| unrelated photographs (negatives) | ≤ 0.05 | ≈ 0.00 |
| chelsea ~ cat | — | identical images (skimage duplicate; correctly merged) |

Two calibration findings shaped the design:

1. **Partial matches have low global coverage.** A crop that keeps 60% of the
   frame explains only 30–50% of the *overlapping* keypoints after the
   transform. A single "coverage ≥ 0.5" rule would miss every crop. Hence the
   global bar is 0.55 and crops are caught by the *local* branch.
2. **Small, repetitive crops need upsampling.** At native scale the top-N SIFT
   selection keeps ambiguous keypoints, and the ratio test drops the matches.
   A 2× internal upsample with a lower contrast threshold raised crop matches
   from ~7 to ~110 on the hardest case. The canonical image is still never
   upsampled on disk, and keypoints are mapped back to canonical coordinates so
   regions stay resolution-stable.

## Local repeats by cell agreement

The first implementation grouped RANSAC inlier keypoints into components. That
failed on the "same background, different subject" case: a changed subject in
the middle is *surrounded* by agreeing background, so a proximity graph bridged
the subject and the selected region straddled the change, diluting the NCC.

The current implementation warps B into A's canonical frame and compares a
16×16 lattice cell by cell (locally normalised absolute difference). The
agreeing cells form the repeated region — typically a ring of background strips
around the subject. Coverage is measured over the *valid* cells, and the
photometric gate is the mean agreement over the agreeing cells, so an altered
middle cannot dilute it. The reported region is the explicit rectangle per row
of agreeing cells, so a ring is localised as four strips rather than one
misleading full-frame box.

## Candidate indexing at scale

O(n²) verification is unacceptable for 10k images. Two indexes propose pairs:

* **rare-code index** — 64-bit hyperplane codes of the descriptors, indexed for
  codes occurring in at most `rare_code_max_df` images. Codes shared by a
  multi-member cluster have df equal to the cluster size, so the cap must be
  large enough to keep them; `store_codes_per_image` bounds what is *persisted*
  for cross-batch work without affecting verification.
* **banded min-hash** — a compact signature of the descriptor set; whole-scene
  repeats and crops share bands.

Measured candidate recall on the controlled set: every planted duplicate pair
is proposed, while proposed pairs stay well under half of all pairs. The
default per-run cap (`max_total_pairs`) is a deterministic safety valve and any
truncation is reported, never silent.

## Determinism

* SIFT with a fixed seed, `nfeatures`, and thresholds.
* Keypoints sorted by `(x, y, -size)` regardless of OpenCV's internal order.
* Candidate order and pair order are by sorted sample key.
* Cluster partition from union-find with a lexicographic-merge rule, so the
  result does not depend on edge insertion order.
* Pair measurement is canonicalised internally (descriptor-digest tie-break),
  so `score_pair(a, b) == score_pair(b, a)`.

Verified: two independent runs over the same data produce identical cluster
ids, member sets and edge scores (asserted in `test_determinism.py`).

## Why the disabled path is genuinely inert

`detectron2/data/audit/loader_bridge.py` imports only JSON helpers; it does not
import OpenCV, numpy, or the engine. `get_detection_dataset_dicts` calls it only
after the dataset dicts are built, and the call returns immediately unless
`DETECTRON2_AUDIT_OVERVIEW` is set. Registration functions, metadata and the
returned dicts are untouched. The engine (OpenCV/scipy) is imported lazily, so
`import detectron2.data` pulls in nothing the audit needs.

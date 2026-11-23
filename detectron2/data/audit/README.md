# Near-duplicate and cross-split leakage audit

The third batch of supplier data contained burst frames of the same road corner
in both the training and validation splits: adjacent frames a few milliseconds
apart, nearly overlapping boxes, and a validation score that climbed during
training and collapsed after deployment. This package finds exactly that class
of problem **before** a measurement is trusted:

* it detects near-duplicate images (burst frames, crops, rescales, re-encodes,
  watermarked copies, and *local* repeats such as the same background with a
  different subject);
* it groups them into duplicate clusters and reports how close each pair is and
  **why** it was judged a duplicate;
* it reports cross-split leakage from the **actual cluster membership and the
  split each sample was registered under** — never from a file name;
* it produces a threshold sweep so the cluster count and leakage rate can be
  read at several settings;
* it locates every result to a concrete file, and to a concrete image region
  for local repeats;
* it is reproducible, incremental (new batches only), and leaves the normal
  dataset registration and loading untouched, including when it is switched
  off.

Nothing is downloaded and no pre-trained weights are used. The audit needs
OpenCV **with SIFT** (`opencv-python-headless` or `opencv-contrib-python`);
that is the only feature in this repository that requires it.

## Why not a checksum, and not "whole image similarity"

* **Checksums are useless here.** A re-encode changes every byte while the
  pixels are identical. The audit compares *content*: SIFT keypoints and their
  descriptors, which survive re-encoding, rescaling and watermarks.
* **The whole-image similarity test alone misses local repeats.** A pair is
  verified with a geometric model (translation, similarity or affine),
  RANSAC-consistent inliers, an overlap-aware coverage ratio, and a
  photometric NCC check on the shared pixels. For partial repeats (a crop, a
  reused background) the audit additionally locates a *repeated region* by
  comparing the two images on a lattice of cells after warping. A changed
  subject in the middle of an otherwise identical frame therefore shows up as a
  local duplicate with an explicit region, not as a missed duplicate or a
  false whole-frame match.

## Quick start

```python
from detectron2.data.audit import AuditConfig, run_audit

result = run_audit(
    {
        "train": ["my_train_dataset"],
        "val":   ["my_val_dataset"],
        "test":  ["my_test_dataset"],
    },
    AuditConfig(enabled=True, store_dir="reports/duplicate_audit", batch_id="batch3"),
    report_dir="reports/duplicate_audit",
)
print(result.overview["headline"])
# batch batch3: N duplicate clusters (x% of M samples), K clusters spanning splits (...)
```

`run_audit` resolves the dataset names through the existing `DatasetCatalog`
exactly as the loader does; it does not register or modify anything.

The artifacts written to `report_dir` are:

| file | contents |
|---|---|
| `clusters.json` | clusters, members, per-edge score, kind (`global`/`local`), the basis of the decision, and the repeated region (canonical **and** original coordinates) |
| `pairs.json` | every verified pair with its measurements (inliers, coverage, scale, rotation, NCC) |
| `split_leaks.json` | the leakage list: clusters spanning splits and the affected samples |
| `dataset_stats.json` | duplicate rate, cross-split leakage rate, affected samples, per-split rates |
| `threshold_sweep.json` | cluster count and leakage rate at each configured threshold |
| `annotation_manifest.json` | a list an annotation platform can ingest: cluster number, files, suggested action |
| `audit_overview.json` | compact per-batch summary read at data-entry time |
| `summary.json`, `skipped.json`, `history_reconciliation.json`, `audit_store.json` | human summary, skipped files, reconciliation record, persistent history |

All files are written atomically, and the store is updated **last**, so an
interrupted run cannot leave a half-written leakage list.

## The important knobs

Most callers only need `enabled`, `dup_threshold`, and `sweep_thresholds`.

* `dup_threshold` (default `0.45`) — pairs at or above this similarity become
  cluster edges. Edges within `borderline_margin` of it are marked
  `borderline` for review.
* `sweep_thresholds` — the thresholds reported in the sweep (the default
  threshold is always included).
* `min_coverage` (global, `0.55`) and `min_local_coverage` (local, `0.12`) —
  how much of the overlapping keypoint field a global / local duplicate must
  explain.
* `min_local_ncc` / `local_cell_agreement` — the photometric bar for local
  repeats.
* `max_side`, `target_pixels`, `max_pixels` — canonicalisation and the size
  guard. Images are never upsampled for storage and their aspect ratio is
  preserved; only the `upsample` factor is applied *internally* before SIFT so
  a repeated crop keeps enough distinctive keypoints.
* `rare_code_max_df`, `band_count`, `band_rows` — candidate indexing. At 10k
  scale, pairs are proposed from a rare-descriptor index and a banded min-hash
  signature; the full geometric check runs only on the proposed pairs. The
  default candidate budget is sub-quadratic.
* `max_total_pairs` — a deterministic safety valve. If exceeded, the pairs are
  truncated in a reported, reproducible order (never silently).
* `history_policy` — `"suggest"` (default) records merge/split suggestions for
  a new batch against historical clusters; `"apply_safe"` additionally applies
  unambiguous extensions.

## Incremental batches

Pass the same `store_dir` for a later batch:

```python
run_audit({"test": ["late_test_batch"]},
          AuditConfig(enabled=True, store_dir="reports/duplicate_audit", batch_id="batch4"),
          report_dir="reports/duplicate_audit")
```

Only the new batch's images are read (`stats["images_scanned_in_this_run"]`).
Historical pair measurements are reused, and the report still covers the whole
accumulated corpus. Reconciliation is **additive**: a historical cluster gains
a new member only when that member has a verified edge to *every* existing
member; otherwise a merge/split suggestion is written to
`history_reconciliation.json` for review. Historical clusters are never
rewritten from a partially-computed batch.

## Reading the result from the data entry point

The audit is off by default and changes nothing when off. To surface the
per-batch summary while loading, opt in explicitly:

```bash
export DETECTRON2_AUDIT_OVERVIEW=1
export DETECTRON2_AUDIT_REPORT_DIR=reports/duplicate_audit
```

`get_detection_dataset_dicts` then attaches `MetadataCatalog.get(name).audit_overview`
and logs the batch headline. The returned dataset dicts, registered functions
and metadata used for loading are unchanged; a stale overview (one whose
`batch_id` does not match) is never presented as current. With the environment
variables unset, the hook is a complete no-op.

## Robustness

* A corrupt, truncated, tiny or absurdly large image is skipped and counted;
  the run continues and the skip reason is written to `skipped.json`.
* Images with no usable keypoints are skipped with `no_features`.
* Every write is atomic, so a crash cannot leave a partial leakage list.
* The measurement of a pair does not depend on argument order (it is
  canonicalised internally), so two runs or two call sites agree.

## Reproducing the acceptance report

```bash
python tools/audit_report.py --out reports/duplicate_audit
# add --run-data-tests to also run the (slow) original tests/data suite
```

This builds a controlled dataset with burst, crop and re-encode duplicates,
runs the audit twice to demonstrate determinism, produces the threshold sweep,
and writes the per-parent artifacts. See `reports/duplicate_audit/README.md`
after running it. The original `tests/data` suite is run separately (it takes
several minutes); its only failures in this environment are pre-existing
environment issues, reproducible on the baseline commit without the audit.

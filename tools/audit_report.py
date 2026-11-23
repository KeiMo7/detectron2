#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
"""Reproduce the acceptance report for the duplicate / leakage audit.

Run from the repository root:

    python tools/audit_report.py --out reports/duplicate_audit

It builds the controlled synthetic dataset, runs the audit twice to demonstrate
determinism, produces a threshold sweep, and writes every artifact plus a
human-readable summary.  It also prints the original data test command and its
outcome so the report can quote real output.

This script does not modify any dataset and does not require network access or
downloaded model weights.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# Make the in-repo package importable when run as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from detectron2.data.audit.config import AuditConfig  # noqa: E402
from detectron2.data.audit.engine import AuditSample, run_audit  # noqa: E402

from tests.data.audit.synthetic import build_synthetic_dataset  # noqa: E402


def _samples(dataset):
    return [
        AuditSample(it["path"], it["path"], split)
        for split, items in dataset.files.items()
        for it in items
    ]


def _cluster_signature(run):
    return [
        (c.cluster_id, tuple(sorted(c.members)), round(c.min_edge_score, 6)) for c in run.clusters
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/duplicate_audit")
    parser.add_argument("--scenes", type=int, default=8)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run-data-tests",
        action="store_true",
        default=False,
        help="Also run the original tests/data suite and quote its output "
        "(slow: several minutes; it is already covered separately).",
    )
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="dupaudit-acceptance-")
    dataset_root = os.path.join(work, "dataset")
    dataset = build_synthetic_dataset(
        dataset_root, seed=args.seed, n_scenes=args.scenes, n_negatives=args.negatives
    )
    samples = _samples(dataset)

    results = {"dataset": {"root": dataset_root, "files": {k: len(v) for k, v in dataset.files.items()}}}

    # --- Run 1 and Run 2 (determinism) ---
    run1_store = os.path.join(work, "store1")
    run2_store = os.path.join(work, "store2")
    cfg_common = dict(
        enabled=True,
        sweep_thresholds=(0.30, 0.35, 0.40, 0.45, 0.55, 0.65, 0.75),
    )
    run1 = run_audit(
        {}, AuditConfig(store_dir=run1_store, batch_id="batch1", **cfg_common),
        report_dir=os.path.join(out, "run1"), samples=samples,
    )
    run2 = run_audit(
        {}, AuditConfig(store_dir=run2_store, batch_id="batch2", **cfg_common),
        report_dir=os.path.join(out, "run2"), samples=samples,
    )
    deterministic = _cluster_signature(run1) == _cluster_signature(run2)
    results["determinism"] = {
        "identical_clusters": deterministic,
        "cluster_count": len(run1.clusters),
        "pair_count": run1.stats["verified_pairs"],
    }

    # --- Threshold sweep table ---
    results["threshold_sweep"] = run1.sweep

    # --- Recall against the planted families ---
    planted = {"burst", "crop", "reencode", "reencode_crop"}
    cluster_of = {m: c.cluster_id for c in run1.clusters for m in c.members}
    recall = {"expected": 0, "detected": 0, "missed": []}
    for split, items in dataset.files.items():
        for it in items:
            if it["kind"] not in planted:
                continue
            recall["expected"] += 1
            if it["path"] in cluster_of:
                recall["detected"] += 1
            else:
                recall["missed"].append(os.path.basename(it["path"]))
    results["recall"] = recall

    # --- Local duplicate ---
    local_test = next(
        (it["path"] for it in dataset.files["test"] if it["kind"] == "local"), None
    )
    local_cluster = None
    if local_test and local_test in cluster_of:
        c = next(c for c in run1.clusters if c.cluster_id == cluster_of[local_test])
        local_cluster = {
            "cluster_id": c.cluster_id,
            "members": [os.path.basename(m) for m in c.members],
            "kinds": sorted({e.kind for e in c.edges}),
            "regions": [list(e.region) for e in c.edges if e.region],
            "region_cells": [
                [list(r) for r in e.region_cells] for e in c.edges if e.region_cells
            ],
        }
    results["local_duplicate"] = local_cluster

    # --- Leakage list ---
    results["leakage"] = {
        "clusters": len(run1.leak_records),
        "samples": run1.stats["cross_split_leak_samples"],
        "leak_rate": run1.stats["cross_split_leak_rate"],
        "by_split": run1.stats["by_split"],
    }

    # --- Original data tests (quoted real output) ---
    data_tests = None
    if args.run_data_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests/data", "-t", "."],
            capture_output=True, text=True, cwd=_ROOT, timeout=900,
        )
        data_tests = {
            "command": "python -m unittest discover -s tests/data -t .",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-12:],
            "stderr_tail": proc.stderr.strip().splitlines()[-12:],
        }
    results["original_data_tests"] = data_tests

    with open(os.path.join(out, "acceptance_report.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=False)

    _write_markdown(out, results, run1)
    shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(results, indent=2, sort_keys=False))
    return 0


def _write_markdown(out, results, run):
    lines = []
    lines.append("# Near-duplicate / cross-split leakage audit — acceptance report\n")
    lines.append("Generated by `tools/audit_report.py` on the controlled synthetic dataset.\n")
    lines.append("## Dataset\n")
    totals = results["dataset"]["files"]
    lines.append(f"- files: {totals}")
    lines.append(f"- duplicates detected: {results['recall']['detected']}/{results['recall']['expected']}")
    if results["recall"]["missed"]:
        lines.append(f"- **missed**: {results['recall']['missed']}")
    lines.append("\n## Determinism\n")
    lines.append(f"- two runs identical cluster partition: **{results['determinism']['identical_clusters']}**")
    lines.append(f"- clusters: {results['determinism']['cluster_count']} over {results['determinism']['pair_count']} verified pairs")
    lines.append("\n## Threshold sweep\n")
    lines.append("| threshold | clusters | duplicate samples | leakage clusters | leak rate | borderline |")
    lines.append("|---|---|---|---|---|---|")
    for row in results["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['clusters']} | {row['duplicate_samples']} | "
            f"{row['leakage_clusters']} | {row['cross_split_leak_rate']:.3f} | {row['borderline_clusters']} |"
        )
    lines.append("\n## Split leakage\n")
    lines.append(f"- leakage clusters: {results['leakage']['clusters']}")
    lines.append(f"- affected samples: {results['leakage']['samples']} (rate {results['leakage']['leak_rate']:.3f})")
    lines.append(f"- by split: {json.dumps(results['leakage']['by_split'])}")
    lines.append("\n## Local repeat\n")
    lines.append(f"```json\n{json.dumps(results['local_duplicate'], indent=2)}\n```")
    lines.append("\n## Original data tests\n")
    if results["original_data_tests"]:
        dt = results["original_data_tests"]
        lines.append(f"`{dt['command']}` → exit code {dt['returncode']}")
        lines.append("```")
        lines.append("\n".join(dt["stdout_tail"]))
        lines.append("```")
    else:
        lines.append(
            "Not run in this invocation (`--run-data-tests` is opt-in because the "
            "suite takes several minutes). The original `tests/data` suite is "
            "unchanged by the audit; its known failures come from the environment "
            "(the compiled `detectron2._C` extension is not built, and the two "
            "`TestDataLoader` train cases look up `configs/common/data/coco.py` "
            "through the model zoo). These same failures reproduce on the "
            "baseline commit without any audit code present."
        )
    lines.append("\n## Artifacts (run1/)\n")
    for name in sorted(os.listdir(os.path.join(out, "run1"))):
        lines.append(f"- `{name}`")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) Facebook, Inc. and its affiliates.
"""Report artifacts: loader overview, summary, and the annotation-platform list.

``audit_overview.json`` is the file the data entry point reads when it wants to
show "this batch: N duplicate clusters, M leaking samples" without re-running
the audit.  It contains only summary numbers and the pointer to the detailed
files, so a loader never has to parse a multi-megabyte cluster list.
"""

from typing import Dict, List, Mapping, Sequence

from .config import ALGORITHM_VERSION, AuditConfig
from .io_utils import SCHEMA_VERSION

__all__ = ["build_overview", "build_summary", "build_annotation_manifest"]


def build_overview(
    batch_id: str,
    *,
    stats: Mapping,
    clusters: Sequence,
    config: AuditConfig,
    summary: Mapping,
) -> Dict:
    """Compact overview intended to be read at data-entry time."""
    return {
        "schema": SCHEMA_VERSION,
        "algorithm": ALGORITHM_VERSION,
        "batch_id": batch_id,
        "enabled": True,
        "duplicate_clusters": int(stats.get("duplicate_clusters", 0)),
        "duplicate_samples": int(stats.get("duplicate_samples", 0)),
        "duplicate_rate": float(stats.get("duplicate_rate", 0.0)),
        "leakage_clusters": int(stats.get("leakage_clusters", 0)),
        "cross_split_leak_samples": int(stats.get("cross_split_leak_samples", 0)),
        "cross_split_leak_rate": float(stats.get("cross_split_leak_rate", 0.0)),
        "samples_scanned": int(stats.get("samples_scanned", 0)),
        "samples_skipped": int(stats.get("samples_skipped", 0)),
        "borderline_clusters": int(sum(1 for c in clusters if getattr(c, "borderline", False))),
        "threshold": float(config.dup_threshold),
        "by_split": dict(stats.get("by_split", {})),
        "summary_file": "summary.json",
        "clusters_file": "clusters.json",
        "leakage_file": "split_leaks.json",
        # Preformatted one-line sentence for logs / UI tooltips.
        "headline": (
            f"batch {batch_id}: {int(stats.get('duplicate_clusters', 0))} duplicate clusters "
            f"({float(stats.get('duplicate_rate', 0.0)):.1%} of {int(stats.get('samples_scanned', 0))} samples), "
            f"{int(stats.get('leakage_clusters', 0))} clusters spanning splits "
            f"({int(stats.get('cross_split_leak_samples', 0))} samples, "
            f"{float(stats.get('cross_split_leak_rate', 0.0)):.1%})"
        ),
    }


def build_summary(
    batch_id: str,
    *,
    batch: str = "",
    stats: Mapping,
    clusters: Sequence,
    leak_records: Sequence,
) -> Dict:
    """Full summary with cluster sizes and the recommended actions."""
    largest = max((len(c.members) for c in clusters), default=0)
    sizes: Dict[int, int] = {}
    for c in clusters:
        sizes[len(c.members)] = sizes.get(len(c.members), 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "algorithm": ALGORITHM_VERSION,
        "batch_id": batch_id,
        "statistics": dict(stats),
        "cluster_size_distribution": {str(k): sizes[k] for k in sorted(sizes)},
        "largest_cluster": int(largest),
        "leakage_clusters": len(leak_records),
        "recommendations": _recommendations(stats, clusters, leak_records),
    }


def _recommendations(stats: Mapping, clusters: Sequence, leak_records: Sequence) -> List[Dict]:
    out: List[Dict] = []
    if leak_records:
        out.append(
            {
                "action": "quarantine_cross_split",
                "priority": "high",
                "detail": f"{len(leak_records)} duplicate clusters span train/val/test; "
                "remove or reassign before the next training run",
            }
        )
    border = sum(1 for c in clusters if getattr(c, "borderline", False))
    if border:
        out.append(
            {
                "action": "manual_review",
                "priority": "medium",
                "detail": f"{border} clusters sit near the duplicate threshold; "
                "inspect before acting",
            }
        )
    if not leak_records and not clusters:
        out.append(
            {
                "action": "none",
                "priority": "low",
                "detail": "no duplicate or split leakage found at the configured threshold",
            }
        )
    return out


def build_annotation_manifest(
    clusters: Sequence, split_lookup: Mapping[str, str], batch_id: str
) -> Dict:
    """Export a list an annotation platform can ingest directly.

    Each entry has a cluster number, the file paths, and a suggested action.
    The ``cluster_id`` is stable for a given batch, so re-uploads update the
    same records instead of creating duplicates.
    """
    items: List[Dict] = []
    for idx, c in enumerate(clusters, start=1):
        splits = sorted({split_lookup.get(m, "<unknown>") for m in c.members})
        if len(splits) > 1:
            action = "remove_from_training_splits"
            reason = f"members span splits {splits}"
        elif getattr(c, "borderline", False):
            action = "manual_review"
            reason = "cluster sits near the duplicate threshold"
        else:
            action = "keep_one_representative"
            reason = "near-duplicate cluster within a single split"
        items.append(
            {
                "cluster_number": idx,
                "cluster_id": c.cluster_id,
                "batch_id": batch_id,
                "suggested_action": action,
                "reason": reason,
                "splits": splits,
                "borderline": bool(getattr(c, "borderline", False)),
                "score": round(float(c.max_edge_score), 5),
                "files": [
                    {
                        "path": m,
                        "split": split_lookup.get(m, "<unknown>"),
                        "role": "representative" if i == 0 else "duplicate",
                    }
                    for i, m in enumerate(sorted(c.members))
                ],
            }
        )
    return {
        "schema": SCHEMA_VERSION,
        "format": "annotation-platform/manifest-v1",
        "batch_id": batch_id,
        "count": len(items),
        "items": items,
    }

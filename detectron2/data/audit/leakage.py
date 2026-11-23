# Copyright (c) Facebook, Inc. and its affiliates.
"""Cross-split leakage detection and dataset-level statistics.

Leakage is decided **only** from the cluster partition and the split each
sample actually belongs to (the split is taken from the dataset registration
that produced the sample, never guessed from a file name).  A cluster that
spans more than one split is a leakage cluster; the listed samples are the ones
in that split whose content also appears in another split.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["LeakRecord", "leakage_from_clusters", "dataset_statistics"]


@dataclass
class LeakRecord:
    cluster_id: str
    splits: List[str]
    # split -> sample keys that share content with another split
    members_by_split: Dict[str, List[str]]
    # every sample that lies in a leaking split (for "affected samples")
    total_members: int
    min_edge_score: float

    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "splits": list(self.splits),
            "members_by_split": {k: list(v) for k, v in self.members_by_split.items()},
            "total_members": self.total_members,
            "min_edge_score": round(self.min_edge_score, 5),
        }


def leakage_from_clusters(
    clusters: Sequence, split_of: Mapping[str, str], clusters_by_id: Optional[Sequence] = None
) -> Tuple[List[LeakRecord], List[str]]:
    """Return leakage records and the set of sample keys affected.

    ``clusters`` are objects with ``cluster_id``, ``members`` and
    ``min_edge_score`` (see :class:`~.clustering.Cluster`).
    """
    records: List[LeakRecord] = []
    affected = set()
    for c in clusters:
        by_split: Dict[str, List[str]] = defaultdict(list)
        for key in c.members:
            by_split[split_of.get(key, "<unknown>")].append(key)
        if len(by_split) < 2:
            continue
        # A sample is at risk when its split holds a duplicate of content that
        # also exists in at least one *other* split.  All members of a leaking
        # cluster qualify, because each was seen by at least one other split's
        # samples somewhere in the connected component.
        members_by_split = {k: sorted(v) for k, v in sorted(by_split.items())}
        records.append(
            LeakRecord(
                cluster_id=c.cluster_id,
                splits=sorted(by_split.keys()),
                members_by_split=members_by_split,
                total_members=len(c.members),
                min_edge_score=float(c.min_edge_score),
            )
        )
        affected.update(c.members)
    return records, sorted(affected)


def dataset_statistics(
    *,
    samples: int,
    skipped: int,
    clusters: Sequence,
    leak_records: Sequence[LeakRecord],
    duplicate_sample_count: int,
    split_sizes: Optional[Mapping[str, int]] = None,
    cross_split_sample_count: int = 0,
) -> Dict:
    """Aggregate the dataset-level numbers required by the audit report."""
    total = max(1, samples)
    dup_rate = duplicate_sample_count / total
    leak_rate = cross_split_sample_count / total
    by_split: Dict[str, Dict[str, float]] = {}
    if split_sizes:
        # How many samples of each split are duplicated *across* splits.
        per_split_leaked: Dict[str, int] = defaultdict(int)
        for rec in leak_records:
            for split, members in rec.members_by_split.items():
                # Only count a split member if another split is in the record.
                per_split_leaked[split] += len(members)
        for split, size in sorted(split_sizes.items()):
            n = per_split_leaked.get(split, 0)
            by_split[split] = {
                "samples": int(size),
                "leaked_samples": int(n),
                "leak_rate": round(n / max(1, size), 6),
            }
    return {
        "samples_scanned": int(samples),
        "samples_skipped": int(skipped),
        "duplicate_clusters": len(clusters),
        "duplicate_samples": int(duplicate_sample_count),
        "duplicate_rate": round(dup_rate, 6),
        "leakage_clusters": len(leak_records),
        "cross_split_leak_samples": int(cross_split_sample_count),
        "cross_split_leak_rate": round(leak_rate, 6),
        "by_split": by_split,
    }

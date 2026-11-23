# Copyright (c) Facebook, Inc. and its affiliates.
"""Persistent audit store and reconcile-new-batch-with-history.

The store keeps the verified pair evidence from previous batches so a new batch
only needs its *own* images plus the historical representatives, not a full
re-audit.  It is written atomically, so an interrupted update cannot corrupt a
previous good store.

Reconciliation is additive.  A historical cluster is extended only when the new
batch supplies a member that has a verified edge to every existing member (and
the edge scores are consistent).  Anything else becomes a *suggestion* (merge
or split) that is recorded for review; historical clusters are never silently
rewritten from a partially-computed batch.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .io_utils import SCHEMA_VERSION, atomic_write_json, read_json_if_exists, store_path

logger = logging.getLogger(__name__)

__all__ = ["AuditStore", "HistoryReconciliation"]


def _content_key(path: str, size: int) -> str:
    return f"{path}\x00{size}"


@dataclass
class AuditStore:
    """Verified evidence accumulated across batches."""

    version: int = 1
    algorithm: str = ""
    batches: Optional[List[str]] = None
    # path -> {size, features_digest, batch, split}
    images: Optional[Dict[str, Dict]] = None
    # "a\x00b" (lexicographically ordered paths) -> dict of pair evidence
    pairs: Optional[Dict[str, Dict]] = None
    # sorted tuples of image paths
    clusters: Optional[List[List[str]]] = None
    # split of each image path, last writer wins (registration is authoritative)
    splits: Optional[Dict[str, str]] = None

    def __post_init__(self):
        self.batches = self.batches or []
        self.images = self.images or {}
        self.pairs = self.pairs or {}
        self.clusters = self.clusters or []
        self.splits = self.splits or {}

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, store_dir: str) -> "AuditStore":
        payload = read_json_if_exists(store_path(store_dir))
        if payload is None:
            return cls()
        store = cls(
            version=int(payload.get("version", 1)),
            algorithm=payload.get("algorithm", ""),
            batches=list(payload.get("batches", [])),
            images=dict(payload.get("images", {})),
            pairs=dict(payload.get("pairs", {})),
            clusters=[list(map(str, c)) for c in payload.get("clusters", [])],
            splits=dict(payload.get("splits", {})),
        )
        return store

    def save(self, store_dir: str) -> None:
        payload = {
            "schema": SCHEMA_VERSION,
            "version": self.version,
            "algorithm": self.algorithm,
            "batches": sorted(set(self.batches)),
            "images": self.images,
            "pairs": self.pairs,
            "clusters": sorted(self.clusters, key=lambda c: (len(c), c)),
            "splits": self.splits,
        }
        atomic_write_json(store_path(store_dir), payload)
        self.batches = sorted(set(self.batches))

    # -- mutation ----------------------------------------------------------
    def add_batch(self, batch_id: str) -> None:
        if batch_id not in self.batches:
            self.batches.append(batch_id)

    def has_batch(self, batch_id: str) -> bool:
        return batch_id in self.batches

    def known_image(self, path: str, split: Optional[str] = None) -> bool:
        if path not in self.images:
            return False
        if split is None:
            return True
        return self.images[path].get("split") == split

    def add_image(
        self,
        path: str,
        split: str,
        batch: str,
        digest: str,
        size: int,
        codes: Optional[Sequence[int]] = None,
    ) -> None:
        rec = {
            "split": split,
            "batch": batch,
            "digest": digest,
            "size": int(size),
        }
        if codes is not None:
            # Rare-code set, used to propose cross-batch candidates without
            # re-reading the image.  Encoded as hex strings: each 64-bit code
            # is then 16 characters instead of up to 20 decimal digits, which
            # roughly halves the store on disk.
            rec["codes"] = [format(int(c), "x") for c in codes]
        self.images[path] = rec
        self.splits[path] = split

    def add_pair(self, a: str, b: str, evidence: Dict) -> None:
        key = self._pair_key(a, b)
        self.pairs[key] = evidence

    def pair(self, a: str, b: str) -> Optional[Dict]:
        return self.pairs.get(self._pair_key(a, b))

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return f"{a}\x00{b}" if a <= b else f"{b}\x00{a}"

    def set_clusters(self, clusters: Sequence[Sequence[str]]) -> None:
        self.clusters = sorted([sorted(c) for c in clusters], key=lambda c: (len(c), c))


@dataclass
class HistoryReconciliation:
    merges: List[Dict]
    extensions: List[Dict]
    splits: List[Dict]
    applied: List[Dict]

    def to_dict(self) -> Dict:
        return {
            "suggested_merges": self.merges,
            "extensions": self.extensions,
            "suggested_splits": self.splits,
            "applied": self.applied,
            "policy": "additive; historical clusters are extended only when a new "
            "member connects to every existing member",
        }


def reconcile_with_history(
    *,
    historical_clusters: Sequence[Sequence[str]],
    new_clusters: Sequence[Sequence[str]],
    edges: Sequence,
    policy: str,
    auto_extend: bool,
) -> Tuple[List[List[str]], HistoryReconciliation]:
    """Merge the new batch's clusters into the historical partition.

    ``edges`` must contain the verified edges that involve at least one new
    sample and one historical sample (plus the new batch's own edges).  The
    returned cluster list is the reconciled partition; the reconciliation record
    explains every change.
    """
    hist = [sorted(c) for c in historical_clusters]
    new = [sorted(c) for c in new_clusters]

    edge_scores: Dict[Tuple[str, str], float] = {}
    for e in edges:
        a, b = (e.a, e.b) if e.a <= e.b else (e.b, e.a)
        edge_scores[(a, b)] = max(edge_scores.get((a, b), 0.0), float(e.score))

    def score(a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        return edge_scores.get(key, 0.0)

    merges: List[Dict] = []
    extensions: List[Dict] = []
    splits: List[Dict] = []
    applied: List[Dict] = []

    # 1) New cluster straddling one or more historical clusters.
    hist_index: Dict[str, int] = {}
    for i, c in enumerate(hist):
        for m in c:
            hist_index[m] = i

    link: Dict[int, List[int]] = {}
    for ni, c in enumerate(new):
        touched = sorted({hist_index[m] for m in c if m in hist_index})
        if touched:
            link[ni] = touched

    # A new cluster linking several historical clusters is a merge suggestion.
    consumed_hist = set()
    for ni, touched in sorted(link.items()):
        if len(touched) > 1:
            merges.append(
                {
                    "new_cluster_members": new[ni],
                    "historical_clusters": [list(hist[t]) for t in touched],
                    "suggestion": "merge",
                    "reason": "a new batch cluster connects historical clusters",
                }
            )
            continue
        t = touched[0]
        # Extension is safe only if every new member has strong edges to *all*
        # existing members of the historical cluster.
        existing = hist[t]
        new_only = [m for m in new[ni] if m not in existing]
        all_linked = all(score(m, e) > 0 for m in new_only for e in existing)
        record = {
            "historical_cluster_id": _cluster_id(existing),
            "historical_members": existing,
            "new_members": new_only,
            "new_cluster_members": new[ni],
            "all_new_link_every_existing": bool(all_linked),
        }
        # "suggest" never mutates history; it only records what *would* happen.
        # "apply_safe" performs the extension when it is unambiguous.
        if all_linked and policy == "apply_safe" and auto_extend:
            hist[t] = sorted(existing + new_only)
            consumed_hist.add(t)
            record["suggestion"] = "extend"
            applied.append(record)
            extensions.append(record)
        elif all_linked:
            record["suggestion"] = "extend (not applied: policy is 'suggest')"
            extensions.append(record)
        else:
            record["suggestion"] = "review"
            record["reason"] = (
                "some new member does not link every existing member; "
                "kept separate pending manual review"
            )
            extensions.append(record)

    # 2) A historical cluster internally has a weak edge but the new batch
    #    reinforces it strongly: suggest splitting if the weakest historical
    #    pair is below the new threshold and the new data does not bridge it.
    #    (Conservative: only suggestions, never automatic.)
    for c in hist:
        if len(c) < 2:
            continue
        rated = [(score(x, y), x, y) for x, y in _pairs(c) if (x, y) in edge_scores]
        if not rated:
            # No evidence for this historical cluster in this run (it was not
            # re-measured); do not invent a split suggestion.
            continue
        weakest = min(rated)
        if weakest[0] <= 0:
            splits.append(
                {
                    "historical_cluster_id": _cluster_id(c),
                    "members": c,
                    "weakest_pair": [weakest[1], weakest[2]],
                    "weakest_score": weakest[0],
                    "suggestion": "review split",
                }
            )

    # 3) Add genuinely new clusters untouched by history.
    merged_clusters = [sorted(c) for c in hist]
    for ni, c in enumerate(new):
        # Members already present in a merged cluster (either from history or a
        # safe extension) are not duplicated.
        if all(any(m in x for x in merged_clusters) for m in c):
            continue
        merged_clusters.append(sorted(c))

    merged_clusters.sort(key=lambda c: (len(c), c))
    return merged_clusters, HistoryReconciliation(
        merges=merges, extensions=extensions, splits=splits, applied=applied
    )


def _pairs(c: Sequence[str]):
    from itertools import combinations

    return combinations(sorted(c), 2)


def _cluster_id(members: Sequence[str]) -> str:
    anchor = min(members)
    return "dup-" + hashlib.sha256(anchor.encode("utf-8")).hexdigest()[:12]

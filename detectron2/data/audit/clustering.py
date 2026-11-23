# Copyright (c) Facebook, Inc. and its affiliates.
"""Deterministic duplicate clustering.

Clusters are connected components of the verified duplicate graph.  Ties are
broken by sorted member key (never by dict/insertion order), so two runs over
the same data and configuration always produce the same partition and the same
cluster ids.  Cluster ids are derived from the smallest member key, which makes
them stable for a given batch.
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Edge", "Cluster", "build_clusters", "cluster_id_for"]


@dataclass(frozen=True, order=True)
class Edge:
    a: str  # sample key
    b: str
    score: float
    kind: str  # "global" | "local"
    borderline: bool = False
    # canonical xyxy of the repeated area (for local duplicates)
    region: Optional[Tuple[int, int, int, int]] = None
    reasons: Tuple[str, ...] = ()
    # The region is expressed in the frame of the second endpoint when set.
    region_swapped: bool = False
    # Rectangles (canonical xyxy) covering the agreeing cells; a repeated area
    # can be ring-shaped, so a single bbox is not enough to localise it.
    region_cells: Optional[Tuple[Tuple[int, int, int, int], ...]] = None


@dataclass
class Cluster:
    cluster_id: str
    members: List[str]
    edges: List[Edge]
    min_edge_score: float
    max_edge_score: float
    borderline: bool

    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "members": list(self.members),
            "size": len(self.members),
            "min_edge_score": round(self.min_edge_score, 5),
            "max_edge_score": round(self.max_edge_score, 5),
            "borderline": bool(self.borderline),
            "edges": [
                {
                    "a": e.a,
                    "b": e.b,
                    "score": round(e.score, 5),
                    "kind": e.kind,
                    "borderline": e.borderline,
                    "region": list(e.region) if e.region else None,
                    "region_cells": [list(r) for r in e.region_cells] if e.region_cells else None,
                    "region_anchor_side": "b" if e.region_swapped else "a",
                    "basis": list(e.reasons),
                }
                for e in self.edges
            ],
        }


def cluster_id_for(members: Sequence[str]) -> str:
    """Stable cluster id derived from the sorted member keys."""
    anchor = min(members)
    h = hashlib.sha256(anchor.encode("utf-8")).hexdigest()[:12]
    return f"dup-{h}"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression, still order-independent for the final roots.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Always attach the lexicographically larger root to the smaller one:
        # this makes the resulting partition independent of edge order.
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


def build_clusters(
    edges: Iterable[Edge],
    threshold: float,
    min_cluster_size: int,
    all_samples: Optional[Iterable[str]] = None,
) -> List[Cluster]:
    """Partition the duplicate graph at ``threshold``.

    Edges with ``score >= threshold`` are included.  Borderline flags come from
    the evidence and are propagated to the cluster.
    """
    uf = _UnionFind()
    if all_samples is not None:
        for s in all_samples:
            uf.add(s)
    accepted = [e for e in edges if e.score >= threshold]
    for e in accepted:
        uf.add(e.a)
        uf.add(e.b)
        uf.union(e.a, e.b)

    groups: Dict[str, List[str]] = {}
    for node in uf.parent:
        groups.setdefault(uf.find(node), []).append(node)

    by_group: Dict[str, List[Edge]] = {}
    for e in accepted:
        by_group.setdefault(uf.find(e.a), []).append(e)

    clusters: List[Cluster] = []
    for root, members in groups.items():
        if len(members) < min_cluster_size:
            continue
        members = sorted(members)
        edges = sorted(by_group.get(root, []), key=lambda e: (e.a, e.b))
        scores = [e.score for e in edges]
        clusters.append(
            Cluster(
                cluster_id=cluster_id_for(members),
                members=members,
                edges=edges,
                min_edge_score=min(scores),
                max_edge_score=max(scores),
                borderline=any(e.borderline for e in edges),
            )
        )
    # Deterministic order: largest first, then by id.
    clusters.sort(key=lambda c: (-len(c.members), c.cluster_id))
    return clusters

# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end orchestration of a duplicate / split-leakage audit run.

A run is a pure function of ``(samples, splits, config, historical store)``:
every expensive step is deterministic, candidate ordering is by sorted sample
key, and clusters are derived from a partition of the verified graph, so two
runs over the same inputs produce byte-identical cluster assignments.

Nothing in this module registers datasets, changes metadata or writes to the
dataset.  It reads image files only.
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .clustering import Edge, build_clusters
from .config import ALGORITHM_VERSION, AuditConfig
from .features import Features, PairEvidence, classify_evidence, extract_features, score_pair
from .image_ops import LoadedImage, SkippedImage, load_canonical_image
from .index import CandidateIndex
from .io_utils import atomic_write_json, report_dir_layout
from .leakage import LeakRecord, dataset_statistics, leakage_from_clusters
from .store import AuditStore, reconcile_with_history
from .artifacts import build_annotation_manifest, build_overview, build_summary

logger = logging.getLogger(__name__)

__all__ = ["AuditSample", "AuditRun", "run_audit"]


@dataclass
class AuditSample:
    """One image taking part in the audit, with its *actual* split.

    ``split`` must come from the registration that produced the sample; it is
    never inferred from ``key``.
    """

    key: str  # unique, stable identifier (usually the resolved file path)
    file_name: str
    split: str
    dataset: str = ""

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "file_name": self.file_name,
            "split": self.split,
            "dataset": self.dataset,
        }


@dataclass
class AuditRun:
    """Result object returned by :func:`run_audit`."""

    config: AuditConfig
    clusters: List
    leak_records: List[LeakRecord]
    stats: Dict
    sweep: List[Dict]
    skipped: List[Dict]
    summary: Dict
    overview: Dict
    history: Dict
    store: AuditStore
    report_dir: str
    paths: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------


def _resolve_file_name(d: Mapping[str, Any]) -> Optional[str]:
    fn = d.get("file_name") or d.get("filename") or d.get("image_path") or d.get("image")
    if fn is None:
        return None
    return os.fspath(fn)


def collect_samples(split_datasets: Mapping[str, Sequence[str]]) -> List[AuditSample]:
    """Build audit samples from ``{split_name: [dataset_name, ...]}``.

    Dataset names are resolved through :class:`DatasetCatalog`, exactly as the
    normal loader does; the audit does not register or modify anything.
    """
    from detectron2.data.catalog import DatasetCatalog

    samples: List[AuditSample] = []
    seen: Dict[str, str] = {}
    for split, names in split_datasets.items():
        for name in names:
            dicts = DatasetCatalog.get(name)
            if isinstance(dicts, dict) or not isinstance(dicts, (list, tuple)):
                raise TypeError(
                    f"Dataset '{name}' did not return a list of dicts; the audit "
                    "cannot enumerate it. Use get_detection_dataset_dicts first."
                )
            for item in dicts:
                fn = _resolve_file_name(item)
                if fn is None:
                    continue
                key = os.path.abspath(os.fspath(fn)) if isinstance(fn, str) else os.fspath(fn)
                if key in seen:
                    continue
                seen[key] = split
                samples.append(AuditSample(key=key, file_name=key, split=split, dataset=name))
    samples.sort(key=lambda s: s.key)
    return samples


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _FeatureCache:
    """Bounded LRU of canonical images and their features.

    Memory is bounded by ``cfg.verify_cache_size`` images regardless of how
    many images the dataset contains: pixel arrays and descriptors are *not*
    retained for the whole run.  The compact candidate index keeps only a
    handful of 64-bit codes per image, so total memory grows with the image
    count only through that small summary, not through image data.

    Failed loads are remembered as errors (small) so a corrupt file is not
    re-read for every pair it appears in.
    """

    def __init__(self, cfg: AuditConfig):
        self.cfg = cfg
        self._capacity = max(2, int(cfg.verify_cache_size))
        self._entries: "OrderedDict[str, Dict]" = OrderedDict()
        self._errors: Dict[str, SkippedImage] = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> Tuple[Optional[LoadedImage], Optional[Features], Optional[SkippedImage]]:
        with self._lock:
            err = self._errors.get(path)
            if err is not None:
                return None, None, err
            entry = self._entries.get(path)
            if entry is not None:
                self._entries.move_to_end(path)
                return entry["image"], entry["features"], None
        # Load outside the lock; duplicate concurrent loads are harmless because
        # extraction is deterministic.
        try:
            image = load_canonical_image(path, self.cfg)
            features = extract_features(image.gray, self.cfg)
        except SkippedImage as e:
            with self._lock:
                self._errors[path] = e
            return None, None, e
        self.put(path, image, features)
        return image, features, None

    def put(self, path: str, image: LoadedImage, features: Features) -> None:
        with self._lock:
            self._entries[path] = {"image": image, "features": features}
            self._entries.move_to_end(path)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def drop(self) -> None:
        with self._lock:
            self._entries.clear()


class _PairEngine:
    """Verifies pairs with a small deterministic cache."""

    def __init__(self, cache: _FeatureCache, cfg: AuditConfig):
        self.cache = cache
        self.cfg = cfg

    def measure(self, a: str, b: str, skips: Dict[str, SkippedImage]):
        ia, fa, ea = self.cache.get(a)
        ib, fb, eb = self.cache.get(b)
        if ea is not None or eb is not None:
            for p, e in ((a, ea), (b, eb)):
                if e is not None:
                    skips[p] = e
            return None
        if fa.count == 0 or fb.count == 0:
            return None
        return score_pair(fa, fb, ia.gray, ib.gray, self.cfg)


def _feature_digest(features: Features) -> str:
    """Content digest of the extracted features (not of the file bytes).

    Two files with identical content but different encodings produce the same
    digest; this is deliberately *not* a checksum-based judgement, it is only a
    cache key recorded for diagnostics.
    """
    if features.count == 0:
        return "empty"
    h = hashlib.sha256()
    h.update(features.descriptors.astype(np.float32).tobytes())
    h.update(str(features.shape).encode("utf-8"))
    return h.hexdigest()


def _sample_digest(samples: Sequence[AuditSample]) -> str:
    h = hashlib.sha256()
    for s in sorted(samples, key=lambda x: x.key):
        h.update(s.key.encode("utf-8"))
        h.update(b"\0")
        h.update(s.split.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def run_audit(
    split_datasets: Mapping[str, Sequence[str]],
    config: AuditConfig,
    *,
    report_dir: Optional[str] = None,
    samples: Optional[Sequence[AuditSample]] = None,
    store: Optional[AuditStore] = None,
    batch_id: Optional[str] = None,
) -> AuditRun:
    """Audit a batch of registered datasets.

    Args:
        split_datasets: ``{"train": ["ds_train"], "val": ["ds_val"], ...}``.
            The split label is the one recorded for every sample from that
            dataset; it is never derived from the file name.
        config: audit configuration (must be enabled).
        report_dir: where to write JSON artifacts (``reports/`` by default).
        samples: pre-collected samples (bypasses the catalog); used by tests.
        store: reuse / update an existing history store. When given, only new
            samples are scanned and missing history evidence is recomputed
            lazily for cross-batch pairs.
        batch_id: label for this batch in the store.

    Returns:
        :class:`AuditRun`.  Artifacts are already on disk.
    """
    if not config.enabled:
        raise ValueError("refusing to run the audit with config.enabled=False")
    cfg = config
    if cfg.num_workers > 1:
        import cv2

        cv2.setNumThreads(1)

    report_dir = report_dir or cfg.store_dir or "reports"
    store_dir = cfg.store_dir or report_dir
    if store is None:
        store = AuditStore.load(store_dir)
    batch = batch_id or cfg.batch_id

    sample_list = list(samples) if samples is not None else collect_samples(split_datasets)
    if not sample_list:
        raise ValueError("no samples to audit")
    sample_keys = {s.key for s in sample_list}
    split_of = {s.key: s.split for s in sample_list}

    # Historical images are only referenced; their features are recomputed on
    # demand for cross-batch candidate pairs.
    historical_images = {p for p in store.images if p not in sample_keys} if store.images else set()
    historical_splits = {p: store.images[p].get("split", "<unknown>") for p in historical_images}

    # Incremental mode: only images not already in the store are re-read.  A
    # key is considered known when its split matches, because a sample moved
    # between splits must be re-examined for leakage.  Historical evidence is
    # reused from the store, never recomputed with different inputs.
    new_samples = [s for s in sample_list if not store.known_image(s.key, s.split)]
    # Incremental whenever a previous store exists: history is never re-read,
    # only the batch's own new images are.
    incremental = bool(store.images)
    scan_samples = new_samples

    # ---- load and extract ------------------------------------------------
    cache = _FeatureCache(cfg)
    skips: Dict[str, SkippedImage] = {}
    index = CandidateIndex(cfg)
    # Only compact per-image metadata is retained for the whole batch; the
    # pixel arrays and descriptors stay inside the bounded LRU cache.
    image_meta: Dict[str, Dict] = {}
    digests: Dict[str, str] = {}
    skip_lock = threading.Lock()

    def process(sample: AuditSample):
        image, feats, err = cache.get(sample.key)
        if err is not None:
            with skip_lock:
                skips[sample.key] = err
            return None
        if feats.count == 0:
            # A blurry or flat image yields no usable keypoints; it cannot take
            # part in pairwise comparison and is reported, not fatal.
            with skip_lock:
                skips[sample.key] = SkippedImage("no_features", "no SIFT keypoints")
            return None
        index.add(sample.key, feats)
        return sample.key, image, feats

    if cfg.num_workers > 1:
        order = sorted(scan_samples, key=lambda s: s.key)
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as ex:
            futures = [ex.submit(process, s) for s in order]
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    key, image, feats = result
                    image_meta[key] = image.to_dict()
                    digests[key] = _feature_digest(feats)
    else:
        for s in sorted(scan_samples, key=lambda x: x.key):
            result = process(s)
            if result is not None:
                key, image, feats = result
                image_meta[key] = image.to_dict()
                digests[key] = _feature_digest(feats)

    logger.info(
        "audit: loaded %d/%d images (%d skipped)",
        len(image_meta),
        len(scan_samples),
        len(skips),
    )

    # ---- candidate pairs -------------------------------------------------
    proposals = index.propose()  # only among the batch's own images
    pair_candidates: Dict[Tuple[str, str], Dict[str, int]] = dict(proposals)

    # Cross-batch candidate generation: index the historical codes too, but
    # without loading their images.  Re-extract features lazily for the pairs
    # that survive.
    if store.images:
        pair_candidates = _add_history_candidates(
            pair_candidates, store, index, sample_keys, cfg
        )

    if len(pair_candidates) > cfg.max_total_pairs:
        # Deterministic truncation with an explicit warning.
        keep = sorted(pair_candidates, key=lambda k: (-_candidate_rank(pair_candidates[k]), k))
        pair_candidates = {k: pair_candidates[k] for k in keep[: cfg.max_total_pairs]}
        logger.warning(
            "audit: candidate pairs %d exceeded max_total_pairs; truncated deterministically",
            len(pair_candidates),
        )

    # Historical evidence is reused from the store when both endpoints are
    # known and unchanged; otherwise the pair is re-measured (historical images
    # are loaded lazily here).
    engine = _PairEngine(cache, cfg)
    evidence: Dict[Tuple[str, str], PairEvidence] = {}
    reused = 0
    pair_order = sorted(pair_candidates.keys())
    known_scan = set(image_meta)
    for a, b in pair_order:
        cached_rec = store.pair(a, b)
        if (
            incremental
            and cached_rec is not None
            and (a in known_scan or store.known_image(a))
            and (b in known_scan or store.known_image(b))
        ):
            ev = _evidence_from_record(cached_rec)
            if ev is not None:
                evidence[(a, b)] = ev
                reused += 1
                continue
        ev = engine.measure(a, b, skips)
        if ev is None:
            continue
        evidence[(a, b)] = ev

    # An incremental run reports the *whole* accumulated corpus, not just this
    # batch: otherwise re-running a known batch would silently drop the
    # historical pairs and produce an empty report.  Stored evidence for
    # endpoints that are both historical is merged in (new batches are never
    # re-measured here, they were just measured above).
    if incremental:
        for pair_key, cached_rec in store.pairs.items():
            a, b = pair_key.split("\x00", 1)
            if (a, b) in evidence:
                continue
            if store.known_image(a) and store.known_image(b):
                ev = _evidence_from_record(cached_rec)
                if ev is not None:
                    evidence[(a, b)] = ev
                    reused += 1
    logger.info("audit: reused %d stored pair measurements", reused)

    # ---- edges, clusters, sweep -----------------------------------------
    def edges_at(threshold: float) -> List[Edge]:
        out = []
        for (a, b), ev in evidence.items():
            cls = classify_evidence(ev, cfg, threshold)
            if cls.kind == "none":
                continue
            region = cls.region
            out.append(
                Edge(
                    a=a,
                    b=b,
                    score=cls.score,
                    kind=cls.kind,
                    borderline=cls.borderline,
                    region=region,
                    reasons=tuple(cls.reasons),
                    region_swapped=bool(cls.region_swapped),
                    region_cells=cls.region_cells,
                )
            )
        out.sort(key=lambda e: (e.a, e.b))
        return out

    measured_keys = set()
    for a, b in evidence:
        measured_keys.add(a)
        measured_keys.add(b)
    all_measured = sorted(measured_keys)

    edges = edges_at(cfg.dup_threshold)
    clusters = build_clusters(edges, cfg.dup_threshold, cfg.min_cluster_size, all_samples=all_measured)

    sweep = []
    for t in sorted(set(cfg.sweep_thresholds) | {float(cfg.dup_threshold)}):
        e_t = edges_at(t)
        c_t = build_clusters(e_t, t, cfg.min_cluster_size, all_samples=all_measured)
        rec_t, aff_t = leakage_from_clusters(c_t, _split_lookup(split_of, historical_splits, c_t))
        sweep.append(
            {
                "threshold": round(float(t), 5),
                "clusters": len(c_t),
                "duplicate_samples": len(aff_t),
                "leakage_clusters": len(rec_t),
                "cross_split_leak_samples": len(aff_t),
                "cross_split_leak_rate": round(len(aff_t) / max(1, len(all_measured)), 6),
                "borderline_clusters": sum(1 for c in c_t if c.borderline),
            }
        )

    # ---- leakage & statistics -------------------------------------------
    split_lookup = _split_lookup(split_of, historical_splits, clusters)
    leak_records, leak_affected = leakage_from_clusters(clusters, split_lookup)
    duplicated = sorted({m for c in clusters for m in c.members})
    # Samples of a leaking cluster that belong to a split also present in the
    # cluster are the ones a user must act on; every listed member of a leaking
    # cluster is by construction in a split that also appears elsewhere.
    split_sizes = defaultdict(int)
    for k, s in split_lookup.items():
        split_sizes[s] += 1
    stats = dataset_statistics(
        samples=len(split_lookup),
        skipped=len(skips),
        clusters=clusters,
        leak_records=leak_records,
        duplicate_sample_count=len(duplicated),
        split_sizes=split_sizes,
        cross_split_sample_count=len(leak_affected),
    )
    stats["duplicate_sample_keys"] = duplicated
    stats["per_edge_kind"] = {
        "global": sum(1 for e in edges if e.kind == "global"),
        "local": sum(1 for e in edges if e.kind == "local"),
        "borderline_edges": sum(1 for e in edges if e.borderline),
    }
    stats["candidate_pairs"] = len(pair_candidates)
    stats["verified_pairs"] = len(evidence)
    stats["reused_pair_measurements"] = reused
    stats["incremental"] = bool(incremental)
    stats["images_scanned_in_this_run"] = len(image_meta)
    stats["batch_id"] = batch

    # ---- history reconciliation -----------------------------------------
    historical_clusters = [list(c) for c in store.clusters]
    reconciled, recon = reconcile_with_history(
        historical_clusters=historical_clusters,
        new_clusters=[list(c.members) for c in clusters],
        edges=edges,
        policy=cfg.history_policy,
        auto_extend=cfg.auto_extend_history,
    )
    history_record = recon.to_dict()
    history_record["batch"] = batch
    history_record["reconciled_clusters"] = [sorted(c) for c in reconciled]

    # ---- build artifacts -------------------------------------------------
    paths = report_dir_layout(report_dir)
    skipped_records = [
        {"path": p, "reason": e.reason, "detail": e.detail}
        for p, e in sorted(skips.items())
    ]
    images_by_key = image_meta
    pairs_payload = _pair_payload(evidence, cfg, skipped_records)

    cluster_payload = _clusters_payload(clusters, split_lookup, images_by_key)
    leaks_payload = {
        "generated_by": ALGORITHM_VERSION,
        "batch": batch,
        "rule": "a cluster whose members belong to more than one split, using the "
        "split recorded at registration",
        "leakage_clusters": [r.to_dict() for r in leak_records],
        "affected_samples": sorted(leak_affected),
    }
    manifest = build_annotation_manifest(clusters, split_lookup, batch)
    summary = build_summary(batch, stats=stats, clusters=clusters, leak_records=leak_records)
    overview = build_overview(batch, stats=stats, clusters=clusters, config=cfg, summary=summary)

    # ---- persist (atomic) -----------------------------------------------
    os.makedirs(report_dir, exist_ok=True)
    atomic_write_json(paths["pairs"], pairs_payload)
    atomic_write_json(paths["clusters"], cluster_payload)
    atomic_write_json(paths["leaks"], leaks_payload)
    atomic_write_json(paths["stats"], stats)
    atomic_write_json(paths["sweep"], {"batch": batch, "thresholds": sweep})
    atomic_write_json(paths["skipped"], {"batch": batch, "skipped": skipped_records})
    atomic_write_json(paths["history"], history_record)
    if cfg.export_annotation_manifest:
        atomic_write_json(paths["annotation_manifest"], manifest)
    atomic_write_json(paths["summary"], summary)
    if cfg.write_loader_overview:
        atomic_write_json(paths["overview"], overview)

    # Store update happens last: a crash before this line leaves the previous
    # complete store in place, and the report files are each atomic.
    store.add_batch(batch)
    for p in sorted(image_meta):
        store.add_image(
            p,
            split_of.get(p, historical_splits.get(p, "<unknown>")),
            batch,
            digest=digests.get(p, "unknown"),
            size=int(os.path.getsize(p)) if os.path.exists(p) else 0,
            codes=index.rare_codes(p),
        )
    for (a, b), ev in evidence.items():
        store.add_pair(a, b, _evidence_record(ev, cfg))
    store.algorithm = ALGORITHM_VERSION
    store.set_clusters(reconciled)
    store.save(store_dir)

    cache.drop()
    return AuditRun(
        config=cfg,
        clusters=clusters,
        leak_records=leak_records,
        stats=stats,
        sweep=sweep,
        skipped=skipped_records,
        summary=summary,
        overview=overview,
        history=history_record,
        store=store,
        report_dir=report_dir,
        paths=paths,
    )


def _evidence_from_record(rec: Dict) -> Optional[PairEvidence]:
    """Reconstruct pair measurements stored by a previous batch.

    Only the raw measurements are restored; classification is always re-derived
    from the current configuration, so changing a threshold never reuses a
    stale decision.
    """
    try:
        region = rec.get("region")
        return PairEvidence(
            score=float(rec.get("score", 0.0)),
            inliers=int(rec.get("inliers", 0)),
            matches=int(rec.get("matches", 0)),
            coverage=float(rec.get("coverage", 0.0)),
            local_coverage=float(rec.get("local_coverage", 0.0)),
            local_inliers=int(rec.get("local_inliers", 0)),
            region=tuple(int(v) for v in region) if region else None,
            scale=float(rec.get("scale", 1.0)),
            rotation_deg=float(rec.get("rotation_deg", 0.0)),
            ncc=float(rec.get("ncc", 0.0)),
            ncc_overlap=float(rec.get("ncc_overlap", 0.0)),
            local_ncc=float(rec.get("local_ncc", 0.0)),
            region_area_ref=float(rec.get("region_area_ref", 0.0)),
            kind=str(rec.get("kind", "none")),
            reasons=list(rec.get("reasons", [])),
            borderline=bool(rec.get("borderline", False)),
            match_ratio=float(rec.get("match_ratio", 0.0)),
            region_swapped=bool(rec.get("region_swapped", False)),
            region_cells=(
                tuple(tuple(int(v) for v in r) for r in rec["region_cells"])
                if rec.get("region_cells")
                else None
            ),
        )
    except (TypeError, ValueError):
        return None


def _candidate_rank(ev: Dict[str, int]) -> int:
    return int(ev.get("rare", 0)) + 2 * int(ev.get("band", 0))


def _split_lookup(
    batch_split: Mapping[str, str],
    historical_split: Mapping[str, str],
    clusters: Sequence,
) -> Dict[str, str]:
    lookup = dict(historical_split)
    lookup.update(batch_split)
    for c in clusters:
        for m in c.members:
            lookup.setdefault(m, "<unknown>")
    return lookup


def _add_history_candidates(
    pair_candidates: Dict[Tuple[str, str], Dict[str, int]],
    store: AuditStore,
    index: CandidateIndex,
    sample_keys: set,
    cfg: AuditConfig,
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Propose (new, historical) pairs using the historical code signatures.

    Only keys that are not already in the batch index are considered, so the
    batch is never compared with itself again.  Historical images are loaded
    lazily during verification.
    """
    new_keys = set(index.ids)
    hist_keys = sorted(set(store.images) - new_keys)
    if not hist_keys:
        return pair_candidates
    # Reconstruct per-image code sets from the store's compact digest.
    code_sets: Dict[str, set] = {}
    for p in sorted(store.images):
        rec = store.images[p]
        codes = rec.get("codes")
        if codes:
            code_sets[p] = {int(c, 16) if isinstance(c, str) else int(c) for c in codes}
    if not code_sets:
        return pair_candidates
    df = defaultdict(int)
    for p, codes in code_sets.items():
        for c in codes:
            df[c] += 1
    index_by_code = defaultdict(list)
    for p, codes in code_sets.items():
        for c in codes:
            if df[c] <= cfg.rare_code_max_df:
                index_by_code[c].append(p)
    added: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"rare": 0, "band": 0})
    # Batch images' rare codes were already computed by CandidateIndex; use its
    # internal code arrays for symmetric lookup.
    for new_key in sorted(new_keys):
        codes = np.unique(index._codes[new_key]) if new_key in index._codes else []
        for c in codes:
            for h in index_by_code.get(int(c), ()):
                added[tuple(sorted((new_key, h)))]["rare"] += 1
    out: Dict[Tuple[str, str], Dict[str, int]] = dict(pair_candidates)
    for pair, ev in added.items():
        if ev["rare"] >= cfg.min_shared_rare:
            out[pair] = ev
    return out


def _safe_round(x: float, n: int = 5) -> float:
    return round(float(x), n)


def _evidence_record(ev: PairEvidence, cfg: AuditConfig) -> Dict:
    rec = ev.to_dict()
    rec["dup_threshold"] = _safe_round(cfg.dup_threshold)
    rec["kind_at_default"] = classify_evidence(ev, cfg).kind
    return rec


def _pair_payload(
    evidence: Dict[Tuple[str, str], PairEvidence], cfg: AuditConfig, skipped: List[Dict]
) -> Dict:
    items = []
    for (a, b), ev in evidence.items():
        rec = _evidence_record(ev, cfg)
        rec["a"] = a
        rec["b"] = b
        items.append(rec)
    # Deterministic and stable regardless of key ordering.
    items.sort(key=lambda r: (min(r["a"], r["b"]), max(r["a"], r["b"])))
    return {"batch": cfg.batch_id, "pairs": items, "skipped": skipped}


def _clusters_payload(clusters, split_lookup, images_by_key) -> Dict:
    """Serialize clusters, mapping local regions back to original pixels.

    Each local edge's ``region`` is in the canonical frame of the *first*
    member (lexicographically).  ``regions_original`` converts it to file
    coordinates for that member so an operator can locate the repeated area in
    the actual image.
    """
    out = []
    for c in clusters:
        item = {**c.to_dict(), "splits": sorted({split_lookup.get(m, "<unknown>") for m in c.members})}
        members = []
        for m in c.members:
            info = dict(images_by_key.get(m, {}))
            info["key"] = m
            info["split"] = split_lookup.get(m, "<unknown>")
            members.append(info)
        item["members_detail"] = members
        regions = []
        for e in c.edges:
            if e.region is None:
                continue
            # The region lives in the canonical frame of the image the
            # measurement was anchored on; honour that when mapping back.
            anchor_key = e.b if e.region_swapped else e.a
            anchor = images_by_key.get(anchor_key, {})
            scale = float(anchor.get("scale", 1.0)) or 1.0
            ow = int(anchor.get("orig_width", 0))
            oh = int(anchor.get("orig_height", 0))

            def _to_original(box):
                mapped = [int(round(v / scale)) for v in box]
                if ow > 0 and oh > 0:
                    mapped = [
                        max(0, min(mapped[0], ow)),
                        max(0, min(mapped[1], oh)),
                        max(0, min(mapped[2], ow)),
                        max(0, min(mapped[3], oh)),
                    ]
                return mapped

            orig = _to_original(e.region)
            cells = None
            if e.region_cells:
                cells = [_to_original(r) for r in e.region_cells]
            regions.append(
                {
                    "pair": [e.a, e.b],
                    "region_canonical_xyxy": list(e.region),
                    "region_original_xyxy": orig,
                    "region_cells_canonical": [list(r) for r in e.region_cells] if e.region_cells else None,
                    "region_cells_original": cells,
                    "anchor": anchor_key,
                    "anchor_scale": scale,
                }
            )
        if regions:
            item["local_regions"] = regions
        out.append(item)
    return {"clusters": out}

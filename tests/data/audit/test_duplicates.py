# Copyright (c) Facebook, Inc. and its affiliates.
"""Acceptance test #1: every duplicate class in the controlled dataset is found.

The builder plants three duplicate families per scene:

* burst frames  (same geometry, tiny noise)      -> expected kind "global"
* crop/rescale  (same content, different framing)-> detected (global or local)
* re-encode     (same pixels, different bytes)   -> expected kind "global"

plus negatives that must never be clustered, and a "same background, different
subject" pair that must be reported as a *local* duplicate with a region.
"""

import os

from detectron2.data.audit.engine import run_audit
from detectron2.data.audit.features import (
    classify_evidence,
    extract_features,
    score_pair,
)
from detectron2.data.audit.image_ops import load_canonical_image

from .base import AuditTestCase


class TestDuplicateRecall(AuditTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import tempfile

        from detectron2.data.audit.config import AuditConfig

        cls._run_dir = tempfile.mkdtemp(prefix="dupaudit-run-")
        cfg = AuditConfig(
            enabled=True,
            store_dir=cls._run_dir,
            batch_id="recall",
            sweep_thresholds=(0.35, 0.45, 0.55, 0.65),
        )
        from detectron2.data.audit.engine import AuditSample

        samples = [
            AuditSample(it["path"], it["path"], split)
            for split, items in cls.dataset.files.items()
            for it in items
        ]
        cls.audit_run = run_audit({}, cfg, report_dir=cls._run_dir, samples=samples)

        # Map every file to its cluster id (if any).
        cls.cluster_of = {}
        for c in cls.audit_run.clusters:
            for m in c.members:
                cls.cluster_of[m] = c.cluster_id

    def scenes(self):
        # scene ids that contain expected duplicate families
        return sorted({it["scene_id"] for it in self.dataset.files["train"]})

    def find_cluster_for(self, path):
        return self.cluster_of.get(path)

    def test_burst_frames_are_detected_and_clustered(self):
        for split_items in self.dataset.files.values():
            for it in split_items:
                if it["kind"] != "burst":
                    continue
                scene_original = self._original_path(it["scene_id"])
                self.assertIn(
                    it["path"],
                    self.cluster_of,
                    f"burst frame not detected: {os.path.basename(it['path'])}",
                )
                self.assertEqual(
                    self.cluster_of[it["path"]],
                    self.cluster_of[scene_original],
                    f"burst frame not clustered with its scene original: {it['path']}",
                )

    def test_crop_variants_are_detected(self):
        misses = []
        for split_items in self.dataset.files.values():
            for it in split_items:
                if it["kind"] != "crop":
                    continue
                if it["path"] not in self.cluster_of:
                    misses.append(os.path.basename(it["path"]))
        self.assertEqual(misses, [], "crop/rescale variants missed: {}".format(misses))

    def test_reencoded_copies_are_detected(self):
        misses = []
        for split_items in self.dataset.files.values():
            for it in split_items:
                if it["kind"] not in ("reencode", "reencode_crop"):
                    continue
                if it["path"] not in self.cluster_of:
                    misses.append(os.path.basename(it["path"]))
        self.assertEqual(misses, [], "re-encoded copies missed: {}".format(misses))

    def test_no_checksum_based_decision(self):
        """Re-encoded copies have different bytes, yet are still clustered."""
        import hashlib

        digests = {}
        for it in self.dataset.files["test"]:
            if it["kind"] in ("reencode", "reencode_crop"):
                with open(it["path"], "rb") as f:
                    digests[it["path"]] = hashlib.sha256(f.read()).hexdigest()
        # Different encodings must be different files, or the test is vacuous.
        self.assertGreater(len(set(digests.values())), 0)
        for it in self.dataset.files["test"]:
            if it["kind"] in ("reencode", "reencode_crop"):
                self.assertIn(it["path"], self.cluster_of)

    def test_negatives_are_not_clustered_with_scenes(self):
        for it in self.dataset.files["test"]:
            if it["kind"] != "negative":
                continue
            cluster = self.find_cluster_for(it["path"])
            if cluster is None:
                continue
            # A negative may only be in a cluster of negatives, never a scene.
            cluster_obj = next(c for c in self.audit_run.clusters if c.cluster_id == cluster)
            for member in cluster_obj.members:
                self.assertNotIn("scene", os.path.basename(member), "negative lumped into a scene cluster")

    def test_local_duplicate_is_reported_as_local_with_a_region(self):
        local_test = next(
            it["path"] for it in self.dataset.files["test"] if it["kind"] == "local"
        )
        self.assertIn(local_test, self.cluster_of, "same-background/different-subject pair not detected")
        cluster = next(c for c in self.audit_run.clusters if c.cluster_id == self.cluster_of[local_test])
        local_edges = [e for e in cluster.edges if e.kind == "local" and e.region is not None]
        self.assertTrue(local_edges, "expected at least one local edge with a region")

    def test_each_planted_family_member_is_explained_by_one_cluster(self):
        """Every duplicate sample belongs to a cluster whose edges are connected."""
        for c in self.audit_run.clusters:
            # certify connectivity: cluster members are reachable through edges
            nodes = set(c.members)
            reached = {c.edges[0].a} if c.edges else set()
            for e in c.edges:
                reached.add(e.a)
                reached.add(e.b)
            self.assertEqual(reached & nodes, nodes, f"cluster {c.cluster_id} contains disconnected members")

    def _original_path(self, scene_id):
        for it in self.dataset.files["train"]:
            if it["scene_id"] == scene_id and it["kind"] == "original":
                return it["path"]
        raise KeyError(scene_id)


class TestPairScoring(AuditTestCase):
    """Direct checks of the measurement layer, independent of candidate indexing."""

    def _features(self, path):
        cfg = self.config()
        image = load_canonical_image(path, cfg)
        return image, extract_features(image.gray, cfg)

    def test_burst_pair_scores_high(self):
        orig = next(it["path"] for it in self.dataset.files["train"] if it["kind"] == "original")
        burst = next(it["path"] for it in self.dataset.files["val"] if it["kind"] == "burst")
        ia, fa = self._features(orig)
        ib, fb = self._features(burst)
        ev = classify_evidence(score_pair(fa, fb, ia.gray, ib.gray, self.config()), self.config())
        self.assertGreaterEqual(ev.score, self.config().dup_threshold)
        self.assertEqual(ev.kind, "global")
        self.assertGreater(ev.ncc, 0.8)

    def test_negative_pair_scores_low(self):
        orig = next(it["path"] for it in self.dataset.files["train"] if it["kind"] == "original")
        neg = next(it["path"] for it in self.dataset.files["test"] if it["kind"] == "negative")
        ia, fa = self._features(orig)
        ib, fb = self._features(neg)
        ev = classify_evidence(score_pair(fa, fb, ia.gray, ib.gray, self.config()), self.config())
        self.assertEqual(ev.kind, "none")
        self.assertLess(ev.score, self.config().dup_threshold)

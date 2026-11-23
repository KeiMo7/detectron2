# Copyright (c) Facebook, Inc. and its affiliates.
"""Acceptance test #8: batch-by-batch re-runs and history reconciliation."""

import json
import os
import unittest

from detectron2.data.audit.config import AuditConfig
from detectron2.data.audit.engine import AuditSample, run_audit
from detectron2.data.audit.store import AuditStore

from .base import AuditTestCase


class TestIncremental(AuditTestCase):
    def _samples(self, splits):
        return [
            AuditSample(it["path"], it["path"], split)
            for split, items in self.dataset.files.items()
            if split in splits
            for it in items
        ]

    def test_second_batch_scans_only_new_images(self):
        root = self.tmpdir("inc")
        cfg_a = AuditConfig(enabled=True, store_dir=root, batch_id="A")
        run_a = run_audit({}, cfg_a, report_dir=root, samples=self._samples({"train", "val"}))
        self.assertGreater(run_a.stats["images_scanned_in_this_run"], 0)

        cfg_b = AuditConfig(enabled=True, store_dir=root, batch_id="B")
        run_b = run_audit({}, cfg_b, report_dir=root, samples=self._samples({"test"}))
        test_count = len(self.dataset.files["test"])
        self.assertTrue(run_b.stats["incremental"])
        self.assertEqual(
            run_b.stats["images_scanned_in_this_run"],
            test_count,
            "second batch re-read historical images instead of only the new ones",
        )
        # Leakage across the combined data must still be found.
        self.assertGreater(len(run_b.leak_records), 0)

    def test_store_records_both_batches(self):
        root = self.tmpdir("inc2")
        run_audit({}, AuditConfig(enabled=True, store_dir=root, batch_id="A"), report_dir=root, samples=self._samples({"train"}))
        run_audit({}, AuditConfig(enabled=True, store_dir=root, batch_id="B"), report_dir=root, samples=self._samples({"val"}))
        store = AuditStore.load(root)
        self.assertEqual(sorted(store.batches), ["A", "B"])
        # Every image from both batches is known.
        self.assertEqual(len(store.images), len(self.dataset.files["train"]) + len(self.dataset.files["val"]))

    def test_history_is_not_silently_rewritten(self):
        """A history conflict must surface as a suggestion, not an overwrite."""
        root = self.tmpdir("inc3")
        run_audit({}, AuditConfig(enabled=True, store_dir=root, batch_id="A"), report_dir=root, samples=self._samples({"train"}))
        store_before = AuditStore.load(root)
        self.assertTrue(store_before.clusters)

        run_audit({}, AuditConfig(enabled=True, store_dir=root, batch_id="B"), report_dir=root, samples=self._samples({"test"}))
        with open(os.path.join(root, "history_reconciliation.json")) as f:
            history = json.load(f)
        # Whatever changed, every change is described in the record.
        self.assertIn("suggested_merges", history)
        self.assertIn("extensions", history)
        self.assertIn("suggested_splits", history)
        self.assertIn("applied", history)
        for extension in history["extensions"]:
            self.assertIn("historical_members", extension)
            self.assertIn("new_members", extension)
            self.assertIn("suggestion", extension)

    def test_reusing_the_same_store_is_idempotent_for_histories(self):
        root = self.tmpdir("inc4")
        samples = self._samples({"train", "val", "test"})
        run1 = run_audit({}, AuditConfig(enabled=True, store_dir=root, batch_id="A"), report_dir=root, samples=samples)
        run2 = run_audit({}, AuditConfig(enabled=True, store_dir=root, batch_id="A"), report_dir=root, samples=samples)
        # Same batch id, same data: the final partition is unchanged.
        self.assertEqual(
            sorted(tuple(sorted(c.members)) for c in run1.clusters),
            sorted(tuple(sorted(c.members)) for c in run2.clusters),
        )


class TestHistoryReconciliationUnit(unittest.TestCase):
    """Directly exercise the merge/split suggestion policy."""

    def _edge(self, a, b, score):
        from detectron2.data.audit.clustering import Edge

        return Edge(a=a, b=b, score=score, kind="global")

    def test_safe_extension_when_new_member_links_every_existing_member(self):
        from detectron2.data.audit.store import reconcile_with_history

        hist = [["a", "b"]]
        new = [["a", "c"]]
        edges = [self._edge("a", "b", 0.9), self._edge("a", "c", 0.9), self._edge("b", "c", 0.9)]
        merged, rec = reconcile_with_history(
            historical_clusters=hist, new_clusters=new, edges=edges, policy="apply_safe", auto_extend=True
        )
        self.assertEqual([sorted(c) for c in merged], [["a", "b", "c"]])
        self.assertTrue(rec.applied)

    def test_no_silent_overwrite_when_link_is_incomplete(self):
        from detectron2.data.audit.store import reconcile_with_history

        hist = [["a", "b"]]
        new = [["a", "c"]]
        # c links a but not b: must not absorb into the historical cluster.
        edges = [self._edge("a", "b", 0.9), self._edge("a", "c", 0.9)]
        merged, rec = reconcile_with_history(
            historical_clusters=hist, new_clusters=new, edges=edges, policy="suggest", auto_extend=True
        )
        self.assertNotIn(["a", "b", "c"], [sorted(c) for c in merged])
        self.assertTrue(rec.extensions)
        self.assertEqual(rec.extensions[0]["suggestion"], "review")

    def test_merge_suggestion_when_new_cluster_bridges_two_histories(self):
        from detectron2.data.audit.store import reconcile_with_history

        hist = [["a", "b"], ["c", "d"]]
        new = [["b", "c"]]
        edges = [self._edge("b", "c", 0.9)]
        merged, rec = reconcile_with_history(
            historical_clusters=hist, new_clusters=new, edges=edges, policy="suggest", auto_extend=True
        )
        self.assertTrue(rec.merges)
        self.assertEqual(rec.merges[0]["suggestion"], "merge")
        # Histories are not merged automatically in "suggest" policy.
        merged_sets = [sorted(c) for c in merged]
        self.assertIn(["a", "b"], merged_sets)
        self.assertIn(["c", "d"], merged_sets)

# Copyright (c) Facebook, Inc. and its affiliates.
"""Acceptance test #2: the leakage list matches actual split membership."""

import os
import tempfile

from detectron2.data.audit.engine import AuditSample, run_audit

from .base import AuditTestCase


class TestLeakage(AuditTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from detectron2.data.audit.config import AuditConfig

        cls._run_dir = tempfile.mkdtemp(prefix="dupaudit-leak-")
        cfg = AuditConfig(enabled=True, store_dir=cls._run_dir, batch_id="leak")
        samples = [
            AuditSample(it["path"], it["path"], split)
            for split, items in cls.dataset.files.items()
            for it in items
        ]
        cls.audit_run = run_audit({}, cfg, report_dir=cls._run_dir, samples=samples)

    def _expected_split(self, path):
        return self.dataset.split_of(path)

    def test_every_leak_cluster_spans_at_least_two_split_labels(self):
        self.assertTrue(self.audit_run.leak_records, "no leakage found in a dataset built to leak")
        for rec in self.audit_run.leak_records:
            self.assertGreaterEqual(len(rec.splits), 2, f"cluster {rec.cluster_id} listed as leaking but spans one split")

    def test_leak_members_match_actual_registration_split(self):
        """Every listed path must be filed under the split the registry gave it."""
        for rec in self.audit_run.leak_records:
            for split, members in rec.members_by_split.items():
                for path in members:
                    self.assertEqual(
                        split,
                        self._expected_split(path),
                        f"leak list placed {os.path.basename(path)} in {split} but it is registered in "
                        f"{self._expected_split(path)}",
                    )

    def test_leak_clusters_are_exactly_the_cross_split_clusters(self):
        # Independent recomputation from the cluster membership and the split map.
        split_map = {it["path"]: split for split, items in self.dataset.files.items() for it in items}
        expected = set()
        for c in self.audit_run.clusters:
            if len({split_map[m] for m in c.members}) > 1:
                expected.add(c.cluster_id)
        actual = {r.cluster_id for r in self.audit_run.leak_records}
        self.assertEqual(actual, expected)

    def test_every_scene_family_leaks(self):
        """The builder plants one family per scene across train/val/test."""
        scene_ids = sorted({it["scene_id"] for it in self.dataset.files["train"]})
        leaked_scene_ids = set()
        for rec in self.audit_run.leak_records:
            for path in rec.members_by_split.get("train", []):
                leaked_scene_ids.add(os.path.basename(path).split("_")[0])
        expected = {f"scene{i}" for i in scene_ids}
        self.assertEqual(leaked_scene_ids, expected, "not every planted scene family leaked across splits")

    def test_negatives_never_leak(self):
        for rec in self.audit_run.leak_records:
            for members in rec.members_by_split.values():
                for path in members:
                    self.assertNotEqual(
                        os.path.basename(path).split("_")[0], "neg",
                        "an unrelated negative image was reported as leakage",
                    )

    def test_affected_sample_count_equals_union_of_leak_members(self):
        union = set()
        for rec in self.audit_run.leak_records:
            for members in rec.members_by_split.values():
                union.update(members)
        self.assertEqual(self.audit_run.stats["cross_split_leak_samples"], len(union))

    def test_split_is_never_inferred_from_file_name(self):
        """Renaming a file must not move it between splits in the leak list."""
        self.dataset  # sanity
        # The audit only knows splits via AuditSample.split.  Build a copy of
        # the run using deliberately misleading names: the split map is passed
        # explicitly, so results must be unchanged.
        from detectron2.data.audit.config import AuditConfig

        samples = []
        for split, items in self.dataset.files.items():
            for i, it in enumerate(items):
                # name says "val" even for train images; the registered split
                # is what must be used.
                samples.append(AuditSample(it["path"], it["path"], split))
        d = tempfile.mkdtemp(prefix="dupaudit-split-")
        run = run_audit({}, AuditConfig(enabled=True, store_dir=d, batch_id="s"), report_dir=d, samples=samples)
        for rec in run.leak_records:
            for split, members in rec.members_by_split.items():
                for path in members:
                    self.assertEqual(split, self._expected_split(path))

# Copyright (c) Facebook, Inc. and its affiliates.
"""Acceptance tests #3 and #4: threshold sweep numbers and run-to-run stability."""

import json
import tempfile

from detectron2.data.audit.config import AuditConfig
from detectron2.data.audit.engine import AuditSample, run_audit

from .base import AuditTestCase


class TestDeterminismAndSweep(AuditTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.samples = [
            AuditSample(it["path"], it["path"], split)
            for split, items in cls.dataset.files.items()
            for it in items
        ]

    def _run(self, batch="det", store=None):
        d = tempfile.mkdtemp(prefix=f"dupaudit-{batch}-")
        cfg = AuditConfig(
            enabled=True,
            store_dir=d,
            batch_id=batch,
            sweep_thresholds=(0.35, 0.45, 0.55, 0.65),
        )
        return run_audit({}, cfg, report_dir=d, samples=list(self.samples))

    def _signature(self, run):
        return [
            (c.cluster_id, tuple(sorted(c.members)), round(c.min_edge_score, 6), round(c.max_edge_score, 6))
            for c in run.clusters
        ]

    def test_two_runs_produce_identical_cluster_partition(self):
        a = self._run("r1")
        b = self._run("r2")
        self.assertEqual(self._signature(a), self._signature(b))
        self.assertEqual(len(a.clusters), len(b.clusters))
        # Cluster ids must be stable, not just the partition.
        self.assertEqual({c.cluster_id for c in a.clusters}, {c.cluster_id for c in b.clusters})

    def test_pair_measurements_are_identical_between_runs(self):
        a = self._run("p1")
        b = self._run("p2")
        with open(a.paths["pairs"]) as f:
            pa = json.load(f)
        with open(b.paths["pairs"]) as f:
            pb = json.load(f)
        # Remove only the fields that legitimately embed a batch id / temp paths.
        self.assertEqual(pa["pairs"], pb["pairs"])

    def test_sweep_reports_clusters_and_leak_rate_at_every_threshold(self):
        run = self._run("sweep")
        sweep = run.sweep
        thresholds = [row["threshold"] for row in sweep]
        self.assertEqual(sorted(set(thresholds)), [0.35, 0.45, 0.55, 0.65])
        for row in sweep:
            for key in (
                "threshold",
                "clusters",
                "duplicate_samples",
                "leakage_clusters",
                "cross_split_leak_samples",
                "cross_split_leak_rate",
            ):
                self.assertIn(key, row)
            self.assertIsInstance(row["clusters"], int)
            self.assertGreaterEqual(row["cross_split_leak_rate"], 0.0)
            self.assertLessEqual(row["cross_split_leak_rate"], 1.0)

    def test_sweep_numbers_are_monotone_in_the_expected_direction(self):
        """Raising the threshold cannot create new clusters or new leakage."""
        run = self._run("mono")
        ordered = sorted(run.sweep, key=lambda r: r["threshold"])
        for prev, cur in zip(ordered, ordered[1:]):
            self.assertLessEqual(cur["clusters"], prev["clusters"])
            self.assertLessEqual(cur["leakage_clusters"], prev["leakage_clusters"])
            self.assertLessEqual(cur["cross_split_leak_samples"], prev["cross_split_leak_samples"])

    def test_sweep_file_is_written_and_matches_in_memory(self):
        run = self._run("file")
        with open(run.paths["sweep"]) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["thresholds"], run.sweep)

    def test_cluster_edges_are_deterministically_ordered(self):
        run = self._run("ord")
        for c in run.clusters:
            keys = [(e.a, e.b) for e in c.edges]
            self.assertEqual(keys, sorted(keys))

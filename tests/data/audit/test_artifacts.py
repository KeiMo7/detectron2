# Copyright (c) Facebook, Inc. and its affiliates.
"""Acceptance tests for the report artifacts and the resource bounds."""

import json
import os
import tempfile

from detectron2.data.audit.config import AuditConfig
from detectron2.data.audit.engine import AuditSample, run_audit

from .base import AuditTestCase


class TestArtifacts(AuditTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._run_dir = tempfile.mkdtemp(prefix="dupaudit-art-")
        cls.cfg = AuditConfig(enabled=True, store_dir=cls._run_dir, batch_id="artifacts")
        samples = [
            AuditSample(it["path"], it["path"], split)
            for split, items in cls.dataset.files.items()
            for it in items
        ]
        cls.audit_run = run_audit({}, cls.cfg, report_dir=cls._run_dir, samples=samples)

    def _load(self, key):
        with open(self.audit_run.paths[key]) as f:
            return json.load(f)

    def test_all_expected_artifacts_exist(self):
        for key in (
            "store",
            "overview",
            "clusters",
            "pairs",
            "leaks",
            "stats",
            "sweep",
            "annotation_manifest",
            "summary",
            "skipped",
            "history",
        ):
            self.assertTrue(os.path.exists(self.audit_run.paths[key]), f"missing artifact {key}")

    def test_clusters_identify_files_and_region(self):
        payload = self._load("clusters")
        by_id = {c["cluster_id"]: c for c in payload["clusters"]}
        for c in payload["clusters"]:
            self.assertGreaterEqual(c["size"], 2)
            # every member resolves to an actual file path on disk
            for m in c["members_detail"]:
                self.assertTrue(os.path.exists(m["key"]))
                self.assertIn("canon_width", m)
                self.assertIn("scale", m)
        # The local duplicate must carry a region (either bbox or cells).
        local_ids = [
            c["cluster_id"]
            for c in payload["clusters"]
            if any("local" in (e["kind"] or "") for e in c["edges"])
        ]
        self.assertTrue(local_ids)
        for cid in local_ids:
            c = by_id[cid]
            self.assertTrue(any(e.get("region") or e.get("region_cells") for e in c["edges"]))

    def test_region_original_coordinates_are_within_the_file(self):
        payload = self._load("clusters")
        for c in payload["clusters"]:
            for region in c.get("local_regions", []):
                anchor = next(m for m in c["members_detail"] if m["key"] == region["anchor"])
                x0, y0, x1, y1 = region["region_original_xyxy"]
                self.assertGreaterEqual(x0, 0)
                self.assertGreaterEqual(y0, 0)
                self.assertLessEqual(x1, anchor["orig_width"])
                self.assertLessEqual(y1, anchor["orig_height"])
                self.assertGreater(x1, x0)
                self.assertGreater(y1, y0)

    def test_annotation_manifest_is_consumable(self):
        manifest = self._load("annotation_manifest")
        self.assertEqual(manifest["format"], "annotation-platform/manifest-v1")
        self.assertEqual(manifest["count"], len(manifest["items"]))
        for item in manifest["items"]:
            self.assertIn("cluster_number", item)
            self.assertIn("cluster_id", item)
            self.assertIn("suggested_action", item)
            self.assertIn("files", item)
            self.assertTrue(item["files"])
            for f in item["files"]:
                self.assertIn("path", f)
                self.assertIn("split", f)
                self.assertIn("role", f)

    def test_annotation_manifest_cluster_numbers_are_unique_and_ordered(self):
        manifest = self._load("annotation_manifest")
        numbers = [i["cluster_number"] for i in manifest["items"]]
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_overview_contains_the_batch_summary_numbers(self):
        overview = self._load("overview")
        for key in (
            "batch_id",
            "duplicate_clusters",
            "duplicate_rate",
            "leakage_clusters",
            "cross_split_leak_rate",
            "samples_scanned",
            "samples_skipped",
            "headline",
        ):
            self.assertIn(key, overview)
        self.assertEqual(overview["batch_id"], "artifacts")
        self.assertEqual(overview["duplicate_clusters"], len(self.audit_run.clusters))

    def test_summary_statistics_are_consistent_with_stats_file(self):
        summary = self._load("summary")
        stats = self._load("stats")
        self.assertEqual(summary["statistics"]["duplicate_clusters"], stats["duplicate_clusters"])
        self.assertEqual(summary["statistics"]["cross_split_leak_samples"], stats["cross_split_leak_samples"])
        self.assertIn("recommendations", summary)

    def test_stats_report_duplicate_and_leak_rates(self):
        stats = self._load("stats")
        for key in (
            "duplicate_rate",
            "cross_split_leak_rate",
            "duplicate_samples",
            "cross_split_leak_samples",
            "duplicate_clusters",
            "leakage_clusters",
            "samples_scanned",
            "samples_skipped",
        ):
            self.assertIn(key, stats)
        self.assertGreater(stats["duplicate_clusters"], 0)
        self.assertGreaterEqual(stats["duplicate_rate"], 0.0)
        self.assertLessEqual(stats["duplicate_rate"], 1.0)


class TestResourceBounds(AuditTestCase):
    def test_feature_count_is_bounded_regardless_of_image_count(self):
        """Per-image feature vectors are fixed-size; memory is O(workers), not O(n)."""
        from detectron2.data.audit.image_ops import load_canonical_image
        from detectron2.data.audit.features import extract_features

        cfg = AuditConfig(enabled=True, max_features=600)
        for it in self.dataset.files["train"][:3]:
            image = load_canonical_image(it["path"], cfg)
            feats = extract_features(image.gray, cfg)
            self.assertLessEqual(feats.count, cfg.max_features)

    def test_large_image_is_downscaled_to_the_canonical_budget(self):
        from PIL import Image

        from detectron2.data.audit.image_ops import load_canonical_image

        root = self.tmpdir("big")
        path = os.path.join(root, "huge.png")
        Image.new("RGB", (3000, 2000), (30, 60, 90)).save(path)
        cfg = AuditConfig(enabled=True, max_side=1024, target_pixels=600_000)
        image = load_canonical_image(path, cfg)
        self.assertLessEqual(max(image.gray.shape), 1024)
        self.assertLessEqual(image.gray.shape[0] * image.gray.shape[1], 600_000)
        # Aspect ratio preserved within rounding.
        self.assertAlmostEqual(
            image.gray.shape[1] / image.gray.shape[0], 3000 / 2000, delta=0.01
        )

    def test_extreme_aspect_ratio_image_is_handled(self):
        from PIL import Image

        from detectron2.data.audit.image_ops import load_canonical_image

        root = self.tmpdir("aspect")
        path = os.path.join(root, "panorama.png")
        Image.new("RGB", (2400, 200), (10, 10, 10)).save(path)
        # The pixel budget may downscale a panorama further than max_side; what
        # matters is that the aspect ratio survives and compute is bounded.
        cfg = AuditConfig(enabled=True, max_side=1024, target_pixels=600_000)
        image = load_canonical_image(path, cfg)
        self.assertAlmostEqual(image.gray.shape[1] / image.gray.shape[0], 12.0, delta=0.05)
        self.assertLessEqual(image.gray.shape[1], 1024)
        self.assertLessEqual(image.gray.shape[0] * image.gray.shape[1], 600_000)

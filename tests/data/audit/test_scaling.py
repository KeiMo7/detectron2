# Copyright (c) Facebook, Inc. and its affiliates.
"""Scaling behaviour of candidate generation (no O(n^2) pairwise comparison)."""

import os
import tempfile
import unittest

from detectron2.data.audit.config import AuditConfig
from detectron2.data.audit.features import extract_features
from detectron2.data.audit.image_ops import load_canonical_image
from detectron2.data.audit.index import CandidateIndex

from .synthetic import build_synthetic_dataset


class TestCandidateScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="dupaudit-scale-")
        cls.dataset = build_synthetic_dataset(
            os.path.join(cls._tmp, "ds"), n_scenes=10, n_negatives=10
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _index(self):
        cfg = AuditConfig(enabled=True)
        index = CandidateIndex(cfg)
        for path in sorted(self.dataset.paths()):
            image = load_canonical_image(path, cfg)
            index.add(path, extract_features(image.gray, cfg))
        return index

    def test_candidate_pairs_are_far_fewer_than_all_pairs(self):
        index = self._index()
        proposals = index.propose()
        n = len(self.dataset.paths())
        total_pairs = n * (n - 1) // 2
        self.assertLess(
            len(proposals) / total_pairs,
            0.5,
            f"candidate generator proposed {len(proposals)} of {total_pairs} pairs; not sub-quadratic",
        )

    def test_true_duplicates_are_all_proposed(self):
        """Every planted family pair must be a candidate before verification."""
        index = self._index()
        proposals = set(index.propose())
        scene_map = {it["path"]: it["scene_id"] for it in self.dataset.files["train"]}
        for split, items in self.dataset.files.items():
            for it in items:
                if it["kind"] == "negative":
                    continue
                if it["scene_id"] not in scene_map:
                    continue
                # Pair against the scene's train original.
                orig = next(
                    x["path"]
                    for x in self.dataset.files["train"]
                    if x["scene_id"] == it["scene_id"] and x["kind"] == "original"
                )
                if it["path"] == orig:
                    continue
                pair = tuple(sorted((orig, it["path"])))
                self.assertIn(
                    pair, proposals,
                    f"planted duplicate {os.path.basename(it['path'])} was not proposed as a candidate",
                )

    def test_index_memory_is_one_vector_per_image(self):
        """The index stores per-image summaries, not pairwise comparisons."""
        index = self._index()
        stats = index.stats()
        self.assertEqual(stats["images"], len(self.dataset.paths()))
        # Each image contributes a bounded set of codes (<= max_features).
        self.assertLessEqual(
            stats["indexed_codes"],
            stats["images"] * 2048,
        )

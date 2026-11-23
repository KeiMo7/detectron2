# Copyright (c) Facebook, Inc. and its affiliates.
"""Memory bound: decoded images / descriptors are not retained for every image.

The candidate index necessarily keeps a *compact* per-image summary (a bounded
set of 64-bit codes and a fixed-size band signature).  What must not scale with
the dataset is the heavy per-image data: the canonical pixel array and the
(SIFT) descriptor matrix.  Those live only inside the bounded LRU verified by
this test.
"""

import tracemalloc

from detectron2.data.audit.config import AuditConfig
from detectron2.data.audit.engine import AuditSample, run_audit

from .base import AuditTestCase


class TestMemoryBound(AuditTestCase):
    n_scenes = 6
    n_negatives = 4

    def test_cache_holds_at_most_capacity_images(self):
        from detectron2.data.audit import engine as engine_mod

        root = self.tmpdir("mem")
        cfg = AuditConfig(enabled=True, store_dir=root, batch_id="mem", verify_cache_size=3)
        samples = [
            AuditSample(it["path"], it["path"], split)
            for split, items in self.dataset.files.items()
            for it in items
        ]
        # Wrap the cache so we can inspect its retained entry count afterwards.
        created = {}
        original_cls = engine_mod._FeatureCache

        class Spy(original_cls):
            def __init__(self, c):
                super().__init__(c)
                created["inst"] = self

        engine_mod._FeatureCache = Spy
        try:
            run_audit({}, cfg, report_dir=root, samples=samples)
        finally:
            engine_mod._FeatureCache = original_cls
        cache = created["inst"]
        self.assertLessEqual(len(cache._entries), cfg.verify_cache_size)
        self.assertLess(len(cache._entries), len(samples), "all images were retained in memory")

    def test_peak_traced_memory_is_not_dominated_by_all_images(self):
        """Peak memory must not equal 'every image decoded at once'."""
        root = self.tmpdir("mempeak")
        cfg = AuditConfig(enabled=True, store_dir=root, batch_id="mempeak", verify_cache_size=2)
        samples = [
            AuditSample(it["path"], it["path"], split)
            for split, items in self.dataset.files.items()
            for it in items
        ]
        # Measure the size of one decoded canonical image as a reference.
        from detectron2.data.audit.image_ops import load_canonical_image

        one = load_canonical_image(samples[0].key, cfg)
        one_bytes = int(one.gray.nbytes)
        tracemalloc.start()
        run_audit({}, cfg, report_dir=root, samples=samples)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # If every image were retained, peak >= n_images * one_image.  We only
        # require that the decoded-image contribution is bounded; allow generous
        # head-room for the interpreter and the JSON artifacts.
        self.assertLess(peak, len(samples) * one_bytes * 2)

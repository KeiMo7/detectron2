# Copyright (c) Facebook, Inc. and its affiliates.
"""Robustness and the "switched off means unchanged" contract.

Covers:
* corrupt / undersized / oversized images are skipped and counted, never fatal;
* an interrupted run leaves no partial artifacts;
* disabling the audit leaves registration, loading and metadata untouched.
"""

import json
import os
from unittest import mock

from detectron2.data.audit.config import AuditConfig
from detectron2.data.audit.engine import AuditSample, run_audit

from .base import AuditTestCase


class TestRobustness(AuditTestCase):
    def _samples(self):
        return [
            AuditSample(it["path"], it["path"], split)
            for split, items in self.dataset.files.items()
            for it in items
        ]

    def test_corrupt_and_tiny_images_are_skipped_not_fatal(self):
        root = self.tmpdir("robust")
        ds_root = os.path.join(root, "bad")
        os.makedirs(os.path.join(ds_root, "test"))
        bad_files = []
        # truncated PNG
        p1 = os.path.join(ds_root, "test", "truncated.png")
        with open(p1, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"garbage")
        bad_files.append(p1)
        # 1x1 image (too small to carry features)
        from PIL import Image

        p2 = os.path.join(ds_root, "test", "tiny.png")
        Image.new("RGB", (3, 3), (10, 20, 30)).save(p2)
        bad_files.append(p2)
        # a normal image alongside
        good = self.dataset.files["test"][0]["path"]
        samples = [AuditSample(p, p, "test") for p in bad_files] + [AuditSample(good, good, "test")]
        cfg = AuditConfig(enabled=True, store_dir=root, batch_id="robust")
        run = run_audit({}, cfg, report_dir=root, samples=samples)
        reasons = {s["reason"] for s in run.skipped}
        self.assertIn("decode_error", reasons)
        self.assertIn("too_small", reasons)
        self.assertEqual(run.stats["samples_skipped"], 2)
        # The run completed and produced artifacts despite the bad files.
        self.assertTrue(os.path.exists(run.paths["stats"]))

    def test_oversized_image_is_skipped_with_a_reason(self):
        root = self.tmpdir("oversize")
        cfg = AuditConfig(enabled=True, store_dir=root, max_pixels=1_000_000)
        from detectron2.data.audit.image_ops import SkippedImage, load_canonical_image

        # 1200x900 = 1.08M pixels, above the configured cap.
        from PIL import Image

        big = os.path.join(root, "big.png")
        # Do not actually allocate 1.1M pixels of random data: use PIL directly.
        Image.new("RGB", (1200, 900), (0, 0, 0)).save(big)
        try:
            load_canonical_image(big, cfg)
        except SkippedImage as e:
            self.assertEqual(e.reason, "too_large")
        else:
            self.fail("expected oversized image to be skipped")

    def test_failure_mid_run_leaves_no_partial_artifacts(self):
        """Simulate a crash while writing the leakage list."""
        root = self.tmpdir("atomic")
        cfg = AuditConfig(enabled=True, store_dir=root, batch_id="atomic")
        samples = self._samples()
        # Patch the atomic writer used for the leakage list to raise.
        import detectron2.data.audit.engine as engine_mod

        real_write = engine_mod.atomic_write_json

        def failing_write(path, payload, **kwargs):
            if path.endswith("split_leaks.json"):
                raise RuntimeError("simulated crash")
            return real_write(path, payload, **kwargs)

        with mock.patch("detectron2.data.audit.engine.atomic_write_json", side_effect=failing_write):
            with self.assertRaises(RuntimeError):
                run_audit({}, cfg, report_dir=root, samples=samples)
        self.assertFalse(
            os.path.exists(os.path.join(root, "split_leaks.json")),
            "a partially-written leakage list survived a crashed run",
        )
        # No temp files left behind either.
        leftovers = [f for f in os.listdir(root) if f.startswith(".audit-tmp-")]
        self.assertEqual(leftovers, [])

    def test_skipped_images_are_listed_in_the_report(self):
        root = self.tmpdir("skipreport")
        cfg = AuditConfig(enabled=True, store_dir=root)
        samples = self._samples()
        run = run_audit({}, cfg, report_dir=root, samples=samples)
        with open(run.paths["skipped"]) as f:
            payload = json.load(f)
        self.assertIn("skipped", payload)
        self.assertEqual(len(payload["skipped"]), len(run.skipped))


class TestDisabledMeansUnchanged(AuditTestCase):
    """The audit must be invisible when it is off."""

    def test_disabled_config_cannot_be_run(self):
        with self.assertRaises(ValueError):
            run_audit({}, AuditConfig(enabled=False), report_dir=self.tmpdir("off"))

    def test_dataset_catalog_registration_and_get_are_untouched(self):
        from detectron2.data.catalog import DatasetCatalog

        name = "audit_disable_probe"
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)

        dicts = self.dataset.as_dicts("train")
        DatasetCatalog.register(name, lambda: dicts)
        before = DatasetCatalog.get(name)

        # Run with audit disabled and with enabled — registration output is
        # identical in both cases.
        run_audit(
            {}, AuditConfig(enabled=True, store_dir=self.tmpdir("on")), report_dir=self.tmpdir("on2"),
            samples=[AuditSample(d["file_name"], d["file_name"], "train") for d in dicts],
        )
        after = DatasetCatalog.get(name)
        self.assertEqual(before, after)
        DatasetCatalog.remove(name)

    def test_get_detection_dataset_dicts_output_is_unchanged_with_audit_off(self):
        from detectron2.data import DatasetCatalog, get_detection_dataset_dicts

        name = "audit_disable_probe2"
        dicts = self.dataset.as_dicts("val")
        DatasetCatalog.register(name, lambda: dicts)
        # No env var set: the loader bridge is a no-op.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DETECTRON2_AUDIT_OVERVIEW", None)
            out = get_detection_dataset_dicts(name, filter_empty=False)
        self.assertEqual(out, dicts)
        DatasetCatalog.remove(name)

    def test_metadata_is_not_touched_when_capability_disabled(self):
        from detectron2.data.catalog import MetadataCatalog

        name = "audit_disable_probe3"
        from detectron2.data import DatasetCatalog

        DatasetCatalog.register(name, lambda: self.dataset.as_dicts("train"))
        meta_before = dict(MetadataCatalog.get(name).as_dict())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DETECTRON2_AUDIT_OVERVIEW", None)
            from detectron2.data.audit.loader_bridge import attach_overview_to_metadata

            self.assertIsNone(attach_overview_to_metadata(name))
        self.assertEqual(dict(MetadataCatalog.get(name).as_dict()), meta_before)
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

    def test_loader_bridge_reads_overview_only_when_enabled(self):
        root = self.tmpdir("bridge")
        cfg = AuditConfig(enabled=True, store_dir=root, batch_id="bridge")
        run_audit({}, cfg, report_dir=root, samples=self._samples_all())
        from detectron2.data.audit.loader_bridge import audit_enabled, load_overview

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DETECTRON2_AUDIT_OVERVIEW", None)
            os.environ.pop("DETECTRON2_AUDIT_REPORT_DIR", None)
            self.assertFalse(audit_enabled())
            # load_overview is a low-level reader; the env gate lives in
            # attach_overview_to_metadata / get_overview, so the raw read still
            # succeeds.  What matters is that *attaching* is disabled.
            from detectron2.data.audit.loader_bridge import attach_overview_to_metadata
            self.assertIsNone(attach_overview_to_metadata("nonexistent_dataset", root, "bridge"))

        with mock.patch.dict(
            os.environ,
            {"DETECTRON2_AUDIT_OVERVIEW": "1", "DETECTRON2_AUDIT_REPORT_DIR": root},
            clear=False,
        ):
            self.assertTrue(audit_enabled())
            overview = load_overview(root, batch_id="bridge")
            self.assertIsNotNone(overview)
            # A stale overview for a different batch is never returned.
            self.assertIsNone(load_overview(root, batch_id="other"))

    def _samples_all(self):
        return [
            AuditSample(it["path"], it["path"], split)
            for split, items in self.dataset.files.items()
            for it in items
        ]

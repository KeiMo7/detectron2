# Copyright (c) Facebook, Inc. and its affiliates.
"""Shared fixtures for the duplicate-audit tests."""

import os
import shutil
import tempfile
import unittest

from detectron2.data.audit.config import AuditConfig

from .synthetic import build_synthetic_dataset


class AuditTestCase(unittest.TestCase):
    """Base class that builds a controlled synthetic dataset once per class."""

    n_scenes = 4
    n_negatives = 2

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="dupaudit-test-")
        cls.dataset = build_synthetic_dataset(
            os.path.join(cls._tmp, "dataset"),
            n_scenes=cls.n_scenes,
            n_negatives=cls.n_negatives,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def tmpdir(self, name):
        path = os.path.join(self._tmp, name)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        return path

    def config(self, **kwargs):
        tmp = self._tmp
        base = dict(
            enabled=True,
            store_dir=os.path.join(tmp, "store"),
            batch_id=kwargs.pop("batch_id", "test"),
        )
        base.update(kwargs)
        return AuditConfig(**base)

    def samples(self, splits=None):
        from detectron2.data.audit.engine import AuditSample

        out = []
        for split, items in self.dataset.files.items():
            if splits and split not in splits:
                continue
            for it in items:
                out.append(AuditSample(key=it["path"], file_name=it["path"], split=split))
        return out

# Copyright (c) Facebook, Inc. and its affiliates.
"""Near-duplicate and cross-split leakage audit for registered datasets.

This package is *side-mounted*: it reads datasets through the existing
``DatasetCatalog``, never registers or rewrites anything, and is dormant unless
a caller explicitly enables it.  OpenCV (with SIFT) is imported only by the
detection modules; the read-side helpers stay pure-Python so that importing
them cannot alter the behaviour of the standard data pipeline.

Typical use::

    from detectron2.data.audit import AuditConfig, run_audit

    cfg = AuditConfig(enabled=True, store_dir="reports")
    result = run_audit(
        {"train": ["my_train"], "val": ["my_val"], "test": ["my_test"]},
        cfg,
        report_dir="reports",
    )
    print(result.overview["headline"])

The loader-side reader is exposed separately so that a data entry point can
show the per-batch overview::

    from detectron2.data.audit.loader_bridge import attach_overview_to_metadata

It requires ``DETECTRON2_AUDIT_OVERVIEW=1`` and is otherwise a no-op.
"""

from .config import ALGORITHM_VERSION, AuditConfig

__all__ = ["AuditConfig", "ALGORITHM_VERSION", "run_audit", "collect_samples", "AuditSample"]


def __getattr__(name):
    # Lazy re-export: importing the detection engine pulls in OpenCV and scipy,
    # which must not happen for the read-side only use.
    if name in ("run_audit", "collect_samples", "AuditSample"):
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

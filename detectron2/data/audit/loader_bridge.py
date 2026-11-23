# Copyright (c) Facebook, Inc. and its affiliates.
"""Read-side access to audit results from the data entry point.

This module is intentionally light: it reads JSON, never imports OpenCV or
numpy, and never runs detection.  It is used to attach a per-batch duplicate /
leakage overview to loading, without changing any registration interface.

All functions are no-ops when the capability is disabled (the default), so the
existing data registration and loading behaviour is byte-identical with the
audit switched off.
"""

import logging
import os
from typing import Any, Dict, Optional

from .io_utils import overview_path, read_json_if_exists

logger = logging.getLogger(__name__)

__all__ = [
    "audit_enabled",
    "load_overview",
    "attach_overview_to_metadata",
    "get_overview",
]

#: Environment variable that switches the read-side overview on.  Reading a
#: small JSON file is harmless, but the default-off contract is explicit.
ENABLE_ENV = "DETECTRON2_AUDIT_OVERVIEW"
_REPORTS_ENV = "DETECTRON2_AUDIT_REPORT_DIR"


def audit_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _report_dir() -> Optional[str]:
    return os.environ.get(_REPORTS_ENV) or None


def load_overview(report_dir: Optional[str] = None, batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load ``audit_overview.json`` if present.

    Args:
        report_dir: directory holding the artifacts; defaults to
            ``DETECTRON2_AUDIT_REPORT_DIR``.
        batch_id: when given, the overview is returned only if it belongs to
            that batch (otherwise ``None``), so a stale overview is never
            presented as this batch's result.
    """
    rd = report_dir or _report_dir()
    if not rd:
        return None
    payload = read_json_if_exists(overview_path(rd))
    if payload is None:
        return None
    if batch_id is not None and payload.get("batch_id") != batch_id:
        return None
    return payload


def get_overview(dataset_name: str, report_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Overview for a registered dataset name, if one was attached."""
    if not audit_enabled():
        return None
    from detectron2.data.catalog import MetadataCatalog

    try:
        meta = MetadataCatalog.get(dataset_name)
    except Exception:
        return None
    overview = getattr(meta, "audit_overview", None)
    if isinstance(overview, dict):
        return overview
    return None


def attach_overview_to_metadata(
    dataset_name: str,
    report_dir: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Attach the batch overview to a dataset's metadata as a side field.

    This does not modify the dataset dicts or the registration function.  The
    metadata field ``audit_overview`` is only set when the capability is
    enabled and a matching overview exists; otherwise the call is a no-op.
    """
    if not audit_enabled():
        return None
    overview = load_overview(report_dir, batch_id)
    if overview is None:
        return None
    from detectron2.data.catalog import MetadataCatalog

    meta = MetadataCatalog.get(dataset_name)
    if getattr(meta, "audit_overview", None) == overview:
        return overview
    try:
        meta.audit_overview = overview
    except AssertionError:
        # Metadata refuses to change an existing value; identical payloads are
        # already handled above, differing ones are left alone by design.
        logger.warning("audit_overview already set for %s; leaving it unchanged", dataset_name)
        return getattr(meta, "audit_overview", None)
    return overview

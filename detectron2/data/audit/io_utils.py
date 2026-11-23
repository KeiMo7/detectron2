# Copyright (c) Facebook, Inc. and its affiliates.
"""Small, dependency-light helpers shared by the writer and the reader.

Everything here deliberately imports only the standard library and ``iopath``'s
PathManager (through detectron2's wrapper).  The data-loading path imports this
module, so it must stay free of OpenCV / numpy so that a disabled audit cannot
change what a plain ``import detectron2.data`` pulls in.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

from detectron2.utils.file_io import PathManager

__all__ = [
    "SCHEMA_VERSION",
    "atomic_write_json",
    "read_json",
    "read_json_if_exists",
    "overview_path",
    "store_path",
    "report_dir_layout",
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def atomic_write_json(path: str, payload: Any, *, indent: Optional[int] = 2) -> None:
    """Write ``payload`` to ``path`` atomically.

    The content is first written to a temporary file in the same directory and
    then moved into place.  A crash therefore leaves either the old file or the
    complete new file, never a half-written one.  This is the guarantee behind
    "a failed run must not leave a partial leakage list behind".
    """
    if PathManager.isdir(os.path.dirname(path) or "."):
        PathManager.mkdirs(os.path.dirname(path))
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".audit-tmp-", dir=dirname or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_if_exists(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        return read_json(path)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Ignoring unreadable audit file %s: %s", path, e)
        return None


def overview_path(report_dir: str) -> str:
    return os.path.join(report_dir, "audit_overview.json")


def store_path(store_dir: str) -> str:
    return os.path.join(store_dir, "audit_store.json")


def report_dir_layout(report_dir: str) -> Dict[str, str]:
    """Names of the artifacts produced inside ``report_dir``."""
    return {
        "store": store_path(report_dir),
        "overview": overview_path(report_dir),
        "clusters": os.path.join(report_dir, "clusters.json"),
        "pairs": os.path.join(report_dir, "pairs.json"),
        "leaks": os.path.join(report_dir, "split_leaks.json"),
        "stats": os.path.join(report_dir, "dataset_stats.json"),
        "sweep": os.path.join(report_dir, "threshold_sweep.json"),
        "annotation_manifest": os.path.join(report_dir, "annotation_manifest.json"),
        "summary": os.path.join(report_dir, "summary.json"),
        "skipped": os.path.join(report_dir, "skipped.json"),
        "history": os.path.join(report_dir, "history_reconciliation.json"),
    }

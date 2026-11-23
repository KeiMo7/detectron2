# Copyright (c) Facebook, Inc. and its affiliates.
"""Image loading and canonicalisation for the duplicate audit.

No image is ever written back: the audit is strictly read-only with respect to
the dataset.  Every failure mode (unreadable file, zero-size image, absurdly
large image, too few local features) is converted into a ``SkippedImage``
record so that one bad file cannot abort a whole batch.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

__all__ = ["SkippedImage", "LoadedImage", "load_canonical_image", "canonical_scale"]


class SkippedImage(Exception):
    """Raised when an image cannot take part in the audit.

    ``reason`` is a short machine-readable tag that is aggregated in the
    dataset statistics; ``detail`` carries the human-readable cause.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class LoadedImage:
    path: str
    gray: np.ndarray  # canonical grayscale, float32-free uint8, HW
    orig_width: int
    orig_height: int
    canon_width: int
    canon_height: int
    # canonical coordinate = original coordinate * scale
    scale: float

    def to_original_bbox(self, bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """Map an axis-aligned box from the canonical frame back to the file."""
        if self.scale <= 0:
            return (0, 0, 0, 0)
        x0, y0, x1, y1 = bbox_xyxy
        inv = 1.0 / self.scale
        return (
            int(round(max(0.0, x0 * inv))),
            int(round(max(0.0, y0 * inv))),
            int(round(min(float(self.orig_width), x1 * inv))),
            int(round(min(float(self.orig_height), y1 * inv))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "orig_width": self.orig_width,
            "orig_height": self.orig_height,
            "canon_width": self.canon_width,
            "canon_height": self.canon_height,
            "scale": round(float(self.scale), 8),
        }


def _apply_exif_orientation(image):
    """Best-effort EXIF orientation, matching detectron2's data loading."""
    try:
        from detectron2.data.detection_utils import _apply_exif_orientation as _fn
    except Exception:  # pragma: no cover - internal helper moved
        return image
    return _fn(image)


def _canonical_size(width: int, height: int, cfg) -> Tuple[int, int, float]:
    """Compute a size that bounds compute while preserving aspect ratio.

    The image is never upsampled.  Only a uniform scale factor is applied, so
    the "same scene at a different resolution" comparison keeps its geometry
    and no aspect ratio is distorted.
    """
    if width <= 0 or height <= 0:
        raise SkippedImage("invalid_size", f"{width}x{height}")
    if width * height > cfg.max_pixels:
        raise SkippedImage(
            "too_large",
            f"{width}x{height} exceeds max_pixels={cfg.max_pixels}",
        )
    scale = 1.0
    if max(width, height) > cfg.max_side:
        scale = min(scale, cfg.max_side / float(max(width, height)))
    if width * height * scale * scale > cfg.target_pixels:
        scale = min(scale, math.sqrt(cfg.target_pixels / float(width * height)))
    # Very small images still need a meaningful descriptor field.
    cw = max(1, int(round(width * scale)))
    ch = max(1, int(round(height * scale)))
    if cw < 8 or ch < 8:
        raise SkippedImage("too_small", f"canonical size {cw}x{ch}")
    return cw, ch, float(cw) / float(width) if width else 0.0


def load_canonical_image(path: str, cfg) -> LoadedImage:
    """Load ``path`` read-only and return a canonical grayscale image.

    Raises :class:`SkippedImage` for every recoverable failure.
    """
    try:
        with PathManager.open(path, "rb") as f:
            # ``Image.open`` is lazy; force the header read inside the error guard.
            pil = Image.open(f)
            pil = _apply_exif_orientation(pil)
            width, height = pil.size
            cw, ch, scale = _canonical_size(width, height, cfg)
            # ``convert`` forces a full decode here.
            rgb = pil.convert("RGB")
            if scale != 1.0:
                rgb = rgb.resize((cw, ch), Image.LANCZOS)
            gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
            if gray.ndim != 2 or gray.shape[0] != ch or gray.shape[1] != cw:
                raise SkippedImage(
                    "decode_error",
                    f"unexpected decoded shape {gray.shape} (expected {(ch, cw)})",
                )
    except SkippedImage:
        raise
    except Exception as e:  # corrupt file, truncated JPEG, unsupported mode...
        raise SkippedImage("decode_error", f"{type(e).__name__}: {e}") from e

    return LoadedImage(
        path=str(path),
        gray=gray,
        orig_width=int(width),
        orig_height=int(height),
        canon_width=int(cw),
        canon_height=int(ch),
        scale=float(scale),
    )


def canonical_scale(width: int, height: int, cfg) -> float:
    """Scale factor that :func:`load_canonical_image` would apply."""
    _, _, s = _canonical_size(width, height, cfg)
    return s

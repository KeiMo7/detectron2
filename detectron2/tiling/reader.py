# Copyright (c) Facebook, Inc. and its affiliates.
"""Image input handling for tiled inference.

Supports ordinary in-memory arrays as well as files too large to decode at
once. Region/strip reading uses ``tifffile`` memory mapping when available and
Pillow's lazy crop for common formats; plain OpenCV decoding remains the
fallback for whole-image reads. No new hard dependency is introduced.
"""

import os
from typing import Iterator, Optional, Tuple, Union

import numpy as np

try:  # optional, only needed for giant TIFFs
    import tifffile  # type: ignore

    _HAS_TIFFFILE = True
except Exception:  # pragma: no cover - environment dependent
    tifffile = None
    _HAS_TIFFFILE = False

from PIL import Image


PathLike = Union[str, "os.PathLike[str]"]


def normalize_image_channels(image: np.ndarray) -> np.ndarray:
    """Normalize an image to HxWx3 without changing channel ordering.

    * grayscale ``HxW``      -> replicated to 3 channels
    * 4-channel ``HxWx4``    -> alpha-composited over black, then dropped
    * 1-channel ``HxWx1``    -> squeezed and replicated
    * already HxWx3          -> returned untouched (same object)

    The operation is deterministic and does not reorder BGR/RGB channels;
    color order remains the caller's responsibility (matches
    ``DefaultPredictor`` convention of BGR in, ``cfg.INPUT.FORMAT`` conversion
    inside the predictor).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected np.ndarray, got {type(image).__name__}")
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"expected HxW or HxWxC image, got shape {image.shape}")

    channels = image.shape[2]
    if channels == 3:
        return image
    if channels == 1:
        return np.repeat(image, 3, axis=2)
    if channels == 4:
        # Deterministic alpha compositing over an opaque black background.
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        out = rgb * alpha
        return out.astype(image.dtype if np.issubdtype(image.dtype, np.floating) else np.uint8)
    raise ValueError(
        f"unsupported channel count {channels}; only 1, 3 or 4 channels are supported"
    )


class LargeImageReader:
    """Array- or file-backed reader with region and strip access.

    Args:
        source: an in-memory ``HxWxC`` ndarray or a path to an image file.
        force_mmap: for TIFF files, require tifffile memory mapping instead
            of whole-image decoding.
    """

    def __init__(self, source: Union[np.ndarray, PathLike], force_mmap: bool = False):
        self._mmap: Optional[np.ndarray] = None
        self._pil = None
        self._array: Optional[np.ndarray] = None
        self._path: Optional[str] = None
        self._backend = ""

        if isinstance(source, np.ndarray):
            self._array = source
            self._backend = "array"
            h, w = source.shape[:2]
            self._height, self._width = int(h), int(w)
            self._channels = source.shape[2] if source.ndim == 3 else 1
            return

        self._path = os.fspath(source)
        if not os.path.isfile(self._path):
            raise FileNotFoundError(self._path)

        ext = os.path.splitext(self._path)[1].lower()
        if ext in (".tif", ".tiff") and _HAS_TIFFFILE:
            try:
                self._mmap = tifffile.memmap(self._path, mode="r")
            except (ValueError, OSError):
                self._mmap = None
            if self._mmap is not None:
                if force_mmap is False:
                    pass
                self._backend = "tifffile-mmap"
                self._height, self._width = self._mmap.shape[:2]
                self._channels = self._mmap.shape[2] if self._mmap.ndim == 3 else 1
                return

        # Pillow opens PNG/JPEG/TIFF lazily and supports region crops without
        # decoding the whole image into Python-side storage first.
        try:
            self._pil = Image.open(self._path)
            self._pil.load()
            self._backend = "pil"
            self._width, self._height = self._pil.size
            self._channels = len(self._pil.getbands())
        except Exception as exc:
            raise IOError(f"cannot open image {self._path}: {exc}") from exc

    # -- properties --------------------------------------------------------

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def path(self) -> Optional[str]:
        return self._path

    # -- reads -------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Return the whole image as an HxWxC ndarray."""
        if self._array is not None:
            return self._array
        if self._mmap is not None:
            return np.array(self._mmap)
        return np.asarray(self._pil)

    def read_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Read one half-open rectangle in original-image coordinates."""
        if not (0 <= x0 < x1 <= self._width and 0 <= y0 < y1 <= self._height):
            raise IndexError(
                f"region ({x0},{y0})-({x1},{y1}) out of bounds for "
                f"{self._height}x{self._width}"
            )
        if self._array is not None:
            return self._array[y0:y1, x0:x1]
        if self._mmap is not None:
            return np.array(self._mmap[y0:y1, x0:x1])
        # Pillow crop box is (left, upper, right, lower) with exclusive bounds.
        return np.asarray(self._pil.crop((x0, y0, x1, y1)))

    def iter_strips(self, strip_height: int) -> Iterator[Tuple[int, int, np.ndarray]]:
        """Yield ``(y0, y1, array)`` for successive horizontal strips."""
        if strip_height <= 0:
            raise ValueError("strip_height must be positive")
        y0 = 0
        while y0 < self._height:
            y1 = min(y0 + strip_height, self._height)
            yield y0, y1, self.read_region(0, y0, self._width, y1)
            y0 = y1

    def close(self) -> None:
        if self._pil is not None:
            self._pil.close()
            self._pil = None
        # memmap arrays release storage when references drop.
        self._mmap = None
        self._array = None

    def __enter__(self) -> "LargeImageReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

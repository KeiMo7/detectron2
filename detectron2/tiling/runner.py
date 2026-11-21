# Copyright (c) Facebook, Inc. and its affiliates.
"""Batch entry points: single image, directory, dataset, optional parallelism.

Determinism rules:

* input paths are sorted before processing, so results never depend on
  filesystem enumeration order;
* each image keeps its own ``image_id``; different images are never merged;
* when workers process tiles/images concurrently, results are reordered by
  stable tile/image indices before merging, so the number of workers and the
  scheduling order cannot change the output;
* parallelism is across *images* by default. An optional tile-level thread
  pool exists for single very large images, but the merge always consumes
  tiles in row-major order regardless of completion order.
"""

import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .config import TileInferenceError
from .geometry import plan_tiles
from .merger import merge_instances, merge_sem_seg
from .predictor import TiledPredictor
from .reader import LargeImageReader, normalize_image_channels

PathLike = Union[str, "os.PathLike[str]"]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageResult:
    image_id: str
    path: Optional[str]
    predictions: Optional[Dict[str, Any]]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------------
# Single image
# --------------------------------------------------------------------------


def run_image(
    predictor: TiledPredictor,
    source: Union[np.ndarray, LargeImageReader, PathLike],
    image_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one image from an array, reader or file path."""
    if isinstance(source, (np.ndarray, LargeImageReader)):
        image = source
        resolved_id = image_id or "image"
    else:
        path = os.fspath(source)
        resolved_id = image_id or os.path.splitext(os.path.basename(path))[0]
        image = LargeImageReader(path)
    try:
        return predictor(image, image_id=resolved_id)
    finally:
        if isinstance(image, LargeImageReader) and image.path is not None:
            image.close()


# --------------------------------------------------------------------------
# Directory / dataset
# --------------------------------------------------------------------------


def _expand_inputs(inputs: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    if isinstance(inputs, (str, os.PathLike)):
        root = os.fspath(inputs)
        if os.path.isdir(root):
            paths = [
                os.path.join(root, f)
                for f in os.listdir(root)
                if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
            ]
        elif os.path.isfile(root):
            paths = [root]
        else:
            raise FileNotFoundError(root)
    else:
        paths = [os.fspath(p) for p in inputs]
    # Sorting makes batch order and image ids reproducible.
    return sorted(paths)


def run_directory(
    predictor_factory: Callable[[], TiledPredictor],
    inputs: Union[PathLike, Sequence[PathLike]],
    num_workers: int = 0,
    fail_fast: bool = True,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Sequence[Any] = (),
) -> List[ImageResult]:
    """Run every image in a directory, file list or dataset manifest.

    Args:
        predictor_factory: zero-arg callable returning a fresh
            :class:`TiledPredictor`. A factory (not an instance) is required
            so worker processes can build their own model copy.
        inputs: directory path, single file path, or explicit list of paths.
        num_workers: 0 runs serially in-process; >0 spawns worker processes
            (one model per worker). Output ordering is identical either way.
        fail_fast: when False, per-image failures are captured as
            :class:`ImageResult` errors instead of aborting the batch. Tile
            failures themselves are still never silently treated as empty.
    """
    paths = _expand_inputs(inputs)
    if not paths:
        return []

    if num_workers and num_workers > 1:
        return _run_parallel(
            predictor_factory, paths, num_workers, fail_fast, initializer, initargs
        )

    predictor = predictor_factory()
    results: List[ImageResult] = []
    for path in paths:
        image_id = os.path.splitext(os.path.basename(path))[0]
        try:
            pred = run_image(predictor, path, image_id=image_id)
            results.append(ImageResult(image_id, path, pred))
        except Exception as exc:
            if fail_fast:
                raise
            results.append(ImageResult(image_id, path, None, f"{type(exc).__name__}: {exc}"))
    return results


def _worker_entry(
    predictor_factory: Callable[[], TiledPredictor],
    path: str,
) -> ImageResult:
    image_id = os.path.splitext(os.path.basename(path))[0]
    try:
        predictor = predictor_factory()
        pred = run_image(predictor, path, image_id=image_id)
        return ImageResult(image_id, path, pred)
    except Exception as exc:
        return ImageResult(image_id, path, None, f"{type(exc).__name__}: {exc}")


def _run_parallel(
    predictor_factory: Callable[[], TiledPredictor],
    paths: List[str],
    num_workers: int,
    fail_fast: bool,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Sequence[Any] = (),
) -> List[ImageResult]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=initializer,
        initargs=tuple(initargs),
    ) as pool:
        # Preserve submission order explicitly; never rely on completion order.
        async_results = [
            pool.apply_async(_worker_entry, (predictor_factory, p)) for p in paths
        ]
        ordered = [r.get() for r in async_results]

    if fail_fast:
        for res in ordered:
            if not res.ok:
                raise TileInferenceError(
                    f"image {res.image_id} failed: {res.error}"
                )
    return ordered


# --------------------------------------------------------------------------
# Optional tile-level threading for one very large image
# --------------------------------------------------------------------------


def run_image_parallel_tiles(
    predictor: TiledPredictor,
    image: np.ndarray,
    image_id: str = "image",
    num_threads: int = 4,
) -> Dict[str, Any]:
    """Infer tiles in a thread pool while keeping merge order deterministic.

    Useful when the base predictor releases the GIL (e.g. native CUDA calls).
    Tiles carry their stable indices; results are sorted back to row-major
    before merging, so the output equals the serial run exactly.
    """
    image = normalize_image_channels(image)
    height, width = image.shape[:2]
    tiles = plan_tiles(height, width, predictor.config, image_id=image_id)

    if len(tiles) == 1 and (tiles[0].x0, tiles[0].y0, tiles[0].x1, tiles[0].y1) == (
        0,
        0,
        width,
        height,
    ):
        return predictor.base_predictor(image)

    def infer(tile):
        crop = image[tile.y0 : tile.y1, tile.x0 : tile.x1]
        return tile, predictor.base_predictor(crop)

    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as pool:
        outputs = list(pool.map(infer, tiles))
    outputs.sort(key=lambda pair: pair[0].tile_index)

    if "sem_seg" in outputs[0][1]:
        return merge_sem_seg(outputs, height, width)
    return merge_instances(outputs, height, width, predictor.config)

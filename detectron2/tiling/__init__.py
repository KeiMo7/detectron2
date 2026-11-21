# Copyright (c) Facebook, Inc. and its affiliates.
"""
Tiled (sliced) inference for very large images.

This package is an *optional* add-on to the standard single-image inference
path. It never replaces :class:`detectron2.engine.DefaultPredictor`: when an
image fits in one tile, or when tiling is not enabled, the original path and
its output are unchanged.

Public entry points:

* :class:`TilingConfig`     – validated configuration for the tiling geometry
* :class:`TiledPredictor`   – predictor that slices, runs a base predictor on
                              each tile and merges back to original-image coords
* :func:`plan_tiles`        – pure geometry: tile grid for an image size
* :func:`run_directory`     – batch over a directory / file list / dataset
* :class:`LargeImageReader` – whole-image or strip-wise reading of huge files
"""

from .config import (
    TilingConfig,
    TilingConfigError,
    TileInferenceError,
)
from .geometry import Tile, plan_tiles, axis_offsets
from .reader import LargeImageReader, normalize_image_channels
from .predictor import TiledPredictor
from .runner import run_image, run_directory

__all__ = [
    "TilingConfig",
    "TilingConfigError",
    "TileInferenceError",
    "Tile",
    "plan_tiles",
    "axis_offsets",
    "LargeImageReader",
    "normalize_image_channels",
    "TiledPredictor",
    "run_image",
    "run_directory",
]

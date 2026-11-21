# Copyright (c) Facebook, Inc. and its affiliates.
"""Tiled predictor: the user-facing counterpart of DefaultPredictor."""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from .config import TilingConfig, TileInferenceError
from .geometry import plan_tiles
from .merger import merge_instances, merge_sem_seg
from .reader import LargeImageReader, normalize_image_channels


class TiledPredictor:
    """Run a base predictor tile by tile and merge in original-image coords.

    Args:
        base_predictor: callable mapping an ``HxWxC`` ndarray to the standard
            detectron2 output dict (e.g. a :class:`DefaultPredictor`).
        config: a validated :class:`TilingConfig`.
        image_id: stable identity embedded in every tile of one call.

    The output dict keeps the standard keys/fields (``instances`` with
    ``pred_boxes``/``scores``/``pred_classes`` and, when produced by the
    model, ``pred_masks``/``pred_keypoints``; or ``sem_seg``). A private
    ``_tiled`` debug key carries tile statistics and is never required by the
    existing output contract.

    When the whole image fits in one tile, tiling is skipped entirely and the
    base predictor is invoked exactly once on the full image, so the new path
    is bit-for-bit identical to the original single-image path.
    """

    def __init__(
        self,
        base_predictor: Callable[[np.ndarray], Dict[str, Any]],
        config: TilingConfig,
        image_id: str = "image",
    ):
        if not isinstance(config, TilingConfig):
            raise TypeError("config must be a TilingConfig instance")
        self.base_predictor = base_predictor
        self.config = config
        self.image_id = image_id

    # ------------------------------------------------------------------

    def _plan(self, height: int, width: int):
        tiles = plan_tiles(height, width, self.config, image_id=self.image_id)
        return tiles

    def __call__(
        self,
        image: Union[np.ndarray, LargeImageReader],
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_id = image_id or self.image_id

        if isinstance(image, LargeImageReader):
            reader = image
            height, width = reader.height, reader.width
            crop = lambda t: normalize_image_channels(  # noqa: E731
                reader.read_region(t.x0, t.y0, t.x1, t.y1)
            )
        else:
            image = normalize_image_channels(image)
            height, width = image.shape[:2]
            crop = lambda t: image[t.y0 : t.y1, t.x0 : t.x1]  # noqa: E731

        tiles = plan_tiles(height, width, self.config, image_id=image_id)

        # Optional-enhancement guarantee: image smaller than / equal to one
        # tile -> original path, untouched output structure.
        if len(tiles) == 1:
            t = tiles[0]
            if (t.x0, t.y0, t.x1, t.y1) == (0, 0, width, height):
                return self.base_predictor(crop(t))

        kind = self._infer_output_kind(tiles, crop)
        prefetched = self._prefetched
        tile_outputs = []
        failures = []
        for tile in tiles:  # row-major order; parallel runners reorder later
            if tile.tile_index in prefetched:
                tile_outputs.append((tile, prefetched[tile.tile_index]))
                continue
            tile_image = crop(tile)
            try:
                pred = self.base_predictor(tile_image)
            except Exception as exc:  # never swallow as "no detections"
                if self.config.on_tile_error == "raise":
                    raise TileInferenceError(
                        f"image {image_id} tile {tile.tile_index} "
                        f"(row={tile.row},col={tile.col}, region="
                        f"[{tile.x0},{tile.y0},{tile.x1},{tile.y1}]) failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                failures.append((tile.tile_index, repr(exc)))
                continue
            tile_outputs.append((tile, pred))

        if failures:
            detail = "; ".join(f"tile {idx}: {err}" for idx, err in failures)
            raise TileInferenceError(
                f"image {image_id}: {len(failures)} of {len(tiles)} tiles failed "
                f"and cannot be treated as empty: {detail}"
            )

        if kind == "sem_seg":
            return merge_sem_seg(tile_outputs, height, width)
        return merge_instances(tile_outputs, height, width, self.config)

    # ------------------------------------------------------------------

    def _infer_output_kind(self, tiles, crop) -> str:
        """Peek at one tile's output keys to choose the merge routine.

        Inference on this tile is not wasted: its result is reused below via a
        tiny per-call cache keyed by tile index.
        """
        first = tiles[0]
        out = self.base_predictor(crop(first))
        self._prefetched = {first.tile_index: out}
        if "instances" in out:
            return "instances"
        if "sem_seg" in out:
            return "sem_seg"
        raise TileInferenceError(
            f"tile {first.tile_index}: output has neither 'instances' nor "
            "'sem_seg'; tiling merger cannot handle its keys"
        )

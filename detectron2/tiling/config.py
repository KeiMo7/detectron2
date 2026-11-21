# Copyright (c) Facebook, Inc. and its affiliates.
"""Configuration and exception types for tiled inference."""

from dataclasses import dataclass
from typing import Optional


class TilingConfigError(ValueError):
    """Raised *before* inference starts when tiling parameters are invalid."""


class TileInferenceError(RuntimeError):
    """Raised when a tile cannot be inferred and the failure must not be silent."""


# Hard safety bound so that a pathological overlap ratio can never turn an
# image into an unbounded number of tiles.
_MAX_TILES_HARD_LIMIT = 10_000
_MIN_TILE_SIDE = 32


def _validate_ratio(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TilingConfigError(f"{name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not 0.0 < value < 1.0:
        raise TilingConfigError(
            f"{name}={value} is illegal: overlap ratio must be strictly between "
            "0.0 and 1.0 (exclusive). Zero overlap would split seam objects."
        )
    return value


@dataclass(frozen=True)
class TilingConfig:
    """Geometry / budget / merge configuration for tiled inference.

    The config is frozen and hashable so that ``(input, config)`` fully
    determines the result: two runs with identical inputs and parameters are
    reproducible regardless of execution order or number of workers.

    Tile sizing has two mutually exclusive modes:

    * explicit ``tile_size`` (square side in source-image pixels), or
    * budget mode: leave ``tile_size`` as ``None`` and provide
      ``max_pixels_per_tile`` (or ``vram_budget_bytes``); the largest square
      tile within the budget is derived during planning.

    Args:
        tile_size: square tile side in *original-image* pixels. ``None``
            selects budget mode.
        overlap_ratio: fraction of a tile side shared with the neighbour on
            each axis. Must be in the open interval (0, 1).
        max_tiles: per-image cap on ``rows * cols``. Protects against tile
            count explosion when overlap is large.
        max_pixels_per_tile: optional hard pixel budget for one tile, measured
            on the source image (e.g. 1024*1024). Planning fails if no legal
            tile fits.
        vram_budget_bytes: optional device memory budget. Combined with
            ``vram_bytes_per_pixel`` it derives a pixel budget. Ignored when
            ``max_pixels_per_tile`` is set explicitly.
        vram_bytes_per_pixel: bytes of device memory consumed per input pixel
            (model dependent; conservative default).
        merge_iou_threshold: IoU above which two detections from different
            tiles are considered the same seam-crossing object.
        merge_ioa_threshold: intersection-over-area-of-the-smaller-box
            threshold. Seam-split halves have low IoU (each is only part of
            the whole) but very high IoA, so this is the primary seam
            criterion: a box whose area is mostly contained in a neighbour is
            merged into it.
        min_score: detections below this score are discarded before merging.
        detections_per_image: final per-image cap, mirroring
            ``cfg.TEST.DETECTIONS_PER_IMAGE``. ``None`` means no extra cap.
        on_tile_error: ``"raise"`` (default) propagates the first tile error;
            ``"collect"`` records it and still fails the image, naming every
            failed tile. A failed tile is never silently treated as
            "no objects here".
        merge_unknown_fields: if False, an output field the merger does not
            know how to reproject raises instead of being dropped.
    """

    tile_size: Optional[int] = None
    overlap_ratio: float = 0.2
    max_tiles: int = 256
    max_pixels_per_tile: Optional[int] = None
    vram_budget_bytes: Optional[int] = None
    vram_bytes_per_pixel: float = 12.0
    merge_iou_threshold: float = 0.5
    merge_ioa_threshold: float = 0.3
    min_score: float = 0.0
    detections_per_image: Optional[int] = 100
    on_tile_error: str = "raise"
    merge_unknown_fields: bool = False

    def __post_init__(self):
        # overlap
        ratio = _validate_ratio("overlap_ratio", self.overlap_ratio)
        object.__setattr__(self, "overlap_ratio", ratio)

        # tile_size
        if self.tile_size is not None:
            if not isinstance(self.tile_size, int) or isinstance(self.tile_size, bool):
                raise TilingConfigError(
                    f"tile_size must be an int number of pixels, got {self.tile_size!r}"
                )
            if self.tile_size <= 0:
                raise TilingConfigError(
                    f"tile_size={self.tile_size} is illegal: must be positive"
                )

        # max_tiles
        if not isinstance(self.max_tiles, int) or isinstance(self.max_tiles, bool):
            raise TilingConfigError("max_tiles must be an int")
        if self.max_tiles < 1:
            raise TilingConfigError(
                f"max_tiles={self.max_tiles} is illegal: at least one tile is required"
            )
        if self.max_tiles > _MAX_TILES_HARD_LIMIT:
            raise TilingConfigError(
                f"max_tiles={self.max_tiles} exceeds the hard safety limit "
                f"{_MAX_TILES_HARD_LIMIT}; refusing to plan a tile grid this large"
            )

        # budgets
        if self.max_pixels_per_tile is not None and self.max_pixels_per_tile <= 0:
            raise TilingConfigError("max_pixels_per_tile must be positive")
        if self.vram_budget_bytes is not None and self.vram_budget_bytes <= 0:
            raise TilingConfigError("vram_budget_bytes must be positive")
        if not isinstance(self.vram_bytes_per_pixel, (int, float)) or (
            self.vram_bytes_per_pixel <= 0
        ):
            raise TilingConfigError("vram_bytes_per_pixel must be a positive number")

        # merge / score knobs
        if not 0.0 <= self.merge_iou_threshold <= 1.0:
            raise TilingConfigError(
                f"merge_iou_threshold={self.merge_iou_threshold} must be in [0, 1]"
            )
        if not 0.0 <= self.merge_ioa_threshold <= 1.0:
            raise TilingConfigError(
                f"merge_ioa_threshold={self.merge_ioa_threshold} must be in [0, 1]"
            )
        if not 0.0 <= self.min_score <= 1.0:
            raise TilingConfigError(f"min_score={self.min_score} must be in [0, 1]")
        if self.detections_per_image is not None and self.detections_per_image < 1:
            raise TilingConfigError("detections_per_image must be >= 1 or None")
        if self.on_tile_error not in ("raise", "collect"):
            raise TilingConfigError(
                f"on_tile_error={self.on_tile_error!r} must be 'raise' or 'collect'"
            )

    # -- derived budget ----------------------------------------------------

    def pixel_budget(self) -> Optional[int]:
        """Effective max pixels per tile, or ``None`` when unconstrained."""
        if self.max_pixels_per_tile is not None:
            return int(self.max_pixels_per_tile)
        if self.vram_budget_bytes is not None:
            return int(self.vram_budget_bytes // self.vram_bytes_per_pixel)
        return None

    def resolve_tile_size(self, image_height: int, image_width: int) -> int:
        """Return the square tile side to use for a given image.

        In explicit mode the requested side is used as-is; geometry planning
        later clamps the *last* row/column to the image boundary. In budget
        mode the largest square fitting the pixel budget is derived.

        Raises :class:`TilingConfigError` before inference when the budget
        cannot fit a minimally usable tile.
        """
        if self.tile_size is not None:
            return int(self.tile_size)

        budget = self.pixel_budget()
        if budget is None:
            # No size and no budget: default to a 1024 px square tile.
            return 1024

        side = int(budget ** 0.5)
        side = min(side, image_height, image_width)
        if side < _MIN_TILE_SIDE:
            raise TilingConfigError(
                "Memory budget cannot fit a usable tile: derived side "
                f"{side}px < minimum {_MIN_TILE_SIDE}px for an image of "
                f"{image_height}x{image_width} (pixel budget={budget}). "
                "Raise max_pixels_per_tile / vram_budget_bytes."
            )
        return side

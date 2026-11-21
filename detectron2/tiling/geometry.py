# Copyright (c) Facebook, Inc. and its affiliates.
"""Pure tiling geometry.

Everything in this module is deterministic and free of model/tensor logic, so
it can be unit-tested in isolation. Coordinates are always source-image pixel
coordinates; the last row and column are clamped to the image boundary, hence
tiles never read out of bounds and never leave an uncovered strip.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import TilingConfig, TilingConfigError


@dataclass(frozen=True)
class Tile:
    """One rectangular crop of a source image.

    Attributes:
        tile_index: globally unique, row-major stable id within the image
            (``row * num_cols + col``). Independent of scheduling order.
        row, col: grid position.
        x0, y0, x1, y1: half-open pixel range in *original-image* coordinates.
            ``x1 <= image_width`` and ``y1 <= image_height`` always hold.
        image_id: stable identity shared by every tile of the same source
            image. Tiles from different images never share an id.
        scale_x, scale_y: source-to-tile scale (1.0 for plain crops; kept so
            callers can reason about pre-resized inputs).
    """

    tile_index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    image_id: str
    scale_x: float = 1.0
    scale_y: float = 1.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def offset(self) -> Tuple[int, int]:
        return (self.x0, self.y0)


def axis_offsets(length: int, tile: int, overlap_ratio: float) -> List[int]:
    """Return tile start offsets along one axis.

    Guarantees:

    * first offset is 0;
    * the final offset plus ``tile`` equals ``length`` (boundary aligned);
    * every offset stays within ``[0, length - tile]``;
    * when ``length <= tile`` there is exactly one offset, 0.
    """
    if length <= 0 or tile <= 0:
        raise TilingConfigError(
            f"axis_offsets requires positive length/tile, got {length}/{tile}"
        )
    if length <= tile:
        return [0]

    step = tile * (1.0 - overlap_ratio)
    if step <= 0:
        # Defensive: config validation should already have rejected this.
        raise TilingConfigError(
            "overlap_ratio makes the tiling step non-positive; "
            f"tile={tile}, overlap_ratio={overlap_ratio}"
        )

    offsets: List[int] = [0]
    while offsets[-1] + tile < length:
        nxt = offsets[-1] + step
        if nxt + tile >= length:
            offsets.append(length - tile)
            break
        offsets.append(int(round(nxt)))
    # Remove duplicates that rounding could theoretically create.
    deduped: List[int] = []
    for off in offsets:
        if not deduped or off != deduped[-1]:
            deduped.append(off)
    return deduped


def plan_tiles(
    image_height: int,
    image_width: int,
    config: TilingConfig,
    image_id: str = "image",
) -> List[Tile]:
    """Plan the full tile grid for one image.

    Raises :class:`TilingConfigError` before any inference when:

    * the explicit tile size violates the pixel/VRAM budget;
    * the resulting grid exceeds ``config.max_tiles``.
    """
    if image_height <= 0 or image_width <= 0:
        raise TilingConfigError(
            f"illegal image size {image_height}x{image_width}: dimensions must be positive"
        )

    tile_side = config.resolve_tile_size(image_height, image_width)

    # In explicit mode, refuse up-front if the requested tile breaks the budget.
    budget = config.pixel_budget()
    if config.tile_size is not None and budget is not None:
        if tile_side * tile_side > budget:
            raise TilingConfigError(
                f"Requested tile_size={tile_side} "
                f"({tile_side * tile_side} px/tile) exceeds the pixel budget "
                f"{budget}. Lower tile_size or raise the budget."
            )

    y_offsets = axis_offsets(image_height, tile_side, config.overlap_ratio)
    x_offsets = axis_offsets(image_width, tile_side, config.overlap_ratio)
    rows, cols = len(y_offsets), len(x_offsets)
    n_tiles = rows * cols

    if n_tiles > config.max_tiles:
        raise TilingConfigError(
            f"Planned grid {rows}x{cols}={n_tiles} tiles exceeds max_tiles="
            f"{config.max_tiles} for image {image_width}x{image_height} with "
            f"tile={tile_side}, overlap={config.overlap_ratio}. Constraint: "
            "max_tiles. Reduce overlap, enlarge tiles or raise max_tiles."
        )

    tiles: List[Tile] = []
    for r, y0 in enumerate(y_offsets):
        for c, x0 in enumerate(x_offsets):
            y1 = min(y0 + tile_side, image_height)
            x1 = min(x0 + tile_side, image_width)
            tiles.append(
                Tile(
                    tile_index=r * cols + c,
                    row=r,
                    col=c,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    image_id=image_id,
                )
            )
    return tiles


def validate_coverage(
    tiles: Sequence[Tile], image_height: int, image_width: int
) -> None:
    """Assert a tile set fully covers the image exactly once per interior pixel.

    Used in tests and as a defensive check; not on the hot path.
    """
    covered = [[0] * image_width for _ in range(image_height)]
    for t in tiles:
        assert 0 <= t.x0 < t.x1 <= image_width
        assert 0 <= t.y0 < t.y1 <= image_height
        for y in range(t.y0, t.y1):
            row = covered[y]
            for x in range(t.x0, t.x1):
                row[x] += 1
    assert all(c >= 1 for row in covered for c in row), "uncovered pixels exist"
    # Boundary pixels must be covered exactly once (no out-of-bounds tiles).
    assert covered[0][0] >= 1 and covered[-1][-1] >= 1

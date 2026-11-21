# Tiled Inference for Very Large Images

## Why

The standard inference path resizes every image so its longest edge is at most
`cfg.INPUT.MAX_SIZE_TEST` (default 1333). On a 12000×9000 aerial frame that
collapses ~108 MP to under 2 MP; a car shrinks to a few pixels and the model
cannot see it. Naively cutting the image into nine pieces makes it worse:
objects on a cut are split into two half-boxes or counted twice.

`detectron2.tiling` is an **optional** add-on that:

1. plans an overlapping tile grid within an explicit size or memory budget;
2. runs the *existing* predictor independently on each tile;
3. reprojects every result into original-image coordinates and merges
   seam-crossing objects into exactly one complete detection;
4. keeps the standard output structure, field names, class ids and
   `TEST.DETECTIONS_PER_IMAGE` semantics unchanged.

It does **not** replace the whole-image resize path. An image that fits in one
tile is sent straight through the original predictor, byte-for-byte.

## Concepts

| Piece | Responsibility |
|---|---|
| `TilingConfig` | Frozen, validated parameters (tile size / budget, overlap, caps, merge thresholds). Illegal values raise `TilingConfigError` before any inference. |
| `plan_tiles` | Pure geometry. Row-major stable tile ids; last row/column glued to the image boundary, so tiles never read out of bounds and never leave a gap. |
| `LargeImageReader` | Array or file input; region and horizontal-strip reads (Pillow lazy crop, tifffile memmap when installed) for images too large to decode at once. |
| `TiledPredictor` | Slices, calls a base predictor per tile, merges results. |
| `merger` | Deterministic class-aware grouping, union boxes, mask OR, keypoint arbitration, semantic-canvas accumulation. |
| `runner` | Single image / directory / dataset batching, optional multiprocessing with order-preserving reassembly. |

## Quick start

```python
from detectron2.engine.defaults import DefaultPredictor
from detectron2.tiling import TiledPredictor, TilingConfig

base = DefaultPredictor(cfg)
tiled = TiledPredictor(
    base,
    TilingConfig(
        tile_size=1300,        # source-image pixels per tile side
        overlap_ratio=0.2,     # strictly between 0 and 1
        max_tiles=256,         # hard cap against overlap-driven explosion
        merge_ioa_threshold=0.3,
    ),
)
predictions = tiled(image)   # same dict shape as DefaultPredictor(image)
```

Budget mode — derive the tile side from memory instead of naming it:

```python
TilingConfig(vram_budget_bytes=6 * 1024**3, vram_bytes_per_pixel=12.0)
# or a direct pixel budget
TilingConfig(max_pixels_per_tile=1300 * 1300)
```

Directory / dataset batching:

```python
from detectron2.tiling import run_directory

results = run_directory(factory, "images/", num_workers=2)
# factory() builds a fresh TiledPredictor; output order is identical to the
# serial run regardless of worker scheduling.
```

CLI:

```bash
python -m detectron2.tiling.cli --config-file cfg.yaml --weights model.pkl \
    --input aerial.png --output out/ --tile-size 1300 --overlap 0.2
```

## How seam objects become one detection

Two boxes from different tiles are grouped as one object when they overlap by
**either** IoU > `merge_iou_threshold` **or** IoA (intersection over the
smaller box) > `merge_ioa_threshold`. IoA matters because two halves of a cut
object barely overlap relative to their union (low IoU) while the smaller is
almost fully contained (high IoA). Grouping is class-aware and greedy over the
total order `(-score, class_id, row_index)`, so:

* a merged box is the coordinate union of its members — never two half-boxes;
* each object is counted once;
* if adjacent tiles label the same seam object with different classes, the
  higher-scoring label wins, ties resolved by class id then row index — never
  at random, so repeated runs are identical.

Masks are OR-composited onto a full-image bool canvas of the exact source
size. Keypoints pick, per joint, the candidate nearest the merged box centre
with a row-index tie-break. Semantic segmentation accumulates class scores per
pixel over every contributing tile and takes the argmax, leaving no seam band
or gap.

## Coordinates and identity

* Every tile carries the same `image_id`, its grid position and its
  `(x0, y0)` offset; tiles of different images are never combined.
* Tile-local boxes are clipped to the tile, translated by the offset, then
  clipped to the image — output coordinates can never leave the source image.
* Stitched masks and segmentation maps have exactly the source `H×W`.

## Failure semantics

* Bad parameters (zero/one overlap, tile larger than allowed by the budget,
  grid over `max_tiles`, unusably small budget tile) raise
  `TilingConfigError` at planning time, naming the violated constraint.
* A tile that errors is never silently treated as "no detections":
  `on_tile_error="raise"` surfaces the first failure with tile coordinates;
  `"collect"` still fails the whole image after naming every failed tile.

## Reproducibility contract

`TilingConfig` is frozen and hashable; tile indices are row-major; grouping and
tie-breaks are fully ordered; parallel runners reassemble by stable index.
Identical input and parameters therefore produce identical output on one or
many workers, in any scheduling order.

## Supported output types

Detection boxes, instance masks, keypoints (`instances` container with the
standard fields) and semantic segmentation (`sem_seg`). An unknown per-instance
field raises rather than being dropped unless
`TilingConfig(merge_unknown_fields=True)` is set explicitly.

## Input edges handled

* image smaller than one tile → original single-image path;
* dimensions not an integer multiple of the tile → last row/column clamped;
* extreme aspect ratios;
* grayscale and 1/4-channel images (replicated channels / deterministic alpha
  composite over black);
* files too large to decode at once → region / strip reads.

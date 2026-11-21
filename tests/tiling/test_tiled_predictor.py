# Copyright (c) Facebook, Inc. and its affiliates.
"""End-to-end tests for TiledPredictor: seam merging, coords, determinism, errors."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _d2_shim import install_shims  # noqa: E402

install_shims()

from detectron2.tiling import (  # noqa: E402
    TileInferenceError,
    TiledPredictor,
    TilingConfig,
    plan_tiles,
)
from detectron2.tiling.runner import run_image_parallel_tiles  # noqa: E402

from fakes import FakeInstancePredictor, FailingOncePredictor, draw_squares  # noqa: E402


def _image(h, w, squares):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return draw_squares(img, squares)


# --------------------------------------------------------------------------
# Optional-enhancement: tiny image takes the original single-image path
# --------------------------------------------------------------------------


def test_small_image_identical_to_direct_path():
    cfg = TilingConfig(tile_size=1024, overlap_ratio=0.2)
    base = FakeInstancePredictor()
    tiled = TiledPredictor(base, cfg)
    img = _image(300, 400, [(50, 60, 90, 100)])

    direct = base(img)
    out = tiled(img)

    # exact same call payload and identical boxes/classes/scores
    assert len(base.calls) == 2  # one direct, one via tiled
    assert base.calls[0] == base.calls[1]
    torch.testing.assert_close(
        out["instances"].pred_boxes.tensor, direct["instances"].pred_boxes.tensor
    )
    torch.testing.assert_close(out["instances"].scores, direct["instances"].scores)
    torch.testing.assert_close(
        out["instances"].pred_classes, direct["instances"].pred_classes
    )
    assert "_tiled" not in out  # single-tile path returns base output untouched


# --------------------------------------------------------------------------
# The headline scenario: 12000x9000 aerial image, seam-crossing car
# --------------------------------------------------------------------------


def test_seam_crossing_object_merged_not_split():
    H, W = 900, 1200  # scaled stand-in for 9000x12000, same geometry ratios
    cfg = TilingConfig(tile_size=400, overlap_ratio=0.25)
    tiles = plan_tiles(H, W, cfg, "aerial")
    assert len(tiles) > 1
    # place a 60px car straddling the first vertical seam
    first_seam_x = min(t.x0 for t in tiles if t.col == 1)
    car = (first_seam_x - 30, 100, first_seam_x + 30, 160)
    img = _image(H, W, [car])

    base = FakeInstancePredictor()
    out = TiledPredictor(base, cfg)(img)
    inst = out["instances"]

    assert len(inst) == 1, "seam object must produce exactly one detection"
    box = inst.pred_boxes.tensor[0].tolist()
    assert box == pytest.approx(list(car), abs=1.0)
    # coordinates never escape the image
    assert 0 <= box[0] and box[2] <= W and 0 <= box[1] and box[3] <= H
    assert int(inst.pred_classes[0]) == 0


def test_separate_objects_not_double_counted():
    H, W = 900, 1200
    cfg = TilingConfig(tile_size=400, overlap_ratio=0.25)
    cars = [(60, 60, 110, 110), (900, 700, 960, 760), (500, 480, 560, 540)]
    img = _image(H, W, cars)
    out = TiledPredictor(FakeInstancePredictor(), cfg)(img)
    assert len(out["instances"]) == 3
    boxes = out["instances"].pred_boxes.tensor
    # every source car recovered, in original coords
    for car in cars:
        ious = np.array(
            [_iou(box.tolist(), list(car)) for box in boxes]
        )
        assert ious.max() > 0.99


def _iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# --------------------------------------------------------------------------
# Determinism: same input/params -> same result, incl. parallel scheduling
# --------------------------------------------------------------------------


def test_reproducible_serial_vs_threaded():
    H, W = 900, 1200
    cfg = TilingConfig(tile_size=400, overlap_ratio=0.25)
    cars = [(100, 100, 150, 150), (450, 300, 510, 360), (820, 700, 880, 760)]
    img = _image(H, W, cars)

    serial = TiledPredictor(FakeInstancePredictor(), cfg)(img)
    threaded = run_image_parallel_tiles(
        TiledPredictor(FakeInstancePredictor(), cfg), img, num_threads=3
    )
    torch.testing.assert_close(
        serial["instances"].pred_boxes.tensor,
        threaded["instances"].pred_boxes.tensor,
    )
    torch.testing.assert_close(
        serial["instances"].scores, threaded["instances"].scores
    )


def test_two_runs_identical():
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.3)
    img = _image(700, 700, [(40, 40, 90, 90), (350, 350, 420, 420), (600, 600, 660, 660)])
    a = TiledPredictor(FakeInstancePredictor(), cfg)(img)
    b = TiledPredictor(FakeInstancePredictor(), cfg)(img)
    torch.testing.assert_close(
        a["instances"].pred_boxes.tensor, b["instances"].pred_boxes.tensor
    )


# --------------------------------------------------------------------------
# Masks and keypoints reproject and merge
# --------------------------------------------------------------------------


def test_masks_stitch_to_full_canvas_size():
    H, W = 500, 900
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.25)
    # The actual tile *boundary* is the first tile's right edge; overlap makes
    # the second tile start earlier, but a seam object is cut at x=tile_size.
    boundary = min(t.x1 for t in plan_tiles(H, W, cfg) if t.col == 0)
    car = (boundary - 30, 200, boundary + 30, 260)  # straddles the cut
    img = _image(H, W, [car])
    out = TiledPredictor(FakeInstancePredictor(with_masks=True), cfg)(img)
    inst = out["instances"]
    assert len(inst) == 1
    mask = inst.pred_masks.tensor
    assert mask.shape == (1, H, W), "stitched mask must match original size exactly"
    # no blank band at the cut: union mask covers pixels on both sides
    assert mask[0, :, boundary - 1].any() and mask[0, :, boundary].any()
    # mask aligns with the true square
    expected = np.zeros((H, W), dtype=bool)
    expected[car[1] : car[3], car[0] : car[2]] = True
    iou = (
        np.logical_and(mask[0].numpy(), expected).sum()
        / np.logical_or(mask[0].numpy(), expected).sum()
    )
    assert iou > 0.95


def test_keypoints_reprojected_and_deterministic():
    H, W = 500, 900
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.25)
    car = (280, 200, 340, 260)
    img = _image(H, W, [car])
    out = TiledPredictor(FakeInstancePredictor(with_keypoints=True), cfg)(img)
    kp = out["instances"].pred_keypoints.tensor
    assert kp.shape == (1, 1, 3)
    cx, cy = (car[0] + car[2]) / 2, (car[1] + car[3]) / 2
    assert abs(kp[0, 0, 0].item() - cx) < 1.0
    assert abs(kp[0, 0, 1].item() - cy) < 1.0


# --------------------------------------------------------------------------
# Failure handling: a failed tile is never silently treated as empty
# --------------------------------------------------------------------------


def test_tile_failure_raises_with_location():
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.25, on_tile_error="raise")
    img = _image(700, 700, [(50, 50, 90, 90)])
    predictor = TiledPredictor(FailingOncePredictor(FakeInstancePredictor(), 1), cfg)
    with pytest.raises(TileInferenceError, match="tile 1"):
        predictor(img)


def test_tile_failure_collect_still_fails_image():
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.25, on_tile_error="collect")
    img = _image(700, 700, [(50, 50, 90, 90)])
    predictor = TiledPredictor(FailingOncePredictor(FakeInstancePredictor(), 2), cfg)
    with pytest.raises(TileInferenceError, match="1 of"):
        predictor(img)


# --------------------------------------------------------------------------
# Output contract: cap semantics and clipping
# --------------------------------------------------------------------------


def test_detections_per_image_cap_respected():
    cfg = TilingConfig(
        tile_size=200, overlap_ratio=0.2, detections_per_image=2
    )
    squares = [(20 + i * 60, 60, 60 + i * 60, 100) for i in range(6)]
    img = _image(400, 900, squares)
    out = TiledPredictor(FakeInstancePredictor(), cfg)(img)
    assert len(out["instances"]) == 2


def test_boxes_always_clipped_to_image():
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.25)
    img = _image(500, 500, [(470, 470, 520, 520)])  # extends beyond canvas
    out = TiledPredictor(FakeInstancePredictor(), cfg)(img)
    box = out["instances"].pred_boxes.tensor[0]
    assert box[2] <= 500 and box[3] <= 500


def test_min_score_filter_keeps_masks_aligned_with_boxes():
    # Regression: score filtering must not desynchronise the mask pool from
    # the surviving box rows.
    H, W = 500, 900
    cfg = TilingConfig(
        tile_size=300, overlap_ratio=0.25, min_score=0.95
    )

    class MixedScore(FakeInstancePredictor):
        def __call__(self, image):
            out = super().__call__(image)
            n = len(out["instances"])
            if n:
                # alternate low/high scores so some rows get filtered
                out["instances"].scores = torch.tensor(
                    [0.5 if i % 2 else 0.99 for i in range(n)]
                )
            return out

    boundary = min(t.x1 for t in plan_tiles(H, W, cfg) if t.col == 0)
    car = (boundary - 30, 200, boundary + 30, 260)
    img = _image(H, W, [car])
    out = TiledPredictor(MixedScore(with_masks=True), cfg)(img)
    inst = out["instances"]
    assert len(inst) == 1
    # mask count matches box count and has full-canvas shape
    assert inst.pred_masks.tensor.shape == (1, H, W)
    expected = np.zeros((H, W), dtype=bool)
    expected[car[1] : car[3], car[0] : car[2]] = True
    iou = (
        np.logical_and(inst.pred_masks.tensor[0].numpy(), expected).sum()
        / np.logical_or(inst.pred_masks.tensor[0].numpy(), expected).sum()
    )
    assert iou > 0.9

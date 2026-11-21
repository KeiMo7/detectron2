# Copyright (c) Facebook, Inc. and its affiliates.
"""Determinism and arbitration rules that the spec calls out explicitly."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _d2_shim import install_shims  # noqa: E402

install_shims()

from detectron2.structures import Boxes, Instances  # noqa: E402
from detectron2.tiling import TilingConfig, TiledPredictor, plan_tiles  # noqa: E402
from detectron2.tiling.merger import merge_instances  # noqa: E402


class ScriptedClassPredictor:
    """Emits a scripted list of (local box, class, score) once per tile call."""

    def __init__(self, script_by_tile):
        self.script = script_by_tile
        self.n = 0

    def __call__(self, image):
        h, w = image.shape[:2]
        items = self.script.get(self.n, [])
        self.n += 1
        inst = Instances((h, w))
        boxes = [b for b, _, _ in items]
        inst.pred_boxes = Boxes(
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        )
        inst.scores = torch.tensor([s for _, _, s in items], dtype=torch.float32)
        inst.pred_classes = torch.tensor([c for _, c, _ in items], dtype=torch.long)
        return {"instances": inst}


def test_cross_class_seam_conflict_deterministic_winner():
    # Two overlapping halves of one seam object receive different classes.
    # The higher-score detection must win, identically on every run.
    H, W = 400, 700
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.25)
    tiles = plan_tiles(H, W, cfg)
    assert len(tiles) >= 2
    t0 = tiles[0]
    # tile 0 right edge
    cut = t0.x1
    script = {
        0: [((cut - 60, 100, cut, 160), 1, 0.80)],   # class 1, weaker
        1: [((0, 100, 60, 160), 2, 0.95)],           # class 2, stronger
    }
    pred = ScriptedClassPredictor(script)
    tp = TiledPredictor(pred, cfg)
    img = np.zeros((H, W, 3), np.uint8)
    out1 = tp(img)
    out2 = TiledPredictor(ScriptedClassPredictor(script), cfg)(img)

    assert len(out1["instances"]) == 1
    assert int(out1["instances"].pred_classes[0]) == 2  # stronger wins
    torch.testing.assert_close(
        out1["instances"].pred_boxes.tensor,
        out2["instances"].pred_boxes.tensor,
    )
    assert int(out2["instances"].pred_classes[0]) == 2


def test_cross_class_tie_broken_by_class_id_not_random():
    H, W = 400, 700
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.25)
    tiles = plan_tiles(H, W, cfg)
    cut = tiles[0].x1
    # identical scores -> class id is the next deterministic tie-break
    script = {
        0: [((cut - 60, 100, cut, 160), 5, 0.90)],
        1: [((0, 100, 60, 160), 3, 0.90)],
    }
    out = TiledPredictor(ScriptedClassPredictor(script), cfg)(
        np.zeros((H, W, 3), np.uint8)
    )
    assert len(out["instances"]) == 1
    assert int(out["instances"].pred_classes[0]) == 3


def test_full_resolution_12000x9000_runs_and_merges_seam_car():
    # The headline case at true resolution, built lazily as a memmap-style
    # array so we never materialise 9000x12000x3 (~324 MB) up front.
    H, W = 9000, 12000
    cfg = TilingConfig(tile_size=1300, overlap_ratio=0.2, max_tiles=256)
    tiles = plan_tiles(H, W, cfg, "aerial_full")
    assert len(tiles) <= 256
    # A car placed across the first vertical cut (tile 0 right edge).
    cut = tiles[0].x1
    car = (cut - 25, 4000, cut + 25, 4050)

    # Drive the merger directly with fully-specified tile outputs, which is
    # the exact contract TiledPredictor feeds it. This avoids materialising a
    # 324 MB image while still exercising true-resolution geometry.
    tile_preds = []
    for t in tiles:
        inst = Instances((t.height, t.width))
        local_boxes, scores, classes = [], [], []
        # overlap of car with this tile
        ix0, iy0 = max(car[0], t.x0), max(car[1], t.y0)
        ix1, iy1 = min(car[2], t.x1), min(car[3], t.y1)
        if ix0 < ix1 and iy0 < iy1:
            local_boxes.append([ix0 - t.x0, iy0 - t.y0, ix1 - t.x0, iy1 - t.y0])
            scores.append(0.9)
            classes.append(0)
        inst.pred_boxes = Boxes(
            torch.tensor(local_boxes, dtype=torch.float32)
            if local_boxes
            else torch.zeros(0, 4)
        )
        inst.scores = torch.tensor(scores, dtype=torch.float32)
        inst.pred_classes = torch.tensor(classes, dtype=torch.long)
        tile_preds.append((t, {"instances": inst}))

    merged = merge_instances(tile_preds, H, W, cfg)
    inst = merged["instances"]
    assert len(inst) == 1, "seam car must be exactly one detection"
    box = inst.pred_boxes.tensor[0].tolist()
    assert box == pytest.approx(list(car), abs=1.0)
    assert box[0] >= 0 and box[2] <= W and box[1] >= 0 and box[3] <= H
    assert merged["_tiled"]["num_tiles"] == len(tiles)

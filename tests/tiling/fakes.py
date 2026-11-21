# Copyright (c) Facebook, Inc. and its affiliates.
"""Shared synthetic helpers: a deterministic 'model' for tiled inference tests."""

import numpy as np
import torch

from detectron2.structures import Boxes, Instances
from detectron2.structures.masks import BitMasks


def draw_squares(canvas, squares, value=255):
    """Draw filled squares: list of (x0, y0, x1, y1) onto an HxW uint8 canvas."""
    for x0, y0, x1, y1 in squares:
        canvas[y0:y1, x0:x1] = value
    return canvas


class FakeInstancePredictor:
    """Detects bright filled rectangles inside the tile it receives.

    Output is a standard detectron2 ``{"instances": Instances}`` dict with
    coordinates expressed in *tile-local* pixels — exactly the contract a
    real model satisfies. Optionally emits masks and keypoints.
    """

    def __init__(self, with_masks=False, with_keypoints=False, class_id=0, score=0.9):
        self.with_masks = with_masks
        self.with_keypoints = with_keypoints
        self.class_id = class_id
        self.score = score
        self.calls = []

    def __call__(self, image: np.ndarray):
        self.calls.append(image.shape)
        if image.ndim == 3:
            gray = image[:, :, 0]
        else:
            gray = image
        h, w = gray.shape
        binary = gray > 127
        # Connected components via simple flood fill (tiny test images only).
        seen = np.zeros_like(binary, dtype=bool)
        boxes, masks, kpts = [], [], []
        for y in range(h):
            for x in range(w):
                if binary[y, x] and not seen[y, x]:
                    stack = [(x, y)]
                    xs, ys = [], []
                    while stack:
                        cx, cy = stack.pop()
                        if not (0 <= cx < w and 0 <= cy < h):
                            continue
                        if seen[cy, cx] or not binary[cy, cx]:
                            continue
                        seen[cy, cx] = True
                        xs.append(cx)
                        ys.append(cy)
                        stack.extend(
                            [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]
                        )
                    x0, x1 = min(xs), max(xs) + 1
                    y0, y1 = min(ys), max(ys) + 1
                    boxes.append([x0, y0, x1, y1])
                    if self.with_masks:
                        # Only the pixels actually present inside the tile,
                        # like a real mask head's tile-local output.
                        m = np.zeros((h, w), dtype=bool)
                        m[y0:y1, x0:x1] = binary[y0:y1, x0:x1]
                        masks.append(m)
                    if self.with_keypoints:
                        # center keypoint, confident
                        kpts.append([[(x0 + x1) / 2.0, (y0 + y1) / 2.0, 2.0]])

        n = len(boxes)
        inst = Instances((h, w))
        inst.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32) if n else torch.zeros(0, 4))
        inst.scores = torch.full((n,), self.score)
        inst.pred_classes = torch.full((n,), self.class_id, dtype=torch.long)
        if self.with_masks:
            if n:
                inst.pred_masks = BitMasks(np.stack(masks))
            else:
                inst.pred_masks = BitMasks(torch.zeros(0, h, w, dtype=torch.bool))
        if self.with_keypoints:
            if n:
                inst.pred_keypoints = torch.tensor(kpts, dtype=torch.float32)
            else:
                inst.pred_keypoints = torch.zeros(0, 1, 3)
        return {"instances": inst}


class FailingOncePredictor:
    """Wraps a predictor; the tile at ``fail_index`` raises once."""

    def __init__(self, inner, fail_index):
        self.inner = inner
        self.fail_index = fail_index
        self._seen = 0

    def __call__(self, image):
        idx = self._seen
        self._seen += 1
        if idx == self.fail_index:
            raise RuntimeError("simulated tile OOM")
        return self.inner(image)


class FakeSemSegPredictor:
    """Returns class scores where bright pixels -> class 1, else class 0."""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def __call__(self, image):
        gray = image[:, :, 0] if image.ndim == 3 else image
        h, w = gray.shape
        scores = torch.zeros(self.num_classes, h, w)
        bright = torch.from_numpy((gray > 127).astype(np.float32))
        scores[0] = 1.0 - bright
        scores[1] = bright
        return {"sem_seg": scores}


# Module-level so multiprocessing 'spawn' workers can pickle the factory.
def make_tiled_predictor():
    from detectron2.tiling import TiledPredictor, TilingConfig

    cfg = TilingConfig(tile_size=120, overlap_ratio=0.25)
    return TiledPredictor(FakeInstancePredictor(), cfg)

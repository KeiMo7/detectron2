# Copyright (c) Facebook, Inc. and its affiliates.
"""Command-line entry point for tiled inference on large images.

Examples
--------
Single huge image, explicit tile side::

    python -m detectron2.tiling.cli \\
        --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \\
        --weights model.pkl --input aerial_12000x9000.png --output out/ \\
        --tile-size 1300 --overlap 0.2

Budget mode (derive tile side from a pixel budget) over a whole directory::

    python -m detectron2.tiling.cli --config-file ... --weights ... \\
        --input images/ --output out/ \\
        --max-pixels-per-tile 1690000 --max-tiles 256 --num-workers 2

The standard whole-image path remains the default unless tiling parameters are
given; an image that fits inside one tile is routed through the original
predictor unchanged.
"""

import argparse
import json
import os
from typing import Any, Dict, List

from .config import TilingConfig
from .predictor import TiledPredictor
from .reader import LargeImageReader
from .runner import run_directory, run_image


def _build_base_predictor(config_file: str, weights: str, opts: List[str]):
    # Imported lazily so the tiling package itself has no heavy/training deps.
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    if weights:
        cfg.MODEL.WEIGHTS = weights
    cfg.freeze()
    return DefaultPredictor(cfg), cfg


def _make_config(args) -> TilingConfig:
    return TilingConfig(
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        max_tiles=args.max_tiles,
        max_pixels_per_tile=args.max_pixels_per_tile,
        vram_budget_bytes=args.vram_budget_bytes,
        merge_iou_threshold=args.merge_iou,
        merge_ioa_threshold=args.merge_ioa,
        min_score=args.min_score,
        detections_per_image=args.detections_per_image,
        on_tile_error="raise",
    )


def _serialize(predictions: Dict[str, Any], height: int, width: int) -> Dict[str, Any]:
    """Project predictions to JSON-safe structures in original-image coords."""
    out: Dict[str, Any] = {"height": height, "width": width}
    if "instances" in predictions:
        inst = predictions["instances"].to("cpu")
        d: Dict[str, Any] = {
            "pred_boxes": inst.pred_boxes.tensor.tolist(),
            "scores": inst.scores.tolist(),
            "pred_classes": inst.pred_classes.tolist(),
        }
        if inst.has("pred_masks"):
            d["pred_masks_rle_shape"] = list(inst.pred_masks.tensor.shape)
        out["instances"] = d
    if "sem_seg" in predictions:
        out["sem_seg_shape"] = list(predictions["sem_seg"].shape)
        out["sem_seg_labels_shape"] = list(
            predictions["sem_seg"].argmax(dim=0).shape
        )
    if "_tiled" in predictions:
        out["tiling"] = predictions["_tiled"]
    return out


def main(argv: List[str] = None) -> None:
    ap = argparse.ArgumentParser(description="Tiled inference for large images")
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--output", required=True, help="output directory for JSON")
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--max-tiles", type=int, default=256)
    ap.add_argument("--max-pixels-per-tile", type=int, default=None)
    ap.add_argument("--vram-budget-bytes", type=int, default=None)
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--merge-ioa", type=float, default=0.3)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--detections-per-image", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="CfgNode key-value pairs")
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    _, cfg = _build_base_predictor(args.config_file, args.weights, args.opts)
    tiling_cfg = _make_config(args)

    def factory() -> TiledPredictor:
        base, _ = _build_base_predictor(
            args.config_file, args.weights, args.opts
        )
        return TiledPredictor(base, tiling_cfg)

    if os.path.isfile(args.input):
        reader = LargeImageReader(args.input)
        try:
            h, w = reader.height, reader.width
            preds = run_image(factory(), reader, image_id=os.path.basename(args.input))
        finally:
            reader.close()
        results = [(os.path.basename(args.input), _serialize(preds, h, w))]
    else:
        batch = run_directory(factory, args.input, num_workers=args.num_workers)
        results = []
        for r in batch:
            if not r.ok:
                raise RuntimeError(f"{r.image_id} failed: {r.error}")
            with LargeImageReader(r.path) as rd:
                h, w = rd.height, rd.width
            results.append((r.image_id, _serialize(r.predictions, h, w)))

    for name, payload in results:
        out_path = os.path.join(args.output, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f"wrote {out_path}: {len(payload.get('instances', {}).get('pred_boxes', []))} objects")


if __name__ == "__main__":
    main()

# Copyright (c) Facebook, Inc. and its affiliates.
"""Deterministic merging of per-tile predictions back into one image.

Design rules enforced here:

* every coordinate is translated into original-image space and clipped to the
  image bounds, so output can never exceed the source image;
* seam-crossing objects are *merged*, not double counted. Grouping is
  class-aware greedy NMS with a fully ordered tie-break, hence reproducible;
* when the same seam object was labelled with different classes by adjacent
  tiles, the winner is decided deterministically (highest score, then class
  id, then earliest tile/instance index) — never at random;
* semantic segmentation accumulates class scores over every contributing
  tile (weighted per pixel), so overlap bands show neither gaps nor seams;
* unknown per-instance fields raise unless explicitly allowed, so nothing is
  silently dropped.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .config import TilingConfig, TileInferenceError
from .geometry import Tile

# Fields the merger knows how to reproject / merge. Anything else on an
# Instances object triggers an explicit error unless disabled by config.
_KNOWN_INSTANCE_FIELDS = {
    "pred_boxes",
    "scores",
    "pred_classes",
    "pred_masks",
    "pred_keypoints",
    "pred_keypoint_heatmaps",
}


# --------------------------------------------------------------------------
# Geometry helpers (pure torch, no torchvision dependency)
# --------------------------------------------------------------------------


def _boxes_to_xyxy(boxes: Any) -> torch.Tensor:
    if hasattr(boxes, "tensor"):  # detectron2 Boxes / ROIMasks
        return boxes.tensor.detach().float().clone()
    return torch.as_tensor(boxes, dtype=torch.float32).clone()


def _clip_boxes(boxes_xyxy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, width)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, height)
    return boxes_xyxy


def _pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for Nx4 XYXY boxes (vectorised, CPU-safe)."""
    area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    lt = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area[:, None] + area[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _pairwise_ioa(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise intersection / area-of-the-smaller-box for Nx4 XYXY boxes.

    This is the seam-split criterion: two halves of one object cut by a tile
    boundary barely overlap relative to their union (low IoU) but the smaller
    is almost fully contained in the other's extent (high IoA).
    """
    area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    lt = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    smaller = torch.minimum(area[:, None], area[None, :])
    return torch.where(smaller > 0, inter / smaller, torch.zeros_like(inter))


def _greedy_class_aware_groups(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
    ioa_threshold: float,
) -> List[List[int]]:
    """Greedy grouping, class aware, with a total deterministic order.

    Two boxes join when they overlap strongly by *either* IoU or IoA. The
    ordering key is ``(-score, class_id, row_index)``: every term is defined,
    so identical scores never introduce nondeterminism.
    """
    n = boxes.shape[0]
    if n == 0:
        return []
    order = sorted(
        range(n), key=lambda i: (-float(scores[i]), int(classes[i]), i)
    )
    suppressed = torch.zeros(n, dtype=torch.bool)
    groups: List[List[int]] = []
    iou_full = _pairwise_iou(boxes)
    ioa_full = _pairwise_ioa(boxes)

    for anchor in order:
        if suppressed[anchor]:
            continue
        group = [anchor]
        suppressed[anchor] = True
        for j in order:
            if suppressed[j]:
                continue
            # Only merge same-class neighbours; cross-class overlap is settled
            # separately by the arbitration rule below.
            if int(classes[j]) != int(classes[anchor]):
                continue
            same_object = (
                float(iou_full[anchor, j]) > iou_threshold
                or float(ioa_full[anchor, j]) > ioa_threshold
            )
            if same_object:
                group.append(j)
                suppressed[j] = True
        groups.append(group)

    # Cross-class seam conflicts: deterministically attach the lower-priority
    # detection to the higher-priority group instead of emitting two objects.
    groups.sort(key=lambda g: (-float(scores[g[0]]), int(classes[g[0]]), g[0]))
    remaining = [g for g in groups]
    merged_groups: List[List[int]] = []
    assigned = torch.zeros(n, dtype=torch.bool)
    for g in remaining:
        members = [i for i in g if not assigned[i]]
        if not members:
            continue
        anchor = members[0]
        assigned[anchor] = True
        for j in range(n):
            if assigned[j] or int(classes[j]) == int(classes[anchor]):
                continue
            same_object = (
                float(iou_full[anchor, j]) > iou_threshold
                or float(ioa_full[anchor, j]) > ioa_threshold
            )
            if same_object:
                members.append(j)
                assigned[j] = True
        merged_groups.append(members)
    return merged_groups


# --------------------------------------------------------------------------
# Per-tile reproject helpers
# --------------------------------------------------------------------------


def _mask_to_bool_tensor(pred_masks: Any) -> torch.Tensor:
    """Normalise the many mask representations to an NxHxW bool tensor."""
    if hasattr(pred_masks, "tensor"):  # BitMasks
        return pred_masks.tensor.detach().to(torch.bool)
    if pred_masks.__class__.__name__ == "PolygonMasks":
        # Lazy import: pycocotools may be absent; only needed in this branch.
        h = int(pred_masks.image_size[0])
        w = int(pred_masks.image_size[1])
        from detectron2.structures.masks import BitMasks

        return BitMasks.from_polygon_masks(pred_masks, h, w).tensor
    t = torch.as_tensor(pred_masks)
    if t.dtype != torch.bool:
        t = t > 0.5
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


@dataclass
class _TileRecords:
    """All detections of one tile already expressed in full-image coords."""

    boxes: torch.Tensor  # Nx4 XYXY, full image, clipped
    scores: torch.Tensor  # N
    classes: torch.Tensor  # N
    masks: Optional[torch.Tensor]  # N x H x W full-image bool, or None
    keypoints: Optional[torch.Tensor]  # N x K x 3 full-image, or None
    extras: Dict[str, List[Any]]  # field name -> per-instance values
    tile_index: int


def _reproject_tile(
    predictions: Dict[str, Any],
    tile: Tile,
    height: int,
    width: int,
    config: TilingConfig,
) -> _TileRecords:
    if "instances" not in predictions:
        raise TileInferenceError(
            f"tile {tile.tile_index} of {tile.image_id}: tiled inference "
            "currently merges 'instances' outputs"
        )
    inst = predictions["instances"]
    fields = set(inst.get_fields().keys()) if hasattr(inst, "get_fields") else set()
    unknown = fields - _KNOWN_INSTANCE_FIELDS
    if unknown and not config.merge_unknown_fields:
        raise TileInferenceError(
            f"tile {tile.tile_index} of {tile.image_id}: output contains "
            f"fields the tiling merger cannot reproject: {sorted(unknown)}. "
            "Set TilingConfig(merge_unknown_fields=True) to drop them explicitly."
        )

    n = len(inst)
    if n == 0:
        return _TileRecords(
            boxes=torch.zeros(0, 4),
            scores=torch.zeros(0),
            classes=torch.zeros(0, dtype=torch.long),
            masks=None,
            keypoints=None,
            extras={},
            tile_index=tile.tile_index,
        )

    # A model's boxes are expressed in its own input (the tile) and must lie
    # inside it. Clip to the tile bounds before translating to full-image
    # coordinates so a box can never claim pixels its tile never saw.
    boxes = _boxes_to_xyxy(inst.pred_boxes)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, tile.width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, tile.height)
    boxes[:, 0::2] += tile.x0
    boxes[:, 1::2] += tile.y0
    boxes = _clip_boxes(boxes, height, width)
    scores = inst.scores.detach().float().cpu()
    classes = inst.pred_classes.detach().long().cpu()

    full_masks: Optional[torch.Tensor] = None
    if "pred_masks" in fields:
        tile_masks = _mask_to_bool_tensor(inst.pred_masks)
        th, tw = tile.height, tile.width
        if tile_masks.shape[-2:] != (th, tw):
            # Some heads return ROI-sized masks; resize onto the tile first.
            tile_masks = (
                torch.nn.functional.interpolate(
                    tile_masks[:, None].float(),
                    size=(th, tw),
                    mode="nearest",
                )[:, 0]
                > 0.5
            )
        full_masks = torch.zeros((n, height, width), dtype=torch.bool)
        full_masks[:, tile.y0 : tile.y1, tile.x0 : tile.x1] = tile_masks.cpu()

    full_kpts: Optional[torch.Tensor] = None
    if "pred_keypoints" in fields:
        kpts = inst.pred_keypoints.detach().float().cpu().clone()  # N x K x 3
        kpts[..., 0] += tile.x0
        kpts[..., 1] += tile.y0
        in_tile = (
            (kpts[..., 0] >= tile.x0)
            & (kpts[..., 0] < tile.x1)
            & (kpts[..., 1] >= tile.y0)
            & (kpts[..., 1] < tile.y1)
        )
        kpts[..., 2] = torch.where(in_tile, kpts[..., 2], torch.zeros_like(kpts[..., 2]))
        kpts[..., 0::2] = kpts[..., 0::2].clamp(0, width)
        kpts[..., 1::2] = kpts[..., 1::2].clamp(0, height)
        full_kpts = kpts

    extras: Dict[str, List[Any]] = {}
    for name in unknown:  # only populated when caller opted into dropping
        vals = inst.get_fields()[name]
        extras[name] = [vals[i] for i in range(n)]

    return _TileRecords(
        boxes=boxes,
        scores=scores,
        classes=classes,
        masks=full_masks,
        keypoints=full_kpts,
        extras=extras,
        tile_index=tile.tile_index,
    )


# --------------------------------------------------------------------------
# Instance merging
# --------------------------------------------------------------------------


def _union_box(rows: Sequence[int], all_boxes: torch.Tensor) -> torch.Tensor:
    sel = all_boxes[rows]
    out = torch.empty(4)
    out[0] = sel[:, 0].min()
    out[1] = sel[:, 1].min()
    out[2] = sel[:, 2].max()
    out[3] = sel[:, 3].max()
    return out


def merge_instances(
    tile_predictions: Sequence[Tuple[Tile, Dict[str, Any]]],
    height: int,
    width: int,
    config: TilingConfig,
) -> Dict[str, Any]:
    """Merge per-tile ``{"instances": ...}`` outputs into one output dict."""
    records: List[_TileRecords] = []
    for tile, pred in tile_predictions:
        rec = _reproject_tile(pred, tile, height, width, config)
        records.append(rec)

    boxes = torch.cat([r.boxes for r in records], dim=0) if records else torch.zeros(0, 4)
    scores = torch.cat([r.scores for r in records]) if records else torch.zeros(0)
    classes = torch.cat([r.classes for r in records]) if records else torch.zeros(0, dtype=torch.long)

    # Stable provenance: for each pooled global row, which record (tile) and
    # which local index inside it it came from. Built in tile order.
    record_of_row: List[int] = []
    local_of_row: List[int] = []
    for rec_idx, r in enumerate(records):
        for local in range(len(r.scores)):
            record_of_row.append(rec_idx)
            local_of_row.append(local)

    if config.min_score > 0 and len(scores):
        keep = scores >= config.min_score
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        record_of_row = [t for t, k in zip(record_of_row, keep.tolist()) if k]
        local_of_row = [t for t, k in zip(local_of_row, keep.tolist()) if k]

    groups = _greedy_class_aware_groups(
        boxes,
        scores,
        classes,
        config.merge_iou_threshold,
        config.merge_ioa_threshold,
    )

    m_boxes, m_scores, m_classes = [], [], []
    m_masks: List[torch.Tensor] = []
    m_kpts: List[torch.Tensor] = []
    have_masks = any(r.masks is not None for r in records)
    have_kpts = any(r.keypoints is not None for r in records)

    # Build mask / keypoint pools over the *same filtered global rows* as
    # boxes/scores/classes. A row only enters a pool when its tile actually
    # produced that field; global_row -> pool row is recorded explicitly so
    # the two index spaces can never drift apart.
    mask_pool_rows: List[torch.Tensor] = []
    mask_index_of_global: Dict[int, int] = {}
    kpt_pool_rows: List[torch.Tensor] = []
    kpt_index_of_global: Dict[int, int] = {}
    for global_row, (rec_idx, local) in enumerate(
        zip(record_of_row, local_of_row)
    ):
        rec = records[rec_idx]
        if have_masks and rec.masks is not None:
            mask_index_of_global[global_row] = len(mask_pool_rows)
            mask_pool_rows.append(rec.masks[local])
        if have_kpts and rec.keypoints is not None:
            kpt_index_of_global[global_row] = len(kpt_pool_rows)
            kpt_pool_rows.append(rec.keypoints[local])

    pooled_masks = torch.stack(mask_pool_rows) if mask_pool_rows else None
    pooled_kpts = torch.stack(kpt_pool_rows) if kpt_pool_rows else None

    for group in groups:
        # Deterministic winner: highest score, then class id, then global row.
        winner = min(group, key=lambda i: (-float(scores[i]), int(classes[i]), i))
        m_boxes.append(_union_box(group, boxes))
        m_scores.append(float(scores[winner]))
        m_classes.append(int(classes[winner]))

        if have_masks:
            idxs = [mask_index_of_global[i] for i in group if i in mask_index_of_global]
            if idxs:
                m_masks.append(pooled_masks[idxs].any(dim=0))
            else:
                m_masks.append(torch.zeros(height, width, dtype=torch.bool))
        if have_kpts:
            idxs = [kpt_index_of_global[i] for i in group if i in kpt_index_of_global]
            if idxs:
                # Per joint choose the candidate nearest the merged box
                # centre, breaking ties by global row index. This prefers the
                # whole-object observation over a seam-truncated one and is
                # fully deterministic even at equal confidence.
                cand = pooled_kpts[idxs]  # G x K x 3
                k = cand.shape[1]
                gx = (m_boxes[-1][0] + m_boxes[-1][2]) / 2.0
                gy = (m_boxes[-1][1] + m_boxes[-1][3]) / 2.0
                dist = (cand[..., 0] - gx) ** 2 + (cand[..., 1] - gy) ** 2
                # add a vanishing, row-index tie-breaker to the distance
                row_bias = torch.arange(len(idxs), dtype=torch.float32)[:, None] * 1e-6
                best = (dist + row_bias).argmin(dim=0)  # K
                m_kpts.append(cand[best, torch.arange(k)])
            else:
                m_kpts.append(torch.zeros_like(pooled_kpts[0]))

    out_boxes = (
        torch.stack(m_boxes) if m_boxes else torch.zeros(0, 4)
    )
    out_scores = torch.tensor(m_scores, dtype=torch.float32)
    out_classes = torch.tensor(m_classes, dtype=torch.long)

    # Final per-image cap mirrors cfg.TEST.DETECTIONS_PER_IMAGE semantics.
    if config.detections_per_image is not None and len(out_scores) > config.detections_per_image:
        cap_order = sorted(
            range(len(out_scores)),
            key=lambda i: (-out_scores[i].item(), out_classes[i].item(), i),
        )[: config.detections_per_image]
        cap_order.sort()
        out_boxes = out_boxes[cap_order]
        out_scores = out_scores[cap_order]
        out_classes = out_classes[cap_order]
        if m_masks:
            m_masks = [m_masks[i] for i in cap_order]
        if m_kpts:
            m_kpts = [m_kpts[i] for i in cap_order]

    result: Dict[str, Any] = {}
    result["_tiled"] = {
        "num_tiles": len(tile_predictions),
        "num_raw_detections": int(len(scores)),
        "num_merged_detections": int(len(out_scores)),
    }

    # Build an Instances object without importing detectron2 at module load.
    from detectron2.structures import Boxes, Instances

    inst = Instances((height, width))
    inst.pred_boxes = Boxes(out_boxes)
    inst.scores = out_scores
    inst.pred_classes = out_classes
    if have_masks:
        from detectron2.structures.masks import BitMasks

        stack = torch.stack(m_masks) if m_masks else torch.zeros(0, height, width, dtype=torch.bool)
        inst.pred_masks = BitMasks(stack)
    if have_kpts:
        from detectron2.structures import Keypoints

        inst.pred_keypoints = Keypoints(
            torch.stack(m_kpts) if m_kpts else torch.zeros(0, 1, 3)
        )
    result["instances"] = inst
    return result


# --------------------------------------------------------------------------
# Semantic segmentation merging
# --------------------------------------------------------------------------


def merge_sem_seg(
    tile_predictions: Sequence[Tuple[Tile, Dict[str, Any]]],
    height: int,
    width: int,
) -> Dict[str, Any]:
    """Stitch per-tile ``sem_seg`` (CxHxW class scores) into a full canvas.

    Every tile contributes its scores into the overlapping region; the final
    argmax is taken over the averaged scores. The output tensor is exactly
    ``C x height x width``, pixel-aligned with the source image.
    """
    accum: Optional[torch.Tensor] = None
    weight = torch.zeros(height, width, dtype=torch.float32)
    for tile, pred in tile_predictions:
        if "sem_seg" not in pred:
            continue
        s = pred["sem_seg"].detach().float().cpu()
        if s.dim() != 3:
            raise TileInferenceError(
                f"tile {tile.tile_index}: sem_seg must be CxHxW, got {tuple(s.shape)}"
            )
        c, th, tw = s.shape
        if (th, tw) != (tile.height, tile.width):
            s = torch.nn.functional.interpolate(
                s[None], size=(tile.height, tile.width), mode="bilinear", align_corners=False
            )[0]
        if accum is None:
            accum = torch.zeros(c, height, width, dtype=torch.float32)
        elif accum.shape[0] != c:
            raise TileInferenceError(
                f"tile {tile.tile_index}: sem_seg class count {c} disagrees "
                f"with earlier tiles ({accum.shape[0]})"
            )
        accum[:, tile.y0 : tile.y1, tile.x0 : tile.x1] += s
        weight[tile.y0 : tile.y1, tile.x0 : tile.x1] += 1.0

    if accum is None:
        raise TileInferenceError("no tile produced a 'sem_seg' output")
    norm = weight.clamp(min=1.0)
    return {"sem_seg": accum / norm}

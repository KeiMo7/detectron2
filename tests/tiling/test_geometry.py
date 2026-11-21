# Copyright (c) Facebook, Inc. and its affiliates.
"""Tests for tiled inference geometry, config validation and coverage."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _d2_shim import install_shims  # noqa: E402

install_shims()

from detectron2.tiling import (  # noqa: E402
    TilingConfig,
    TilingConfigError,
    axis_offsets,
    plan_tiles,
)
from detectron2.tiling.geometry import validate_coverage  # noqa: E402


# --------------------------------------------------------------------------
# axis_offsets
# --------------------------------------------------------------------------


def test_axis_smaller_than_tile_single_offset():
    assert axis_offsets(500, 1024, 0.2) == [0]


def test_axis_last_offset_aligned_to_boundary():
    offs = axis_offsets(3000, 1024, 0.2)
    assert offs[0] == 0
    assert offs[-1] + 1024 == 3000
    assert all(o + 1024 <= 3000 for o in offs)
    # monotonic, strictly increasing
    assert all(b > a for a, b in zip(offs, offs[1:]))


def test_axis_exact_multiple():
    # No-overlap-equivalent spacing when step lands exactly on the boundary:
    # 2048 = 2 * 1024, and with overlap_ratio->0 the grid is exactly two tiles.
    offs = axis_offsets(2048, 1024, 1e-6)
    assert offs[0] == 0 and offs[-1] == 1024 and len(offs) == 2
    # With real overlap a middle tile appears but the boundary still aligns.
    offs_overlap = axis_offsets(2048, 1024, 0.2)
    assert offs_overlap[0] == 0 and offs_overlap[-1] == 1024
    assert offs_overlap[-1] + 1024 == 2048


def test_axis_coverage_no_gaps_or_oob():
    length, tile, overlap = 9000, 1300, 0.25
    offs = axis_offsets(length, tile, overlap)
    covered = np.zeros(length, dtype=np.int32)
    for o in offs:
        covered[o : o + tile] += 1
    assert (covered >= 1).all(), "every pixel must be covered"
    assert covered[0] >= 1 and covered[-1] >= 1


# --------------------------------------------------------------------------
# plan_tiles grid semantics
# --------------------------------------------------------------------------


def test_plan_grid_row_major_stable_indices():
    cfg = TilingConfig(tile_size=1000, overlap_ratio=0.2)
    tiles = plan_tiles(2500, 3200, cfg, image_id="big")
    cols = len({t.col for t in tiles})
    for t in tiles:
        assert t.tile_index == t.row * cols + t.col
        assert t.image_id == "big"
        assert 0 <= t.x0 < t.x1 <= 3200
        assert 0 <= t.y0 < t.y1 <= 2500
    validate_coverage(tiles, 2500, 3200)
    # last row/col glued to boundary
    last_row = max(t.row for t in tiles)
    last_col = max(t.col for t in tiles)
    for t in tiles:
        if t.row == last_row:
            assert t.y1 == 2500
        if t.col == last_col:
            assert t.x1 == 3200


def test_plan_single_tile_for_tiny_image():
    cfg = TilingConfig(tile_size=1024, overlap_ratio=0.2)
    tiles = plan_tiles(480, 640, cfg)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.x0, t.y0, t.x1, t.y1) == (0, 0, 640, 480)


def test_plan_extreme_aspect_ratio():
    cfg = TilingConfig(tile_size=512, overlap_ratio=0.1)
    tiles = plan_tiles(200, 8000, cfg)
    assert all(t.y1 <= 200 for t in tiles)
    assert len({t.row for t in tiles}) == 1
    validate_coverage(tiles, 200, 8000)


# --------------------------------------------------------------------------
# config validation — all failures happen before inference
# --------------------------------------------------------------------------


def test_overlap_zero_rejected():
    with pytest.raises(TilingConfigError, match="overlap"):
        TilingConfig(tile_size=512, overlap_ratio=0.0)


def test_overlap_one_rejected():
    with pytest.raises(TilingConfigError, match="overlap"):
        TilingConfig(tile_size=512, overlap_ratio=1.0)


def test_negative_tile_rejected():
    with pytest.raises(TilingConfigError):
        TilingConfig(tile_size=0)


def test_tile_count_explosion_capped():
    # huge overlap on a big image -> many tiles; cap must reject it
    cfg = TilingConfig(tile_size=512, overlap_ratio=0.95, max_tiles=64)
    with pytest.raises(TilingConfigError, match="max_tiles"):
        plan_tiles(9000, 12000, cfg)


def test_explicit_tile_over_budget_rejected_at_planning():
    cfg = TilingConfig(tile_size=1024, max_pixels_per_tile=512 * 512)
    with pytest.raises(TilingConfigError, match="budget"):
        plan_tiles(3000, 3000, cfg)


def test_budget_mode_derives_side_and_fits():
    cfg = TilingConfig(max_pixels_per_tile=512 * 512, overlap_ratio=0.2)
    tiles = plan_tiles(2000, 2000, cfg)
    side = max(t.width for t in tiles)
    assert side <= 512
    validate_coverage(tiles, 2000, 2000)


def test_vram_budget_too_small_fails_readably():
    cfg = TilingConfig(vram_budget_bytes=1024, vram_bytes_per_pixel=12.0)
    with pytest.raises(TilingConfigError, match="Memory budget"):
        plan_tiles(4000, 4000, cfg)


def test_illegal_dimensions_rejected():
    cfg = TilingConfig(tile_size=512)
    with pytest.raises(TilingConfigError):
        plan_tiles(0, 100, cfg)


def test_config_is_frozen():
    cfg = TilingConfig(tile_size=512)
    with pytest.raises(Exception):
        cfg.tile_size = 256  # type: ignore[misc]


def test_deterministic_identical_plans():
    cfg = TilingConfig(tile_size=900, overlap_ratio=0.237)
    a = plan_tiles(9000, 12000, cfg, "img")
    b = plan_tiles(9000, 12000, cfg, "img")
    assert [ (t.x0, t.y0, t.x1, t.y1, t.tile_index) for t in a ] == [
        (t.x0, t.y0, t.x1, t.y1, t.tile_index) for t in b
    ]

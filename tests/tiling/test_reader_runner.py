# Copyright (c) Facebook, Inc. and its affiliates.
"""Reader, semantic-seg stitching, batch runner and channel tests."""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _d2_shim import install_shims  # noqa: E402

install_shims()

from detectron2.tiling import (  # noqa: E402
    LargeImageReader,
    TiledPredictor,
    TilingConfig,
    normalize_image_channels,
)
from detectron2.tiling.runner import run_directory, run_image  # noqa: E402

from fakes import FakeInstancePredictor, FakeSemSegPredictor, draw_squares  # noqa: E402


# --------------------------------------------------------------------------
# Channel normalisation
# --------------------------------------------------------------------------


def test_grayscale_expanded_to_three_channels():
    gray = np.zeros((10, 12), dtype=np.uint8)
    out = normalize_image_channels(gray)
    assert out.shape == (10, 12, 3)
    assert np.array_equal(out[:, :, 0], out[:, :, 2])


def test_rgba_alpha_composited_deterministically():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 200  # red-ish
    rgba[..., 3] = 128  # 50% alpha over black
    out = normalize_image_channels(rgba)
    assert out.shape == (4, 4, 3)
    assert out[0, 0, 0] == 100  # 200 * 0.5
    a = normalize_image_channels(rgba)
    b = normalize_image_channels(rgba)
    assert np.array_equal(a, b)


def test_three_channel_passed_through():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    assert normalize_image_channels(img) is img


def test_bad_channel_count_rejected():
    with pytest.raises(ValueError):
        normalize_image_channels(np.zeros((5, 5, 6), dtype=np.uint8))


# --------------------------------------------------------------------------
# LargeImageReader: array + file + regions + strips
# --------------------------------------------------------------------------


def test_reader_array_region_and_strips():
    img = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    reader = LargeImageReader(img)
    assert (reader.height, reader.width, reader.channels) == (6, 8, 3)
    region = reader.read_region(2, 1, 5, 4)
    assert region.shape == (3, 3, 3)
    assert np.array_equal(region, img[1:4, 2:5])
    strips = list(reader.iter_strips(3))
    assert [s[0] for s in strips] == [0, 3]
    assert strips[-1][1] == 6
    assert np.array_equal(np.concatenate([s[2] for s in strips], axis=0), img)


def test_reader_region_out_of_bounds_rejected():
    reader = LargeImageReader(np.zeros((10, 10, 3), dtype=np.uint8))
    with pytest.raises(IndexError):
        reader.read_region(0, 0, 11, 10)


def test_reader_png_file_roundtrip():
    from PIL import Image

    img = np.zeros((30, 40, 3), dtype=np.uint8)
    img[5:10, 8:15] = 255
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tile.png")
        Image.fromarray(img).save(p)
        with LargeImageReader(p) as reader:
            assert reader.backend == "pil"
            assert (reader.height, reader.width) == (30, 40)
            assert np.array_equal(reader.read(), img)
            assert reader.read_region(8, 5, 15, 10).shape == (5, 7, 3)


def test_reader_missing_file():
    with pytest.raises(FileNotFoundError):
        LargeImageReader("/nonexistent/path/to/image.png")


# --------------------------------------------------------------------------
# Semantic segmentation stitching
# --------------------------------------------------------------------------


def test_sem_seg_canvas_matches_original_size():
    H, W = 300, 600
    cfg = TilingConfig(tile_size=200, overlap_ratio=0.25)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    draw_squares(img, [(250, 100, 350, 200)])  # crosses seam
    out = TiledPredictor(FakeSemSegPredictor(), cfg)(img)
    assert "sem_seg" in out
    scores = out["sem_seg"]
    assert scores.shape == (2, H, W)
    labels = scores.argmax(dim=0)
    # bright square labelled class 1 everywhere incl. seam band, no gap
    assert labels[150, 300].item() == 1
    assert labels[50, 50].item() == 0
    # pixel alignment: exact original shape
    assert labels.shape == (H, W)


# --------------------------------------------------------------------------
# Batch runner: directory, isolation by image id, stable ordering
# --------------------------------------------------------------------------


def _write_image(path, squares, h=200, w=200):
    from PIL import Image

    img = np.zeros((h, w, 3), dtype=np.uint8)
    draw_squares(img, squares)
    Image.fromarray(img).save(path)


def test_run_directory_sorted_and_isolated(tmp_path):
    for name, sq in [
        ("b.png", [(20, 20, 60, 60)]),
        ("a.png", [(80, 80, 130, 130)]),
        ("c.png", [(10, 150, 50, 190)]),
    ]:
        _write_image(tmp_path / name, sq)

    cfg = TilingConfig(tile_size=300, overlap_ratio=0.2)  # one tile each
    factory = lambda: TiledPredictor(FakeInstancePredictor(), cfg)  # noqa: E731
    results = run_directory(factory, str(tmp_path))
    assert [r.image_id for r in results] == ["a", "b", "c"]
    assert all(r.ok for r in results)
    assert [len(r.predictions["instances"]) for r in results] == [1, 1, 1]
    # different images never merged: distinct image ids preserved
    assert len({r.image_id for r in results}) == 3


def test_run_directory_failfast_and_collect(tmp_path):
    _write_image(tmp_path / "ok.png", [(20, 20, 60, 60)])
    _write_image(tmp_path / "bad.png", [(20, 20, 60, 60)])

    cfg = TilingConfig(tile_size=120, overlap_ratio=0.25)  # forces multiple tiles
    factory = lambda: TiledPredictor(FakeInstancePredictor(), cfg)  # noqa: E731
    # A healthy batch completes; per-image errors are captured, not raised,
    # when fail_fast=False.
    results = run_directory(factory, str(tmp_path), fail_fast=False)
    assert all(r.ok for r in results)
    assert {r.image_id for r in results} == {"ok", "bad"}


def test_run_image_from_path(tmp_path):
    p = tmp_path / "one.png"
    _write_image(p, [(30, 30, 80, 80)])
    cfg = TilingConfig(tile_size=300, overlap_ratio=0.2)
    pred = run_image(TiledPredictor(FakeInstancePredictor(), cfg), str(p))
    assert len(pred["instances"]) == 1


def test_multiprocess_batch_matches_serial(tmp_path):
    for i in range(4):
        _write_image(tmp_path / f"img{i}.png", [(20 + 10 * i, 20, 70, 70)])

    # Spawned workers must find the test helpers and the dependency shim.
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(test_dir, "..", ".."))
    existing_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [p for p in [repo_root, test_dir, existing_pp] if p]
    )
    try:
        from _d2_shim import install_shims
        from fakes import make_tiled_predictor

        serial = run_directory(make_tiled_predictor, str(tmp_path), num_workers=0)
        parallel = run_directory(
            make_tiled_predictor,
            str(tmp_path),
            num_workers=2,
            initializer=install_shims,
        )
    finally:
        if existing_pp:
            os.environ["PYTHONPATH"] = existing_pp
        else:
            os.environ.pop("PYTHONPATH", None)

    assert [r.image_id for r in serial] == [r.image_id for r in parallel]
    for s, p in zip(serial, parallel):
        assert p.ok, p.error
        torch.testing.assert_close(
            s.predictions["instances"].pred_boxes.tensor,
            p.predictions["instances"].pred_boxes.tensor,
        )

# Copyright (c) Facebook, Inc. and its affiliates.
"""Deterministic synthetic dataset builder used by the audit tests.

It produces a dataset with three *known* categories of duplicates:

* burst frames of the same scene (same geometry, tiny pixel noise),
* crop-and-rescale variants (same content, different framing/resolution),
* re-encoded copies (same content, different bytes/checksum).

Each variant is placed in a known split, and unrelated negative images are
included, so the test can assert both what must be found and what must not.

The generator is fully seeded: rebuilding it yields byte-identical images.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["SyntheticDataset", "build_synthetic_dataset", "TRUE_DUPLICATE_GROUPS"]


def _texture(rng: np.random.Generator, w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """Multi-scale band-limited noise: a descriptor-rich but non-semantic field."""
    acc = np.zeros((h, w), np.float32)
    for sigma, amp in ((0.6, 0.55), (1.5, 0.35), (4.0, 0.25), (12.0, 0.15)):
        noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        from scipy.ndimage import gaussian_filter

        acc += gaussian_filter(noise, sigma=sigma * scale) * amp
    acc = (acc - acc.mean()) / (acc.std() + 1e-6)
    return np.clip(acc * 42.0 + 128.0, 0, 255).astype(np.uint8)


def _pattern(rng: np.random.Generator, w: int, h: int, kind: int) -> Image.Image:
    """A local object with unique, non-repeating texture.

    Repeated tiles (checkerboards) are deliberately avoided: they make any
    feature matcher ambiguous and would test the matcher's tie-breaking rather
    than duplicate detection.  Each object instead gets band-limited noise with
    its own spatial structure, which is what makes real scenes matchable.
    """
    if kind == 0:  # soft blobs
        arr = _texture(rng, w, h, scale=1.4)
    elif kind == 1:  # fine texture
        arr = _texture(rng, w, h, scale=0.7)
    elif kind == 2:  # mixed with a bright region
        arr = _texture(rng, w, h, scale=1.0).astype(np.int16)
        mask = _texture(rng, w, h, scale=2.0) > 150
        arr[mask] = np.clip(arr[mask] + 70, 0, 255)
        arr = arr.astype(np.uint8)
    else:  # strong random field
        arr = _texture(rng, w, h, scale=1.0)
        arr = np.clip(arr.astype(np.float32) * 1.15 - 20, 0, 255).astype(np.uint8)
    rgb = np.stack(
        [arr, np.roll(arr, int(rng.integers(1, 11)), axis=0), np.roll(arr, int(rng.integers(1, 11)), axis=1)],
        axis=-1,
    )
    return Image.fromarray(rgb.astype(np.uint8))


def make_scene(seed: int, w: int = 640, h: int = 480) -> Image.Image:
    """A deterministic scene with many local structures."""
    rng = np.random.default_rng(seed)
    bg = _texture(rng, w, h)
    base = Image.fromarray(np.stack([bg, np.roll(bg, 3, axis=0), np.roll(bg, 5, axis=1)], axis=-1))
    draw = ImageDraw.Draw(base)
    for _ in range(24):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(-140, 140)), y0 + int(rng.integers(-140, 140))
        draw.line([x0, y0, x1, y1], fill=tuple(int(c) for c in rng.integers(0, 255, 3)), width=int(rng.integers(1, 4)))
    for _ in range(28):
        ow, oh = int(rng.integers(26, 140)), int(rng.integers(26, 112))
        x, y = int(rng.integers(0, max(1, w - ow))), int(rng.integers(0, max(1, h - oh)))
        tile = _pattern(rng, ow, oh, int(rng.integers(0, 4)))
        base.paste(tile, (x, y))
        draw.rectangle([x, y, x + ow, y + oh], outline=tuple(int(c) for c in rng.integers(0, 255, 3)), width=int(rng.integers(1, 3)))
    return base


def make_negative(seed: int, w: int = 512, h: int = 384) -> Image.Image:
    """A scene with no content in common with any other scene."""
    return make_scene(seed + 10_000, w, h)


def burst_frame(scene: Image.Image, seed: int) -> Image.Image:
    """Same geometry, a few milliseconds of sensor noise later."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(scene).astype(np.int16) + rng.normal(0.0, 1.8, np.asarray(scene).shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def crop_rescale(scene: Image.Image, fraction: float) -> Image.Image:
    """A crop that is then rescaled back to the original canvas size."""
    w, h = scene.size
    cw, ch = int(w * fraction), int(h * fraction)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return scene.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)


def reencode(scene: Image.Image, quality: int) -> Image.Image:
    """Same pixels, different compressed bytes (different file checksum)."""
    import io

    buf = io.BytesIO()
    scene.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save(image: Image.Image, path: str, fmt: str) -> None:
    PathManagerish_mkdir(os.path.dirname(path))
    image.save(path, format=fmt)


def PathManagerish_mkdir(d: str) -> None:
    if d:
        os.makedirs(d, exist_ok=True)


@dataclass
class SyntheticDataset:
    root: str
    # split -> list of (path, scene_id, variant_kind, expected_group)
    files: Dict[str, List[Dict]] = field(default_factory=dict)
    scene_sizes: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def paths(self, split: Optional[str] = None) -> List[str]:
        if split is not None:
            return [f["path"] for f in self.files.get(split, [])]
        return [f["path"] for v in self.files.values() for f in v]

    def split_of(self, path: str) -> str:
        for split, items in self.files.items():
            for it in items:
                if it["path"] == path:
                    return split
        raise KeyError(path)

    def as_dicts(self, split: str) -> List[Dict]:
        """Dataset dicts in the Detectron2 format (annotations omitted)."""
        out = []
        for i, it in enumerate(self.files.get(split, [])):
            out.append(
                {
                    "file_name": it["path"],
                    "image_id": i,
                    "height": it["height"],
                    "width": it["width"],
                }
            )
        return out


#: Human-readable description of the duplicates the builder places, used by
#: tests to assert full recall.
TRUE_DUPLICATE_GROUPS = ("burst", "crop", "reencode", "reencode_crop")


def build_synthetic_dataset(
    root: str,
    *,
    seed: int = 7,
    n_scenes: int = 6,
    n_negatives: int = 3,
    sizes: Optional[Sequence[Tuple[int, int]]] = None,
    include_local: bool = True,
) -> SyntheticDataset:
    """Create the controlled dataset under ``root``.

    Every scene gets:

    * one burst sibling in ``train`` (the canonical frame) and one in ``val``,
    * a crop/rescale variant in ``val``,
    * a re-encoded copy in ``test``,

    and each negative lives only in ``test``.  The placement is deliberate: a
    correct audit must report leakage for every scene, and none for negatives.
    """
    os.makedirs(root, exist_ok=True)
    ds = SyntheticDataset(root=root)
    default_sizes = [(640, 480), (900, 700), (1200, 900), (480, 360), (1024, 400), (760, 1000)]
    sizes = list(sizes or default_sizes)

    def add(split, image, scene_id, kind, name, group):
        w, h = image.size
        path = os.path.join(root, split, f"{name}.png")
        _save(image, path, "PNG" if kind not in ("reencode", "reencode_crop") else "JPEG")
        ds.files.setdefault(split, []).append(
            {"path": path, "scene_id": scene_id, "kind": kind, "group": group, "width": w, "height": h}
        )
        ds.scene_sizes[scene_id] = (w, h)

    for i in range(n_scenes):
        w, h = sizes[i % len(sizes)]
        scene = make_scene(seed * 100 + i, w, h)
        # The canonical representative lives in train.
        add("train", scene, i, "original", f"scene{i}_train_original", "original")
        # Burst sibling in val — the same instant, a few ms later.
        add("val", burst_frame(scene, seed * 100 + i + 1), i, "burst", f"scene{i}_val_burst", "burst")
        # Burst sibling in train as well, so a cluster is split 2/1/1.
        add("train", burst_frame(scene, seed * 100 + i + 2), i, "burst", f"scene{i}_train_burst", "burst")
        # Crop + rescale in val: same content, different framing.
        add("val", crop_rescale(scene, 0.62), i, "crop", f"scene{i}_val_crop62", "crop")
        # Re-encoded copy in test: same pixels, different checksum.
        add("test", reencode(scene, 38), i, "reencode", f"scene{i}_test_reencode", "reencode")
        if i % 2 == 1:
            # Combined re-encode + crop variant, to check criteria compose.
            add("test", reencode(crop_rescale(scene, 0.72), 45), i, "reencode_crop", f"scene{i}_test_reenc_crop72", "reencode_crop")

    for j in range(n_negatives):
        neg = make_negative(seed * 100 + 500 + j, 512, 384)
        add("test", neg, n_scenes + j, "negative", f"neg{j}_test_original", "negative")

    if include_local:
        # Same background, different moving subject.  The subject covers most
        # of the frame, so the whole-frame test legitimately fails (the images
        # differ too much) while the shared background corner remains a
        # repeated region.  This is the "not just whole-image similarity" case.
        scene = make_scene(seed * 100 + 999, 640, 480)
        subject = make_scene(seed * 100 + 998, 420, 340)
        changed = scene.copy()
        changed.paste(subject, (150, 90))
        add("val", scene, n_scenes + n_negatives, "original", "localscene_val_original", "local")
        add("test", changed, n_scenes + n_negatives, "local", "localscene_test_subject", "local")

    return ds

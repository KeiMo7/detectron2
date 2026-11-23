# Copyright (c) Facebook, Inc. and its affiliates.
"""Candidate generation for large datasets.

Comparing every pair of images is O(n^2) and unacceptable at 10k+ scale.  Two
complementary indexes propose only a small number of pairs, and every proposal
is then verified with the full geometric check:

1. **Rare-code index** - quantised descriptors that occur in at most
   ``rare_code_max_df`` images are indexed directly.  Images sharing several
   rare codes are proposed.  This is what finds *local* repeats and crops: the
   part that survived the crop still has rare codes.
2. **Banded min-hash index** - a compact signature of the descriptor set.
   Whole-scene repeats (bursts, re-encodes, watermarks) share many bands.

Both are deterministic: the hash families are derived from
``cfg.deterministic_seed`` and insertion order is by sorted image id.
"""

import hashlib
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from .config import AuditConfig
from .features import Features

__all__ = ["CandidateIndex", "quantized_codes"]


def _projection_matrix(bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + bits * 7919)
    return rng.normal(0.0, 1.0, (128, bits)).astype(np.float32)


_EMPTY_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _projection(bits: int, seed: int) -> np.ndarray:
    key = (bits, seed)
    if key not in _EMPTY_CACHE:
        _EMPTY_CACHE[key] = _projection_matrix(bits, seed)
    return _EMPTY_CACHE[key]


def quantized_codes(features: Features, cfg: AuditConfig, bits: int = 64) -> np.ndarray:
    """Random-hyperplane codes for the descriptors, as uint32.

    Deterministic for a fixed seed.  Root-SIFT descriptors are already
    L2-normalised, so a sign-of-projection code is a stable approximation of
    descriptor identity.
    """
    if features.count == 0:
        return np.zeros((0,), np.uint64)
    desc = features.descriptors
    proj = _projection(bits, cfg.deterministic_seed)
    signs = (desc @ proj) > 0
    weights = (np.uint64(1) << np.arange(bits, dtype=np.uint64)).astype(np.uint64)
    codes = (signs.astype(np.uint64) * weights).sum(axis=1)
    return codes.astype(np.uint64)


def _stable_priority(code: int) -> int:
    """Order-independent pseudo-random priority for a code."""
    h = hashlib.blake2b(str(code).encode("ascii"), digest_size=8).digest()
    return int.from_bytes(h, "big")


def _band_signature(codes: np.ndarray, cfg: AuditConfig) -> np.ndarray:
    """``(band_count, band_rows)`` min-hash signature over the code set."""
    nb, nr = int(cfg.band_count), int(cfg.band_rows)
    if codes.shape[0] == 0:
        return np.full((nb, nr), np.uint64(0xFFFFFFFF), np.uint64)
    rng = np.random.default_rng(cfg.deterministic_seed + 13)
    a = rng.integers(1, 2 ** 31, size=(nb * nr,), dtype=np.uint64)
    b = rng.integers(0, 2 ** 31, size=(nb * nr,), dtype=np.uint64)
    c = codes.astype(np.uint64)
    # Hash each code nr*nb times; use 32-bit multiply (no overflow warnings).
    h = (np.multiply.outer(a, c) + b[:, None]) & np.uint64(0xFFFFFFFF)
    h = h.reshape(nb, nr, codes.shape[0]).min(axis=2)
    return h


class CandidateIndex:
    """Accumulates signatures and proposes image pairs."""

    def __init__(self, cfg: AuditConfig):
        self.cfg = cfg
        self._codes: Dict[str, np.ndarray] = {}  # sample key -> uint32 codes
        self._sig: Dict[str, np.ndarray] = {}
        self._ids: List[str] = []

    def add(self, key: str, features: Features) -> None:
        codes = quantized_codes(features, self.cfg)
        self._codes[key] = codes
        self._sig[key] = _band_signature(codes, self.cfg)
        self._ids.append(key)

    def _rare_pairs(self) -> Dict[Tuple[str, str], int]:
        cfg = self.cfg
        df = self._df()
        inverse: Dict[int, List[str]] = defaultdict(list)
        hits: Dict[Tuple[str, str], int] = defaultdict(int)
        for key in self._ids:
            codes = self._codes[key]
            rare = [int(c) for c in np.unique(codes) if df[int(c)] <= cfg.rare_code_max_df]
            # Bound per-image work deterministically.  Select by a stable hash
            # priority (not by document frequency: the codes shared inside a
            # duplicate cluster are the ones we need, and they have a higher
            # df than the incidental singletons).
            if len(rare) > cfg.max_rare_codes_per_image:
                rare = sorted(rare, key=lambda c: (_stable_priority(c), c))[
                    : cfg.max_rare_codes_per_image
                ]
            for code in rare:
                inverse[code].append(key)
        for code, owners in inverse.items():
            if len(owners) < 2 or len(owners) > cfg.max_band_df * 2:
                continue
            for x, y in combinations(sorted(owners), 2):
                hits[(x, y)] += 1
        return hits

    def _band_pairs(self) -> Dict[Tuple[str, str], int]:
        cfg = self.cfg
        buckets: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
        for key in self._ids:
            sig = self._sig[key]
            for band in range(sig.shape[0]):
                buckets[(band, int(sig[band, 0]), int(sig[band, 1]))].append(key)
        hits: Dict[Tuple[str, str], int] = defaultdict(int)
        for _, owners in buckets.items():
            if len(owners) < 2 or len(owners) > cfg.max_band_df:
                continue
            for x, y in combinations(sorted(owners), 2):
                hits[(x, y)] += 1
        return hits

    def propose(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Return candidate pairs with the evidence that nominated them."""
        rare = self._rare_pairs()
        band = self._band_pairs()
        merged: Dict[Tuple[str, str], Dict[str, int]] = {}
        for pair, count in rare.items():
            merged.setdefault(pair, {"rare": 0, "band": 0})["rare"] = count
        for pair, count in band.items():
            merged.setdefault(pair, {"rare": 0, "band": 0})["band"] = count

        candidates = {}
        for pair, ev in merged.items():
            if (
                ev["rare"] >= self.cfg.min_shared_rare
                or ev["band"] >= self.cfg.min_shared_bands
            ):
                candidates[pair] = ev
        return candidates

    def rare_codes(self, key: str) -> List[int]:
        """The document-frequency-filtered code set recorded for ``key``.

        Persisted in the store so a later incremental batch can propose
        cross-batch pairs without re-reading the image.
        """
        codes = self._codes.get(key)
        if codes is None:
            return []
        df = self._df()
        rare = sorted(
            int(c) for c in np.unique(codes) if df.get(int(c), 0) <= self.cfg.rare_code_max_df
        )
        # The store only needs enough codes to *propose* cross-batch candidates.
        # Codes shared by at least two images matter most (df >= 2); singletons
        # cannot link anything on their own but may do so against a future
        # batch, so a fixed share of them is kept as well.  Selection is
        # deterministic (stable hash order), so stores are reproducible.
        cap = int(self.cfg.store_codes_per_image)
        if len(rare) > cap:
            shared = [c for c in rare if df.get(c, 0) >= 2]
            singles = [c for c in rare if df.get(c, 0) < 2]
            shared = sorted(shared, key=lambda c: _stable_priority(c))
            singles = sorted(singles, key=lambda c: _stable_priority(c))
            keep = shared[:cap]
            remaining = cap - len(keep)
            if remaining > 0:
                keep += singles[:remaining]
            rare = sorted(keep)
        return rare

    def _df(self) -> Dict[int, int]:
        df: Dict[int, int] = defaultdict(int)
        for key in self._ids:
            for code in np.unique(self._codes[key]):
                df[int(code)] += 1
        return df

    @property
    def ids(self) -> List[str]:
        """Keys added so far, in sorted order (deterministic)."""
        return sorted(self._ids)

    def stats(self) -> Dict[str, int]:
        return {"images": len(self._ids), "indexed_codes": int(sum(len(c) for c in self._codes.values()))}

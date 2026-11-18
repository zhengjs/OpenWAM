"""FAST action codec: quantize → base-256 digit stream → BPE, and its inverse
(spec_zh.md §6.4). Model-agnostic — lives on the data side because training
encodes action chunks into FAST code sequences (labels) here, and deploy
decodes generated codes back into trajectories; ``model/utils/fast_vocab``
owns only the LM-vocabulary segment those codes map onto.

Round-trip guarantee: ``decode(encode(x))`` recovers the QUANTIZED action
exactly (``quantize(dequantize(u8)) == u8`` bit-exact); float actions carry
the standard 256-bin quantization error only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["FASTCodec", "quantize_actions", "dequantize_actions", "fit_bpe"]

BASE_VOCAB = 256  # one code per byte


def quantize_actions(actions: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """[-1, 1] floats → uint8 bins (π0.5 / FAST convention)."""
    clipped = np.clip(actions, -1.0, 1.0)
    scaled = (clipped + 1.0) / 2.0 * (num_bins - 1)
    return np.round(scaled).astype(np.uint8)


def dequantize_actions(u8: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """uint8 bins → [-1, 1] floats; the exact inverse of :func:`quantize_actions`
    on the bin grid."""
    return u8.astype(np.float32) / (num_bins - 1) * 2.0 - 1.0


# ---------------------------------------------------------------- BPE core --


def _merge_all(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Left-to-right replacement of every ``pair`` occurrence with ``new_id``."""
    out: list[int] = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def fit_bpe(seqs: list[bytes], target_vocab: int) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Greedy most-frequent-pair merges until the vocab budget (256 base + merges).

    Works entirely in token-id space (bytes = ids 0..255, merges = 256+), so
    pair counting and replacement stay correct after earlier merges consume
    the raw bytes. Returns the int-pair merges plus every token id's bytes —
    the persisted ``fast_bpe_codes.txt`` format (hex byte pairs) is derived
    from those.
    """
    rows = [list(row) for row in seqs]
    merges: list[tuple[int, int]] = []
    token_bytes = {i: bytes([i]) for i in range(BASE_VOCAB)}
    next_id = BASE_VOCAB
    while next_id < target_vocab:
        counts: Counter = Counter()
        for row in rows:
            for pair in zip(row, row[1:]):
                counts[pair] += 1
        if not counts:
            break
        best, best_n = counts.most_common(1)[0]
        if best_n < 2:
            break
        merges.append(best)
        token_bytes[next_id] = token_bytes[best[0]] + token_bytes[best[1]]
        rows = [_merge_all(row, best, next_id) for row in rows]
        next_id += 1
    return merges, token_bytes


# ------------------------------------------------------------------ codec ---


class FASTCodec:
    """Byte-level BPE over quantized action chunks.

    Codes are ints: ``0..255`` = raw bytes, ``256 + i`` = the i-th learned
    merge. ``codebook_size`` is the number of LM vocabulary slots to reserve
    (``fast_vocab.add_robot_action_tokens`` budget).
    """

    def __init__(
        self,
        merges: Sequence[tuple[bytes, bytes]],
        action_dim: int,
        action_horizon: int,
        num_bins: int = 256,
    ):
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.num_bins = int(num_bins)
        self.codebook_size = BASE_VOCAB + len(merges)
        # Merge sides are single bytes or previously-learned merges — resolve
        # everything into token-id space so encode ranks match int tokens.
        self._expansions = {BASE_VOCAB + i: a + b for i, (a, b) in enumerate(merges)}
        id_of = {bytes([i]): i for i in range(BASE_VOCAB)}
        id_of.update({token: BASE_VOCAB + i for i, token in self._expansions.items()})
        self.merges = [(id_of[a], id_of[b]) for a, b in merges]
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dir(cls, fit_dir: str | Path) -> "FASTCodec":
        """Load a codec from ``fit_fast_tokenizer.py``'s output directory
        (expects ``fast_tokenizer/{config.json, fast_bpe_codes.txt}``)."""
        root = Path(fit_dir) / "fast_tokenizer"
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        merges = []
        for line in (root / "fast_bpe_codes.txt").read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            a_hex, b_hex = line.split("\t")
            merges.append((bytes.fromhex(a_hex), bytes.fromhex(b_hex)))
        return cls(
            merges,
            action_dim=config["action_dim"],
            action_horizon=config["action_horizon"],
            num_bins=config.get("num_bins", 256),
        )

    # -- encode / decode ----------------------------------------------------

    def encode(self, actions: np.ndarray) -> list[int]:
        """``[T, D]`` float actions in [-1, 1] → FAST code sequence."""
        u8 = quantize_actions(actions, self.num_bins)
        if u8.shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"FASTCodec expects actions shaped {(self.action_horizon, self.action_dim)}, got {u8.shape}"
            )
        tokens: list[int] = list(u8.tobytes())
        while True:
            best_rank = None
            for i in range(len(tokens) - 1):
                rank = self._ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
            if best_rank is None:
                break
            tokens = _merge_all(tokens, self.merges[best_rank], BASE_VOCAB + best_rank)
        return tokens

    def decode(
        self,
        codes: Sequence[int],
        action_horizon: int | None = None,
        action_dim: int | None = None,
    ) -> np.ndarray:
        """FAST code sequence → ``[T, D]`` float actions in [-1, 1].

        ``action_horizon`` / ``action_dim`` default to the fitted values; the
        byte-stream length must equal their product (a wrong-length stream is
        a hard error, never a silent truncation)."""
        horizon = self.action_horizon if action_horizon is None else int(action_horizon)
        dim = self.action_dim if action_dim is None else int(action_dim)
        parts = []
        for code in codes:
            code = int(code)
            if code < 0 or code >= self.codebook_size:
                raise ValueError(
                    f"FAST code {code} out of range [0, {self.codebook_size}) — "
                    "not produced by this codec"
                )
            parts.append(bytes([code]) if code < BASE_VOCAB else self._expansions[code])
        raw = b"".join(parts)
        expected = horizon * dim
        if len(raw) != expected:
            raise ValueError(
                f"FAST byte stream length {len(raw)} != horizon*dim {expected} — "
                "the code sequence is truncated or corrupt"
            )
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(horizon, dim)
        return dequantize_actions(u8, self.num_bins)

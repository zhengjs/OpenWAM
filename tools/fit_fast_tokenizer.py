"""FAST tokenizer fitting tool (spec_zh.md §6.4): fit a FAST BPE over a
target action space and save the mapping + codebook size for training/
deploy.

``fit_fast_tokenizer(input, --out-dir)`` where ``input`` is a ``.npy`` of
shape ``(N, T_action, D_action)`` (normalized action chunks). Writes
``fast_tokenizer/{config.json, fast_bpe_codes.txt}``.

The BPE is the physical-intelligence FAST mapping (base-256 digit encoding
of a quantized action matrix, then BPE compression); this port keeps the
dependency-light core: quantize -> digit-encode -> BPE over byte pairs. The
resulting ``codebook_size`` feeds
``vlm_backbone/fast_vocab.py::add_robot_action_tokens``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

NUM_BINS = 256  # per-dim quantization bins over [-1, 1]


def quantize_actions(actions: np.ndarray) -> np.ndarray:
    """[-1, 1] floats -> uint8 bins (π0.5 / FAST convention)."""
    clipped = np.clip(actions, -1.0, 1.0)
    scaled = (clipped + 1.0) / 2.0 * (NUM_BINS - 1)
    return np.round(scaled).astype(np.uint8)


def digit_encode(u8: np.ndarray) -> list[bytes]:
    """(N, T, D) uint8 -> base-256 digit strings (row-major flatten per
    chunk), the FAST time-minor layout that groups consecutive timesteps."""
    n = u8.shape[0]
    flat = u8.reshape(n, -1)
    return [row.tobytes() for row in flat]


def fit_bpe(seqs: list[bytes], target_vocab: int) -> list[tuple[bytes, bytes]]:
    """Greedy byte-pair merges until the vocab budget (256 base + merges)."""
    merges: list[tuple[bytes, bytes]] = []
    corpus = list(seqs)
    vocab = 256
    while vocab < target_vocab:
        counts = Counter()
        for row in corpus:
            for i in range(len(row) - 1):
                counts[(row[i:i + 1], row[i + 1:i + 2])] += 1
        if not counts:
            break
        best, best_n = counts.most_common(1)[0]
        if best_n < 2:
            break
        merges.append(best)
        vocab += 1
        corpus = [apply_merge(row, best) for row in corpus]
    return merges


def apply_merge(row: bytes, pair: tuple[bytes, bytes]) -> bytes:
    a, b = pair
    out = bytearray()
    i = 0
    while i < len(row):
        if i < len(row) - 1 and row[i:i + 1] == a and row[i + 1:i + 2] == b:
            out += a + b
            i += 2
        else:
            out += row[i:i + 1]
            i += 1
    return bytes(out)


def fit(input_path: str, out_dir: str, target_vocab: int = 1024) -> dict:
    """Fit and persist; returns the summary dict (also written as config.json)."""
    actions = np.load(input_path)
    if actions.ndim != 3:
        raise ValueError(f"expected (N, T_action, D_action), got {actions.shape}")
    u8 = quantize_actions(actions)
    seqs = digit_encode(u8)
    merges = fit_bpe(seqs, target_vocab)

    out = Path(out_dir) / "fast_tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fast_bpe_codes.txt").write_text(
        "\n".join(f"{a.hex()}\t{b.hex()}" for a, b in merges), encoding="utf-8"
    )
    config = {
        "num_bins": NUM_BINS,
        "action_dim": int(actions.shape[2]),
        "action_horizon": int(actions.shape[1]),
        "target_vocab": target_vocab,
        "codebook_size": 256 + len(merges),
        "num_merges": len(merges),
        "num_sequences": int(actions.shape[0]),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help=".npy of (N, T_action, D_action) normalized action chunks")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-vocab", type=int, default=1024)
    args = parser.parse_args()
    config = fit(args.input, args.out_dir, args.target_vocab)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

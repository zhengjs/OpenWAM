"""FAST tokenizer fitting tool (spec_zh.md §6.4): fit a FAST BPE over a
target action space and save the mapping + codebook size for training/deploy.

``fit_fast_tokenizer(input, --out-dir)`` where ``input`` is a ``.npy`` of
shape ``(N, T_action, D_action)`` (normalized action chunks). Writes
``fast_tokenizer/{config.json, fast_bpe_codes.txt}`` — loadable by
``openwam.dataloader.utils.fast_codec.FASTCodec.from_dir`` (the shared
quantize/BPE core lives there; this tool only fits and persists).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openwam.dataloader.utils.fast_codec import fit_bpe, quantize_actions


def digit_encode(u8: np.ndarray) -> list[bytes]:
    """(N, T, D) uint8 -> base-256 digit strings (row-major flatten per
    chunk), the FAST time-minor layout that groups consecutive timesteps."""
    n = u8.shape[0]
    flat = u8.reshape(n, -1)
    return [row.tobytes() for row in flat]


def fit(input_path: str, out_dir: str, target_vocab: int = 1024) -> dict:
    """Fit and persist; returns the summary dict (also written as config.json)."""
    actions = np.load(input_path)
    if actions.ndim != 3:
        raise ValueError(f"expected (N, T_action, D_action), got {actions.shape}")
    u8 = quantize_actions(actions)
    seqs = digit_encode(u8)
    merges, token_bytes = fit_bpe(seqs, target_vocab)

    out = Path(out_dir) / "fast_tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fast_bpe_codes.txt").write_text(
        "\n".join(f"{token_bytes[a].hex()}\t{token_bytes[b].hex()}" for a, b in merges),
        encoding="utf-8",
    )
    config = {
        "num_bins": 256,
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

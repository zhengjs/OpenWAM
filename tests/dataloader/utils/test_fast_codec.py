"""FAST codec round-trip (spec_zh.md §6.4): quantize → digit stream → BPE.

Pins: ``decode(encode(x))`` recovers the QUANTIZED action bit-exactly, the
codec actually compresses structured chunks (the encode merges must fire —
a no-op BPE would still roundtrip and silently defeat the point), and the
malformed-code/length/shape error paths fail loudly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openwam.dataloader.utils.fast_codec import (
    FASTCodec,
    dequantize_actions,
    fit_bpe,
    quantize_actions,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fit_dir(tmp_path: Path, corpus: np.ndarray, target_vocab: int = 1024) -> tuple[Path, dict]:
    inp = tmp_path / "actions.npy"
    np.save(inp, corpus)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "fit_fast_tokenizer.py"),
         str(inp), "--out-dir", str(tmp_path), "--target-vocab", str(target_vocab)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return tmp_path, json.loads(result.stdout)


def test_quantize_dequantize_exact_on_bin_grid():
    u8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    back = quantize_actions(dequantize_actions(u8))
    assert np.array_equal(back, u8)  # bit-exact inverse on the bin grid


def test_fit_then_roundtrip_with_compression(tmp_path):
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 0.05, (64, 8, 7)), axis=1)  # smooth -> compressible
    corpus = np.clip(base, -1, 1).astype(np.float32)
    fit_dir, config = _fit_dir(tmp_path, corpus)
    codec = FASTCodec.from_dir(fit_dir)
    assert codec.codebook_size == config["codebook_size"] == 256 + config["num_merges"]
    assert codec.action_dim == 7 and codec.action_horizon == 8

    lengths = [len(codec.encode(corpus[i])) for i in range(0, 64, 8)]
    assert sum(lengths) / len(lengths) < 56  # merges actually fire (not a no-op BPE)

    for i in range(0, 64, 8):
        codes = codec.encode(corpus[i])
        assert max(codes) < codec.codebook_size
        decoded = codec.decode(codes)
        assert np.array_equal(quantize_actions(decoded), quantize_actions(corpus[i]))
        np.testing.assert_allclose(decoded, corpus[i], atol=2.0 / 255.0)  # quantization error only


def test_degenerate_no_merge_codec_roundtrips(tmp_path):
    """``target_vocab=256`` forbids merges by construction — the codec must
    still roundtrip (identity byte stream), the degenerate path."""
    rng = np.random.default_rng(1)
    corpus = rng.uniform(-1, 1, (32, 8, 7)).astype(np.float32)
    fit_dir, config = _fit_dir(tmp_path, corpus, target_vocab=256)
    assert config["num_merges"] == 0
    codec = FASTCodec.from_dir(fit_dir)
    assert codec.codebook_size == 256
    chunk = corpus[0]
    codes = codec.encode(chunk)
    assert max(codes) < 256  # raw bytes only
    assert np.array_equal(quantize_actions(codec.decode(codes)), quantize_actions(chunk))


def test_fit_on_noise_roundtrips(tmp_path):
    """Uniform-random chunks: whatever few chance-collision merges the fitter
    learns, encode/decode stays exact."""
    rng = np.random.default_rng(1)
    corpus = rng.uniform(-1, 1, (32, 8, 7)).astype(np.float32)
    fit_dir, config = _fit_dir(tmp_path, corpus, target_vocab=512)
    codec = FASTCodec.from_dir(fit_dir)
    assert codec.codebook_size == 256 + config["num_merges"]
    for i in (0, 7, 31):
        decoded = codec.decode(codec.encode(corpus[i]))
        assert np.array_equal(quantize_actions(decoded), quantize_actions(corpus[i]))


def test_decode_rejects_bad_codes_and_lengths(tmp_path):
    rng = np.random.default_rng(2)
    corpus = np.clip(np.cumsum(rng.normal(0, 0.05, (16, 8, 7)), axis=1), -1, 1).astype(np.float32)
    fit_dir, _ = _fit_dir(tmp_path, corpus)
    codec = FASTCodec.from_dir(fit_dir)
    codes = codec.encode(corpus[0])

    with pytest.raises(ValueError, match="out of range"):
        codec.decode([codec.codebook_size])
    with pytest.raises(ValueError, match="truncated or corrupt"):
        codec.decode(codes[:-1])
    with pytest.raises(ValueError, match="expects actions shaped"):
        codec.encode(rng.uniform(-1, 1, (4, 7)).astype(np.float32))


def test_fit_bpe_counts_pairs_in_token_space():
    """After a merge consumes a byte pair, the freed neighbors must be counted
    as TOKEN pairs (the original byte-slice counting double-counted)."""
    # "aa" everywhere: merging (97,97) should leave no (97,97) pairs to recount
    seqs = [b"aaaa"] * 4
    merges, token_bytes = fit_bpe(seqs, target_vocab=258)
    assert merges[0] == (97, 97)
    assert token_bytes[256] == b"aa"
    # one round of "aa"->256 collapses each row to [256, 256]; next best merge
    # is (256, 256) — proving counting sees merged tokens, not raw bytes
    assert merges[1] == (256, 256)
    assert token_bytes[257] == b"aaaa"

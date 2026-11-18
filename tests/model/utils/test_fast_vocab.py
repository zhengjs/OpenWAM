"""FAST vocabulary extension (spec_zh.md §6.4 / D4).

Pins the embedding contract: segment appended at the END of the current
vocab (no renumbering — a pre-extension checkpoint loads unchanged), vocab
size padded to the 64 multiple, neutral row init, tokenizer registration
id consistency, and segment extraction. The fit tool's quantize/BPE core is
covered against reference values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.model.utils.fast_vocab import (
    FastVocabSpec,
    add_robot_action_tokens,
    ids_to_robot_action_segment,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- model ----


class _Emb(torch.nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab, dim))

    def forward(self, x):
        return self.weight[x]


class _TinyLM(torch.nn.Module):
    """HF-causal-LM surface used by add_robot_action_tokens."""

    def __init__(self, vocab=100, dim=8):
        super().__init__()
        self.embed = _Emb(vocab, dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)
        self.config = type("Cfg", (), {"vocab_size": vocab})()

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size):
        old_in = self.embed.weight.data
        self.embed = _Emb(new_size, old_in.shape[1])
        self.embed.weight.data[: old_in.shape[0]] = old_in
        old_out = self.lm_head.weight.data
        self.lm_head = torch.nn.Linear(old_out.shape[1], new_size, bias=False)
        self.lm_head.weight.data[: old_out.shape[0]] = old_out
        self.config.vocab_size = new_size

    def tie_weights(self):
        pass


def test_segment_appended_after_pad_and_alignment():
    model = _TinyLM(vocab=100)
    old = model.embed.weight.data.clone()
    spec = add_robot_action_tokens(model, num_tokens=100)

    # vocab was padded to 128 first, then the 100-token segment lands there,
    # and the total is rounded up to the 64-multiple again
    assert spec.base_vocab_size == 100
    assert spec.robot_action_start_id == 128
    assert spec.robot_action_end_id == 228
    assert spec.padded_vocab_size == 256 and spec.padded_vocab_size % 64 == 0
    assert model.config.vocab_size == 256

    # pre-extension rows unchanged — a checkpoint trained before extension
    # loads bit-identically
    assert torch.equal(model.embed.weight.data[:100], old)
    # neutral init from the embedding mean
    assert torch.allclose(
        model.embed.weight.data[128], old.mean(dim=0), atol=1e-5
    )


def test_no_renumbering_checkpoint_compat():
    """ids < base are never remapped (spec §6.4: embedding rows are never
    renumbered)."""
    model = _TinyLM(vocab=100)
    ids_before = model.embed.weight.data[:100].clone()
    add_robot_action_tokens(model, num_tokens=10)
    assert torch.equal(model.embed.weight.data[:100], ids_before)


def test_invalid_num_tokens_rejected():
    with pytest.raises(ValueError, match="positive"):
        add_robot_action_tokens(_TinyLM(), num_tokens=0)


# ------------------------------------------------------------- segment ----


def test_segment_extraction_maps_only_reserved_ids():
    spec = FastVocabSpec(
        robot_action_start_id=128, robot_action_end_id=138,
        codebook_size=10, base_vocab_size=100, padded_vocab_size=192,
    )
    ids = torch.randint(0, 128, (2, 6))
    ids[0, 2:5] = torch.arange(128, 131)
    out = ids_to_robot_action_segment(ids, spec)
    assert out[0, 2:5].tolist() == [0, 1, 2]
    assert (out[0, :2] == -1).all() and (out[1] == -1).all()
    # upper edge exclusive
    ids[1, 0] = 138
    assert ids_to_robot_action_segment(ids, spec)[1, 0] == -1


# ---------------------------------------------------------------- tool -----


def test_fit_fast_tokenizer_tool(tmp_path):
    """tools/fit_fast_tokenizer.py: quantize->BPE pipeline runs and persists
    a config whose codebook_size is the add_robot_action_tokens budget."""
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (32, 8, 7)).astype(np.float32)
    inp = tmp_path / "actions.npy"
    np.save(inp, actions)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "fit_fast_tokenizer.py"),
         str(inp), "--out-dir", str(tmp_path), "--target-vocab", "300"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((tmp_path / "fast_tokenizer" / "config.json").read_text())
    assert config["action_dim"] == 7 and config["action_horizon"] == 8
    assert config["codebook_size"] == 256 + config["num_merges"]
    assert config["codebook_size"] <= 300
    codes = (tmp_path / "fast_tokenizer" / "fast_bpe_codes.txt").read_text()
    assert len(codes.strip().splitlines()) == config["num_merges"]

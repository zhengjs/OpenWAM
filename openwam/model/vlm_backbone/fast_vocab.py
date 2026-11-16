"""FAST vocabulary extension for Qwen-family VLM tokenizers (spec_zh.md §6.4).

Port of starVLA ``add_qwen_special_tokens``: reserves a contiguous
``<robot_action_00> ... <robot_action_NN>`` id segment inside the LM's
embedding table, pads the vocab size to a multiple of 64 (tensor-core
friendly), and resizes the embedding + LM head accordingly.

Design (D4 "vocabulary embedded in the model"):
- The FAST BPE itself lives OUTSIDE the LM vocabulary — an action chunk is
  BPE-encoded into integer codes which map onto the reserved segment ids;
  the LM never needs its text vocabulary retrained.
- The segment is appended at the END of the current vocab (embedding rows
  are never renumbered — a checkpoint trained before extension loads
  unchanged).
- New rows initialize from the embedding mean (starVLA convention): the LM
  head logits for unseen tokens start neutral rather than random-extreme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

VOCAB_SIZE_MULTIPLE = 64  # pad to this multiple (starVLA add_qwen_special_tokens)


@dataclass
class FastVocabSpec:
    """Result of a successful extension; persisted next to the checkpoint so
    deploy can recover the segment bounds without re-inspecting the model."""

    robot_action_start_id: int  # first reserved id (inclusive)
    robot_action_end_id: int  # one past the last reserved id
    codebook_size: int  # number of reserved slots = FAST BPE vocab size
    base_vocab_size: int  # LM text vocab before extension
    padded_vocab_size: int  # embedding rows after padding

    @property
    def segment_size(self) -> int:
        return self.robot_action_end_id - self.robot_action_start_id


def add_robot_action_tokens(model, num_tokens: int) -> FastVocabSpec:
    """Extend ``model``'s tokenizer-agnostic embedding table with a
    ``<robot_action_xx>`` segment.

    ``model`` is a HF causal-LM (``Qwen3VLForConditionalGeneration`` or any
    module exposing ``resize_token_embeddings`` + ``config``). Call BEFORE
    loading the tokenizer-side additions: the returned spec's ids must be
    registered on the tokenizer with ``add_tokens(AddedToken(...))`` so the
    text ``<robot_action_07>`` maps to the reserved id.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    config = model.config
    text_config = getattr(config, "text_config", config)
    base_vocab = int(getattr(text_config, "vocab_size", config.vocab_size))
    start = base_vocab

    # 1. Pad the CURRENT vocab to the multiple FIRST (resize is cheap while
    #    no new tokens are pending), then reserve the segment on top.
    padded = -(-base_vocab // VOCAB_SIZE_MULTIPLE) * VOCAB_SIZE_MULTIPLE
    pad_tokens = padded - base_vocab
    # 2. Round the TOTAL (pad + segment) up to the multiple as well, so the
    #    embedding matrix keeps the alignment invariant.
    total = -(-(base_vocab + pad_tokens + num_tokens) // VOCAB_SIZE_MULTIPLE) * VOCAB_SIZE_MULTIPLE
    add_count = total - base_vocab

    old_embedding = model.get_input_embeddings().weight.data
    init = old_embedding.mean(dim=0)  # neutral init for unseen rows

    model.resize_token_embeddings(base_vocab + add_count)
    text_config.vocab_size = base_vocab + add_count
    if hasattr(config, "vocab_size"):
        config.vocab_size = base_vocab + add_count
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    new_embedding = model.get_input_embeddings().weight.data
    new_embedding[base_vocab:] = init
    # Also neutral-init the (possibly tied) LM head rows for unseen tokens.
    if model.get_output_embeddings() is not None:
        out = model.get_output_embeddings()
        if out.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
            out.weight.data[base_vocab:] = 0.0

    spec = FastVocabSpec(
        robot_action_start_id=base_vocab + pad_tokens,
        robot_action_end_id=base_vocab + pad_tokens + num_tokens,
        codebook_size=num_tokens,
        base_vocab_size=base_vocab,
        padded_vocab_size=base_vocab + add_count,
    )
    logger.info(
        "FAST vocab extension: base=%d pad=%d segment=[%d, %d) size=%d padded=%d",
        base_vocab, pad_tokens, spec.robot_action_start_id, spec.robot_action_end_id,
        num_tokens, spec.padded_vocab_size,
    )
    return spec


def register_robot_action_tokens(tokenizer, spec: FastVocabSpec):
    """Register the segment on a tokenizer so ``<robot_action_NN>`` text maps
    into the reserved ids. Returns the added tokens (id == index in spec)."""
    from transformers import AddedToken

    tokens = [
        AddedToken(f"<robot_action_{i:02d}>", single_word=False, normalized=False)
        for i in range(spec.codebook_size)
    ]
    added = tokenizer.add_tokens(tokens)
    expected = spec.robot_action_start_id
    first = tokenizer.convert_tokens_to_ids("<robot_action_00>")
    if first != expected:
        raise ValueError(
            f"tokenizer assigned <robot_action_00> id {first}, expected {expected} — "
            "extend the MODEL embeddings before registering tokenizer tokens"
        )
    assert added == spec.codebook_size
    return tokens


def ids_to_robot_action_segment(input_ids: torch.Tensor, spec: FastVocabSpec) -> torch.Tensor:
    """Extract the FAST segment ids per sample (deploy: ``model.generate`` ->
    segment -> FAST decode). Non-segment positions map to -1."""
    lo, hi = spec.robot_action_start_id, spec.robot_action_end_id
    mask = (input_ids >= lo) & (input_ids < hi)
    out = torch.where(mask, input_ids - lo, torch.full_like(input_ids, -1))
    return out

"""VLA prompt tokenization processor (spec_data_zh.md §5).

Builds the language half of the context: camera-token prefix + optional state
readout (``EgoStateProcessor.state2str``) + navigation instruction (+ optional
chain-of-thought text) → ``input_ids`` / ``attention_mask`` / ``vlm_labels``.

Supervision convention: when ``cot_text`` is present, the prefix+instruction
tokens are masked to ``-100`` and only the CoT span is supervised; without
``cot_text`` the full sequence is supervised (world-model/VLA prompt loss).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.preprocessors.base import BasePreprocessor

__all__ = ["VLAPromptTokenizeProcessor"]

_LABEL_IGNORE = -100


class VLAPromptTokenizeProcessor(BasePreprocessor):
    """``nav_instruction`` (+ state readout / camera tokens / cot_text) → token ids.

    Args:
        tokenizer: HF-style tokenizer (``tokenizer(text)["input_ids"]``).
        camera_tokens: optional token string(s) prepended as visual placeholders
            (expansion order is the configured camera order).
        use_state2str: prepend an ``EgoStateProcessor.state2str`` readout of
            ``ctx.ego_state_tensor`` (requires that processor to run first).
        state2str_fn: override for the state readout callable.
    """

    def __init__(
        self,
        tokenizer,
        camera_tokens: Optional[str] = None,
        use_state2str: bool = False,
        state2str_fn=None,
    ):
        self.tokenizer = tokenizer
        self.camera_tokens = camera_tokens
        self.use_state2str = use_state2str
        self.state2str_fn = state2str_fn

    def _build_prompt(self, ctx: PipelineContext) -> str:
        parts: List[str] = []
        if self.camera_tokens:
            parts.append(self.camera_tokens)
        if self.use_state2str:
            if self.state2str_fn is None:
                raise ValueError("vla_prompt: use_state2str=True requires state2str_fn")
            if ctx.ego_state_tensor is not None:
                parts.append(self.state2str_fn(ctx.ego_state_tensor.numpy()))
        parts.append(ctx.nav_instruction)
        return " ".join(p for p in parts if p)

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.nav_instruction == "" and not self.camera_tokens:
            return ctx  # pure world-model sample: no language path
        prompt = self._build_prompt(ctx)
        encoded = self.tokenizer(prompt)
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)

        if ctx.cot_text:
            cot_ids = torch.as_tensor(
                self.tokenizer(" " + ctx.cot_text)["input_ids"], dtype=torch.long
            )
            input_ids = torch.cat([input_ids, cot_ids], dim=0)
            labels = input_ids.clone()
            labels[: len(encoded["input_ids"])] = _LABEL_IGNORE  # supervise CoT only
        else:
            labels = input_ids.clone()  # supervise the full prompt

        ctx.prompt = None  # token path replaces the raw-string path
        ctx.input_ids = input_ids
        ctx.attention_mask = torch.ones_like(input_ids)
        ctx.vlm_labels = labels
        return ctx

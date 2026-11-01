# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
#
# Adapted from starVLA (starVLA/model/framework/VLM4A/QwenOFT.py, itself
# inspired by OpenVLA-OFT), modified for OpenWAM: :class:`BaseVLAArchitecture`
# template + the OpenWAM registry / config system; the cross-package image
# tools and the hardcoded autocast are dropped.

"""QwenOFT: action special tokens -> L1 regression over their hidden states.

A lightweight VLA that injects repeated action special tokens into the VLM
prompt and parallelly predicts continuous actions conditioned on multi-view
images plus a language instruction (shares parameters with the VLM). The
action-token hidden states feed an MLP regression head (L1 loss).
"""


import torch
from torch import Tensor

from openwam.model.action_backbone.vla_heads import build_vla_action_head
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.vla.base_vla import BaseVLAArchitecture

# No spaces between repeated tokens, otherwise they are tokenized to multiple tokens.
ACTION_TOKEN = "🔍"


@register_architecture(
    "vla_qwen_oft",
    status="supported",
    note="VLA QwenOFT: action-token hidden states -> MLP L1 regression (no diffusion).",
    framework="vla",
    variant="oftp",
)
class QwenOFTArchitecture(BaseVLAArchitecture):
    """Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen3-VL backbone for fused language/vision token embeddings
      - Action special tokens injected into the VLM prompt
      - MLP regression head over action-token hidden states (L1 loss)
    """

    DEFAULTS = {
        "action_head": {
            "action_model_type": "MLP",
            "action_dim": 7,
            # Hidden dim for the action MLP — overridden at runtime with the
            # VLM hidden size (the head consumes the VLM hidden states).
            "action_hidden_dim": 2560,
            "action_horizon": 8,
        },
        "repeated_diffusion_steps": 1,
    }
    # Discretised proprio state rides the instruction as text tokens (π₀.5
    # style), not the head's state encoder.
    _STATE_IN_INSTRUCTION = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is None:
            return
        processor = getattr(self.vlm_backbone, "processor", None)
        if processor is None:
            raise RuntimeError("QwenOFTArchitecture requires the VLM backbone to expose a processor.")
        self.action_token = ACTION_TOKEN
        self.action_token_id = int(
            processor.tokenizer(self.action_token, add_special_tokens=False)["input_ids"][0]
        )

    def _init_action_head(self, cfg):
        # The head consumes the VLM hidden states directly: input dim = VLM
        # hidden size (the DEFAULTS value is a documentation placeholder).
        model_type = self._expect_action_model_type(cfg, ("MLP",), "MLP")
        head_cfg = self._cfg_get(cfg, "action_head", None)
        return build_vla_action_head(
            head_cfg, action_model_type=model_type, action_hidden_dim=self.vlm_backbone.hidden_size
        )

    def _build_prompt_suffix(self) -> str:
        action_tokens = self.action_token * self.action_horizon
        return f" Please predict the next {self.action_horizon} robot actions: <action>{action_tokens}<action>."

    def _forward_loss(self, examples) -> Tensor:
        hidden, _, vlm_inputs = self._encode(examples)
        input_ids = vlm_inputs.get("input_ids")
        action_queries = self._gather_action_token_embeddings(hidden, input_ids, self.action_token_id)
        actions = self._stack_actions(examples, self.action_horizon)
        action_mask = self._stack_action_masks(examples, self.action_horizon)
        return self.action_backbone.compute_loss(action_queries, actions, action_mask=action_mask)

    def _predict(self, examples) -> Tensor:
        hidden, _, vlm_inputs = self._encode(examples)
        input_ids = vlm_inputs.get("input_ids")
        action_queries = self._gather_action_token_embeddings(hidden, input_ids, self.action_token_id)
        return self.action_backbone.predict_action(action_queries)

    def _gather_action_token_embeddings(
        self,
        last_hidden: Tensor,  # [B, L, H]
        input_ids: Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> Tensor:
        """Vectorized batch extraction of action-token embeddings: select the
        last ``action_horizon`` action placeholder tokens from each sample."""
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Support multiple ids (e.g., multiple special-token variants)
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.action_horizon).any():
            insufficient = (counts < self.action_horizon).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.action_horizon}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        # Position indices; take the last action_horizon positions (higher
        # indices = later in sequence), sorted back into temporal order.
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1
        topk_pos = masked_pos.topk(k=self.action_horizon, dim=-1).values  # [B, chunk] unsorted
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk]

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk, H]
        return last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk, H]

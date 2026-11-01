# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
#
# Adapted from starVLA
# (starVLA/model/modules/action_model/MLP_ActionHeader.py, itself adapted from
# https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py),
# modified for OpenWAM: plain-dict config + :class:`VLAActionHead` interface,
# masked L1 loss folded into ``compute_loss``.

"""MLP regression action head (QwenOFT): L1 over gathered action-token queries."""

from typing import Optional

import torch
from torch import Tensor, nn

from openwam.model.action_backbone.vla_heads.base import VLAActionHead


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean L1 over valid action cells, preserving legacy behavior without a mask."""
    error = torch.abs(prediction - target)
    if mask is None:
        return error.mean()
    valid = mask.to(device=error.device, dtype=error.dtype)
    if valid.dim() == error.dim() - 1:
        # Per-timestep mask (B, T) — broadcast across the action dims.
        valid = valid.unsqueeze(-1).expand_as(error)
    if valid.shape != error.shape:
        raise ValueError(f"action_mask shape {tuple(valid.shape)} != action shape {tuple(error.shape)}")
    denominator = valid.sum()
    if not torch.any(mask):
        raise ValueError("action_mask contains no valid targets")
    return (error * valid).sum() / denominator


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Feedforward network, "Pre-Layer Normalization" ordering:
        # https://arxiv.org/pdf/2002.04745.pdf
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(VLAActionHead):
    """MLP-based action head that generates continuous actions via L1 regression.

    Consumes gathered action-token queries (B, chunk, H) — the parameters are
    shared with the VLM in the sense that the queries are its hidden states; the
    head itself has no diffusion loop (single-pass regression).
    """

    def __init__(self, *, input_dim: int, hidden_dim: int, action_dim: int, action_horizon: int):
        super().__init__()
        self._action_dim = int(action_dim)
        self._action_horizon = int(action_horizon)
        self.model = MLPResNet(num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    def predict_action(self, encoder_hidden_states: Tensor) -> Tensor:
        """
        encoder_hidden_states: (B, chunk_len, hidden_dim)
        Returns: (B, chunk_len, action_dim)
        """
        batch_size, chunk_len, hidden_dim = encoder_hidden_states.shape
        x = encoder_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)  # (B * chunk_len, action_dim)
        return x.view(batch_size, chunk_len, self._action_dim)

    def forward(self, encoder_hidden_states: Tensor) -> Tensor:
        return self.predict_action(encoder_hidden_states)

    def compute_loss(
        self,
        encoder_hidden_states: Tensor,
        actions: Tensor,
        *,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Masked L1 loss. ``state`` / ``encoder_attention_mask`` unused —
        the queries are already gathered action-token embeddings."""
        pred_actions = self.predict_action(encoder_hidden_states)
        return masked_l1_loss(pred_actions, actions, action_mask)

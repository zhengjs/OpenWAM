"""Model-side output contracts (spec_zh.md §3.2).

Standardized dataclasses flowing between the model layer's four roles:
backbone → connector → action head, with ``SystemOutput`` as the single
``BaseArchitecture.forward`` return. Two orthogonal naming axes (normative):

- *state* fields are named by their SOURCE layer (``last_hidden_state``,
  ``hidden_states`` — which layer);
- *mask* fields are named by the token SEQUENCE they mask
  (``attention_mask``, ``context_mask``, ``condition_mask``, ``video_mask``,
  ``trajectory_mask``). A mask is a property of a token sequence, never of a
  single layer — one mask serves all layers' states, so mask naming never
  follows layer naming. (HF defines no output-side mask convention; this is
  an internal consistency rule.)

These dataclasses are plain ``@dataclass`` (not frozen): backbone/connector
stages assemble them incrementally, unlike the frozen ``ModelInputBatch``
which is the collate/forward boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor

__all__ = [
    "BackboneOutput",
    "ConnectorOutput",
    "ActionHeadOutput",
    "SystemOutput",
]


@dataclass
class BackboneOutput:
    """Standardized VLMBackbone / VideoBackbone output.

    Naming follows the widely known HF convention: ``last_hidden_state`` is
    the final layer; ``hidden_states`` is the per-layer stack where
    ``hidden_states[-1]`` IS ``last_hidden_state`` (alias, not a copy).
    """

    # [B, L_ctx, D_model] — final-layer hidden state; the default
    # conditioning source and the low-frequency compiled unit consumed by
    # encode_scene_tokens.
    last_hidden_state: Tensor

    # [B, L_ctx] — validity over the token sequence (shared by all layers).
    context_mask: Tensor

    # Optional per-layer hidden states, HF ``output_hidden_states``
    # convention — embeddings first, one [B, L_ctx, D] entry per layer,
    # hidden_states[-1] IS last_hidden_state. len() self-describes the
    # layer count — consumers never consult config. Filled only when the
    # run config asks for it (a connector declares its layer requirement,
    # e.g. connector=layerwise_hidden with num_layers=N — the backbone
    # returns exactly those layers). Consumers:
    #   - QwenPI layerwise cross-attention conditioning (§6.3)
    #   - DeepStack-style multi-layer routing (already in Qwen3-VL)
    #   - auxiliary deep-supervision losses (future)
    # Default None: a full stack costs roughly N_layers × L_ctx × D × 2
    # bytes (hundreds of MB per sample on a 4B VLM) — never materialize
    # speculatively.
    hidden_states: Optional[List[Tensor]] = None

    # Video flow-matching loss (WAM world stream).
    world_pred_loss: Optional[Tensor] = None

    # Language CE (VLA co-training, §6.5).
    vlm_loss: Optional[Tensor] = None

    # Decode-stage cache (inference only).
    kv_cache: Optional[Dict[str, Tensor]] = None


@dataclass
class ConnectorOutput:
    """Standardized connector output, consumed by the action head.

    One conditioning stream per consumed VLM layer; the list length
    self-describes the layer count (mirroring ``BackboneOutput.hidden_states``).
    """

    # Per-layer conditioning streams, each [B, L_src, D_head] — all entries
    # live on the SAME token sequence (the VLM's), hence identical lengths
    # and identical per-entry masks:
    #   - standard routing (last_hidden_tokens, last_hidden_cross_attn):
    #     len == 1; single-stream heads take condition_tokens[0].
    #   - layerwise routing (layerwise_hidden / QwenPI): len == N_layers;
    #     DiT layer i cross-attends to condition_tokens[i]. A wrong layer
    #     count fails loudly at the head's len()/shape assertions — nothing
    #     is ever packed into a shape that needs external config to decode.
    # The layer→DiT-layer mapping ("last N layers in order" vs "uniform
    # bridging") is connector CONFIG; the list itself carries zero packing
    # ambiguity.
    condition_tokens: List[Tensor]

    # [B, L_src] — one mask serves all streams (identical per entry).
    condition_mask: Tensor


@dataclass
class ActionHeadOutput:
    """Standardized action head output."""

    # Supervision loss (FM velocity MSE / L1 / FAST token CE).
    action_loss: Optional[Tensor] = None

    # [B, T_pred, D_action] — continuous heads (FM / MLP).
    pred_trajectory_norm: Optional[Tensor] = None

    # [B, L_tok] — FAST-path token ids (§6.4); trajectory is None then.
    pred_action_tokens: Optional[Tensor] = None

    # Reserved for RL post-training (§8.2).
    action_logprobs: Optional[Tensor] = None


@dataclass
class SystemOutput:
    """Top-level return of ``BaseArchitecture.forward()``."""

    total_loss: Optional[Tensor] = None

    action_loss: Optional[Tensor] = None

    world_pred_loss: Optional[Tensor] = None

    vlm_loss: Optional[Tensor] = None

    pred_trajectory_norm: Optional[Tensor] = None

    pred_action_tokens: Optional[Tensor] = None

    # Scalar logging metrics (already detached).
    metrics: Optional[Dict[str, float]] = field(default_factory=dict)

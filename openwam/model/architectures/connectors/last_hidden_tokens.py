"""``last_hidden_tokens`` connector (QwenOFT conditioning, spec_zh.md §6.3).

Gathers the trailing ``action_horizon`` action-placeholder token embeddings
from the last hidden state; the gathered positions feed the MLP L1 head. The
gather logic is shared verbatim with ``QwenOFTArchitecture`` (which delegates
to :func:`gather_action_token_embeddings`).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from openwam.contracts.model_contract import BackboneOutput, ConnectorOutput
from openwam.model.architectures.connectors.base import BaseConnector
from openwam.model.architectures.connectors.registry import register_connector


def gather_action_token_embeddings(
    last_hidden: Tensor,  # [B, L, H]
    input_ids: Tensor,  # [B, L]
    action_token_id: Union[int, list, None],
    action_horizon: int,
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
    if (counts < action_horizon).any():
        insufficient = (counts < action_horizon).nonzero(as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"The following samples have insufficient action tokens (< {action_horizon}): {insufficient} |"
            f" counts={counts.tolist()}"
        )

    # Position indices; take the last action_horizon positions (higher
    # indices = later in sequence), sorted back into temporal order.
    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
    masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1
    topk_pos = masked_pos.topk(k=action_horizon, dim=-1).values  # [B, chunk] unsorted
    selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk]

    expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk, H]
    return last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk, H]


@register_connector("last_hidden_tokens")
class LastHiddenTokensConnector(BaseConnector):
    """Action-token positions from the last hidden state -> single stream."""

    def __init__(self, action_horizon: int, action_token_id: Optional[int] = None):
        """``action_token_id`` may be injected after construction (resolved
        from the VLM processor at build time, QwenOFT convention)."""
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_token_id = action_token_id

    @property
    def required_backbone_layers(self) -> Optional[int]:
        return None

    def forward(self, backbone_output: BackboneOutput, *, input_ids: Tensor, **aux) -> ConnectorOutput:
        gathered = gather_action_token_embeddings(
            backbone_output.last_hidden_state,
            input_ids,
            self.action_token_id,
            self.action_horizon,
        )
        return ConnectorOutput(
            condition_tokens=[gathered],
            condition_mask=torch.ones(gathered.shape[:2], dtype=torch.bool, device=gathered.device),
        )

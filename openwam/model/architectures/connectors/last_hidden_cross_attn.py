"""``last_hidden_cross_attn`` connector (QwenGR00T conditioning, spec_zh.md §6.3).

Passes the last hidden state through as the single conditioning stream for
the DiT-B/L FlowMatching head's cross-attention.
"""

from __future__ import annotations

from typing import Optional

from openwam.contracts.model_contract import BackboneOutput, ConnectorOutput
from openwam.model.architectures.connectors.base import BaseConnector
from openwam.model.architectures.connectors.registry import register_connector


@register_connector("last_hidden_cross_attn")
class LastHiddenCrossAttnConnector(BaseConnector):
    """Identity packaging: last hidden state in, one stream out."""

    @property
    def required_backbone_layers(self) -> Optional[int]:
        return None

    def forward(self, backbone_output: BackboneOutput, **aux) -> ConnectorOutput:
        return ConnectorOutput(
            condition_tokens=[backbone_output.last_hidden_state],
            condition_mask=backbone_output.context_mask,
        )

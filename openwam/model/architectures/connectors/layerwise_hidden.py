"""``layerwise_hidden`` connector (QwenPI conditioning, spec_zh.md §6.3).

Packages the trailing VLM hidden-state layers as one conditioning stream per
layer; the LayerwiseFM DiT cross-attends DiT layer i to
``condition_tokens[i]``. The stream count self-describes the depth pairing —
a mismatch fails at the head's assertions.
"""

from __future__ import annotations

from typing import Optional

from openwam.contracts.model_contract import BackboneOutput, ConnectorOutput
from openwam.model.architectures.connectors.base import BaseConnector
from openwam.model.architectures.connectors.registry import register_connector


@register_connector("layerwise_hidden")
class LayerwiseHiddenConnector(BaseConnector):
    """One stream per trailing VLM layer; identity packaging, no projection
    (QwenPI: DiT depth = VLM layers, width = VLM hidden)."""

    def __init__(self, num_layers: Optional[int] = None):
        """``num_layers``: trailing layers to package (``None`` = everything the
        backbone provided). Also declares the backbone's ``hidden_states``
        materialization requirement."""
        super().__init__()
        self.num_layers = num_layers

    @property
    def required_backbone_layers(self) -> Optional[int]:
        return self.num_layers

    @property
    def num_condition_streams(self) -> Optional[int]:
        return self.num_layers  # None = follow the backbone depth

    def forward(self, backbone_output: BackboneOutput, **aux) -> ConnectorOutput:
        if backbone_output.hidden_states is None:
            raise ValueError(
                "layerwise_hidden requires BackboneOutput.hidden_states — the backbone must be "
                "run with the connector's required_backbone_layers"
            )
        layers = backbone_output.hidden_states
        if self.num_layers is not None:
            layers = layers[-self.num_layers :]
        return ConnectorOutput(
            condition_tokens=list(layers),
            condition_mask=backbone_output.context_mask,
        )

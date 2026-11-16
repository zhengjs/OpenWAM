"""Connector ABC: ``BackboneOutput -> ConnectorOutput`` conditioning layer
(spec_zh.md §6.3).

A connector packages ONE conditioning strategy between a VLM backbone and an
action head, replacing the per-subclass branching that previously lived
inside the VLA architectures (QwenPI / QwenOFT / QwenGR00T). Composition is
config-driven: ``vlm_backbone + connector + action_head`` without subclass
proliferation.

Two properties drive the plumbing around the connector:

- ``required_backbone_layers`` — how many trailing layers the backbone must
  fill into ``BackboneOutput.hidden_states`` (``None`` = last layer only, the
  stack stays unmaterialized);
- ``num_condition_streams`` — the ``ConnectorOutput.condition_tokens`` count
  the action head should expect (``None`` = resolved at runtime from the
  backbone stack, e.g. layerwise following the VLM depth).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn

from openwam.contracts.model_contract import BackboneOutput, ConnectorOutput


class BaseConnector(nn.Module, ABC):
    """One conditioning strategy between backbone and action head."""

    @property
    @abstractmethod
    def required_backbone_layers(self) -> Optional[int]:
        """Trailing ``BackboneOutput.hidden_states`` layers this connector
        consumes; ``None`` = last layer only (stack not materialized)."""

    @property
    def num_condition_streams(self) -> Optional[int]:
        """Len of ``ConnectorOutput.condition_tokens``; ``None`` = runtime
        (follows the backbone's layer count)."""
        return 1

    @abstractmethod
    def forward(self, backbone_output: BackboneOutput, **aux) -> ConnectorOutput:
        """Package the backbone output into per-stream conditioning."""

"""Connectors: explicit conditioning layer between VLM backbone and action
head (spec_zh.md §6.3). Importing the package registers the built-ins."""

from openwam.model.architectures.connectors.base import BaseConnector
from openwam.model.architectures.connectors.layerwise_hidden import LayerwiseHiddenConnector
from openwam.model.architectures.connectors.last_hidden_cross_attn import LastHiddenCrossAttnConnector
from openwam.model.architectures.connectors.last_hidden_tokens import (
    LastHiddenTokensConnector,
    gather_action_token_embeddings,
)
from openwam.model.architectures.connectors.registry import (
    CONNECTOR_REGISTRY,
    build_connector,
    register_connector,
)

__all__ = [
    "BaseConnector",
    "CONNECTOR_REGISTRY",
    "build_connector",
    "register_connector",
    "LayerwiseHiddenConnector",
    "LastHiddenTokensConnector",
    "LastHiddenCrossAttnConnector",
    "gather_action_token_embeddings",
]

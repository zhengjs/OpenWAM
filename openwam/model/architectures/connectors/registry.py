"""Connector registry (spec_zh.md §1.3 rule 1 — same decorator pattern as the
dataloader / preprocessor registries)."""

from __future__ import annotations

from typing import Type

from openwam.model.architectures.connectors.base import BaseConnector

CONNECTOR_REGISTRY: dict = {}

__all__ = ["CONNECTOR_REGISTRY", "register_connector", "build_connector"]


def register_connector(name: str):
    """Decorator registering a connector class by config ``name``."""

    def wrapper(cls: Type[BaseConnector]):
        CONNECTOR_REGISTRY[name] = cls
        return cls

    return wrapper


def build_connector(name: str, **kwargs) -> BaseConnector:
    """Registry-key construction; unknown names fail fast with the available set."""
    if name not in CONNECTOR_REGISTRY:
        available = ", ".join(sorted(CONNECTOR_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown connector '{name}'. Available: {available}")
    return CONNECTOR_REGISTRY[name](**kwargs)

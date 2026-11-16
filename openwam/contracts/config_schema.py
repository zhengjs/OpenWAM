"""Typed config blocks for the data pipeline (spec_data_zh.md §1.3).

Scope control (normative): full DictConfig→dataclass conversion is NOT a goal.
Only the data-side blocks get schemas here; the rest of the Hydra tree keeps
flowing as DictConfig. Validation happens at build time via
:func:`validate_config_block` (``OmegaConf.structured`` merge) — schema errors
fail fast at construction, not at forward.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

__all__ = ["ProcessorSpec", "PreprocessorChainConfig", "validate_config_block"]


@dataclass
class ProcessorSpec:
    """One preprocessor entry: registry key + private kwargs.

    ``kwargs`` is heterogeneous across processors, so there is no rigid schema
    here; each processor's ``from_config`` validates its own kwargs at build
    time (fast fail).
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessorChainConfig:
    """The ``preprocessors`` block: ordered chain (empty = adapter / Track A only)."""

    chain: List[ProcessorSpec] = field(default_factory=list)


def validate_config_block(block: Union[dict, DictConfig], schema_cls):
    """Validate a YAML block against a schema dataclass; return the instance.

    ``OmegaConf.merge`` against the structured schema fails fast on unknown
    keys and wrong types.
    """
    if isinstance(block, DictConfig):
        block = OmegaConf.to_container(block, resolve=True)
    merged = OmegaConf.merge(OmegaConf.structured(schema_cls), block)
    return OmegaConf.to_object(merged)
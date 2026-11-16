"""Data + model contracts (spec_data_zh.md §1, spec_zh.md §3).

Data side: the pure data types (``PipelineContext`` / ``ModelInputBatch``),
the collation entry point, and the typed config blocks for the data pipeline.
Model side (§3.2): the standardized outputs flowing backbone → connector →
action head (``BackboneOutput`` / ``ConnectorOutput`` / ``ActionHeadOutput``
/ ``SystemOutput``).
"""

from openwam.contracts.collate import collate_pipeline_contexts
from openwam.contracts.config_schema import (
    PreprocessorChainConfig,
    ProcessorSpec,
    validate_config_block,
)
from openwam.contracts.data_contract import ModelInputBatch, PipelineContext
from openwam.contracts.model_contract import (
    ActionHeadOutput,
    BackboneOutput,
    ConnectorOutput,
    SystemOutput,
)

__all__ = [
    "PipelineContext",
    "ModelInputBatch",
    "collate_pipeline_contexts",
    "ProcessorSpec",
    "PreprocessorChainConfig",
    "validate_config_block",
    "BackboneOutput",
    "ConnectorOutput",
    "ActionHeadOutput",
    "SystemOutput",
]
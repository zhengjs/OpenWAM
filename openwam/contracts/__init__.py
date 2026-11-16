"""Data-side contracts (spec_data_zh.md §1).

Exposes the pure data types (``PipelineContext`` / ``ModelInputBatch``), the
collation entry point, and the typed config blocks for the data pipeline.
Model-side contracts (``BackboneOutput`` / ``ConnectorOutput`` / ...) live in
their own phase and are intentionally not imported here.
"""

from openwam.contracts.collate import collate_pipeline_contexts
from openwam.contracts.config_schema import (
    PreprocessorChainConfig,
    ProcessorSpec,
    validate_config_block,
)
from openwam.contracts.data_contract import ModelInputBatch, PipelineContext

__all__ = [
    "PipelineContext",
    "ModelInputBatch",
    "collate_pipeline_contexts",
    "ProcessorSpec",
    "PreprocessorChainConfig",
    "validate_config_block",
]
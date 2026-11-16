"""Composable preprocessor chain (spec_data_zh.md §5)."""

from openwam.dataloader.preprocessors.base import BasePreprocessor, PreProcessorPipeline
from openwam.dataloader.preprocessors.registry import (
    PREPROCESSOR_REGISTRY,
    build_preprocessor_chain,
    list_registered_preprocessors,
    register_preprocessor,
)

__all__ = [
    "BasePreprocessor",
    "PreProcessorPipeline",
    "PREPROCESSOR_REGISTRY",
    "register_preprocessor",
    "build_preprocessor_chain",
    "list_registered_preprocessors",
]

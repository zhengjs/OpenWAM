"""Composable preprocessor chain over PipelineContext (spec_data_zh.md §5).

Contract:
- ``BasePreprocessor.process`` mutates and returns one ``PipelineContext``
  (worker-side, per sample). ``is_training_only=True`` marks processors that
  run only on the training path (decode / sync); the inference pipeline filters
  them out and rebuilds from the checkpoint-frozen config.
- ``PreProcessorPipeline`` executes the chain in ``PreprocessorChainConfig``
  order; ordering IS the contract (e.g. decode → sync → ego_state → image →
  prompt).
"""

from __future__ import annotations

from typing import List, Optional

from openwam.contracts.config_schema import PreprocessorChainConfig
from openwam.contracts.data_contract import PipelineContext

__all__ = ["BasePreprocessor", "PreProcessorPipeline"]


class BasePreprocessor:
    """One link of the preprocessor chain: ``ctx -> ctx``."""

    #: True → skipped by the inference-time pipeline (training-only stages).
    is_training_only: bool = False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        return self.process(ctx)


class PreProcessorPipeline:
    """Ordered chain of preprocessors over a single ``PipelineContext``."""

    def __init__(self, preprocessors: Optional[List[BasePreprocessor]] = None):
        self.preprocessors: List[BasePreprocessor] = list(preprocessors or [])

    @classmethod
    def from_config(
        cls,
        cfg: PreprocessorChainConfig,
        *,
        training: bool = True,
    ) -> "PreProcessorPipeline":
        """Build the chain from config via the registry; ``training=False``
        drops ``is_training_only`` stages (inference path, spec §5 dual mode)."""
        from openwam.dataloader.preprocessors.registry import build_preprocessor_chain

        return cls(build_preprocessor_chain(cfg, training=training))

    def process(self, ctx: PipelineContext, *, training: bool = True) -> PipelineContext:
        for proc in self.preprocessors:
            if proc.is_training_only and not training:
                continue
            ctx = proc.process(ctx)
        return ctx

    def __call__(self, ctx: PipelineContext, **kwargs) -> PipelineContext:
        return self.process(ctx, **kwargs)

    def __len__(self) -> int:
        return len(self.preprocessors)

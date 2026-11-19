"""Preprocessor registry (spec_data_zh.md §5).

Same decorator + ``from_config`` discipline as ``dataloader/registry.py``
(D1: registries are the instantiation path — no builders).
"""

from __future__ import annotations

from typing import Dict, List, Type

from openwam.contracts.config_schema import PreprocessorChainConfig
from openwam.dataloader.preprocessors.base import BasePreprocessor
from openwam.dataloader.utils import get_cfg

PREPROCESSOR_REGISTRY: Dict[str, Type[BasePreprocessor]] = {}


def register_preprocessor(name: str):
    """Decorator registering a preprocessor class by config ``name``."""

    def wrapper(cls):
        PREPROCESSOR_REGISTRY[name] = cls
        return cls

    return wrapper


def build_preprocessor_chain(
    cfg: PreprocessorChainConfig,
    *,
    training: bool = True,
) -> List[BasePreprocessor]:
    """Instantiate the configured chain in order.

    ``training=False`` filters out ``is_training_only=True`` stages (inference
    runs ego_state + image + prompt only, spec §5 dual mode).
    """
    out: List[BasePreprocessor] = []
    for spec in cfg.chain:
        name = spec.name
        if name not in PREPROCESSOR_REGISTRY:
            raise ValueError(
                f"Unknown preprocessor {name!r}. Available: {list(PREPROCESSOR_REGISTRY)}"
            )
        proc = PREPROCESSOR_REGISTRY[name](**dict(spec.kwargs))
        if proc.is_training_only and not training:
            continue
        out.append(proc)
    return out


def list_registered_preprocessors() -> List[str]:
    return sorted(PREPROCESSOR_REGISTRY.keys())


def _register_builtins() -> None:
    from openwam.dataloader.preprocessors.ego_state_processor import EgoStateProcessor
    from openwam.dataloader.preprocessors.multiview_image_processor import MultiViewImageProcessor
    from openwam.dataloader.preprocessors.topic_sync_processor import TopicTimeSynchronizer
    from openwam.dataloader.preprocessors.vla_prompt_processor import VLAPromptTokenizeProcessor
    from openwam.dataloader.preprocessors.video_decode_processor import MultiCamVideoDecoder

    register_preprocessor("ego_state")(EgoStateProcessor)
    register_preprocessor("multiview_image")(MultiViewImageProcessor)
    register_preprocessor("topic_sync")(TopicTimeSynchronizer)
    register_preprocessor("vla_prompt")(VLAPromptTokenizeProcessor)
    register_preprocessor("multicam_decode")(MultiCamVideoDecoder)


_register_builtins()

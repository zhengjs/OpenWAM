"""VLA architecture family (``framework=vla``).

One architecture, config-driven (spec_zh.md §6.3): ``vla_system`` composes
``vlm_backbone + connector + action_head`` from yaml, with per-variant
presets (oftp / pi / groot) carrying the behavior the legacy per-variant
subclasses used to own. Those subclasses were deleted after the loss
reproduction gate pinned equivalence (git history 4412820);

    from openwam.model import build_architecture
    arch = build_architecture("vla_system", cfg)   # cfg.variant: oftp | pi | groot
"""

from openwam.model.architectures.vla.base_vla import BaseVLAArchitecture
from openwam.model.architectures.vla.vla_system import VLASystemArchitecture

__all__ = [
    "BaseVLAArchitecture",
    "VLASystemArchitecture",
]

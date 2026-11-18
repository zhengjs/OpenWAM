"""VLM backbone package: the VlmBackbone ABC + its Qwen3-VL implementation.

    from openwam.model.vlm_backbone import build_vlm_backbone
    backbone = build_vlm_backbone("qwen3_vl_2b", checkpoint_path=..., dtype=...)

ONE Qwen3-VL class serves both consumers (tri_system understanding stream and
the VLA family); the two registry names differ only in the yaml-provided
knobs (prompt_style / add_generation_prompt / max_length / padding_side).

Adding a VLM backbone: subclass :class:`VlmBackbone`, then register it at the
bottom of this file via ``register_vlm_backbone("name")(YourClass)`` and set
``vlm_backbone.name: your_name`` in the model config yaml.
"""

from openwam.model.vlm_backbone.base import VlmBackbone
from openwam.model.vlm_backbone.registry import (
    _VLM_BACKBONE_REGISTRY,
    build_vlm_backbone,
    register_vlm_backbone,
)

__all__ = [
    "VlmBackbone",
    "Qwen3VLBackbone",
    "build_vlm_backbone",
    "register_vlm_backbone",
    "_VLM_BACKBONE_REGISTRY",
]

# Built-in registration (at the bottom so the implementation imports only from
# base/registry, with no circular dependency). Both names -> the same class;
# behavior differences live in configs/model/vlm_backbone/*.yaml.
from openwam.model.vlm_backbone.qwen3_vl_backbone import Qwen3VLBackbone  # noqa: E402

register_vlm_backbone("qwen3_vl_2b")(Qwen3VLBackbone)
register_vlm_backbone("qwen3_vl_vla")(Qwen3VLBackbone)

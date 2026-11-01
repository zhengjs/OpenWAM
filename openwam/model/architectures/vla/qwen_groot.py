# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
#
# Adapted from starVLA (starVLA/model/framework/VLM4A/QwenGR00T.py), modified
# for OpenWAM: :class:`BaseVLAArchitecture` template + the OpenWAM registry /
# config system; the sys.path hack, cross-package image tools, and the
# hardcoded autocast are dropped.

"""QwenGR00T: single-layer VLM hidden -> DiT-B flow matching.

A lightweight VLA that feeds the last VLM hidden state (repeated along the
batch axis, GR00T N1.5 style) into a flow-matching DiT action head.
"""

from openwam.model.action_backbone.vla_heads import build_vla_action_head
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.vla.base_vla import BaseVLAArchitecture


@register_architecture(
    "vla_qwen_groot",
    status="supported",
    note="VLA QwenGR00T: single-layer VLM hidden -> DiT-B flow matching.",
    framework="vla",
    variant="groot",
)
class QwenGR00TArchitecture(BaseVLAArchitecture):
    """Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence
        modeling
    """

    DEFAULTS = {
        "action_head": {
            "action_model_type": "DiT-B",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "hidden_size": 1024,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # cross_attention_dim is overridden at runtime with the VLM hidden
            # size (the placeholder value matches the Qwen2.5-VL convention).
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        },
        "repeated_diffusion_steps": 8,
    }

    def _init_action_head(self, cfg):
        # DiT cross-attention dim = VLM hidden size.
        model_type = self._expect_action_model_type(cfg, ("DiT-B", "DiT-L"), "DiT-B")
        head_cfg = self._cfg_get(cfg, "action_head", None)
        return build_vla_action_head(
            head_cfg, action_model_type=model_type, cross_attention_dim=self.vlm_backbone.hidden_size
        )

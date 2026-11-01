# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
#
# Adapted from starVLA (starVLA/model/framework/VLM4A/QwenPI.py, a GR00T N1.5
# layer-wise flow-matching head merged into a starVLA framework), modified for
# OpenWAM: :class:`BaseVLAArchitecture` template + the OpenWAM registry /
# config system; the hardcoded autocast and the RTC path are dropped.

"""QwenPI: layer-wise cross-DiT flow matching over multi-layer VLM hidden states.

The DiT depth equals the VLM layer count and its width equals the LLM hidden
size (no compression projection); the last ``num_hidden_layers`` hidden states
feed the layer-aligned cross-attention action head.
"""

from openwam.model.action_backbone.vla_heads import build_vla_action_head
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.vla.base_vla import BaseVLAArchitecture


@register_architecture(
    "vla_qwen_pi",
    status="supported",
    note="VLA QwenPI: layer-wise cross-DiT flow matching over multi-layer VLM hidden states.",
    framework="vla",
    variant="pi",
)
class QwenPIArchitecture(BaseVLAArchitecture):
    """Multimodal vision-language-action model (PI variant).

    Components:
      - Qwen3-VL backbone for fused language/vision token embeddings
      - Layer-wise cross-DiT flow-matching head fed by multi-layer VLM
        hidden states
    """

    DEFAULTS = {
        "action_head": {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # DiT shape fields (num_layers / hidden_size / cross_attention_dim /
            # num_attention_heads) are overridden at runtime from the loaded VLM.
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        },
        "repeated_diffusion_steps": 2,
    }

    def _init_action_head(self, cfg):
        # DiT depth = VLM layers, width = VLM hidden (no compression); the
        # head itself never reads VLM-specific config.
        model_type = self._expect_action_model_type(cfg, ("LayerwiseFM",), "LayerwiseFM")
        head_cfg = self._cfg_get(cfg, "action_head", None)
        return build_vla_action_head(
            head_cfg,
            action_model_type=model_type,
            hidden_size=self.vlm_backbone.hidden_size,
            num_layers=self.vlm_backbone.num_vl_layers,
        )

    def _num_last_layers(self):
        # ``output_hidden_states=True`` returns (num_hidden_layers + 1) tensors;
        # keep the last num_hidden_layers of them, matching the DiT depth.
        return self.vlm_backbone.num_vl_layers

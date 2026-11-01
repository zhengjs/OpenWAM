"""VLA action heads: self-contained predictors behind the :class:`VLAActionHead` ABC.

Like the WAM action backbones there is no registry — the host VLA architecture
builds its head via :func:`build_vla_action_head`, keyed on the
``action_model_type`` field of the ``architecture.action_head`` config block:

    MLP          -> :class:`L1RegressionActionHead`          (QwenOFT)
    DiT-B / DiT-L-> :class:`FlowMatchingActionHead`          (QwenGR00T)
    LayerwiseFM  -> :class:`LayerwiseFlowMatchingActionHead` (QwenPI)
"""

from openwam.model.action_backbone.vla_heads.base import VLAActionHead
from openwam.model.action_backbone.vla_heads.flow_matching_head import FlowMatchingActionHead
from openwam.model.action_backbone.vla_heads.layerwise_fm_head import LayerwiseFlowMatchingActionHead
from openwam.model.action_backbone.vla_heads.mlp_head import L1RegressionActionHead

__all__ = [
    "VLAActionHead",
    "L1RegressionActionHead",
    "FlowMatchingActionHead",
    "LayerwiseFlowMatchingActionHead",
    "build_vla_action_head",
]


def build_vla_action_head(cfg=None, **overrides) -> VLAActionHead:
    """Build a VLA action head from the ``architecture.action_head`` config block.

    Args:
        cfg: plain dict / DictConfig action-head block (``action_model_type``,
            ``action_dim``, ``action_horizon``, head-specific fields).
        **overrides: explicit kwargs that win over ``cfg`` — used by the host
            architecture to inject VLM-derived dims (e.g. ``cross_attention_dim``
            aligned to the loaded VLM hidden size).
    """
    merged = dict(cfg) if cfg is not None else {}
    merged.update(overrides)
    model_type = str(merged.get("action_model_type", "MLP"))

    if model_type == "MLP":
        hidden = int(merged.get("action_hidden_dim", 2560))
        return L1RegressionActionHead(
            input_dim=hidden,
            hidden_dim=hidden * 2,
            action_dim=int(merged.get("action_dim", 7)),
            action_horizon=int(merged.get("action_horizon", 8)),
        )
    if model_type in ("DiT-B", "DiT-L"):
        return FlowMatchingActionHead(
            action_dim=int(merged.get("action_dim", 7)),
            action_horizon=int(merged.get("action_horizon", 8)),
            model_size=model_type,
            state_dim=int(merged.get("state_dim", 0) or 0),
            hidden_size=int(merged.get("hidden_size", 1024)),
            num_inference_timesteps=int(merged.get("num_inference_timesteps", 4)),
            num_target_vision_tokens=int(merged.get("num_target_vision_tokens", 32)),
            add_pos_embed=bool(merged.get("add_pos_embed", True)),
            max_seq_len=int(merged.get("max_seq_len", 1024)),
            noise_beta_alpha=float(merged.get("noise_beta_alpha", 1.5)),
            noise_beta_beta=float(merged.get("noise_beta_beta", 1.0)),
            noise_s=float(merged.get("noise_s", 0.999)),
            num_timestep_buckets=int(merged.get("num_timestep_buckets", 1000)),
            cross_attention_dim=merged.get("cross_attention_dim"),
            diffusion_model_cfg=merged.get("diffusion_model_cfg"),
        )
    if model_type == "LayerwiseFM":
        return LayerwiseFlowMatchingActionHead(
            action_dim=int(merged.get("action_dim", 7)),
            action_horizon=int(merged.get("action_horizon", 16)),
            num_layers=int(merged.get("num_layers", 36)),
            hidden_size=int(merged.get("hidden_size", 2048)),
            attention_head_dim=int(merged.get("attention_head_dim", 64)),
            state_dim=int(merged.get("state_dim", 0) or 0),
            num_inference_timesteps=int(merged.get("num_inference_timesteps", 4)),
            num_target_vision_tokens=int(merged.get("num_target_vision_tokens", 32)),
            add_pos_embed=bool(merged.get("add_pos_embed", True)),
            max_seq_len=int(merged.get("max_seq_len", 1024)),
            noise_beta_alpha=float(merged.get("noise_beta_alpha", 1.5)),
            noise_beta_beta=float(merged.get("noise_beta_beta", 1.0)),
            noise_s=float(merged.get("noise_s", 0.999)),
            num_timestep_buckets=int(merged.get("num_timestep_buckets", 1000)),
            diffusion_model_cfg=merged.get("diffusion_model_cfg"),
        )
    raise KeyError(f"Unknown VLA action head type '{model_type}'. Available: MLP, DiT-B, DiT-L, LayerwiseFM")

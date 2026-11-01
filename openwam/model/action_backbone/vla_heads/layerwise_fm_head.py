# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Jinhui YE / HKUST] in [2026] for starVLA.
#
# Adapted from starVLA
# (starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py, GR00T N1.5
# layer-wise flow-matching head), modified for OpenWAM: plain-dict config +
# :class:`VLAActionHead` interface; the shared FM components moved to
# ``fm_components.py``; the unused RTC path (``predict_action_realtime``) is
# not carried over.

"""Layer-wise cross-attention DiT action head (QwenPI)."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Beta

from openwam.model.action_backbone.vla_heads.base import VLAActionHead
from openwam.model.action_backbone.vla_heads.cross_attention_dit import DiT
from openwam.model.action_backbone.vla_heads.fm_components import MLP, ActionEncoder


class LayerwiseFlowMatchingActionHead(VLAActionHead):
    """Layer-wise cross-attention DiT action head.

    Configuration boundary: the head is agnostic of any VLM backbone. The host
    architecture (QwenPI) populates the DiT shape from the loaded VLM BEFORE
    construction (via the factory ``overrides``):

        num_layers          = <VLM num_hidden_layers>   (DiT depth)
        hidden_size         = <VLM hidden_size>         (DiT internal hidden, no compression)
        cross_attention_dim = <VLM hidden_size>         (encoder is pre-projected)
        num_attention_heads = hidden_size // attention_head_dim
    """

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        num_layers: int,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        attention_head_dim: int = 64,
        state_dim: int = 0,
        num_inference_timesteps: int = 4,
        num_target_vision_tokens: int = 32,
        add_pos_embed: bool = True,
        max_seq_len: int = 1024,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        diffusion_model_cfg: Optional[dict] = None,
    ):
        super().__init__()
        # DiT shape: host-provided VLM-aligned fields win; the yaml
        # ``diffusion_model_cfg`` subtree overrides/extends; safe defaults for
        # anything left unset.
        shape = dict(diffusion_model_cfg) if diffusion_model_cfg else {}
        shape.setdefault("num_layers", int(num_layers))
        shape.setdefault("input_embedding_dim", int(hidden_size))
        shape.setdefault(
            "cross_attention_dim", int(hidden_size if cross_attention_dim is None else cross_attention_dim)
        )
        head_dim = shape.get("attention_head_dim") or int(attention_head_dim)
        shape["attention_head_dim"] = int(head_dim)
        shape.setdefault("num_attention_heads", int(hidden_size) // int(head_dim))
        shape.setdefault("dropout", 0.2)
        shape.setdefault("final_dropout", True)
        shape.setdefault("interleave_self_attention", True)
        shape.setdefault("norm_type", "ada_norm")
        shape.setdefault("positional_embeddings", None)
        self.model = DiT(**shape)

        self.input_embedding_dim = shape["input_embedding_dim"]
        self._action_dim = int(action_dim)
        self._action_horizon = int(action_horizon)
        self.num_inference_timesteps = int(num_inference_timesteps)
        self.hidden_size = int(hidden_size)

        self.state_encoder = (
            MLP(input_dim=int(state_dim), hidden_dim=1024, output_dim=self.input_embedding_dim)
            if state_dim
            else None
        )
        self.action_encoder = ActionEncoder(action_dim=int(action_dim), hidden_size=self.input_embedding_dim)
        self.action_decoder = MLP(input_dim=self.input_embedding_dim, hidden_dim=1024, output_dim=int(action_dim))
        self.future_tokens = nn.Embedding(int(num_target_vision_tokens), self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self._add_pos_embed = bool(add_pos_embed)
        if self._add_pos_embed:
            self.position_embedding = nn.Embedding(int(max_seq_len), self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(float(noise_beta_alpha), float(noise_beta_beta))
        self._noise_s = float(noise_s)
        self.num_timestep_buckets = int(num_timestep_buckets)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self._noise_s)
        return (self._noise_s - sample) / self._noise_s

    def forward(
        self,
        vl_embs_list: list,
        actions: Tensor,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        vl_embs_list: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        encoder_attention_mask: optional (B, seq_length) bool/int
        """
        device = actions.device

        # Embed noised action trajectory (flow-matching: linear interp + velocity target).
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self._add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs_list[0].shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Layer-wise DiT forward. DiT handles cross/self-attention interleaving.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
        )

        # Decode only the action-token positions.
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        return ((pred_actions - velocity) ** 2).mean()

    def compute_loss(
        self,
        encoder_hidden_states: list,
        actions: Tensor,
        *,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Flow-matching MSE. ``action_mask`` unused — the FM loss is unmasked,
        matching the upstream implementation."""
        return self.forward(
            encoder_hidden_states, actions, state=state, encoder_attention_mask=encoder_attention_mask
        )

    @torch.no_grad()
    def predict_action(
        self,
        encoder_hidden_states: list,
        *,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Set initial actions as the sampled noise.
        batch_size = encoder_hidden_states[0].shape[0]
        device = encoder_hidden_states[0].device
        actions = torch.randn(
            size=(batch_size, self._action_horizon, self._action_dim),
            dtype=encoder_hidden_states[0].dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            action_features = self.action_encoder(actions, timesteps_tensor)
            if self._add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps_tensor,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self._action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
        return actions

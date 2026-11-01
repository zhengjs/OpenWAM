# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from starVLA
# (starVLA/model/modules/action_model/GR00T_ActionHeader.py, the GR00T N1.5
# flow-matching head; action repeat inspired by CogACT), modified for OpenWAM:
# plain-dict config + :class:`VLAActionHead` interface; the shared FM
# components moved to ``fm_components.py``.

"""Flow-matching action head (QwenGR00T): single-layer VLM hidden -> DiT."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Beta

from openwam.model.action_backbone.vla_heads.base import VLAActionHead
from openwam.model.action_backbone.vla_heads.cross_attention_dit import DiT
from openwam.model.action_backbone.vla_heads.fm_components import MLP, ActionEncoder

# DiT model-size presets; ``diffusion_model_cfg`` from the yaml overrides/extends.
#   DiT-B -> input_embedding_dim=768,  heads=12, head_dim=64
#   DiT-L -> input_embedding_dim=1536, heads=32, head_dim=48
DiT_MODEL_SIZES = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowMatchingActionHead(VLAActionHead):
    """Flow-matching (DiT) diffusion head over the last VLM hidden layer.

    Configuration boundary: ``cross_attention_dim`` MUST be set by the host
    architecture to the VLM hidden size (via the factory ``overrides``) BEFORE
    construction; the head itself never reads VLM-specific config.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        model_size: str = "DiT-B",
        state_dim: int = 0,
        hidden_size: int = 1024,
        num_inference_timesteps: int = 4,
        num_target_vision_tokens: int = 32,
        add_pos_embed: bool = True,
        max_seq_len: int = 1024,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        cross_attention_dim: Optional[int] = None,
        diffusion_model_cfg: Optional[dict] = None,
    ):
        super().__init__()
        # DiT shape: preset + yaml override; ``cross_attention_dim`` is set by
        # the host architecture to the VLM hidden size.
        shape = dict(DiT_MODEL_SIZES[model_size])
        shape.update(dict(diffusion_model_cfg) if diffusion_model_cfg else {})
        if cross_attention_dim is not None:
            shape["cross_attention_dim"] = int(cross_attention_dim)
        shape.setdefault("output_dim", int(hidden_size))
        self.model = DiT(**shape)

        # Action horizon (chunk length sent to the DiT), action / state dims.
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self.num_inference_timesteps = int(num_inference_timesteps)
        # Intermediate MLP width for state_encoder / action_decoder; decoupled
        # from input_embedding_dim so the MLP width can differ from the DiT
        # latent size.
        self.hidden_size = int(hidden_size)

        self.state_encoder = (
            MLP(input_dim=int(state_dim), hidden_dim=self.hidden_size, output_dim=shape["input_embedding_dim"])
            if state_dim
            else None
        )
        self.action_encoder = ActionEncoder(action_dim=int(action_dim), hidden_size=shape["input_embedding_dim"])
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim, hidden_dim=self.hidden_size, output_dim=int(action_dim)
        )

        # Learnable query tokens prepended before the action sequence so the
        # DiT has dedicated "planning" slots.
        self.future_tokens = nn.Embedding(int(num_target_vision_tokens), shape["input_embedding_dim"])
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # Positional embedding over the action sequence.
        self._add_pos_embed = bool(add_pos_embed)
        if self._add_pos_embed:
            self.position_embedding = nn.Embedding(int(max_seq_len), shape["input_embedding_dim"])
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Flow-matching noise schedule (Beta distribution).
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
        vl_embs: Tensor,
        actions: Tensor,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        """
        device = vl_embs.device

        # Embed noised action trajectory (flow-matching: linear interp + velocity target).
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete for the timestep encoder inside DiT.
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self._add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        # State and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        return ((pred_actions - velocity) ** 2).mean()

    def compute_loss(
        self,
        encoder_hidden_states: Tensor,
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
        encoder_hidden_states: Tensor,
        *,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Set initial actions as the sampled noise.
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        actions = torch.randn(
            size=(batch_size, self._action_horizon, self._action_dim),
            dtype=encoder_hidden_states.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
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
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self._action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

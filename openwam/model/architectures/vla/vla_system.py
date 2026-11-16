# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# Adapted for OpenWAM (spec_zh.md §6.3): ``VLASystem`` composes
# ``vlm_backbone + connector + action_head`` from config, replacing the
# per-variant subclass branching with the explicit connector layer.

"""VLASystem: config-driven VLA composition over the connector layer.

One architecture for the whole VLA family: ``variant`` only decides the
DEFAULTS preset (head type / connector / state routing / diffusion-step
count); the actual conditioning flows through the registered connector
(``configs/model/connector/*.yaml``), so new combinations are yaml work, not
subclass work.

Per-variant wiring (identical behavior to the legacy subclasses, verified by
the loss-reproduction test):

===============  =====================  ==========================  =========
variant          connector              action head                 state
===============  =====================  ==========================  =========
oftp             last_hidden_tokens     MLP (L1)                    in instruction (π0.5)
pi               layerwise_hidden       LayerwiseFM (FM DiT)        head state encoder
groot            last_hidden_cross_attn DiT-B / DiT-L (FM)         head state encoder
===============  =====================  ==========================  =========
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from openwam.model.action_backbone.vla_heads import build_vla_action_head
from openwam.model.architectures.connectors import build_connector
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.vla.base_vla import (
    BaseVLAArchitecture,
    add_discretized_state_to_instruction,
)


def _global_token_mean(sum_ce: Tensor, n_tokens: Tensor) -> Tensor:
    """Global token-sum CE normalization (§6.5, FluxVLA ``causal_lm`` pattern):
    all-reduce (sum_ce, n_tokens) across ranks when distributed, then divide —
    every rank gets the identical scalar, so long sequences on one rank don't
    dominate per-rank gradients."""
    total = torch.stack([sum_ce.float().reshape(()), n_tokens.float().reshape(())])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return total[0] / total[1].clamp(min=1.0)


class _VariantPreset:
    """Per-variant defaults + wiring facts (replaces subclass overrides)."""

    def __init__(
        self,
        *,
        defaults: dict,
        connector: str,
        connector_kwargs_key_map: Optional[dict] = None,
        head_type: str,
        head_type_allowed: tuple,
        head_dim_overrides: dict,
        state_in_instruction: bool,
    ):
        self.defaults = defaults
        self.connector = connector
        # Maps head-config keys -> connector kwargs (runtime VLM-aligned).
        self.connector_kwargs_key_map = connector_kwargs_key_map or {}
        self.head_type = head_type
        self.head_type_allowed = head_type_allowed
        self.head_dim_overrides = head_dim_overrides
        self.state_in_instruction = state_in_instruction


_VARIANTS: dict = {
    "oftp": _VariantPreset(
        defaults={
            "action_head": {
                "action_model_type": "MLP",
                "action_dim": 7,
                "action_hidden_dim": 2560,
                "action_horizon": 8,
            },
            "repeated_diffusion_steps": 1,
        },
        connector="last_hidden_tokens",
        head_type="MLP",
        head_type_allowed=("MLP",),
        head_dim_overrides={"action_hidden_dim": "hidden_size"},
        state_in_instruction=True,
    ),
    "pi": _VariantPreset(
        defaults={
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
        },
        connector="layerwise_hidden",
        connector_kwargs_key_map={"num_layers": "num_vl_layers"},
        head_type="LayerwiseFM",
        head_type_allowed=("LayerwiseFM",),
        head_dim_overrides={"hidden_size": "hidden_size", "num_layers": "num_vl_layers"},
        state_in_instruction=False,
    ),
    "groot": _VariantPreset(
        defaults={
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
        },
        connector="last_hidden_cross_attn",
        head_type="DiT-B",
        head_type_allowed=("DiT-B", "DiT-L"),
        head_dim_overrides={"cross_attention_dim": "hidden_size"},
        state_in_instruction=False,
    ),
}


@register_architecture(
    "vla_system",
    status="supported",
    note="VLA System: config-driven vlm_backbone + connector + action_head composition (§6.3).",
    framework="vla",
    # Owns the whole framework=vla family: ``system`` (canonical) plus the
    # oftp/pi/groot aliases so legacy (framework=vla, variant=X) configs
    # resolve here — the preset table carries their behavior.
    variant=("system", "oftp", "pi", "groot"),
)
class VLASystemArchitecture(BaseVLAArchitecture):
    """Config-driven VLA architecture (absorbs QwenOFT / QwenPI / QwenGR00T).

    Two composition knobs:

    - ``variant``: preset selector (oftp | pi | groot) — defaults + state
      routing + connector/head pairing. ``resolve_architecture_config``
      maps every framework=vla config here, so there is exactly ONE
      registry entry for the family.
    - ``connector`` (optional): registry name overriding the preset's
      connector, with ``connector_kwargs`` for its constructor. Config wins
      over the preset — new combinations need no subclass.
    """

    def __init__(self, cfg=None):
        self._preset = None  # resolved in _resolve_variant before super().__init__
        if cfg is not None:
            variant = self._cfg_get(cfg, "variant", "oftp")
            if variant not in _VARIANTS:
                raise ValueError(
                    f"VLASystem variant must be one of {sorted(_VARIANTS)}, got '{variant}'"
                )
            self._preset = _VARIANTS[variant]
            cfg = self._deep_merge(self._preset.defaults, cfg)
        super().__init__(cfg)
        if cfg is None:
            return

        # OFT: resolve the action-token id from the VLM processor and inject
        # it into the connector (gather needs the placeholder id).
        if self._preset.state_in_instruction:
            processor = getattr(self.vlm_backbone, "processor", None)
            if processor is None:
                raise RuntimeError("VLASystem (oftp) requires the VLM backbone to expose a processor.")
            action_token = "🔍"
            self.action_token_id = int(
                processor.tokenizer(action_token, add_special_tokens=False)["input_ids"][0]
            )
            if hasattr(self.connector, "action_token_id"):
                self.connector.action_token_id = self.action_token_id

    def _deep_merge(self, base: dict, override) -> dict:
        from openwam.model.architectures.vla.base_vla import deep_merge_defaults

        return deep_merge_defaults(base, override)

    # --- Composition ---

    def _init_vlm_backbone(self, cfg) -> None:
        super()._init_vlm_backbone(cfg)
        # Connector builds AFTER the VLM: its kwargs bind VLM shape fields at
        # runtime (num_layers = VLM depth), mirroring the action head.
        connector_name = self._cfg_get(cfg, "connector", None) or self._preset.connector
        connector_kwargs = dict(self._cfg_get(cfg, "connector_kwargs", None) or {})
        # VLM-aligned connector kwargs, resolved regardless of which connector
        # the config picked (preset or override); only kwargs the connector
        # actually accepts are passed (e.g. cross_attn takes none).
        import inspect

        from openwam.model.architectures.connectors.registry import CONNECTOR_REGISTRY

        params = inspect.signature(CONNECTOR_REGISTRY[connector_name]).parameters
        if "num_layers" in params and "num_layers" not in connector_kwargs:
            connector_kwargs["num_layers"] = self.vlm_backbone.num_vl_layers
        if "action_horizon" in params and "action_horizon" not in connector_kwargs:
            head_cfg = self._cfg_get(cfg, "action_head", None)
            connector_kwargs["action_horizon"] = self._cfg_get(head_cfg, "action_horizon", 8)
        self.connector = build_connector(connector_name, **connector_kwargs)

    def _init_action_head(self, cfg):
        model_type = self._expect_action_model_type(cfg, self._preset.head_type_allowed, self._preset.head_type)
        head_cfg = self._cfg_get(cfg, "action_head", None)
        overrides = {
            target: getattr(self.vlm_backbone, source)
            for target, source in self._preset.head_dim_overrides.items()
        }
        return build_vla_action_head(head_cfg, action_model_type=model_type, **overrides)

    @property
    def _STATE_IN_INSTRUCTION(self) -> bool:  # noqa: N802 — base class hook
        return self._preset.state_in_instruction

    def _build_prompt_suffix(self) -> str:
        if not self._preset.state_in_instruction:
            return ""
        horizon = self.action_backbone.action_horizon
        action_tokens = "🔍" * horizon
        return f" Please predict the next {horizon} robot actions: <action>{action_tokens}<action>."

    def _num_last_layers(self) -> Optional[int]:
        # Route through the connector's declaration (single source of truth).
        return self.connector.required_backbone_layers

    # --- Conditioning through the connector layer ---

    def _coforward(self, examples: list[dict]) -> tuple[Tensor, Optional[Tensor]]:
        """One encode, both legs (§6.5 single-backward rule): the action leg
        through the connector/head, plus the optional language CE leg when
        ``vlm_loss_weight > 0``. With the weight at 0 the VLM leg is skipped
        entirely — RNG consumption is identical to the legacy single-leg path
        (pinned by the loss-reproduction test)."""
        hidden, attention_mask, vlm_inputs = self._encode(
            examples, num_last_layers=self._num_last_layers()
        )
        times = self._repeated_diffusion_steps
        actions = self._stack_actions(examples, self.action_horizon)
        state = self._stack_state(examples, times)
        if times > 1:
            actions = actions.repeat(times, 1, 1)
            hidden = self._repeat_context(hidden, times)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(times, 1).to(dtype=torch.bool)

        cond = self._condition(hidden, attention_mask, vlm_inputs, times)
        action_loss = self.action_backbone.compute_loss(
            self._head_conditioning(cond),
            actions,
            state=state,
            encoder_attention_mask=cond.condition_mask.to(dtype=torch.bool),
        )

        vlm_loss = None
        if self.vlm_loss_weight > 0:
            if not hasattr(self.vlm_backbone, "compute_vlm_loss"):
                raise TypeError(
                    f"vlm_loss_weight > 0 requires the VLM backbone to expose compute_vlm_loss; "
                    f"got {type(self.vlm_backbone).__name__}"
                )
            sum_ce, n_tokens = self.vlm_backbone.compute_vlm_loss(
                vlm_inputs, labels=vlm_inputs.get("vlm_labels")
            )
            vlm_loss = _global_token_mean(sum_ce, n_tokens)
        return action_loss, vlm_loss

    def _forward_loss(self, examples: list[dict]) -> Tensor:
        """Base-class contract: action leg only (legacy trainer path)."""
        return self._coforward(examples)[0]

    def compute_loss(self, *, batch=None, lambda_action: float = 1.0, lambda_video: float = 0.0, **_):
        """Trainer entry with the §6.5 composition:
        ``total = lambda_action * action_loss + vlm_loss_weight * vlm_loss``
        — a single tensor, ONE backward. ``loss_vlm`` rides the trainer dict
        for logging."""
        if batch is None:
            raise ValueError("VLA compute_loss requires the dataset batch (via prepare_inputs).")
        if isinstance(batch, dict):
            batch = [batch]
        action_loss, vlm_loss = self._coforward(self._to_examples(batch))
        total = lambda_action * action_loss
        if vlm_loss is not None:
            total = total + self.vlm_loss_weight * vlm_loss
        return {
            "loss": total,
            "loss_video": torch.tensor(0.0, device=total.device),
            "loss_action": (lambda_action * action_loss).detach(),
            **({"loss_vlm": vlm_loss.detach()} if vlm_loss is not None else {}),
        }

    def _predict(self, examples: list[dict]) -> Tensor:
        hidden, attention_mask, vlm_inputs = self._encode(
            examples, num_last_layers=self._num_last_layers()
        )
        state = self._stack_state(examples)
        cond = self._condition(hidden, attention_mask, vlm_inputs, times=1)
        if self._preset.connector == "last_hidden_tokens":
            # MLP head takes only the gathered queries (no state/mask kwargs).
            return self.action_backbone.predict_action(self._head_conditioning(cond))
        return self.action_backbone.predict_action(
            self._head_conditioning(cond),
            state=state,
            encoder_attention_mask=cond.condition_mask.to(dtype=torch.bool),
        )

    def _condition(self, hidden: Union[Tensor, list], attention_mask, vlm_inputs: dict, times: int):
        """Package the backbone run through the connector into ConnectorOutput."""
        from openwam.contracts.model_contract import BackboneOutput

        if isinstance(hidden, list):
            backbone_out = BackboneOutput(
                last_hidden_state=hidden[-1],
                context_mask=attention_mask.to(dtype=torch.bool) if attention_mask is not None
                else torch.ones(hidden[-1].shape[:2], dtype=torch.bool),
                hidden_states=hidden,
            )
        else:
            backbone_out = BackboneOutput(
                last_hidden_state=hidden,
                context_mask=attention_mask.to(dtype=torch.bool) if attention_mask is not None
                else torch.ones(hidden.shape[:2], dtype=torch.bool),
            )
        aux = {}
        if "input_ids" in (vlm_inputs or {}):
            aux["input_ids"] = vlm_inputs["input_ids"]
            if times > 1:
                aux["input_ids"] = aux["input_ids"].repeat(times, 1)
        return self.connector(backbone_out, **aux)

    def _head_conditioning(self, cond):
        """Unpack ConnectorOutput into the head's per-type shape convention:
        MLP takes the gathered queries, LayerwiseFM the per-layer list,
        FlowMatching the full sequence — all are ``condition_tokens`` in
        their conventional form; single-stream heads take stream [0]."""
        if self._preset.connector == "last_hidden_tokens":
            return cond.condition_tokens[0]  # (B, chunk, H)
        if self._preset.head_type == "LayerwiseFM":
            return cond.condition_tokens  # list of per-layer (B, L, H)
        return cond.condition_tokens[0]  # full sequence (B, L, H)

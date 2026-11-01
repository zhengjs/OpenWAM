"""BaseVLAArchitecture: the VLM-conditioned action-prediction contract shared
by the VLA architecture family (QwenOFT / QwenPI / QwenGR00T, ported from
starVLA's VLM4A frameworks).

Extends the family-agnostic :class:`BaseArchitecture` with a template-method
pair — ``compute_loss(batch)`` and ``predict_action(batch)`` — that owns the
shared plumbing (batch unpacking, action stacking / last-horizon slicing,
VLM encoding, optional state discretization), leaving only the
architecture-specific differences to the subclasses:

- ``_init_action_head(cfg)``   — build the action head with VLM-aligned dims
- ``_build_prompt_suffix()``   — prompt engineering (OFT: action-token prompt)
- ``_num_last_layers()``       — which hidden layers condition the head
                                  (``None`` = last layer only; QwenPI: layer-wise)
- ``_forward_loss(examples)``  — training loss flow (base: flow-matching path
                                  shared by QwenPI / QwenGR00T; QwenOFT overrides
                                  with the action-token gather + L1)
- ``_predict(examples)``       — inference flow (same split)

Unlike the WAM families there is no video stream: no ``video_backbone`` is
built, ``forward`` is a thin batch->loss-dict alias (starVLA-compatible), and
the trainer needs no changes — ``prepare_inputs`` keeps the trainer's
``inputs.get("actions")`` guard satisfied.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from openwam.model.architectures.base import BaseArchitecture
from openwam.model.vlm_backbone import build_vlm_backbone

logger = logging.getLogger(__name__)


def _merge_dict(base: dict, override: dict) -> dict:
    """Recursive dict merge; ``override`` wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def deep_merge_defaults(defaults, cfg=None) -> dict:
    """Deep-merge per-architecture defaults with the user config; config wins.

    Replaces starVLA's ``merge_framework_config`` / AccessTrackedConfig: plain
    dict or dataclass in, plain dict out. ``cfg`` may be a dict, an OmegaConf
    DictConfig, or ``None``.
    """
    base = asdict(defaults) if is_dataclass(defaults) else dict(defaults)
    if cfg is None:
        return base
    if not isinstance(cfg, dict):
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(cfg, resolve=True)
    return _merge_dict(base, cfg)


def state2str_transform(state: "np.ndarray", num_bins: int = 256) -> str:
    """Quantise a state vector into ``num_bins`` uniform bins over [-1, 1]
    and return space-separated bin indices.

    Example: [-0.5, 0.1, 0.8] -> "95 133 203"
    """
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, num_bins + 1)[:-1]) - 1
    return " ".join(map(str, discretized_state))


def add_discretized_state_to_instruction(instructions: "list[str]", states: "list[np.ndarray]", num_bins: int = 256):
    """Append discretised proprioceptive state tokens to each instruction.

    Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
    Lets the VLM attend to the robot state purely through its existing
    text-token pathway — no extra encoder required (π₀.5 style).
    """
    updated_instructions = []
    for instr, state in zip(instructions, states):
        state_str = state2str_transform(state[0], num_bins=num_bins)
        updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
    return updated_instructions


def _coerce_images(frames: list) -> list:
    """Coerce a per-sample frame list (PIL or ndarray) into PIL images."""
    from PIL import Image

    coerced = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            coerced.append(frame)
        else:
            coerced.append(Image.fromarray(frame))
    return coerced


def _resize_images(images: list, target_size: tuple) -> list:
    """Resize per-sample PIL image lists to ``target_size`` (H, W)."""
    resized = []
    for multi_view in images:
        resized.append([img.resize((int(target_size[1]), int(target_size[0]))) for img in multi_view])
    return resized


class BaseVLAArchitecture(BaseArchitecture):
    """Base class for VLA architecture variants.

    Composes a ``vlm_backbone`` (frozen / trainable VLM feature extractor,
    built via the ``vlm_backbone`` registry) and an ``action_backbone`` (a
    :class:`VLAActionHead`, built by the subclass ``_init_action_head`` hook).
    Subclasses declare a ``DEFAULTS`` dict merged under the user config.

    Args:
        cfg: Architecture-specific configuration (OmegaConf DictConfig or dict),
            containing the flattened ``architecture:`` block plus the
            ``vlm_backbone:`` group.
    """

    # Per-architecture defaults, deep-merged under the user config.
    DEFAULTS: dict = {}
    # Whether the proprioceptive state rides the instruction as discretised
    # text tokens (π₀.5 style, QwenOFT) instead of the head's state encoder
    # (QwenPI / QwenGR00T).
    _STATE_IN_INSTRUCTION: bool = False

    # --- Construction ---

    def __init__(self, cfg=None):
        if cfg is not None:
            cfg = deep_merge_defaults(self.DEFAULTS, cfg)
        super().__init__(cfg)
        if cfg is not None:
            self._init_vlm_backbone(cfg)
            self.action_backbone = self._init_action_head(cfg)
            self._repeated_diffusion_steps = int(self._cfg_get(cfg, "repeated_diffusion_steps", 1) or 1)

    def _init_vlm_backbone(self, cfg) -> None:
        """Build the VLM backbone from ``cfg.vlm_backbone`` via the registry.

        Mirrors ``BaseWAMArchitecture._init_video_backbone`` (registry-key
        training path / ``_source`` deploy path are not needed here — the VLM
        is always rebuilt from ``checkpoint_path``).
        """
        vlm_cfg = self._cfg_get(cfg, "vlm_backbone", None)
        if vlm_cfg is None:
            raise ValueError("VLA architectures require a `vlm_backbone` config group.")
        vlm_name = self._cfg_get(vlm_cfg, "name")
        vlm_kwargs = {k: v for k, v in vlm_cfg.items() if k != "name"}
        self.vlm_backbone = build_vlm_backbone(vlm_name, **vlm_kwargs)

    def _init_action_head(self, cfg):
        """Build the :class:`VLAActionHead` from ``cfg.action_head``. Subclasses
        inject the VLM-aligned dims (``hidden_size`` / ``cross_attention_dim`` /
        ``num_layers``) via the factory's ``overrides``."""
        from openwam.model.action_backbone.vla_heads import build_vla_action_head

        head_cfg = self._cfg_get(cfg, "action_head", None)
        return build_vla_action_head(head_cfg)

    def _expect_action_model_type(self, cfg, allowed: tuple, default: str) -> str:
        """Fail-fast when the yaml ``action_head.action_model_type`` conflicts
        with the architecture variant (e.g. ``variant=pi`` left with the OFT
        head type); returns the effective type."""
        head_cfg = self._cfg_get(cfg, "action_head", None)
        model_type = self._cfg_get(head_cfg, "action_model_type", None)
        if model_type is not None and str(model_type) not in allowed:
            raise ValueError(
                f"{type(self).__name__} requires action_head.action_model_type in "
                f"{list(allowed)}, got '{model_type}'. Update the yaml action_head block."
            )
        return default

    # --- Backbone composition ---

    @property
    def action_horizon(self) -> int:
        """Number of future action steps predicted per chunk (owned by the head)."""
        return self.action_backbone.action_horizon

    # --- Trainer bridge ---

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Trainer bridge (``openwam_trainer.compute_loss`` orchestration).

        WAM architectures aggregate the batch into per-key tensors here; VLA
        architectures consume the raw sample list inside ``_forward_loss``
        instead. Return the batch under ``batch`` plus the stacked ground-truth
        ``actions`` (satisfying the trainer's ``inputs.get("actions")`` guard);
        VLA ``compute_loss`` re-derives the targets from the batch.
        """
        if isinstance(batch, dict):
            batch = [batch]
        examples = self._to_examples(batch)
        return {"batch": batch, "actions": self._stack_actions(examples)}

    def compute_loss(
        self,
        *,
        batch=None,
        lambda_action: float = 1.0,
        lambda_video: float = 0.0,
        **_,  # trainer passes the WAM kwarg set (latents / flags / masks); unused here
    ) -> dict:
        """VLA training loss entry (template method).

        The trainer calls ``architecture.compute_loss(**prepare_inputs(batch),
        lambda_video=..., lambda_action=...)`` — this signature accepts that
        call shape unchanged, so the trainer needs no VLA-specific branch.

        Returns the trainer loss-dict contract: ``{"loss", "loss_video",
        "loss_action"}``. There is no video stream in the VLA family —
        ``loss_video`` is always zero.
        """
        if batch is None:
            raise ValueError("VLA compute_loss requires the dataset batch (via prepare_inputs).")
        if isinstance(batch, dict):
            batch = [batch]
        loss = self._forward_loss(self._to_examples(batch))
        return {
            "loss": lambda_action * loss,
            "loss_video": torch.tensor(0.0, device=loss.device),
            "loss_action": (lambda_action * loss).detach(),
        }

    # --- starVLA-compatible standalone forward ---

    def forward(self, batch: "list[dict] | dict | None" = None, **kwargs) -> dict:
        """Training forward: batch -> ``{"action_loss": Tensor}`` (starVLA-compatible)."""
        if batch is None:
            batch = kwargs.get("examples")
        if batch is None:
            raise ValueError("VLA forward requires the dataset batch.")
        if isinstance(batch, dict):
            batch = [batch]
        return {"action_loss": self._forward_loss(self._to_examples(batch))}

    # --- Inference ---

    @torch.no_grad()
    def predict_action(self, batch: "list[dict] | dict") -> dict:
        """Inference entry: predict a normalized action chunk.

        Returns ``{"actions": np.ndarray (B, action_horizon, action_dim)}`` —
        real-scale actions when a normalizer is attached (same contract as the
        WAM families' ``generate``), normalized otherwise.
        """
        if isinstance(batch, dict):
            batch = [batch]
        examples = self._to_examples(batch)
        obs_image_size = self._cfg_get(self.cfg, "obs_image_size", None)
        if obs_image_size:
            examples = [{**ex, "image": _resize_images(ex["image"], obs_image_size)} for ex in examples]
        pred = self._predict(examples)
        actions = pred.detach().float().cpu().numpy()
        if self.normalizer is not None:
            actions = self.normalizer.unnormalize(actions)
        return {"actions": actions}

    # --- Shared plumbing (used by the template methods) ---

    @staticmethod
    def _to_examples(batch: list[dict]) -> list[dict]:
        """OpenWAM dataset samples -> starVLA-style examples.

        Mapping: ``sample["video"]`` (the dataloader's temporal window,
        multi-view views composited per frame) -> ``image`` — VLA semantics
        use a single current observation, so the LATEST frame is taken;
        ``sample["prompt"]`` -> ``lang``, ``sample["proprio"]`` -> ``state``;
        ``action`` / ``action_mask`` pass through.
        """
        examples = []
        for sample in batch:
            example = {
                "image": _coerce_images([sample["video"][-1]]),
                "lang": sample["prompt"],
                "action": sample.get("action"),
            }
            if sample.get("proprio") is not None:
                example["state"] = sample["proprio"]
            if sample.get("action_mask") is not None:
                example["action_mask"] = sample["action_mask"]
            examples.append(example)
        return examples

    def _stack_actions(self, examples: list[dict], horizon: Optional[int] = None) -> Tensor:
        """Stack per-sample actions -> (B, T, action_dim) tensor, taking the
        last ``horizon`` segment (the dataloader may emit longer windows)."""
        actions = [ex["action"] for ex in examples]
        if any(a is None for a in actions):
            raise ValueError("VLA training requires actions in every sample.")
        tensor = torch.as_tensor(np.array(actions), device=self.device, dtype=self.dtype)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if horizon is not None:
            tensor = tensor[:, -horizon:, :]
        return tensor

    def _stack_state(self, examples: list[dict], times: int = 1) -> Optional[Tensor]:
        """Stack per-sample proprio states -> (B, 1, state_dim) tensor (or None)."""
        states = [ex.get("state") for ex in examples]
        if any(s is None for s in states):
            if any(s is not None for s in states):
                raise ValueError("Mixed state presence in batch: all samples must be consistent.")
            return None
        state = torch.as_tensor(np.array(states), device=self.device, dtype=self.dtype)
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if times > 1:
            state = state.repeat(times, 1, 1)
        return state

    def _stack_action_masks(self, examples: list[dict], horizon: Optional[int] = None) -> Optional[Tensor]:
        """Stack per-sample action masks -> (B, T) or (B, T, action_dim) bool
        tensor (or None). Per-timestep (B, T) masks are broadcast across the
        action dims inside the head's loss."""
        masks = [ex.get("action_mask") for ex in examples]
        if masks[0] is None:
            return None
        mask = torch.as_tensor(np.array(masks), device=self.device, dtype=torch.bool)
        if horizon is not None:
            mask = mask[:, -horizon:]
        return mask

    def _repeat_context(self, hidden: Union[Tensor, list], times: int) -> Union[Tensor, list]:
        """Repeat context features along the batch axis (``repeated_diffusion_steps``)."""
        if isinstance(hidden, list):
            return [h.repeat(times, 1, 1) for h in hidden]
        return hidden.repeat(times, 1, 1)

    def _encode(
        self, examples: list[dict], *, num_last_layers: Optional[int] = None
    ) -> tuple:
        """Build VLM inputs and run the VLM; returns (hidden, attention_mask, vlm_inputs).

        ``hidden`` is the last layer (B, L, H) or the last ``num_last_layers``
        layers as a list. Optionally prepends the discretised proprio state to
        the instruction (``_STATE_IN_INSTRUCTION``, π₀.5 style) and the
        per-architecture prompt suffix.
        """
        images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        if state is not None and self._STATE_IN_INSTRUCTION:
            instructions = add_discretized_state_to_instruction(instructions, state)
        prompt_suffix = self._build_prompt_suffix()
        if prompt_suffix:
            instructions = [instruction + prompt_suffix for instruction in instructions]

        vlm_inputs = self.vlm_backbone.prepare_vlm_inputs(prompts=instructions, images=images)
        hidden, attention_mask = self.vlm_backbone.forward_hidden_states(
            vlm_inputs, num_last_layers=num_last_layers
        )
        return hidden, attention_mask, vlm_inputs

    # --- Per-architecture hooks ---

    def _build_prompt_suffix(self) -> str:
        """Prompt engineering appended to each instruction. Default: none."""
        return ""

    def _num_last_layers(self) -> Optional[int]:
        """How many last hidden layers condition the action head.
        ``None`` = last layer only (QwenOFT / QwenGR00T); QwenPI returns the
        layer-wise count."""
        return None

    def _forward_loss(self, examples: list[dict]) -> Tensor:
        """Training loss (flow-matching path, shared by QwenPI / QwenGR00T).

        Repeats the context / targets along the batch axis
        (``repeated_diffusion_steps``) and delegates to the head's
        ``compute_loss``. QwenOFT overrides with the action-token gather + L1.
        """
        hidden, attention_mask, _ = self._encode(examples, num_last_layers=self._num_last_layers())
        times = self._repeated_diffusion_steps
        actions = self._stack_actions(examples, self.action_horizon)
        state = self._stack_state(examples, times)
        if times > 1:
            actions = actions.repeat(times, 1, 1)
            hidden = self._repeat_context(hidden, times)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(times, 1).to(dtype=torch.bool)
        return self.action_backbone.compute_loss(
            hidden,
            actions,
            state=state,
            encoder_attention_mask=attention_mask.to(dtype=torch.bool) if attention_mask is not None else None,
        )

    def _predict(self, examples: list[dict]) -> Tensor:
        """Inference prediction (flow-matching path, shared by QwenPI /
        QwenGR00T). QwenOFT overrides with the action-token gather."""
        hidden, attention_mask, _ = self._encode(examples, num_last_layers=self._num_last_layers())
        state = self._stack_state(examples)
        return self.action_backbone.predict_action(
            hidden,
            state=state,
            encoder_attention_mask=attention_mask.to(dtype=torch.bool) if attention_mask is not None else None,
        )

"""Qwen3-VL backbone — ONE class for both consumers (tri_system understanding
stream and the VLA family), parameterized instead of subclassed.

The two former classes (``Qwen3VLBackbone`` tri variant and
``Qwen3VLVLABackbone``) differed ONLY in constructor-level choices; the
differences live in yaml now:

======================  =====================  ==========================
knob                    tri (qwen3_vl_2b)      VLA (qwen3_vl_vla)
======================  =====================  ==========================
prompt_style            processor_text         chat_messages
add_generation_prompt   false                  true
max_length              512 (truncate)         null (never — action
                                               placeholders must survive)
padding_side            tokenizer default      left
======================  =====================  ==========================

The two prompt-rendering branches are kept VERBATIM from the former classes:
prompt text feeds tokenization, and any character drift would silently change
model behavior (unverifiable offline — no real Qwen3-VL weights in CI), so
this is a config-selected branch, not a unified rewrite. Everything else
(loading, forward core, deploy assets, batching) was literal duplication and
is now written once.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from openwam.model.vlm_backbone.base import VlmBackbone

logger = logging.getLogger(__name__)


class Qwen3VLBackbone(VlmBackbone):
    """Qwen3-VL wrapper shared by tri_system and the VLA architectures.

    Construction loads the pretrained VLM via ``from_pretrained``;
    ``load_pretrained=False`` random-initializes from the local config (same
    stable path as upstream PreTrainedModel — for CPU smoke tests).
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype = torch.bfloat16,
        load_pretrained: bool = True,
        *,
        prompt_style: str = "chat_messages",
        add_generation_prompt: bool = True,
        max_length: Optional[int] = None,
        padding_side: Optional[str] = None,
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        if prompt_style not in ("chat_messages", "processor_text"):
            raise ValueError(
                f"prompt_style must be 'chat_messages' or 'processor_text', got {prompt_style!r}"
            )
        self._prompt_style = prompt_style
        self._add_generation_prompt = add_generation_prompt
        self._max_length = max_length

        self.dtype = dtype
        self._checkpoint_path = checkpoint_path
        # ``trust_remote_code`` is intentionally NOT set: ``Qwen3VLForConditionalGeneration``
        # is imported directly from upstream ``transformers``, so no remote
        # ``modeling_*.py`` is needed. Leaving ``trust_remote_code=True`` would
        # let a config/model directory under ``checkpoint_path`` execute
        # arbitrary Python at load time — an unnecessary code-execution
        # surface in training and deployment.
        if load_pretrained:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_path, attn_implementation=attn_implementation, dtype=dtype
            )
        else:
            # Public PreTrainedModel constructor + ``.to(dtype=...)`` instead of
            # the private ``_from_config`` whose signature shifted across
            # releases — equivalent semantically (random init from cfg, no
            # weights downloaded) and stable across releases.
            cfg = AutoConfig.from_pretrained(checkpoint_path)
            self.vlm_model = Qwen3VLForConditionalGeneration(cfg).to(dtype=dtype)

        self.processor = None
        if checkpoint_path:
            try:
                self.processor = AutoProcessor.from_pretrained(checkpoint_path)
            except Exception:
                if load_pretrained:
                    raise
        if self.processor is not None and padding_side:
            self.processor.tokenizer.padding_side = padding_side

    # --- VlmBackbone ABC ---

    @property
    def hidden_size(self) -> int:
        return int(self.vlm_model.config.text_config.hidden_size)

    @property
    def num_vl_layers(self) -> int:
        """Number of VL transformer layers (Qwen3-VL stores it under text_config)."""
        cfg = self.vlm_model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        return int(text_cfg.num_hidden_layers)

    def get_submodule(self, name: str) -> nn.Module | None:
        return getattr(self, name, None)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.vlm_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype = dtype
        self.to(dtype=dtype, device=device)

    # --- Prompt rendering (two config-selected branches, kept verbatim) ---

    def format_prompt(self, prompt: str) -> str:
        """``processor_text`` branch: render the chat template around ONE image
        placeholder (tri_system convention; the processor call supplies images)."""
        if self.processor is None:
            return prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def prepare_vlm_inputs(self, prompts: list[str], images: list[Any]) -> dict[str, Tensor]:
        """``(prompts, per-sample image lists)`` -> one processor tensor dict."""
        if self._prompt_style == "processor_text":
            if self.processor is None:
                raise RuntimeError("Qwen3VLBackbone.prepare_vlm_inputs requires a loaded AutoProcessor.")
            texts = [self.format_prompt(prompt) for prompt in prompts]
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=self._max_length is not None,
                max_length=self._max_length,
            )
            return {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        if self._prompt_style != "chat_messages":  # constructor guard, re-checked at use
            raise ValueError(f"Unknown prompt_style {self._prompt_style!r}")

        # chat_messages branch: embed every sample's image list into the message
        # content. NO truncation — VLA prompts carry trailing action-token
        # placeholders (oftp) that right-truncation would cut off.
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        messages = []
        for imgs, prompt in zip(images, prompts):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=self._add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.device)

    def batch_vlm_inputs(self, vlm_inputs: dict | list[dict]) -> dict[str, Tensor]:
        """Collate one already-batched dict (passthrough of tensor entries) or a
        list of per-sample dicts (right-pad + concat)."""
        if isinstance(vlm_inputs, dict):
            # Only tensor entries are forwarded to the VLM; non-tensor metadata
            # (processor attributes, image grids stored as lists) is dropped.
            return {
                key: value for key, value in vlm_inputs.items() if isinstance(value, torch.Tensor) and value is not None
            }

        if not isinstance(vlm_inputs, list) or not vlm_inputs:
            raise ValueError("vlm_inputs must be a non-empty dict or list of dicts.")

        input_ids_list = [item["input_ids"] for item in vlm_inputs]
        attention_mask_list = [
            item.get("attention_mask", torch.ones_like(item["input_ids"], dtype=torch.long)) for item in vlm_inputs
        ]
        for idx, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            if ids.ndim != 2:
                raise ValueError(f"vlm_inputs[{idx}]['input_ids'] must be 2D [B, L], got shape {tuple(ids.shape)}")
            if mask.shape != ids.shape:
                raise ValueError(
                    f"vlm_inputs[{idx}]['attention_mask'] must match input_ids shape "
                    f"{tuple(ids.shape)}, got {tuple(mask.shape)}"
                )
        max_seq_len = max(ids.shape[1] for ids in input_ids_list)
        pad_token_id = self._pad_token_id()

        padded_ids = []
        padded_masks = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad = max_seq_len - ids.shape[1]
            if pad > 0:
                padded_ids.append(F.pad(ids, (0, pad), value=pad_token_id))
                padded_masks.append(F.pad(mask, (0, pad), value=0))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)

        batched = {
            "input_ids": torch.cat(padded_ids, dim=0),
            "attention_mask": torch.cat(padded_masks, dim=0),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [item.get(key) for item in vlm_inputs]
            if any(value is not None for value in values):
                if not all(value is not None for value in values):
                    raise ValueError(f"Mixed missing/non-missing {key} in vlm_inputs list.")
                batched[key] = torch.cat(values, dim=0)
        return batched

    def _pad_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)
        config_pad = getattr(getattr(self.vlm_model, "config", None), "pad_token_id", None)
        if config_pad is not None:
            return int(config_pad)
        raise ValueError(
            "Qwen3VLBackbone: cannot determine pad_token_id from processor.tokenizer "
            "or model.config. Pass a processor whose tokenizer has pad_token_id set, "
            "or set pad_token_id on the model config."
        )

    def _move_inputs(self, batch: dict) -> dict:
        model_inputs = {}
        for key, value in batch.items():
            if key in {"pixel_values", "pixel_values_videos"}:
                value = value.to(device=self.device, dtype=self.dtype)
            else:
                value = value.to(device=self.device)
            model_inputs[key] = value
        return model_inputs

    # --- Forward ---

    def forward_hidden_states(
        self, vlm_inputs: dict, *, num_last_layers: Optional[int] = None
    ) -> tuple[Union[Tensor, list[Tensor]], Optional[Tensor]]:
        """Run the Qwen3-VL base model; return (hidden_states, attention_mask).

        ``num_last_layers=None`` -> last layer only (B, L, H);
        ``num_last_layers=k`` -> list of the last k layers (VLA pi-style
        layer-wise DiT conditioning). Runs ``self.vlm_model.model`` (the
        multimodal base, no LM head) — same hidden states the full wrapper
        returns, minus the wasted logits projection.
        """
        batch = self.batch_vlm_inputs(vlm_inputs)
        model_inputs = self._move_inputs(batch)

        outputs = self.vlm_model.model(
            **model_inputs,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        hidden = list(hidden_states[-num_last_layers:]) if num_last_layers else hidden_states[-1]
        return hidden, model_inputs.get("attention_mask")

    def extract_features(self, vlm_inputs: dict | list[dict]) -> Tensor:
        """VlmBackbone contract: frozen last-hidden-state (B, L, hidden_size).

        **Alignment assumption** (Qwen2-VL family convention): the Qwen3-VL
        processor expands ``<image>`` placeholders in ``input_ids`` to one
        token per visual patch before the model forward, so
        ``hidden_states.shape[1] == input_ids.shape[1] == attention_mask.shape[1]``.
        This lets tri_system derive ``UnderstandingState.und_mask`` directly from
        ``vlm_inputs["attention_mask"]``; ``UnderstandingExpert.prepare_state``
        validates this contract at runtime.
        """
        hidden, _ = self.forward_hidden_states(vlm_inputs)
        assert isinstance(hidden, Tensor)
        return hidden

    def compute_vlm_loss(self, vlm_inputs: dict, labels: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Co-training leg (spec §6.5): token-sum language CE over the full VLM.

        Runs the full ``vlm_model`` (with LM head — unlike
        :meth:`forward_hidden_states`, which stops at the base model) and
        returns ``(sum_ce, n_tokens)``, NOT a mean: the caller normalizes
        with a GLOBAL token count (FluxVLA ``losses/causal_lm.py`` pattern).
        ``labels=None`` shifts ``input_ids``; ``-100`` positions are ignored.
        """
        batch = self.batch_vlm_inputs(vlm_inputs)
        model_inputs = self._move_inputs(batch)

        outputs = self.vlm_model(
            **model_inputs,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            labels=labels,
            return_dict=True,
        )
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = (labels if labels is not None else model_inputs["input_ids"])[:, 1:]
        n_tokens = (shift_labels != -100).sum()
        sum_ce = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return sum_ce, n_tokens

    # --- Deploy ---

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Copy the VLM checkpoint into the deploy dir so deploy is self-contained
        (no dependency on the training-time checkpoint_path). Idempotent. Invoked by
        the architecture's save_assets_for_deployment, same hook as video backbones."""
        if not self._checkpoint_path:
            return
        dest = os.path.join(output_dir, "vlm_backbone")
        if os.path.exists(dest):
            return
        shutil.copytree(self._checkpoint_path, dest)
        logger.info("Copied VLM checkpoint to %s", dest)

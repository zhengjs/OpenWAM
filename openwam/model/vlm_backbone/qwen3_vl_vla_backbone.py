"""Qwen3-VL VLA backbone: chat-template wrapper + multi-layer hidden states.

Adapted from starVLA (starVLA/model/modules/vlm/QWen3.py,
``_QWen3_VL_Interface``), modified for OpenWAM:

- subclass of the :class:`VlmBackbone` ABC so the host VLA architecture
  composes it like any other backbone (freeze / dtype / deploy assets all flow
  through ``BaseArchitecture``);
- prompt construction embeds the per-sample multi-view image lists in the
  chat message content (starVLA convention) instead of the
  ``processor(text=, images=)`` path used by the tri_system wrapper;
- ``forward_hidden_states`` returns multi-layer hidden states + the backbone
  attention mask for the VLA action heads (QwenOFT / QwenPI / QwenGR00T);
- the FAST-token label masking of the upstream wrapper is dropped — it is only
  needed by the QwenFast architecture, which is not ported.

**No truncation**: the VLA prompts carry trailing action-token placeholders
(QwenOFT), so right-truncation would cut them off — the chat path is never
truncated here (deliberately different from the tri_system wrapper).

Alignment assumption (Qwen2-VL family convention): the processor expands
``<image>`` placeholders in ``input_ids`` to one token per visual patch before
the model forward, so ``hidden_states.shape[1] == input_ids.shape[1] ==
attention_mask.shape[1]`` (see Qwen3VLBackbone.extract_features).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.vlm_backbone.base import VlmBackbone

logger = logging.getLogger(__name__)


class Qwen3VLVLABackbone(VlmBackbone):
    """Qwen3-VL wrapper for the VLA architectures (OFT / PI / GR00T).

    Construction loads the pretrained VLM via ``from_pretrained``;
    ``load_pretrained=False`` random-initializes from the local config (same
    stable path as :class:`Qwen3VLBackbone` — for CPU smoke tests).
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype = torch.bfloat16,
        load_pretrained: bool = True,
        padding_side: str = "left",
        attn_implementation: str = "sdpa",
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Qwen3VL requires transformers>=4.50 with Qwen3VLForConditionalGeneration. "
                "Install via `pip install -U transformers`."
            ) from e

        self.dtype = dtype
        self._checkpoint_path = checkpoint_path
        if load_pretrained:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                attn_implementation=attn_implementation,
                dtype=dtype,
            )
        else:
            # Public PreTrainedModel constructor + ``.to(dtype=...)`` instead of
            # the private ``_from_config`` whose signature shifted across
            # releases — random init from cfg, no weights downloaded.
            cfg = AutoConfig.from_pretrained(checkpoint_path)
            self.vlm_model = Qwen3VLForConditionalGeneration(cfg).to(dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
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

    def prepare_vlm_inputs(self, prompts: list[str], images: list[Any]) -> dict[str, Tensor]:
        """Build Qwen3-VL chat-template inputs from (prompts, per-sample image lists).

        ``images`` is a list of per-sample image lists (multi-view supported),
        matching the OpenWAM dataloader ``sample["video"]`` shape.
        """
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
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.device)

    def batch_vlm_inputs(self, vlm_inputs: dict | list[dict]) -> dict[str, Tensor]:
        """Collate to a single batched tensor dict.

        The VLA architectures always pass one already-batched dict (the chat
        template batches per-sample messages directly), so the dict path is a
        tensor passthrough; the per-sample list path of the ABC is not needed.
        """
        if not isinstance(vlm_inputs, dict):
            raise NotImplementedError(
                "Qwen3VLVLABackbone.batch_vlm_inputs expects one batched dict "
                "(from prepare_vlm_inputs); per-sample list batching is not supported."
            )
        return {
            key: value for key, value in vlm_inputs.items() if isinstance(value, Tensor) and value is not None
        }

    def extract_features(self, vlm_inputs: dict | list[dict]) -> Tensor:
        """VlmBackbone contract: frozen last-hidden-state (B, L, hidden_size)."""
        hidden, _ = self.forward_hidden_states(vlm_inputs)
        assert isinstance(hidden, Tensor)
        return hidden

    # --- VLA-specific entry point ---

    def forward_hidden_states(
        self, vlm_inputs: dict, *, num_last_layers: Optional[int] = None
    ) -> tuple[Union[Tensor, list[Tensor]], Optional[Tensor]]:
        """Run the Qwen3-VL base model and return (hidden_states, attention_mask).

        ``num_last_layers=None`` -> last layer only (B, L, H);
        ``num_last_layers=k`` -> list of the last k layers (QwenPI-style
        layer-wise DiT conditioning). Runs ``self.vlm_model.model`` (the
        multimodal base, no LM head) — same hidden states the full wrapper
        returns, minus the wasted logits projection.
        """
        batch = self.batch_vlm_inputs(vlm_inputs)
        model_inputs = {}
        for key, value in batch.items():
            if key in {"pixel_values", "pixel_values_videos"}:
                value = value.to(device=self.device, dtype=self.dtype)
            else:
                value = value.to(device=self.device)
            model_inputs[key] = value

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

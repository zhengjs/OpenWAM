"""VLASystem variant behavior pins (spec_zh.md §6.3 / P3 acceptance).

The per-variant subclasses (``vla_qwen_oft/pi/groot``) were deleted after
the loss reproduction gate proved ``vla_system`` bit-identical to them
(git 4412820, 2026-11-16). What remains to pin:

1. **Seeded-loss PIN per variant** — exact float values under the stub VLM;
   any conditioning/head regression moves them.
2. Registry ownership: every ``(framework=vla, variant)`` key resolves to
   ``vla_system`` — the only entry point for the family.
3. Explicit ``connector`` override changes conditioning without new code.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.model.vlm_backbone.base import VlmBackbone

B, SEQ, HIDDEN = 2, 12, 64
ACTION_DIM, HORIZON, STATE_DIM = 4, 3, 5
NUM_VL_LAYERS = 2
ACTION_TOKEN_ID = 151669

# Fixed-seed losses (seed 42 build / seed 0 batch, stub VLM). Pinned at the
# subclass deletion commit; the subclass-equivalence evidence lives in git
# history (4412820: bitwise parity vs vla_qwen_oft/pi/groot).
_PINNED_LOSSES = {
    "oftp": 1.2374356985092163,
    "pi": 2.750852584838867,
    "groot": 2.650782346725464,
}


class _StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ACTION_TOKEN_ID if "🔍" in text else 1]}


class _StubProcessor:
    def __init__(self):
        self.tokenizer = _StubTokenizer()


class _SeedableStubVLM(VlmBackbone):
    """Stub VLM whose hidden states derive from input_ids — deterministic."""

    def __init__(self, hidden_size=HIDDEN, num_layers=NUM_VL_LAYERS, **_):
        super().__init__()
        self._hidden_size, self._num_layers = hidden_size, num_layers
        self.processor = _StubProcessor()

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_vl_layers(self):
        return self._num_layers

    def prepare_vlm_inputs(self, prompts, images):
        b = len(prompts)
        ids = torch.arange(2, 2 + b * (SEQ - HORIZON), dtype=torch.long).reshape(b, SEQ - HORIZON) % 98 + 2
        act = torch.full((b, HORIZON), ACTION_TOKEN_ID, dtype=torch.long)
        return {"input_ids": torch.cat([ids, act], dim=1),
                "attention_mask": torch.ones(b, SEQ, dtype=torch.long)}

    def batch_vlm_inputs(self, vlm_inputs):
        return {k: v for k, v in vlm_inputs.items() if isinstance(v, torch.Tensor)}

    def extract_features(self, vlm_inputs):
        return self.forward_hidden_states(vlm_inputs)[0]

    def forward_hidden_states(self, vlm_inputs, *, num_last_layers=None):
        batch = self.batch_vlm_inputs(vlm_inputs)
        b, length = batch["input_ids"].shape
        h = batch["input_ids"].float().unsqueeze(-1).expand(b, length, self._hidden_size) % 7.0
        h = h + torch.arange(self._hidden_size).float() * 0.01
        if num_last_layers is None:
            return h, batch.get("attention_mask")
        return [h * (1.0 + 0.1 * i) for i in range(num_last_layers)], batch.get("attention_mask")


@pytest.fixture
def stub_vlm(monkeypatch):
    monkeypatch.setattr(
        "openwam.model.architectures.vla.base_vla.build_vlm_backbone",
        lambda name, **kwargs: _SeedableStubVLM(**kwargs),
    )


def _batch():
    torch.manual_seed(0)
    np.random.seed(0)
    return [
        {
            "video": [Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)) for _ in range(3)],
            "prompt": f"instruction {i}",
            "action": torch.randn(HORIZON + 2, ACTION_DIM),
            "proprio": torch.randn(1, STATE_DIM),
        }
        for i in range(B)
    ]


def _cfg(variant: str) -> dict:
    return {
        "framework": "vla",
        "variant": variant,
        "repeated_diffusion_steps": 1,
        "obs_image_size": None,
        "action_head": {"action_dim": ACTION_DIM, "action_horizon": HORIZON, "state_dim": STATE_DIM},
        "vlm_backbone": {"name": "qwen3_vl_vla", "checkpoint_path": "/dummy"},
    }


def _loss_of(variant: str) -> torch.Tensor:
    from openwam.model import build_architecture

    torch.manual_seed(42)
    arch = build_architecture("vla_system", _cfg(variant))
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    inputs = arch.prepare_inputs(_batch())
    return arch.compute_loss(**inputs, lambda_action=1.0)["loss"]


@pytest.mark.parametrize("variant", ["oftp", "pi", "groot"])
def test_variant_loss_matches_pin(stub_vlm, variant):
    loss = _loss_of(variant)
    assert loss.item() == _PINNED_LOSSES[variant]


def test_registry_variants_all_resolve_to_vla_system():
    from openwam.model import resolve_architecture_config
    from types import SimpleNamespace

    for variant in ("system", "oftp", "pi", "groot"):
        cfg = SimpleNamespace(
            architecture={"framework": "vla", "variant": variant, "action_head": {"action_dim": 7}},
            action_backbone=None, video_backbone=None,
            vlm_backbone={"name": "qwen3_vl_vla"},
        )
        assert resolve_architecture_config(cfg).registry_name == "vla_system"


def test_connector_override_changes_conditioning(stub_vlm):
    """Explicit ``connector`` config wins over the preset — the composition
    knob that replaces subclass proliferation."""
    from openwam.model import build_architecture

    cfg = _cfg("groot")
    cfg["connector"] = "layerwise_hidden"  # override the groot preset
    arch = build_architecture("vla_system", cfg)
    assert type(arch.connector).__name__ == "LayerwiseHiddenConnector"
    assert arch.connector.num_layers == NUM_VL_LAYERS  # VLM-aligned at build

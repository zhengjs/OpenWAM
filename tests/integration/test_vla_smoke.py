"""End-to-end smoke tests for the VLA architectures (CPU, stubbed VLM).

The production path loads Qwen3-VL in ``Qwen3VLVLABackbone.__init__``; these
tests monkeypatch ``base_vla.build_vlm_backbone`` with a ``_StubVLM`` exposing
the :class:`VlmBackbone` contract. Each architecture then runs the production
build path:

    build_architecture(registry_name, yaml-shaped cfg)
      -> prepare_inputs (trainer bridge)
      -> compute_loss (the exact ``openwam_trainer`` orchestration shape)
      -> backward
      -> predict_action
      -> checkpoint save/load round-trip

Verifies, per variant (vla_qwen_oft / vla_qwen_pi / vla_qwen_groot):
  1. Builds from a yaml-shaped config (framework=vla + variant).
  2. ``backbones`` scans {vlm_backbone, action_backbone}.
  3. ``compute_loss`` returns the trainer loss-dict contract and backward flows.
  4. ``predict_action`` returns (B, horizon, action_dim).
  5. ``freeze_modules(["vlm_backbone"])`` freezes the VLM subtree only.
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

_VLA_NAMES = {"oftp": "vla_qwen_oft", "pi": "vla_qwen_pi", "groot": "vla_qwen_groot"}


class _StubTokenizer:
    """Tokenizes the OFT action token to a fixed id; anything else -> [1]."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ACTION_TOKEN_ID if "🔍" in text else 1]}


class _StubProcessor:
    def __init__(self):
        self.tokenizer = _StubTokenizer()


class _StubVLM(VlmBackbone):
    """Minimal VlmBackbone double: fabricates hidden states shaped from the
    stubbed ``input_ids`` (OFT's gather needs the action tokens present in
    ``input_ids`` at ``action_horizon`` fixed positions)."""

    def __init__(self, hidden_size=HIDDEN, num_layers=NUM_VL_LAYERS, **_):
        super().__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self.processor = _StubProcessor()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_vl_layers(self) -> int:
        return self._num_layers

    def prepare_vlm_inputs(self, prompts, images):
        B = len(prompts)
        ids = torch.randint(2, 100, (B, SEQ - HORIZON))
        action_ids = torch.full((B, HORIZON), ACTION_TOKEN_ID, dtype=torch.long)
        input_ids = torch.cat([ids, action_ids], dim=1)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def batch_vlm_inputs(self, vlm_inputs):
        return {k: v for k, v in vlm_inputs.items() if isinstance(v, torch.Tensor)}

    def extract_features(self, vlm_inputs):
        hidden, _ = self.forward_hidden_states(vlm_inputs)
        return hidden

    def forward_hidden_states(self, vlm_inputs, *, num_last_layers=None):
        batch = self.batch_vlm_inputs(vlm_inputs)
        B, L = batch["input_ids"].shape
        hidden = torch.randn(B, L, self._hidden_size)
        if num_last_layers is None:
            return hidden, batch.get("attention_mask")
        return [hidden.clone() for _ in range(num_last_layers)], batch.get("attention_mask")


@pytest.fixture
def stub_vlm(monkeypatch):
    monkeypatch.setattr(
        "openwam.model.architectures.vla.base_vla.build_vlm_backbone",
        lambda name, **kwargs: _StubVLM(**kwargs),
    )


def _fake_batch(b: int, num_frames: int = 3) -> list[dict]:
    batch = []
    for i in range(b):
        frames = [Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)) for _ in range(num_frames)]
        batch.append(
            {
                "video": frames,
                "prompt": f"instruction {i}",
                "action": torch.randn(HORIZON + 2, ACTION_DIM),
                "proprio": torch.randn(1, STATE_DIM),
            }
        )
    return batch


def _vla_cfg(variant: str, arch_extra: dict | None = None, head_extra: dict | None = None) -> dict:
    cfg = {
        "framework": "vla",
        "variant": variant,
        "repeated_diffusion_steps": 1,
        "obs_image_size": None,
        "action_head": {
            "action_dim": ACTION_DIM,
            "action_horizon": HORIZON,
            "state_dim": STATE_DIM,
        },
        "vlm_backbone": {"name": "qwen3_vl_vla", "checkpoint_path": "/dummy"},
    }
    cfg.update(arch_extra or {})
    cfg["action_head"].update(head_extra or {})
    return cfg


def _build_vla(variant: str, head_extra: dict | None = None, arch_extra: dict | None = None):
    from openwam.model import build_architecture

    arch = build_architecture(_VLA_NAMES[variant], _vla_cfg(variant, arch_extra, head_extra))
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


# ---------------------------------------------------------------------------
# 1. Build + backbones scan (all variants)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["oftp", "pi", "groot"])
def test_vla_loads_and_backbones_scan(stub_vlm, variant):
    arch = _build_vla(variant, {"diffusion_model_cfg": {"num_layers": 2, "dropout": 0.0, "final_dropout": False}})
    assert set(arch.backbones.keys()) == {"vlm_backbone", "action_backbone"}
    assert arch.action_dim == ACTION_DIM
    assert arch.action_horizon == HORIZON
    assert arch.vlm_backbone.hidden_size == HIDDEN


def test_vla_oft_action_token_id(stub_vlm):
    arch = _build_vla("oftp")
    assert arch.action_token_id == ACTION_TOKEN_ID


# ---------------------------------------------------------------------------
# 2. Trainer-orchestrated compute_loss (mirrors openwam_trainer:555-574)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["oftp", "pi", "groot"])
def test_vla_compute_loss_backward(stub_vlm, variant):
    arch = _build_vla(variant, {"diffusion_model_cfg": {"num_layers": 2, "dropout": 0.0, "final_dropout": False}})

    batch = _fake_batch(B)
    inputs = arch.prepare_inputs(batch)
    # The trainer's action guard (openwam_trainer:561) must pass.
    assert inputs.get("actions") is not None

    result = arch.compute_loss(**inputs, lambda_video=1.0, lambda_action=1.0)
    assert set(result) == {"loss", "loss_video", "loss_action"}
    assert torch.isfinite(result["loss"])
    assert result["loss_video"] == 0.0
    assert torch.isfinite(result["loss_action"])

    result["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in arch.action_backbone.parameters())


def test_vla_last_horizon_slicing(stub_vlm):
    """Actions longer than the horizon are sliced to the last ``action_horizon``."""
    arch = _build_vla("oftp")
    batch = _fake_batch(2)
    stacked = arch._stack_actions(arch._to_examples(batch))
    assert stacked.shape == (2, HORIZON + 2, ACTION_DIM)
    sliced = arch._stack_actions(arch._to_examples(batch), arch.action_horizon)
    assert sliced.shape == (2, HORIZON, ACTION_DIM)
    assert torch.allclose(sliced, stacked[:, -HORIZON:, :])


def test_vla_compute_loss_requires_batch(stub_vlm):
    arch = _build_vla("oftp")
    with pytest.raises(ValueError, match="batch"):
        arch.compute_loss()


# ---------------------------------------------------------------------------
# 3. predict_action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["oftp", "pi", "groot"])
def test_vla_predict_action(stub_vlm, variant):
    arch = _build_vla(variant, {"diffusion_model_cfg": {"num_layers": 2, "dropout": 0.0, "final_dropout": False}})
    out = arch.predict_action(_fake_batch(1))
    assert out["actions"].shape == (1, HORIZON, ACTION_DIM)
    assert np.isfinite(out["actions"]).all()


def test_vla_predict_action_unnormalizes_with_normalizer(stub_vlm):
    class _Normalizer:
        def unnormalize(self, actions):
            return actions * 2.0

    arch = _build_vla("oftp")
    arch.attach_normalizer(_Normalizer())
    # predict_action's noise init is random — seed so both runs see identical noise.
    torch.manual_seed(123)
    out = arch.predict_action(_fake_batch(1))
    torch.manual_seed(123)
    raw = arch._predict(arch._to_examples(_fake_batch(1))).detach().numpy()
    assert np.allclose(out["actions"], raw * 2.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. Freeze / checkpoint round-trip
# ---------------------------------------------------------------------------


def test_vla_freeze_vlm_backbone(stub_vlm):
    arch = _build_vla("groot", {"diffusion_model_cfg": {"num_layers": 2, "dropout": 0.0, "final_dropout": False}})
    frozen = arch.freeze_modules(["vlm_backbone"])
    assert frozen == ["vlm_backbone"]
    assert not any(p.requires_grad for p in arch.vlm_backbone.parameters())
    # Head stays trainable.
    assert any(p.requires_grad for p in arch.action_backbone.parameters())


def test_vla_checkpoint_round_trip(stub_vlm, tmp_path):
    arch = _build_vla("oftp")
    path = str(tmp_path / "model.safetensors")
    arch.save_checkpoint(path)
    arch2 = _build_vla("oftp")
    arch2.load_checkpoint(path)
    for (k1, v1), (k2, v2) in zip(arch.state_dict().items(), arch2.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)


# ---------------------------------------------------------------------------
# 5. Config guards
# ---------------------------------------------------------------------------


def test_vla_oft_rejects_mismatched_head_type(stub_vlm):
    with pytest.raises(ValueError, match="action_model_type"):
        _build_vla("oftp", {"action_model_type": "LayerwiseFM"})


def test_vla_requires_vlm_backbone_group(stub_vlm):
    from openwam.model import build_architecture

    cfg = _vla_cfg("oftp")
    del cfg["vlm_backbone"]
    with pytest.raises(ValueError, match="vlm_backbone"):
        build_architecture(_VLA_NAMES["oftp"], cfg)

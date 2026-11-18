"""Co-training rules (spec_zh.md §6.5 / P3 acceptance).

Pins: single-backward composition (``total = action + w * vlm`` with the
global token-sum CE normalization), the freeze fast-fail
(``vlm_loss_weight > 0`` contradicts a frozen ``vlm_backbone``), grad flow
into the VLM when co-trained, and zero-change at ``vlm_loss_weight=0``
(the P3.4 loss-reproduction gate keeps passing untouched).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from openwam.model.vlm_backbone.base import VlmBackbone

B, SEQ, HIDDEN = 2, 12, 64
ACTION_DIM, HORIZON, STATE_DIM = 4, 3, 5
NUM_VL_LAYERS = 2
ACTION_TOKEN_ID = 151669
VOCAB = ACTION_TOKEN_ID + 10

import sys  # noqa: E402

sys.path.insert(0, "tests/integration")
from test_vla_system_parity import _SeedableStubVLM  # noqa: E402


def _add_lm_head(vlm: "_SeedableStubVLM") -> "_SeedableStubVLM":
    """Extend the parity stub with an LM head + token-sum CE (the contract the
    real ``Qwen3VLBackbone.compute_vlm_loss`` implements)."""
    vlm.lm_head = torch.nn.Linear(HIDDEN, VOCAB)

    def compute_vlm_loss(vlm_inputs, labels=None):
        batch = vlm.batch_vlm_inputs(vlm_inputs)
        ids = batch["input_ids"]
        h = ids.float().unsqueeze(-1).expand(ids.shape[0], ids.shape[1], HIDDEN) % 7.0
        h = h + torch.arange(HIDDEN).float() * 0.01
        logits = vlm.lm_head(h).float()
        shift_logits = logits[:, :-1]
        shift_labels = (labels if labels is not None else ids)[:, 1:]
        n_tokens = (shift_labels != -100).sum()
        sum_ce = F.cross_entropy(
            shift_logits.reshape(-1, VOCAB), shift_labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        return sum_ce, n_tokens

    vlm.compute_vlm_loss = compute_vlm_loss
    return vlm


@pytest.fixture
def cotrain_vlm(monkeypatch):
    monkeypatch.setattr(
        "openwam.model.architectures.vla.base_vla.build_vlm_backbone",
        lambda name, **kwargs: _add_lm_head(_SeedableStubVLM(**kwargs)),
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


def _cfg(**extra):
    cfg = {
        "framework": "vla", "variant": "pi",
        "repeated_diffusion_steps": 1, "obs_image_size": None,
        "action_head": {"action_dim": ACTION_DIM, "action_horizon": HORIZON, "state_dim": STATE_DIM},
        "vlm_backbone": {"name": "qwen3_vl_vla", "checkpoint_path": "/dummy"},
    }
    cfg.update(extra)
    return cfg


def _build(**extra):
    from openwam.model import build_architecture

    torch.manual_seed(42)
    arch = build_architecture("vla_system", _cfg(**extra))
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def test_weight_zero_matches_action_only_and_no_vlm_key(cotrain_vlm):
    arch = _build(vlm_loss_weight=0.0)
    out = arch.compute_loss(**arch.prepare_inputs(_batch()), lambda_action=1.0)
    assert "loss_vlm" not in out
    torch.testing.assert_close(out["loss"], out["loss_action"], rtol=0.0, atol=0.0)


def test_single_backward_composition_math(cotrain_vlm):
    w = 0.5
    arch = _build(vlm_loss_weight=w)
    out = arch.compute_loss(**arch.prepare_inputs(_batch()), lambda_action=1.0)
    assert "loss_vlm" in out
    # total = action + w * vlm — one tensor, one backward
    torch.testing.assert_close(
        out["loss"].detach(), out["loss_action"].detach() + w * out["loss_vlm"].detach(),
        rtol=1e-6, atol=1e-6,
    )
    out["loss"].backward()  # single backward flows
    # grads reach BOTH legs: the head (action) and the VLM LM head (language)
    head_grad = any(p.grad is not None for p in arch.action_backbone.parameters())
    vlm_grad = arch.vlm_backbone.lm_head.weight.grad is not None
    assert head_grad and vlm_grad


def test_ce_normalized_by_token_count(cotrain_vlm):
    """loss_vlm == sum_ce / n_tokens (local; distributed branch is a no-op
    outside torch.distributed init)."""
    arch = _build(vlm_loss_weight=1.0)
    batch = _batch()
    examples = arch._to_examples(batch)
    _, _, vlm_inputs = arch._encode(examples, num_last_layers=arch._num_last_layers())
    sum_ce, n_tokens = arch.vlm_backbone.compute_vlm_loss(vlm_inputs)
    expected = sum_ce / n_tokens
    out = arch.compute_loss(**arch.prepare_inputs(batch), lambda_action=1.0)
    torch.testing.assert_close(out["loss_vlm"], expected, rtol=1e-5, atol=1e-6)


def test_frozen_vlm_with_positive_weight_fast_fails(cotrain_vlm):
    arch = _build(vlm_loss_weight=0.5)
    with pytest.raises(ValueError, match="requires a trainable VLM"):
        arch.freeze_modules(["vlm_backbone"])


def test_frozen_vlm_with_zero_weight_freezes_normally(cotrain_vlm):
    arch = _build(vlm_loss_weight=0.0)
    assert arch.freeze_modules(["vlm_backbone"]) == ["vlm_backbone"]


def test_partial_freeze_below_vlm_path_is_allowed(cotrain_vlm):
    """Freezing a VLM *sub*-module is legitimate co-training (§6.5: "VLM 或其
   声明部分")— only the full-backbone freeze contradicts a positive weight."""
    arch = _build(vlm_loss_weight=0.5)
    # freeze a non-existent path: silently skipped by the base freeze
    assert arch.freeze_modules(["vlm_backbone.nonexistent"]) == []

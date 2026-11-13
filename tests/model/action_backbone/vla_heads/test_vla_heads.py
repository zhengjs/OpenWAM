"""CPU unit tests for the three VLA action heads.

For each of:
    - L1RegressionActionHead   (QwenOFT: gathered action-token queries)
    - FlowMatchingActionHead   (QwenGR00T: single-layer VLM hidden -> DiT)
    - LayerwiseFlowMatchingActionHead (QwenPI: layer-wise VLM hidden list)

verify that:
  1. ``compute_loss`` returns a finite scalar and backward populates gradients.
  2. ``predict_action`` returns (B, horizon, action_dim).
  3. ``build_vla_action_head`` dispatches on ``action_model_type`` and explicit
     overrides win over the config block.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.action_backbone.vla_heads import (
    FlowMatchingActionHead,
    L1RegressionActionHead,
    LayerwiseFlowMatchingActionHead,
    build_vla_action_head,
)

B, HIDDEN = 2, 64
SEQ = 8
ACTION_DIM, HORIZON, STATE_DIM = 4, 3, 5


def _grads_flow(head) -> bool:
    return any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


# ---------------------------------------------------------------------------
# 1. L1RegressionActionHead (QwenOFT)
# ---------------------------------------------------------------------------


def test_mlp_head_predict_and_loss():
    head = L1RegressionActionHead(
        input_dim=HIDDEN, hidden_dim=2 * HIDDEN, action_dim=ACTION_DIM, action_horizon=HORIZON
    )
    queries = torch.randn(B, HORIZON, HIDDEN)
    pred = head.predict_action(queries)
    assert pred.shape == (B, HORIZON, ACTION_DIM)
    loss = head.compute_loss(queries, torch.randn(B, HORIZON, ACTION_DIM))
    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_flow(head)


def test_mlp_head_masked_l1():
    head = L1RegressionActionHead(
        input_dim=HIDDEN, hidden_dim=2 * HIDDEN, action_dim=ACTION_DIM, action_horizon=HORIZON
    )
    queries = torch.randn(B, HORIZON, HIDDEN)
    target = torch.randn(B, HORIZON, ACTION_DIM)
    # Per-timestep mask (B, T) — broadcast across the action dims.
    mask = torch.ones(B, HORIZON, dtype=torch.bool)
    mask[:, -1] = False
    loss_masked = head.compute_loss(queries, target, action_mask=mask)
    loss_full = head.compute_loss(queries, target)
    assert torch.isfinite(loss_masked)
    assert loss_masked != loss_full


# ---------------------------------------------------------------------------
# 2. FlowMatchingActionHead (QwenGR00T, DiT-B)
# ---------------------------------------------------------------------------


def _fm_head(**overrides):
    cfg = dict(
        action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        model_size="DiT-B",
        state_dim=0,
        hidden_size=32,
        num_inference_timesteps=2,
        num_target_vision_tokens=4,
        max_seq_len=16,
        cross_attention_dim=HIDDEN,
        diffusion_model_cfg={"num_layers": 2, "dropout": 0.0, "final_dropout": False},
    )
    cfg.update(overrides)
    return FlowMatchingActionHead(**cfg)


def test_flow_matching_head_loss_backward_predict():
    head = _fm_head()
    hidden = torch.randn(B, SEQ, HIDDEN)
    loss = head.compute_loss(hidden, torch.randn(B, HORIZON, ACTION_DIM))
    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_flow(head)

    pred = head.predict_action(hidden)
    assert pred.shape == (B, HORIZON, ACTION_DIM)


def test_flow_matching_head_with_state():
    head = _fm_head(state_dim=STATE_DIM)
    hidden = torch.randn(B, SEQ, HIDDEN)
    state = torch.randn(B, 1, STATE_DIM)
    loss = head.compute_loss(hidden, torch.randn(B, HORIZON, ACTION_DIM), state=state)
    assert torch.isfinite(loss)
    loss.backward()
    # state_encoder participates
    assert any(p.grad is not None for p in head.state_encoder.parameters())


def test_flow_matching_head_attention_mask():
    head = _fm_head()
    hidden = torch.randn(B, SEQ, HIDDEN)
    mask = torch.ones(B, SEQ, dtype=torch.bool)
    loss = head.compute_loss(hidden, torch.randn(B, HORIZON, ACTION_DIM), encoder_attention_mask=mask)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 3. LayerwiseFlowMatchingActionHead (QwenPI)
# ---------------------------------------------------------------------------


def _layerwise_head(**overrides):
    cfg = dict(
        action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        num_layers=2,
        hidden_size=HIDDEN,
        state_dim=0,
        num_inference_timesteps=2,
        num_target_vision_tokens=4,
        max_seq_len=16,
        diffusion_model_cfg={"dropout": 0.0, "final_dropout": False},
    )
    cfg.update(overrides)
    return LayerwiseFlowMatchingActionHead(**cfg)


def test_layerwise_head_loss_backward_predict():
    head = _layerwise_head()
    hidden_list = [torch.randn(B, SEQ, HIDDEN) for _ in range(2)]
    loss = head.compute_loss(hidden_list, torch.randn(B, HORIZON, ACTION_DIM))
    assert torch.isfinite(loss)
    loss.backward()
    assert _grads_flow(head)

    pred = head.predict_action(hidden_list)
    assert pred.shape == (B, HORIZON, ACTION_DIM)


def test_layerwise_head_with_state():
    head = _layerwise_head(state_dim=STATE_DIM)
    hidden_list = [torch.randn(B, SEQ, HIDDEN) for _ in range(2)]
    state = torch.randn(B, 1, STATE_DIM)
    loss = head.compute_loss(hidden_list, torch.randn(B, HORIZON, ACTION_DIM), state=state)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in head.state_encoder.parameters())


# ---------------------------------------------------------------------------
# 4. build_vla_action_head factory
# ---------------------------------------------------------------------------


def test_build_dispatch():
    head = build_vla_action_head({"action_model_type": "MLP", "action_dim": 4, "action_horizon": 3})
    assert isinstance(head, L1RegressionActionHead)
    assert head.action_dim == 4 and head.action_horizon == 3

    head = build_vla_action_head(
        {"action_model_type": "DiT-B", "action_dim": 4, "action_horizon": 3, "cross_attention_dim": HIDDEN,
         "diffusion_model_cfg": {"num_layers": 2, "dropout": 0.0}}
    )
    assert isinstance(head, FlowMatchingActionHead)

    head = build_vla_action_head(
        {"action_model_type": "LayerwiseFM", "action_dim": 4, "action_horizon": 3,
         "num_layers": 2, "hidden_size": HIDDEN}
    )
    assert isinstance(head, LayerwiseFlowMatchingActionHead)

    with pytest.raises(KeyError):
        build_vla_action_head({"action_model_type": "nope"})


def test_build_overrides_win():
    head = build_vla_action_head({"action_model_type": "MLP", "action_hidden_dim": 32}, action_hidden_dim=64)
    assert head.model.layer_norm1.normalized_shape[0] == 64

"""Connectors (spec_zh.md §6.3): BackboneOutput -> ConnectorOutput packaging.

Pins the three built-ins' shapes/masks/stream counts, the layer-requirement
declaration contract, the OFT gather parity with the original inline
implementation, and registry fast-fail.
"""

import pytest
import torch

from openwam.contracts.model_contract import BackboneOutput
from openwam.model.architectures.connectors import (
    CONNECTOR_REGISTRY,
    build_connector,
    gather_action_token_embeddings,
)


def backbone_out(b=2, l=7, h=8, num_layers=3):
    layers = [torch.randn(b, l, h) for _ in range(num_layers)]
    return BackboneOutput(
        last_hidden_state=layers[-1],
        context_mask=torch.ones(b, l, dtype=torch.bool),
        hidden_states=layers,
    )


def test_registry_has_three_builtins_and_fast_fail():
    assert set(CONNECTOR_REGISTRY) == {"layerwise_hidden", "last_hidden_tokens", "last_hidden_cross_attn"}
    with pytest.raises(KeyError, match="Unknown connector 'nope'"):
        build_connector("nope")


def test_layerwise_hidden_streams_follow_configured_depth():
    conn = build_connector("layerwise_hidden", num_layers=2)
    assert conn.required_backbone_layers == 2
    assert conn.num_condition_streams == 2
    out = conn(backbone_out(num_layers=3))
    assert len(out.condition_tokens) == 2
    assert out.condition_tokens[-1] is out.condition_tokens[-1]
    assert torch.equal(out.condition_tokens[0], out.condition_tokens[0])
    assert out.condition_mask.shape == (2, 7)


def test_layerwise_hidden_follows_backbone_depth_when_unset():
    conn = build_connector("layerwise_hidden")  # num_layers=None
    assert conn.required_backbone_layers is None
    assert conn.num_condition_streams is None
    out = conn(backbone_out(num_layers=3))
    assert len(out.condition_tokens) == 3


def test_layerwise_hidden_requires_materialized_stack():
    conn = build_connector("layerwise_hidden", num_layers=2)
    bo = backbone_out()
    bo.hidden_states = None
    with pytest.raises(ValueError, match="hidden_states"):
        conn(bo)


def test_last_hidden_tokens_gather_matches_manual_selection():
    conn = build_connector("last_hidden_tokens", action_horizon=3, action_token_id=42)
    bo = backbone_out()
    ids = torch.full((2, 7), 5)
    ids[0, [1, 4, 6]] = 42  # sample 0: action tokens at 1, 4, 6
    ids[1, ::2] = 42  # sample 1: at 0, 2, 4, 6 (take last 3 -> 2, 4, 6)
    out = conn(bo, input_ids=ids)
    assert out.condition_tokens[0].shape == (2, 3, 8)
    # mask: all-ones over the gathered window
    assert torch.equal(out.condition_mask, torch.ones(2, 3, dtype=torch.bool))
    for b, expected in ((0, [1, 4, 6]), (1, [2, 4, 6])):
        assert torch.equal(out.condition_tokens[0][b], bo.last_hidden_state[b, expected])


def test_last_hidden_tokens_insufficient_tokens_raise():
    conn = build_connector("last_hidden_tokens", action_horizon=3, action_token_id=42)
    ids = torch.full((2, 7), 5)
    ids[1, 0] = 42  # sample 1 has only 1 action token
    with pytest.raises(RuntimeError, match="insufficient action tokens"):
        conn(backbone_out(), input_ids=ids)


def test_last_hidden_cross_attn_is_identity_packaging():
    conn = build_connector("last_hidden_cross_attn")
    assert conn.required_backbone_layers is None
    bo = backbone_out()
    out = conn(bo)
    assert out.condition_tokens[0] is bo.last_hidden_state  # alias, no copy
    assert out.condition_mask is bo.context_mask


def test_oft_gather_shared_function_parity_with_original_inline():
    """The shared gather must reproduce the original QwenOFT inline logic on
    the reference example (multiple ids variant included)."""
    torch.manual_seed(0)
    last_hidden = torch.randn(2, 9, 4)
    ids = torch.full((2, 9), 7)
    ids[0, [2, 3, 5, 8]] = 11
    ids[1, [0, 1, 4, 6, 7]] = 11
    got = gather_action_token_embeddings(last_hidden, ids, 11, action_horizon=3)
    # manual: last-3 positions in temporal order
    assert torch.equal(got[0], last_hidden[0, [3, 5, 8]])
    assert torch.equal(got[1], last_hidden[1, [4, 6, 7]])
    # multi-id list variant takes the same positions
    got2 = gather_action_token_embeddings(last_hidden, ids, [11, 12], action_horizon=3)
    assert torch.equal(got, got2)

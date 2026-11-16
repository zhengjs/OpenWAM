"""Model-side output contracts (spec_zh.md §3.2).

Beyond construction/defaults, the load-bearing property is the HF alias
convention: ``hidden_states[-1]`` IS ``last_hidden_state`` (same object, no
copy) — consumers may rely on identity.
"""

import pytest
import torch
from torch import Tensor

from openwam.contracts.model_contract import (
    ActionHeadOutput,
    BackboneOutput,
    ConnectorOutput,
    SystemOutput,
)


def test_backbone_output_defaults():
    out = BackboneOutput(last_hidden_state=torch.zeros(2, 5, 8), context_mask=torch.ones(2, 5))
    assert out.hidden_states is None
    assert out.world_pred_loss is None and out.vlm_loss is None and out.kv_cache is None


def test_hidden_states_alias_not_copy():
    """hidden_states[-1] IS last_hidden_state — alias, not a copy (HF convention)."""
    layers = [torch.randn(2, 5, 8) for _ in range(3)]
    out = BackboneOutput(last_hidden_state=layers[-1], context_mask=torch.ones(2, 5),
                         hidden_states=layers)
    assert out.hidden_states[-1] is out.last_hidden_state
    out.last_hidden_state.mul_(1.0)  # in-place ops flow through the alias
    assert torch.equal(out.hidden_states[-1], out.last_hidden_state)


def test_connector_output_stream_count_self_describes():
    mask = torch.ones(2, 7)
    single = ConnectorOutput(condition_tokens=[torch.randn(2, 7, 8)], condition_mask=mask)
    layerwise = ConnectorOutput(condition_tokens=[torch.randn(2, 7, 8) for _ in range(4)],
                                condition_mask=mask)
    assert len(single.condition_tokens) == 1
    assert len(layerwise.condition_tokens) == 4
    # one mask serves all streams — same object requirement is downstream policy,
    # but the contract's shape rule is: identical per-entry lengths.
    assert all(t.shape[:2] == layerwise.condition_mask.shape for t in layerwise.condition_tokens)


def test_action_head_output_fast_path_exclusivity_shape():
    fast = ActionHeadOutput(action_loss=torch.tensor(0.5), pred_action_tokens=torch.randint(0, 100, (2, 6)))
    assert fast.pred_trajectory_norm is None  # FAST path: trajectory is None
    cont = ActionHeadOutput(pred_trajectory_norm=torch.zeros(2, 3, 4))
    assert cont.pred_action_tokens is None


def test_system_output_defaults_and_metrics():
    out = SystemOutput(total_loss=torch.tensor(1.0))
    assert out.metrics == {}  # factory default, not shared mutable None
    out.metrics["lr"] = 1e-4
    assert SystemOutput().metrics == {}


def test_no_frozen_mutation_trap():
    """Assembling stages mutate these incrementally (unlike ModelInputBatch)."""
    out = SystemOutput()
    out.action_loss = torch.tensor(0.1)  # would raise if frozen
    assert isinstance(out.action_loss, Tensor)


def test_mask_naming_docstring_rule_is_enforceable():
    """Naming axes are documented policy; this pins the fields exist with
    mask names that reference SEQUENCES, not layers."""
    for mask_field in ("context_mask", "condition_mask"):
        assert mask_field in BackboneOutput.__dataclass_fields__ or \
            mask_field in ConnectorOutput.__dataclass_fields__

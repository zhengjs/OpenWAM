"""Cosmos flow-match scheduler adapters: predict2.5 base + cosmos3 subclass deltas.

The predict2.5 section pins the four fields the trainer needs and the
flow-match math. The cosmos3 section pins the inheritance relation
(cosmos3 reuses the predict25 math verbatim) and the
``timesteps = sigmas × num_train_timesteps`` linear relation that
generate() / denoise_schedule.py invert externally.
"""

from __future__ import annotations

import torch

from openwam.model.video_backbone.cosmos3.scheduler import Cosmos3FlowSchedulerAdapter
from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter

# ---------------------------------------------------------------------------
# predict2.5 base adapter
# ---------------------------------------------------------------------------


def test_default_shift_video_is_cosmos_upstream_value():
    """Default must match upstream
    cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:99
    (`shift: int = 5`)."""
    sch = CosmosFlowSchedulerAdapter()
    assert sch.shift_video == 5.0


def test_training_weights_are_uniform():
    """Cosmos upstream uses TrainTimeWeight("uniform") — weights must be 1.0
    everywhere (see cosmos_predict2/_src/predict2/schedulers/rectified_flow.py:21-42)."""
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=50, training=True)
    weights = sch.linear_timesteps_weights
    assert weights is not None
    assert sch.num_train_timesteps == 1000
    assert torch.allclose(weights, torch.ones_like(weights))


def test_add_noise_matches_flow_match_form():
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=10, training=True)
    clean = torch.zeros(2, 4)
    noise = torch.ones(2, 4)
    noisy = sch.add_noise(clean, noise, sch.timesteps[0])
    # (1 - sigma)·clean + sigma·noise with clean=0 → noisy == sigma·noise.
    assert torch.allclose(noisy, torch.full_like(noisy, float(sch.sigmas[0])))


def test_training_target_is_noise_minus_sample():
    sch = CosmosFlowSchedulerAdapter()
    sch.set_timesteps(num_inference_steps=5, training=True)
    sample = torch.randn(2, 3)
    noise = torch.randn(2, 3)
    target = sch.training_target(sample, noise, sch.timesteps[0])
    assert torch.allclose(target, noise - sample)


def test_shift_video_changes_sigmas():
    sch_low = CosmosFlowSchedulerAdapter(shift_video=1.0)
    sch_high = CosmosFlowSchedulerAdapter(shift_video=5.0)
    sch_low.set_timesteps(num_inference_steps=20, training=False)
    sch_high.set_timesteps(num_inference_steps=20, training=False)
    # shift=1 collapses to identity; shift=5 pushes sigmas toward 1.
    assert not torch.allclose(sch_low.sigmas, sch_high.sigmas)


# ---------------------------------------------------------------------------
# cosmos3 subclass deltas
# ---------------------------------------------------------------------------


def test_cosmos3_subclass_of_predict25_adapter():
    assert issubclass(Cosmos3FlowSchedulerAdapter, CosmosFlowSchedulerAdapter)


def test_cosmos3_training_grid_matches_shift_formula():
    shift = 5.0
    sched = Cosmos3FlowSchedulerAdapter(shift_video=shift)
    sched.set_timesteps(1000, training=True, shift=shift)

    base = torch.linspace(1.0, 0.0, 1001)[:-1]
    expected = shift * base / (1 + (shift - 1) * base)
    assert torch.allclose(sched.sigmas, expected, atol=1e-6)
    assert torch.allclose(sched.timesteps, sched.sigmas * sched.num_train_timesteps, atol=1e-4)
    assert sched.linear_timesteps_weights is not None
    assert torch.all(sched.linear_timesteps_weights == 1.0)


def test_cosmos3_inference_grid_respects_shift_override():
    sched = Cosmos3FlowSchedulerAdapter(shift_video=5.0)
    sched.set_timesteps(10, shift=3.0)
    base = torch.linspace(1.0, 0.0, 11)[:-1]
    expected = 3.0 * base / (1 + 2.0 * base)
    assert torch.allclose(sched.sigmas, expected, atol=1e-6)
    assert sched.linear_timesteps_weights is None

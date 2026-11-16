"""IterMixtureDataset + WeightAnnealingSampler (spec_data_zh.md §6.2/§6.3).

Covers: plan resolution + tagging, deterministic annealing of dataset_index,
rank-stride coverage of shared sample sequences, resume via
``state_dict``/``load_state_dict`` (byte-identical remaining stream), config
validation, DataLoader composition, and PipelineContext passthrough.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.iter_mixture import IterMixtureDataset, WeightAnnealingSampler


class Fake:
    def __init__(self, n, dim=10):
        self._n, self._dim = n, dim

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return {"action": torch.zeros(4, self._dim), "pos": i}

    @property
    def action_dim(self):
        return self._dim


def make_sampler(total_steps=8, batch_size=4, **kw):
    ds = IterMixtureDataset([Fake(20), Fake(30)], names=["a", "b"])
    defaults = dict(weight_start=[1.0, 0.0], weight_end=[0.0, 1.0], seed=7)
    defaults.update(kw)
    sampler = WeightAnnealingSampler(ds, total_steps=total_steps, batch_size=batch_size, **defaults)
    return ds, sampler


def test_no_plan_is_a_hard_error():
    fresh = IterMixtureDataset([Fake(5), Fake(5)])
    with pytest.raises(RuntimeError, match="no plan attached"):
        fresh[0]


def test_plan_attachment_resolution_and_tagging():
    ds, _ = make_sampler()
    assert ds.plan_attached
    assert len(ds) == 32 and len(ds) == 32
    x0, x_last = ds[0], ds[31]
    # deterministic 0/1 weights: alpha(0) draws source a, final step draws b
    assert x0["_dataset_index"] == 0 and x0["_dataset_name"] == "a"
    assert x_last["_dataset_index"] == 1 and x_last["_dataset_name"] == "b"
    # action padded to max dim across sources
    assert x0["action"].shape[-1] == 10


def test_annealing_endpoint_determinism():
    ds, _ = make_sampler()
    di = ds._plan_di
    assert (di[:4] == 0).all()
    assert (di[-4:] == 1).all()


def test_rank_stride_disjoint_full_coverage():
    ds0 = IterMixtureDataset([Fake(12)], names=["only"])
    WeightAnnealingSampler(ds0, weight_start=[1.0], weight_end=[1.0], total_steps=3,
                           batch_size=2, rank=0, world_size=2, seed=5)
    ds1 = IterMixtureDataset([Fake(12)], names=["only"])
    WeightAnnealingSampler(ds1, weight_start=[1.0], weight_end=[1.0], total_steps=3,
                           batch_size=2, rank=1, world_size=2, seed=5)
    got0 = [ds0[i]["pos"] for i in range(len(ds0))]
    got1 = [ds1[i]["pos"] for i in range(len(ds1))]
    assert set(got0).isdisjoint(got1)
    assert sorted(got0 + got1) == list(range(12))  # ranks jointly cover the permutation


def test_resume_remaining_stream_bit_identical():
    ds, sampler = make_sampler()
    assert sampler.state_dict()["consumed"] == 0
    list(iter(sampler))
    sampler._consumed = 16  # simulate a mid-run checkpoint position
    state = sampler.state_dict()
    assert state["consumed"] == 16 and len(sampler) == 16

    ds2, _ = make_sampler()
    sampler2 = WeightAnnealingSampler(
        ds2, weight_start=[1.0, 0.0], weight_end=[0.0, 1.0], total_steps=8, batch_size=4, seed=7
    )
    sampler2.load_state_dict(state)
    for i in range(16, 32):
        a, b = ds[i], ds2[i]
        assert a["_dataset_index"] == b["_dataset_index"]
        assert a["pos"] == b["pos"]


def test_resume_config_mismatch_guards():
    _, sampler = make_sampler()
    state = {**sampler.state_dict(), "consumed": 4}
    _, s2 = make_sampler()
    with pytest.raises(ValueError, match="total_steps"):
        s2.load_state_dict({**state, "total_steps": 99})
    with pytest.raises(ValueError, match="schedule"):
        s2.load_state_dict({**state, "schedule": "cosine"})
    with pytest.raises(ValueError, match="cursors"):
        s2.load_state_dict({**state, "cursors": [0]})
    with pytest.raises(ValueError, match="outside plan range"):
        s2.load_state_dict({**state, "consumed": 10**9})


def _valid_base():
    return dict(weight_start=[1.0, 1.0], weight_end=[1.0, 1.0], total_steps=4)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"weight_start": [1.0]}, "weight_start/weight_end"),  # wrong arity
        ({"weight_start": [-1.0, 1.0]}, "finite and non-negative"),
        ({"weight_start": [0.0, 0.0]}, "positive at both endpoints"),
        ({"schedule": "step"}, "unknown schedule"),
        ({"total_steps": 0}, "total_steps and batch_size"),
        ({"rank": 2, "world_size": 2}, "invalid rank/world_size"),
    ],
)
def test_validation_errors(overrides, match):
    ds = IterMixtureDataset([Fake(5), Fake(5)])
    kwargs = {**_valid_base(), **overrides}
    with pytest.raises(ValueError, match=match):
        WeightAnnealingSampler(ds, batch_size=2, **kwargs)


def test_dataloader_composition_and_shrunk_len_after_resume():
    ds, sampler = make_sampler(total_steps=5, batch_size=4,
                               weight_start=[0.7, 0.3], weight_end=[0.3, 0.7])
    steps = sum(1 for batch in DataLoader(ds, batch_size=4, sampler=sampler))
    assert steps == 5
    sampler.load_state_dict({**sampler.state_dict(), "consumed": 8})
    assert len(list(DataLoader(ds, batch_size=4, sampler=sampler))) == 3


def test_pipeline_context_passthrough_untouched():
    """Track B path: ctx samples carry their own metadata; no dict tagging."""
    class CtxSource(Fake):
        def __getitem__(self, i):
            return PipelineContext(sample_id=f"s{i}", clip_uri="c", anchor_timestamp_us=0,
                                   cam_frame_offsets={}, source_name="drv")

    ds = IterMixtureDataset([CtxSource(8), Fake(8)], names=["drv", "rob"])
    WeightAnnealingSampler(ds, weight_start=[1.0, 1.0], weight_end=[1.0, 1.0],
                           total_steps=2, batch_size=2, seed=3)
    seen_ctx = False
    for i in range(len(ds)):
        sample = ds[i]
        if isinstance(sample, PipelineContext):
            seen_ctx = True
            assert sample.source_name == "drv"
    assert seen_ctx

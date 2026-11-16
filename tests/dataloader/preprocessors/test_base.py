"""Preprocessor pipeline + registry (spec_data_zh.md §5 infrastructure)."""

import pytest

from openwam.contracts.config_schema import PreprocessorChainConfig, ProcessorSpec
from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.preprocessors import (
    PREPROCESSOR_REGISTRY,
    PreProcessorPipeline,
    build_preprocessor_chain,
    list_registered_preprocessors,
)


class _Mark(PipelineContext):
    pass


def test_registry_has_five_builtins():
    assert set(list_registered_preprocessors()) == {
        "multicam_decode", "topic_sync", "ego_state", "multiview_image", "vla_prompt",
    }


def test_chain_executes_in_config_order(stats_path):
    """Ordering IS the contract — processors run in PreprocessorChainConfig order."""
    p = PreProcessorPipeline.from_config(PreprocessorChainConfig(chain=[
        ProcessorSpec(name="ego_state", kwargs={"stats_path": stats_path()}),
        ProcessorSpec(name="multiview_image", kwargs={"mode": "standard_tensor"}),
    ]))
    assert [type(x).__name__ for x in p.preprocessors] == ["EgoStateProcessor", "MultiViewImageProcessor"]
    assert len(p) == 2


def test_training_filter_drops_training_only(stats_path):
    chain = PreprocessorChainConfig(chain=[
        ProcessorSpec(name="multicam_decode"),
        ProcessorSpec(name="topic_sync", kwargs={"t_hist": 1}),
        ProcessorSpec(name="ego_state", kwargs={"stats_path": stats_path()}),
    ])
    train_chain = build_preprocessor_chain(chain, training=True)
    infer_chain = build_preprocessor_chain(chain, training=False)
    assert len(train_chain) == 3
    assert [type(x).__name__ for x in infer_chain] == ["EgoStateProcessor"]
    # registry classes carry the flag
    assert all(proc.is_training_only for proc in train_chain if type(proc).__name__ != "EgoStateProcessor")


def test_unknown_processor_fails_fast():
    with pytest.raises(ValueError, match="Unknown preprocessor 'nope'"):
        build_preprocessor_chain(PreprocessorChainConfig(chain=[ProcessorSpec(name="nope")]))


def test_direct_call_delegates_to_process(stats_path):
    proc = PREPROCESSOR_REGISTRY["ego_state"](stats_path=stats_path())
    ctx = PipelineContext(sample_id="", clip_uri="c", anchor_timestamp_us=0, cam_frame_offsets={},
                          source_name="d",
                          raw_ego_history=[[0.0, 0.5, 1.0, 0.25], [1.0, 1.0, 1.0, 1.0]])
    out = proc(ctx)  # __call__ path
    assert out.ego_state_tensor.shape == (2, 4)

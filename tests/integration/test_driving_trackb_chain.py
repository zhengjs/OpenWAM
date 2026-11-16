"""Track B worker chain on a synthetic clip: decode → sync → ego_state → image
(spec_data_zh.md §5 end-to-end, CPU, no lance required)."""

import json

import numpy as np
import pytest
import torch

from openwam.contracts.config_schema import PreprocessorChainConfig, ProcessorSpec
from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.clip_reader import ClipReader
from openwam.dataloader.preprocessors import PreProcessorPipeline
from tests.integration.conftest_driving import build_synthetic_clip


@pytest.fixture(scope="module")
def storage(tmp_path_factory):
    return build_synthetic_clip(tmp_path_factory.mktemp("trackb_chain"))


def make_stats(storage, d_state=8):
    path = storage["root"] / "stats.json"
    path.write_text(json.dumps({
        "mode": "min_max",
        "ego_state": {"min": [0.0] * d_state, "max": [10.0] * d_state},
        "trajectory": {"min": [-1.0] * 3, "max": [1.0] * 3},
        "T_hist": 3, "D_state": d_state, "T_pred": 3, "D_action": 3,
    }), encoding="utf-8")
    return str(path)


def make_ctx(storage):
    ts = storage["ts"]
    return PipelineContext(
        sample_id="s0", clip_uri=str(storage["clip"]),
        anchor_timestamp_us=int(ts[3]) // 1000 + 500,
        cam_frame_offsets={"cam_front": [2, 3, 4, 5, 6, 7]}, source_name="ds",
        raw_gt_trajectory=np.zeros((3, 3), dtype=np.float32),
        raw_trajectory_valid_mask=np.ones(3, dtype=np.int8),
    )


def build_chain(storage):
    chain = PreprocessorChainConfig(chain=[
        ProcessorSpec(name="multicam_decode"),
        ProcessorSpec(name="topic_sync", kwargs={"t_hist": 3, "stride_us": 50_000,
                                                 "rot6d_source": "yaw_quat"}),
        ProcessorSpec(name="ego_state", kwargs={"stats_path": make_stats(storage),
                                                "state_dim_names": ["speed", "yaw_rate"]}),
        ProcessorSpec(name="multiview_image", kwargs={"mode": "standard_tensor"}),
    ])
    return PreProcessorPipeline.from_config(chain)


def test_chain_produces_all_processed_slots(storage):
    out = build_chain(storage).process(make_ctx(storage))
    assert out.raw_images is None and out.raw_ego_history is None  # consumed
    assert out.pixel_values_by_cam["cam_front"].shape == (6, 3, 16, 24)
    assert 0.0 <= float(out.pixel_values_by_cam["cam_front"].max()) <= 1.0
    # D_state after quat->rot6d: speed + yaw_rate + 6 = 8
    assert out.ego_state_tensor.shape == (3, 8)
    assert out.gt_trajectory_norm.shape == (3, 3)
    assert out.trajectory_mask.tolist() == [True, True, True]
    # per-read-frame timestamps: T_obs entries, strictly ascending
    assert out.frame_timestamps_ns.shape == (6,)
    assert (out.frame_timestamps_ns[1:] > out.frame_timestamps_ns[:-1]).all()


def test_topic_sync_interpolates_speed_at_anchor(storage):
    out = build_chain(storage).process(make_ctx(storage))
    # anchor sits between topic rows 3/4 (speed 3.0-3.33); history queried at
    # -50ms steps -> strictly before the anchor, all inside [0, 10] pre-norm.
    # min_max (/10*2-1): speed block in [-1, 1]; identity quat -> rot6d
    # normalizes to [-0.8, -1, -1, -1, -0.8, -1].
    assert np.allclose(out.ego_state_tensor.numpy()[0, 2:], [-0.8, -1, -1, -1, -0.8, -1], atol=1e-6)
    assert -1.0 <= out.ego_state_tensor.numpy()[:, 0].max() <= 1.0


def test_tolerance_gate_error_policy(storage):
    chain = PreprocessorChainConfig(chain=[
        ProcessorSpec(name="topic_sync", kwargs={
            "t_hist": 1, "tolerance_us": 1, "overflow": "error",
            "rot6d_source": "yaw_quat"}),
    ])
    with pytest.raises(ValueError, match="exceeds"):
        PreProcessorPipeline.from_config(chain).process(make_ctx(storage))


def test_clip_reader_lru_singleton(storage):
    assert ClipReader.instance() is ClipReader.instance()
    handle = ClipReader.instance().handle(str(storage["clip"]))
    again = ClipReader.instance().handle(str(storage["clip"]))
    assert handle is again

"""EgoStateProcessor (spec_data_zh.md §5): normalization + Fourier + inverses."""

import numpy as np
import pytest
import torch

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.preprocessors.ego_state_processor import EgoStateProcessor


def make_ctx(hist, traj=None, mask=None):
    return PipelineContext(
        sample_id="s", clip_uri="c", anchor_timestamp_us=0, cam_frame_offsets={}, source_name="d",
        raw_ego_history=hist,
        raw_gt_trajectory=traj,
        raw_trajectory_valid_mask=mask,
    )


def test_minmax_normalization_and_mask_promotion(stats_path):
    ctx = make_ctx(
        hist=np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        traj=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        mask=np.array([1, 1, 0], dtype=np.int8),
    )
    proc = EgoStateProcessor(stats_path=stats_path())
    out = proc.process(ctx)
    # min_max: (x - 0)/(1-0)*2 - 1
    assert out.ego_state_tensor.shape == (2, 4)
    assert torch.allclose(out.ego_state_tensor[0], torch.full((4,), 0.0), atol=1e-6)
    assert torch.allclose(out.ego_state_tensor[1], torch.full((4,), 1.0), atol=1e-6)
    assert torch.allclose(out.gt_trajectory_norm[1], torch.full((3,), 1.0), atol=1e-6)
    assert out.trajectory_mask.dtype == torch.bool
    assert out.trajectory_mask.tolist() == [True, True, False]
    # raw slots consumed
    assert out.raw_ego_history is None and out.raw_gt_trajectory is None and out.raw_trajectory_valid_mask is None


def test_fourier_expansion_width(stats_path):
    # raw 0.5 normalizes to 0 under min_max -> Fourier of exactly 0
    ctx = make_ctx(hist=np.full((2, 4), 0.5, dtype=np.float32))
    proc = EgoStateProcessor(stats_path=stats_path(), fourier_num_bands=2)
    out = proc.process(ctx)
    assert out.ego_state_tensor.shape == (2, 4 * 4)  # 2 bands x (sin, cos) blocks
    # x=0 -> every sin block is 0, every cos block is 1 (block order: s0 c0 s1 c1)
    for k in range(2):
        assert torch.allclose(out.ego_state_tensor[0, 4 * 2 * k:4 * 2 * k + 4], torch.zeros(4), atol=1e-6)
        assert torch.allclose(out.ego_state_tensor[0, 4 * (2 * k + 1):4 * (2 * k + 2)], torch.ones(4), atol=1e-6)


def test_denormalize_roundtrip_interior(stats_path):
    proc = EgoStateProcessor(stats_path=stats_path())
    normed = np.array([[0.0, 0.5, -0.5], [1.0, -1.0, 0.25]], dtype=np.float32)
    recovered = proc.denormalize_trajectory(normed)
    # interior points invert exactly through the affine map
    renormed = (recovered - (-1.0)) / 2.0 * 2.0 - 1.0
    assert np.allclose(renormed, normed, atol=1e-6)


def test_state2str_uses_configured_names(stats_path):
    proc = EgoStateProcessor(stats_path=stats_path(), state_dim_names=["speed", "yaw"])
    s = proc.state2str(np.array([[0.5, 0.25, 0.0, 0.0]], dtype=np.float32))
    assert s.startswith("speed=0.500, yaw=0.250")


def test_empty_ctx_passthrough(stats_path):
    proc = EgoStateProcessor(stats_path=stats_path())
    ctx = PipelineContext(sample_id="", clip_uri="c", anchor_timestamp_us=0,
                          cam_frame_offsets={}, source_name="d")
    out = proc.process(ctx)
    assert out.ego_state_tensor is None and out.gt_trajectory_norm is None

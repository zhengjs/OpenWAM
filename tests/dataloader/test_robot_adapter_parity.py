"""Robot adapter (Track A) parity gate (spec_data_zh.md §3).

Pins the adapter→empty-chain→collate path against the model-side conversion
the transforms path performs (``wan/preprocess.py::preprocess_image``):
``uint8 -> float32 * (2/255) - 1``.

The contract layer emits ``pixel_values`` in [0, 1] (``u8 / 255``). The two
rounding paths are NOT bit-identical directly (division vs multiplication),
but ``fl(fl(u8/255) * 255) == u8`` holds exactly for every uint8 value, so the
model-side adaptation ``(x * 255) * (2/255) - 1`` reproduces the transforms
path bits EXACTLY. This file pins:

1. the exact-recovery property of ``pixel_values`` (uint8 round-trip),
2. bitwise equality of the model conversion vs the transforms reference,
3. pure-rename fields (action / masks / proprio) surviving adapter→collate
   bit-identically.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from openwam.contracts.data_contract import ModelInputBatch
from openwam.dataloader.utils.robot_data_adapter import robot_sample_to_context
from openwam.dataloader.utils.normalization import NORM_EPS  # noqa: F401 (parity deps marker)
from openwam.model.video_backbone.wan.preprocess import preprocess_image

T_OBS, H, W, ACTION_DIM, T_PRED = 4, 32, 48, 6, 3


def make_sample(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    frames = [Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)) for _ in range(T_OBS)]
    return {
        "video": frames,
        "vace_video": None,
        "first_frame_image": frames[0],
        "action": rng.standard_normal((T_PRED, ACTION_DIM)).astype(np.float32),
        "action_mask": rng.integers(0, 2, (T_PRED, ACTION_DIM)).astype(bool),
        "video_mask": np.ones(T_OBS, dtype=bool),
        "proprio": rng.standard_normal((1, ACTION_DIM)).astype(np.float32),
        "proprio_mask": np.ones((1, ACTION_DIM), dtype=bool),
        "prompt": "pick and place",
    }


def transforms_path_reference(video: list) -> torch.Tensor:
    """What ``wan.preprocess_video`` produces per clip: [C, T, H, W] in [-1, 1]."""
    frames = [preprocess_image(img, dtype=torch.float32, device="cpu").squeeze(0) for img in video]
    return torch.stack(frames, dim=1)  # [C, T, H, W]


def test_pixel_values_roundtrip_and_bitwise_parity():
    from openwam.contracts.collate import collate_pipeline_contexts

    sample = make_sample(0)
    ctx = robot_sample_to_context(sample, source_name="robotwin")
    batch = collate_pipeline_contexts([ctx])

    pv = batch.pixel_values  # [B, T, C, H, W] in [0, 1]
    assert pv.shape == (1, T_OBS, 3, H, W)
    assert pv.dtype == torch.float32

    # 1. Exact uint8 recovery: division by 255 then *255 round-trips exactly.
    u8 = (pv * 255.0)
    assert torch.equal(u8, u8.round())

    # 2. Model-side conversion reproduces the transforms path bits exactly.
    model_domain = (pv * 255.0) * (2.0 / 255.0) - 1.0  # [B, T, C, H, W] in [-1, 1]
    reference = transforms_path_reference(sample["video"])  # [C, T, H, W]
    assert torch.equal(model_domain[0].permute(1, 0, 2, 3), reference)


def test_pure_rename_fields_bit_identical():
    from openwam.contracts.collate import collate_pipeline_contexts

    sample = make_sample(1)
    ctx = robot_sample_to_context(sample, source_name="robotwin")
    batch = collate_pipeline_contexts([ctx])

    assert torch.equal(batch.gt_trajectory_norm[0], torch.from_numpy(sample["action"]))
    assert torch.equal(batch.trajectory_mask[0], torch.from_numpy(sample["action_mask"]))
    assert torch.equal(batch.ego_state_tensor[0], torch.from_numpy(sample["proprio"]))
    assert torch.equal(batch.ego_state_mask[0], torch.from_numpy(sample["proprio_mask"]))
    assert torch.equal(batch.video_mask[0], torch.from_numpy(sample["video_mask"]))
    assert batch.prompt == [sample["prompt"]]
    assert torch.equal(batch.sample_weights, torch.ones(1))
    assert isinstance(batch, ModelInputBatch)


def test_multi_sample_collate():
    from openwam.contracts.collate import collate_pipeline_contexts

    s0, s1 = make_sample(2), make_sample(3)
    c0 = robot_sample_to_context(s0, source_name="a")
    c1 = robot_sample_to_context(s1, source_name="b")
    batch = collate_pipeline_contexts([c0, c1])

    assert batch.gt_trajectory_norm.shape == (2, T_PRED, ACTION_DIM)
    assert torch.equal(batch.gt_trajectory_norm[1], torch.from_numpy(s1["action"]))
    assert batch.trajectory_mask.shape == (2, T_PRED, ACTION_DIM)
    assert batch.sample_weights.shape == (2,)
    # action-dim padding for narrower sources lives at the mixture layer
    # (MixtureDataset / IterMixtureDataset pad to the suite's max action_dim),
    # NOT at collate: mixed shapes are a hard error there.
    s1["action"] = s1["action"][:, :4]
    with pytest.raises(ValueError, match="shapes differ"):
        collate_pipeline_contexts([c0, robot_sample_to_context(s1, source_name="b")])

"""Contract layer: ``collate_pipeline_contexts`` + config schema fast-fail.

Covers the §1.2 collate rules: three exclusive visual layouts, language path
exclusivity, mask rank reconciliation, all-or-none presence, sample_weights,
and the frozen ``ModelInputBatch`` guarantee. Config schema: OmegaConf
merge-based validation (unknown keys / wrong types raise at build time).
"""

import pytest
import torch

from openwam.contracts.config_schema import PreprocessorChainConfig, ProcessorSpec, validate_config_block
from openwam.contracts.data_contract import ModelInputBatch, PipelineContext
from openwam.contracts.collate import collate_pipeline_contexts


def ctx(**overrides):
    base = dict(sample_id="s", clip_uri="c", anchor_timestamp_us=0,
                cam_frame_offsets={}, source_name="d")
    base.update(overrides)
    return PipelineContext(**base)


# ---------------------------------------------------------------- visuals ----


def test_canvas_layout_batch():
    t, h, w = 2, 8, 10
    batch = collate_pipeline_contexts([
        ctx(pixel_values=torch.rand(t, 3, h, w)) for _ in range(3)
    ])
    assert batch.pixel_values.shape == (3, t, 3, h, w)
    assert batch.pixel_values_by_cam is None and batch.image_grid_thw is None


def test_by_cam_layout_requires_identical_camera_sets():
    pv0 = {"front": torch.rand(1, 3, 4, 4)}
    pv1 = {"front": torch.rand(1, 3, 4, 4), "left": torch.rand(1, 3, 4, 4)}
    with pytest.raises(ValueError, match="camera set"):
        collate_pipeline_contexts([ctx(pixel_values_by_cam=pv0), ctx(pixel_values_by_cam=pv1)])


def test_by_cam_with_camera_order():
    pv = {"left": torch.rand(1, 3, 4, 4), "front": torch.rand(1, 3, 4, 4)}
    batch = collate_pipeline_contexts([ctx(pixel_values_by_cam=pv), ctx(pixel_values_by_cam=pv)],
                                      camera_order=["front", "left"])
    assert set(batch.pixel_values_by_cam) == {"front", "left"}
    assert batch.pixel_values_by_cam["front"].shape == (2, 1, 3, 4, 4)


def test_layouts_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        collate_pipeline_contexts([ctx(pixel_values=torch.rand(1, 3, 4, 4),
                                       pixel_values_by_cam={"f": torch.rand(1, 3, 4, 4)})])


def test_all_or_none_presence_is_hard_error():
    with pytest.raises(ValueError, match="present on some samples and missing"):
        collate_pipeline_contexts([ctx(prompt="a"), ctx(prompt="b"),
                                   ctx(ego_state_tensor=torch.zeros(1, 4))])


# --------------------------------------------------------------- language ----


def test_language_prompt_and_ids_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive language paths"):
        collate_pipeline_contexts([ctx(prompt="go", input_ids=torch.tensor([1, 2, 3]))])


def test_prompt_right_padded_and_mask():
    batch = collate_pipeline_contexts([ctx(prompt="abcdef"), ctx(prompt="ab")])
    assert batch.prompt == ["abcdef", "ab"]
    assert batch.input_ids is None


def test_ids_right_padded_with_labels_ignore():
    ids0 = torch.tensor([1, 2, 3, 4])
    ids1 = torch.tensor([5, 6])
    batch = collate_pipeline_contexts([
        ctx(input_ids=ids0, attention_mask=torch.ones(4, dtype=torch.long), vlm_labels=ids0.clone()),
        ctx(input_ids=ids1, attention_mask=torch.ones(2, dtype=torch.long), vlm_labels=torch.tensor([5, -100])),
    ])
    assert batch.input_ids.shape == (2, 4)
    assert batch.input_ids[1].tolist() == [5, 6, 0, 0]          # id pad value 0
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch.vlm_labels[1].tolist() == [5, -100, -100, -100]  # label pad value -100


# ------------------------------------------------------------ state/action ----


def test_ego_state_1d_promoted_and_mask():
    batch = collate_pipeline_contexts([
        ctx(ego_state_tensor=torch.zeros(4), ego_state_mask=torch.ones(4, dtype=torch.bool)),
        ctx(ego_state_tensor=torch.ones(4), ego_state_mask=torch.ones(4, dtype=torch.bool)),
    ])
    assert batch.ego_state_tensor.shape == (2, 1, 4)
    assert batch.ego_state_mask.shape == (2, 1, 4)


def test_trajectory_mask_rank_reconciliation():
    """Uniform 1-D masks stack to [B, T]; MIXED ranks are a hard error."""
    t, d = 3, 2
    batch = collate_pipeline_contexts([
        ctx(gt_trajectory_norm=torch.zeros(t, d), trajectory_mask=torch.ones(t, dtype=torch.bool)),
        ctx(gt_trajectory_norm=torch.zeros(t, d), trajectory_mask=torch.ones(t, dtype=torch.bool)),
    ])
    assert batch.trajectory_mask.shape == (2, t)
    with pytest.raises(ValueError, match="ranks differ"):
        collate_pipeline_contexts([
            ctx(gt_trajectory_norm=torch.zeros(t, d), trajectory_mask=torch.ones(t, dtype=torch.bool)),
            ctx(gt_trajectory_norm=torch.zeros(t, d), trajectory_mask=torch.ones(t, d, dtype=torch.bool)),
        ])


def test_sample_weights_default_ones():
    batch = collate_pipeline_contexts([ctx(), ctx(sample_weight=2.0)])
    assert torch.equal(batch.sample_weights, torch.tensor([1.0, 2.0]))
    assert batch.sample_weights.dtype == torch.float32


def test_output_is_frozen_model_input_batch():
    batch = collate_pipeline_contexts([ctx()])
    assert isinstance(batch, ModelInputBatch)
    with pytest.raises(Exception):  # frozen dataclass rejects mutation
        batch.sample_weights = None


# ----------------------------------------------------------- config schema ----


def test_validate_config_block_accepts_valid():
    block = {"chain": [{"name": "ego_state", "kwargs": {"stats_path": "/x"}}]}
    cfg = validate_config_block(block, PreprocessorChainConfig)
    assert cfg.chain[0].name == "ego_state"


def test_validate_config_block_rejects_unknown_keys():
    with pytest.raises(Exception):
        validate_config_block({"chain": [], "typo": 1}, PreprocessorChainConfig)


def test_validate_config_block_rejects_wrong_type():
    with pytest.raises(Exception):
        validate_config_block({"chain": "not-a-list"}, PreprocessorChainConfig)


def test_processor_spec_defaults():
    spec = ProcessorSpec(name="ego_state")
    assert spec.kwargs == {}

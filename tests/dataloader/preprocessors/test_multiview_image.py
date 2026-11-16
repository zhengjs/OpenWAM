"""MultiViewImageProcessor (spec_data_zh.md §5): the three visual layouts."""

import numpy as np
import pytest
import torch

from openwam.contracts.config_schema import PreprocessorChainConfig, ProcessorSpec
from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.preprocessors import PreProcessorPipeline

T, H, W = 2, 60, 80
CAMS = ["cam_front", "cam_left", "cam_rear"]


def raw_images(seed=0):
    rng = np.random.default_rng(seed)
    return {cam: rng.integers(0, 256, (T, H, W, 3), dtype=np.uint8) for cam in CAMS}


def make_ctx(images):
    return PipelineContext(sample_id="", clip_uri="", anchor_timestamp_us=0,
                           cam_frame_offsets={}, source_name="d", raw_images=images)


def build(kwargs):
    return PreProcessorPipeline.from_config(PreprocessorChainConfig(
        chain=[ProcessorSpec(name="multiview_image", kwargs=kwargs)]))


def test_wam_canvas_layout():
    out = build({"mode": "wam_canvas", "camera_layout": CAMS, "canvas_size": [128, 160]}).process(
        make_ctx(raw_images()))
    assert out.pixel_values.shape == (T, 3, 128, 160)
    assert 0.0 <= float(out.pixel_values.min()) and float(out.pixel_values.max()) <= 1.0
    assert out.raw_images is None  # consumed


def test_standard_tensor_preserves_resolution_with_resize():
    out = build({"mode": "standard_tensor",
                 "per_cam_resolution": {c: [32, 40] for c in CAMS}}).process(make_ctx(raw_images()))
    assert set(out.pixel_values_by_cam) == set(CAMS)
    assert out.pixel_values_by_cam["cam_front"].shape == (T, 3, 32, 40)


def test_qwen_vl_patch_pack_and_grid():
    out = build({"mode": "qwen_vl_patch"}).process(make_ctx(raw_images()))
    # 60x80 -> 56x84 (28-multiples) -> (4x6) patches per image x (T*3) images
    n_images = T * len(CAMS)
    assert out.pixel_values.shape == (24 * n_images, 588)
    assert out.image_grid_thw.shape == (n_images, 3)
    assert (out.image_grid_thw == torch.tensor([[1, 4, 6]] * n_images)).all()


def test_unknown_mode_and_missing_canvas_args():
    with pytest.raises(ValueError, match="unknown mode"):
        build({"mode": "video_wall"})
    with pytest.raises(ValueError, match="camera_layout"):
        build({"mode": "wam_canvas"})


def test_empty_raw_images_is_error():
    with pytest.raises(ValueError, match="raw_images is empty"):
        build({"mode": "standard_tensor"}).process(
            PipelineContext(sample_id="", clip_uri="", anchor_timestamp_us=0,
                            cam_frame_offsets={}, source_name="d"))

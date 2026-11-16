"""Unit tests for ObsPreprocessor — obs payload validation + preprocessing.

Engine-free / GPU-free: ObsPreprocessor is constructed from an explicit view config,
so every obs-contract branch (single/multi view, missing cameras, bad base64,
proprio passthrough) is covered without a checkpoint or a live server.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from benchmarks.utils import encode_numpy_b64, resize_for_lshape_slot
from openwam.dataloader.utils.multiview import (
    DEFAULT_MULTIVIEW_CAMERA_LAYOUT,
    format_prompt_for_inference,
)
from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError


def _jpeg_b64(h: int = 48, w: int = 64, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _single_view(*, requires_proprio=False) -> ObsPreprocessor:
    return ObsPreprocessor(
        multiview=False,
        camera_layout=["head_camera"],
        img_height=32,
        img_width=32,
        requires_proprio=requires_proprio,
    )


def _multi_view(*, requires_proprio=False) -> ObsPreprocessor:
    return ObsPreprocessor(
        multiview=True,
        camera_layout=list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT),
        img_height=32,
        img_width=32,
        requires_proprio=requires_proprio,
    )


def test_single_view_head_only_resizes_and_wraps_prompt():
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "pick up the bottle"})
    assert obs["image"].size == (32, 32)
    assert isinstance(obs["prompt"], str) and obs["prompt"]  # wrapped, non-empty


def test_single_view_ignores_wrist_inputs():
    obs = _single_view().preprocess(
        {"images": {"head_camera": _jpeg_b64(), "left_wrist_camera": _jpeg_b64(seed=1)}, "prompt": "x"}
    )
    assert obs["image"].size == (32, 32)


def test_multiview_black_fills_missing_wrists():
    obs = _multi_view().preprocess(
        {"images": {"head_camera": _jpeg_b64(), "left_wrist_camera": None, "right_wrist_camera": None}, "prompt": "x"}
    )
    assert obs["image"].size == (32, 32)


def test_missing_images_dict_raises():
    with pytest.raises(ObsValidationError, match="images"):
        _single_view().preprocess({"prompt": "x"})


def test_missing_head_camera_raises():
    with pytest.raises(ObsValidationError, match="head_camera"):
        _single_view().preprocess({"images": {"head_camera": None}, "prompt": "x"})


def test_bad_base64_raises():
    with pytest.raises(ObsValidationError, match="head_camera"):
        _single_view().preprocess({"images": {"head_camera": "!!!not-a-jpeg!!!"}, "prompt": "x"})


def test_multiview_requires_three_camera_layout():
    dec = ObsPreprocessor(multiview=True, camera_layout=["only_one"], img_height=32, img_width=32)
    with pytest.raises(ObsValidationError, match="camera_layout"):
        dec.preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "x"})


@pytest.mark.parametrize("state_dim", [pytest.param(14, id="dim_14"), pytest.param(20, id="dim_20")])
def test_state_any_dim_passes_through(state_dim):
    # State-dim validation was removed: a state of any width is accepted and flattened.
    obs = _single_view().preprocess(
        {"images": {"head_camera": _jpeg_b64()}, "prompt": "x", "state": list(range(state_dim))}
    )
    assert isinstance(obs["state"], np.ndarray) and obs["state"].shape == (state_dim,)


def test_requires_proprio_but_no_state_raises():
    with pytest.raises(ObsValidationError, match="requires obs"):
        _single_view(requires_proprio=True).preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "x"})


def test_from_cfg_resolves_view_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "dataloader": {"multiview": True, "camera_layout": ["a", "b", "c"]},
            "inference": {"height": 384, "width": 320},
            "model": {"architecture": {"use_proprioception": True, "state_dim": 20}},
        }
    )
    dec = ObsPreprocessor.from_cfg(cfg, engine=None)
    assert dec.multiview is True
    assert dec.camera_layout == ["a", "b", "c"]
    assert (dec.img_height, dec.img_width) == (384, 320)
    assert dec.requires_proprio is True


# --- Pixel-level layout checks (migrated from test_policy_server_obs.py) ---


def _solid_b64(color=(240, 240, 240), w: int = 640, h: int = 480) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _multi_view_canvas(h: int = 384, w: int = 320) -> ObsPreprocessor:
    return ObsPreprocessor(
        multiview=True,
        camera_layout=list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT),
        img_height=h,
        img_width=w,
    )


def test_multiview_all_three_bright_canvas():
    obs = _multi_view_canvas().preprocess(
        {
            "images": {
                "head_camera": _solid_b64(),
                "left_wrist_camera": _solid_b64(),
                "right_wrist_camera": _solid_b64(),
            },
            "prompt": "go",
        }
    )
    assert obs["image"].size == (320, 384)  # PIL.size is (width, height)
    assert np.asarray(obs["image"]).mean() > 200


def test_multiview_black_fill_pixels_when_wrists_none():
    obs = _multi_view_canvas().preprocess(
        {
            "images": {"head_camera": _solid_b64(), "left_wrist_camera": None, "right_wrist_camera": None},
            "prompt": "go",
        }
    )
    arr = np.asarray(obs["image"])
    # Layout: top 2/3 (rows 0-255) is head, bottom 1/3 is the two wrist slots.
    assert arr[:256].mean() > 200
    assert arr[256:].max() < 5


def test_multiview_black_fill_one_wrist_missing():
    obs = _multi_view_canvas().preprocess(
        {
            "images": {"head_camera": _solid_b64(), "left_wrist_camera": _solid_b64()},
            "prompt": "go",
        }
    )
    bottom = np.asarray(obs["image"])[256:]
    assert bottom[:, : 320 // 2].mean() > 200
    assert bottom[:, 320 // 2 :].max() < 5


# --- Prompt template contract (migrated from test_policy_server_obs.py) ---


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        pytest.param(
            "pick up the bottle",
            "A video recorded from a robot's point of view executing the following instruction: pick up the bottle",
            id="plain_instruction",
        ),
        pytest.param(
            "pick up the bottle.",
            "A video recorded from a robot's point of view executing the following instruction: pick up the bottle.",
            id="keeps_trailing_period",
        ),
        pytest.param(
            "",
            "A video recorded from a robot's point of view executing the following instruction: ",
            id="empty_instruction",
        ),
    ],
)
def test_prompt_wrap_appends_instruction(prompt, expected):
    # The template is prepended verbatim; a trailing period in the instruction
    # is kept and the empty instruction still yields the bare template suffix.
    assert format_prompt_for_inference(prompt) == expected


def test_prompt_wrap_matches_dataset_training_output():
    """Regression guard: deploy-side wrapping must equal the dataset's
    training-time prompt byte-for-byte."""
    from openwam.dataloader.robotwin import _resolve_prompt

    base = "pick up the red bottle"
    training_output = _resolve_prompt(
        instructions={"episode0.json": {"seen": [base]}},
        ep_file="episode0.hdf5",
        split="val",  # deterministic: picks pool[0]
        task_name="dummy",
    )
    assert training_output == format_prompt_for_inference(base)


def test_decode_passes_prompt_through():
    # The server is prompt-agnostic: it forwards the client's prompt verbatim.
    # Prompt wrapping now lives in each benchmark's client, not the server.
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "pick up the bottle"})
    assert obs["prompt"] == "pick up the bottle"


def test_decode_missing_prompt_normalized_to_empty():
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}})
    assert obs["prompt"] == ""


def test_state_accepts_nested_list_and_flattens():
    obs = _single_view().preprocess(
        {
            "images": {"head_camera": _jpeg_b64()},
            "prompt": "x",
            "state": np.arange(20, dtype=np.float32).reshape(1, 20).tolist(),
        }
    )
    assert obs["state"].shape == (20,)
    assert obs["state"].dtype == np.float32


# ==== from test_eval_image_transport.py ====
# Pixel-level contracts for benchmark eval image transport.


def _random_rgb(height: int, width: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _decode(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_numpy_transport_is_lossless_png():
    source = _random_rgb(47, 61, seed=0)
    decoded = _decode(encode_numpy_b64(source))
    assert decoded.format == "PNG"
    np.testing.assert_array_equal(np.asarray(decoded.convert("RGB")), source)


@pytest.mark.parametrize(
    ("slot", "expected_size"),
    [
        ("head_camera", (320, 256)),
        ("left_wrist_camera", (160, 128)),
        ("right_wrist_camera", (160, 128)),
    ],
)
def test_eval_slot_resize_is_exact_pillow_lanczos(slot, expected_size):
    source = _random_rgb(256, 256, seed=len(slot))
    actual = resize_for_lshape_slot(source, slot)
    expected = np.asarray(Image.fromarray(source).resize(expected_size, Image.Resampling.LANCZOS))
    assert actual.shape == (expected_size[1], expected_size[0], 3)
    np.testing.assert_array_equal(actual, expected)


def test_lanczos_png_server_composition_is_pixel_exact_for_two_views():
    head = resize_for_lshape_slot(_random_rgb(256, 256, seed=1), "head_camera")
    wrist = resize_for_lshape_slot(_random_rgb(256, 256, seed=2), "left_wrist_camera")
    preprocessor = ObsPreprocessor(
        multiview=True,
        camera_layout=["head", "left", "right"],
        img_height=384,
        img_width=320,
    )
    result = preprocessor.preprocess(
        {
            "images": {
                "head_camera": encode_numpy_b64(head),
                "left_wrist_camera": encode_numpy_b64(wrist),
                "right_wrist_camera": None,
            },
            "prompt": "native prompt",
        }
    )

    expected = np.zeros((384, 320, 3), dtype=np.uint8)
    expected[:256, :] = head
    expected[256:, :160] = wrist
    np.testing.assert_array_equal(np.asarray(result["image"]), expected)


def test_slot_resize_rejects_unknown_slot():
    with pytest.raises(ValueError, match="unknown L-shape image slot"):
        resize_for_lshape_slot(np.zeros((8, 8, 3), dtype=np.uint8), "camera4")


def test_vlabench_adapter_resizes_each_eval_slot(monkeypatch):
    from benchmarks.vlabench import openwam2vlabench_interface as interface

    captured = []
    monkeypatch.setattr(
        interface.client,
        "encode_numpy_b64",
        lambda image: captured.append(np.asarray(image).shape) or "png",
    )
    policy = interface.OpenWAMVLABenchPolicy.__new__(interface.OpenWAMVLABenchPolicy)
    obs = {"rgb": np.zeros((4, 32, 48, 3), dtype=np.uint8)}
    assert policy._maybe_encode(obs, 3, "left_wrist", "left_wrist_camera") == "png"
    assert policy._maybe_encode(obs, 0, "right_wrist", "right_wrist_camera") == "png"
    assert captured == [(128, 160, 3), (128, 160, 3)]


def test_robocasa_gr1_adapter_resizes_eval_head_slot(monkeypatch):
    from benchmarks.robocasa_gr1 import openwam2robocasa_gr1_interface as interface

    captured = []
    monkeypatch.setattr(
        interface.client,
        "encode_numpy_b64",
        lambda image: captured.append(np.asarray(image).shape) or "png",
    )
    policy = interface.OpenWAMRoboCasaGR1Policy.__new__(interface.OpenWAMRoboCasaGR1Policy)
    obs = {"head": np.zeros((256, 256, 3), dtype=np.uint8)}
    assert policy._maybe_encode(obs, "head", "head_camera") == "png"
    assert captured == [(256, 320, 3)]

"""API-hygiene guards for the Cosmos vendor backbones (cosmos3_edge + predict2.5).

Shared invariants, parametrized over both vendor backbones:
  * the public surface may only extend the ``VideoBackbone`` ABC contract —
    every vendor-specific helper must be private;
  * subclass of ``VideoBackbone``.

Vendor-specific gates live in their own sections: cosmos3_edge's strict
preprocess validation (cfg / 4k+1 frames / single image) and predict2.5's
registry name + wired-hook checks. Detailed behavior coverage lives in the
per-topic suites (cfg_inference / deploy_artifacts / joint_self_attn / ...).
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from openwam.model.video_backbone import (
    _VIDEO_BACKBONE_REGISTRY,
    Cosmos3EdgeVideoBackbone,
    VideoBackbone,
)
from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone

VENDORS = [Cosmos3EdgeVideoBackbone, CosmosPredict25VideoBackbone]


def _public_members(cls) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


# ---------------------------------------------------------------------------
# Shared invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", VENDORS, ids=["cosmos3_edge", "predict25"])
def test_no_public_surface_beyond_base_contract(cls):
    extra = _public_members(cls) - _public_members(VideoBackbone)
    assert extra == set(), (
        f"{cls.__name__} exposes public members absent from the VideoBackbone "
        f"base contract (must be private helpers): {sorted(extra)}"
    )


@pytest.mark.parametrize("cls", VENDORS, ids=["cosmos3_edge", "predict25"])
def test_is_video_backbone_subclass(cls):
    assert issubclass(cls, VideoBackbone)


# ---------------------------------------------------------------------------
# predict2.5 specifics
# ---------------------------------------------------------------------------


def test_predict25_registered_under_expected_names():
    for name in ("cosmos_predict25_2b",):
        assert _VIDEO_BACKBONE_REGISTRY[name] is CosmosPredict25VideoBackbone


def test_predict25_inference_and_deploy_are_wired_not_stubs():
    """Inference/deploy hooks are implemented — no longer NotImplementedError stubs.

    Detailed behavior lives in test_cosmos_predict25_cfg_inference.py /
    test_cosmos_predict25_deploy_artifacts.py; here we only assert the stubs are gone.
    """
    bb = CosmosPredict25VideoBackbone.__new__(CosmosPredict25VideoBackbone)
    bb.text_encoder = None  # no live source → inference gate raises

    # Inference now reaches the real no-prompt-source gate (ValueError), not the
    # old NotImplementedError stub.
    with pytest.raises(ValueError, match="text encoder"):
        bb.preprocess_input_for_inference(prompt="x", cfg_scale=1.0)

    # save_deploy_assets is a safe no-op when model_path is unreadable (it must
    # not raise NotImplementedError anymore).
    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": "/no/such/dir"}}})
    bb.save_deploy_assets("/tmp/does-not-matter", cfg=cfg)
    assert "components" not in cfg.model.video_backbone  # unreadable path → no mutation


# ---------------------------------------------------------------------------
# cosmos3_edge specifics
# ---------------------------------------------------------------------------


def _cosmos3_bare_instance():
    return Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)


def test_cosmos3_inference_hook_is_wired_not_stub():
    vb = _cosmos3_bare_instance()
    try:
        vb.preprocess_input_for_inference()
    except ValueError:
        pass  # the real gate: missing `prompt`
    except NotImplementedError as exc:  # pragma: no cover
        raise AssertionError("preprocess_input_for_inference is an unwired stub") from exc


def test_cosmos3_decode_hook_is_wired_not_stub():
    vb = _cosmos3_bare_instance()
    try:
        vb.decode_video(None)  # type: ignore[arg-type]
    except ValueError:
        pass  # the real gate: no VAE attached
    except NotImplementedError as exc:  # pragma: no cover
        raise AssertionError("decode_video is an unwired stub") from exc


def _cosmos3_validating_instance():
    vb = _cosmos3_bare_instance()
    object.__setattr__(vb, "_temporal_compression", 4)
    return vb


def test_cosmos3_deploy_cfg_gt1_rejected_loudly():
    vb = _cosmos3_validating_instance()
    with pytest.raises(NotImplementedError, match="cfg_scale"):
        vb.preprocess_input_for_inference(prompt="x", first_frame_image=object(), cfg_scale=2.0)


def test_cosmos3_deploy_num_frames_must_be_4k_plus_1():
    vb = _cosmos3_validating_instance()
    with pytest.raises(ValueError, match="4k\\+1"):
        vb.preprocess_input_for_inference(prompt="x", first_frame_image=object(), num_frames=32)


def test_cosmos3_deploy_multi_image_list_rejected():
    vb = _cosmos3_validating_instance()
    with pytest.raises(ValueError, match="exactly one"):
        vb.preprocess_input_for_inference(prompt="x", first_frame_image=[object(), object()])


def test_cosmos3_save_deploy_assets_noop_on_unreadable_model_path(tmp_path):
    vb = _cosmos3_bare_instance()
    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": str(tmp_path / "missing")}}})
    vb.save_deploy_assets(str(tmp_path), cfg)
    assert OmegaConf.select(cfg, "model.video_backbone.components") is None
    assert not (tmp_path / "text_tokenizer").exists()

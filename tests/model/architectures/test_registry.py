"""Registry + config resolution tests for the VLA architecture family.

Verifies:
  1. ``vla_system`` owns every (framework=vla, variant) index entry — it is
     the ONE registry name for the family (the per-variant subclasses were
     deleted after the loss reproduction gate pinned equivalence).
  2. ``resolve_architecture_config`` flattens the VLA ``architecture:`` block
     and forwards the ``vlm_backbone`` group unchanged (registry needs no edit
     for the VLA family — the resolve path is shared with WAM).
  3. The shipped vla.yaml / qwen3_vl_vla.yaml / robotwin_vla.yaml Hydra-compose
     cleanly through ``train``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_vla_names_registered():
    from openwam.model.architectures.registry import (
        _FRAMEWORK_VARIANT_INDEX,
        ARCHITECTURE_REGISTRY,
        ARCHITECTURE_SUPPORT,
    )

    assert "vla_system" in ARCHITECTURE_REGISTRY
    assert ARCHITECTURE_SUPPORT["vla_system"].supported
    for variant in ("system", "oftp", "pi", "groot"):
        assert _FRAMEWORK_VARIANT_INDEX[("vla", variant)] == "vla_system"


def test_vla_base_classes_inherit_common_root():
    from openwam.model.architectures.base import BaseArchitecture, BaseWAMArchitecture
    from openwam.model.architectures.vla import BaseVLAArchitecture

    assert issubclass(BaseVLAArchitecture, BaseArchitecture)
    assert issubclass(BaseWAMArchitecture, BaseArchitecture)
    assert issubclass(BaseVLAArchitecture, BaseWAMArchitecture) is False


@pytest.mark.parametrize(
    "variant,registry_name",
    [("oftp", "vla_system"), ("pi", "vla_system"), ("groot", "vla_system")],
)
def test_vla_resolve_framework_variant(variant, registry_name):
    from openwam.model import resolve_architecture_config

    model_cfg = SimpleNamespace(
        architecture={"framework": "vla", "variant": variant, "action_head": {"action_dim": 7}},
        action_backbone=None,
        video_backbone=None,
        vlm_backbone={"name": "qwen3_vl_vla"},
    )
    resolved = resolve_architecture_config(model_cfg)
    assert resolved.registry_name == registry_name
    assert resolved.params["framework"] == "vla"
    assert resolved.params["variant"] == variant
    # The vlm_backbone group forwards unchanged for the architecture to consume.
    assert resolved.params["vlm_backbone"] == {"name": "qwen3_vl_vla"}


def test_vla_yaml_hydra_composes():
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["model=vla", "dataloader=robot/robotwin_vla"])

    assert cfg.model.architecture.framework == "vla"
    assert cfg.model.architecture.variant == "oftp"
    assert cfg.model.vlm_backbone.name == "qwen3_vl_vla"
    assert cfg.model.architecture.action_head.action_model_type == "MLP"
    # The VLA dataloader keeps raw-width actions.
    assert cfg.dataloader.unify_action is False

    from openwam.model import resolve_architecture_config

    resolved = resolve_architecture_config(cfg.model)
    assert resolved.registry_name == "vla_system"
    assert resolved.params["vlm_backbone"].name == "qwen3_vl_vla"


def test_vla_yaml_pi_variant_override_composes():
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    overrides = [
        "model=vla",
        "dataloader=robot/robotwin_vla",
        "model.architecture.variant=pi",
        "model.architecture.action_head.action_model_type=LayerwiseFM",
    ]
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=overrides)

    assert cfg.model.architecture.variant == "pi"
    assert cfg.model.architecture.action_head.action_model_type == "LayerwiseFM"

    from openwam.model import resolve_architecture_config

    assert resolve_architecture_config(cfg.model).registry_name == "vla_system"

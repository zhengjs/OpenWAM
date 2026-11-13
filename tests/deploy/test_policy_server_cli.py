"""CLI tests for the OpenWAM policy server execution override wiring.

Exercises ``_apply_execution_cli_overrides`` / ``_build_argparser`` without an
engine, GPU, or weights: these controls are pure config logic.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf


def test_cli_execution_overrides_write_inference_section():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--inference-mode", "async", "--inference-horizon", "24", "--inference-delay-steps", "6"])
    cfg = _apply_execution_cli_overrides(OmegaConf.create({}), args)

    assert OmegaConf.select(cfg, "inference.inference_mode") == "async"
    assert OmegaConf.select(cfg, "inference.inference_horizon") == 24
    assert OmegaConf.select(cfg, "inference.inference_delay_steps") == 6


def test_cli_inference_horizon_is_valid_in_sync_mode():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--inference-mode", "sync", "--inference-horizon", "24"])
    cfg = _apply_execution_cli_overrides(OmegaConf.create({}), args)

    assert OmegaConf.select(cfg, "inference.inference_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.inference_horizon") == 24
    assert OmegaConf.select(cfg, "inference.inference_delay_steps") is None


def test_cli_execution_overrides_fail_fast_on_invalid_ranges():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--inference-mode", "async", "--inference-horizon", "4"])
    args.inference_delay_steps = 4

    with pytest.raises(ValueError, match="inference_delay_steps must be < inference_horizon"):
        _apply_execution_cli_overrides(OmegaConf.create({}), args)


def test_cli_inference_mode_rejects_invalid_value():
    from openwam.deploy.server import _build_argparser

    with pytest.raises(SystemExit):
        _build_argparser().parse_args(["--inference-mode", "unsupported"])


# --- Unified CLI: the package entrypoint is a strict superset of scripts/deploy.py ---


def test_cli_exposes_ckpt_name_and_inference_overrides():
    """Flags absorbed from scripts/deploy.py parse with the documented defaults."""
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args([])
    assert args.ckpt_name is None
    assert args.denoise_steps is None
    assert args.denoise_mode is None
    assert args.device is None  # fallback chain resolves later: CLI > yaml > cuda

    args = _build_argparser().parse_args(["--ckpt-name", "checkpoint_step_42.safetensors", "--denoise-steps", "7"])
    assert args.ckpt_name == "checkpoint_step_42.safetensors"
    assert args.denoise_steps == 7


def test_cli_denoise_mode_accepts_sync_and_async():
    from openwam.deploy.server import _build_argparser

    parser = _build_argparser()
    assert parser.parse_args(["--denoise-mode", "sync"]).denoise_mode == "sync"
    assert parser.parse_args(["--denoise-mode", "async"]).denoise_mode == "async"
    with pytest.raises(SystemExit):
        parser.parse_args(["--denoise-mode", "unsupported"])


def test_cli_async_denoising_overrides_parse():
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args(
        [
            "--lead-modality",
            "video",
            "--variance-shift-alpha",
            "9",
            "--linear-offset",
            "0.2",
        ]
    )
    assert args.lead_modality == "video"
    assert args.variance_shift_alpha == 9.0
    assert args.linear_offset == 0.2


def test_cli_async_denoising_overrides_require_async_mode():
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--variance-shift-alpha", "9"])
    with pytest.raises(ValueError, match="--denoise-mode async"):
        _apply_inference_overrides(OmegaConf.create({}), args)


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    [
        ("--variance-shift-alpha", "0", "variance_shift_alpha must be >= 1"),
        ("--linear-offset", "-0.1", "linear_offset must satisfy"),
        ("--linear-offset", "1", "linear_offset must satisfy"),
    ],
)
def test_cli_async_denoising_overrides_validate_ranges(flag, value, match):
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--denoise-mode", "async", flag, value])
    with pytest.raises(ValueError, match=match):
        _apply_inference_overrides(OmegaConf.create({}), args)


def _async_denoise_cfg():
    return OmegaConf.create(
        {
            "inference": {
                "denoise_mode": "async",
                "lead_modality": "action",
                "variance_shift_alpha": 3.0,
                "linear_offset": 0.1,
            }
        }
    )


def test_cli_denoise_mode_sync_resets_async_controls():
    """Running the sync baseline against an async yaml needs no field-by-field unset."""
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--denoise-mode", "sync"])
    cfg = _apply_inference_overrides(_async_denoise_cfg(), args)

    assert OmegaConf.select(cfg, "inference.denoise_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.lead_modality") == "video"
    assert OmegaConf.select(cfg, "inference.variance_shift_alpha") == 1.0
    assert OmegaConf.select(cfg, "inference.linear_offset") == 0.0


def test_cli_denoise_mode_sync_still_rejects_contradictory_async_flags():
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--denoise-mode", "sync", "--variance-shift-alpha", "3"])
    with pytest.raises(ValueError, match="--variance-shift-alpha"):
        _apply_inference_overrides(_async_denoise_cfg(), args)


def test_cli_async_flags_at_their_defaults_are_accepted_under_sync():
    """The CLI must be no stricter than the same value written in the yaml."""
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(
        ["--denoise-mode", "sync", "--lead-modality", "video", "--variance-shift-alpha", "1.0", "--linear-offset", "0"]
    )
    cfg = _apply_inference_overrides(_async_denoise_cfg(), args)
    assert OmegaConf.select(cfg, "inference.denoise_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.variance_shift_alpha") == 1.0


def test_cli_inference_mode_sync_keeps_horizon_and_resets_async_delay():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    cfg = OmegaConf.create(
        {"inference": {"inference_mode": "async", "inference_horizon": 8, "inference_delay_steps": 2}}
    )
    args = _build_argparser().parse_args(["--inference-mode", "sync"])
    cfg = _apply_execution_cli_overrides(cfg, args)

    assert OmegaConf.select(cfg, "inference.inference_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.inference_horizon") == 8
    assert OmegaConf.select(cfg, "inference.inference_delay_steps") is None


def test_cli_inference_mode_sync_rejects_async_delay_flag():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    cfg = OmegaConf.create({"inference": {"inference_mode": "async", "inference_horizon": 8}})
    args = _build_argparser().parse_args(["--inference-mode", "sync", "--inference-delay-steps", "2"])
    with pytest.raises(ValueError, match="--inference-mode async"):
        _apply_execution_cli_overrides(cfg, args)


def test_noop_async_denoising_warns_at_startup(caplog):
    from openwam.deploy.server import _validate_inference_config

    cfg = OmegaConf.create({"inference": {"denoise_mode": "async"}})
    with caplog.at_level("WARNING", logger="openwam.deploy.server"):
        _validate_inference_config(cfg)
    assert any("reproduces the sync trajectory" in record.getMessage() for record in caplog.records)


def test_shifted_async_denoising_does_not_warn(caplog):
    from openwam.deploy.server import _validate_inference_config

    cfg = OmegaConf.create({"inference": {"denoise_mode": "async", "variance_shift_alpha": 3.0}})
    with caplog.at_level("WARNING", logger="openwam.deploy.server"):
        _validate_inference_config(cfg)
    assert not any("reproduces the sync trajectory" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("inference", "match"),
    [
        ({"denoise_mode": "unsupported"}, "Unsupported denoise mode"),
        ({"denoise_mode": "sync", "variance_shift_alpha": 9.0}, "denoise_mode='async'"),
        ({"inference_mode": "sync", "inference_delay_steps": 2}, "inference_mode='async'"),
    ],
)
def test_deploy_inference_config_is_validated_at_startup(inference, match):
    from openwam.deploy.server import _validate_inference_config

    with pytest.raises(ValueError, match=match):
        _validate_inference_config(OmegaConf.create({"inference": inference}))


def test_cli_dotlist_overrides_coexist_with_value_flags():
    """Positional dotlist overrides must not swallow values of the new flags."""
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args(["--denoise-steps", "7", "foo.bar=1", "inference.shift=9.0"])
    assert args.denoise_steps == 7
    assert args.overrides == ["foo.bar=1", "inference.shift=9.0"]


# ==== from test_deployment_changes.py ====
# Tests for deployment-related changes.
#
# Covers mixed-precision save/load behavior, deploy.yaml inference/optimization
# sections, deploy.py config loading, CLI override logic, attention backend
# logging, and joint_engine compile flags.
# (Config-loading / CLI / server classes migrated here; the openwam.deploy.engine
# classes moved to tests/deploy/test_engine.py.)

# ---------------------------------------------------------------------------
# 4. deploy.py — config loading and CLI override logic
# ---------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class TestDeployConfigLoading:
    """Config loading + CLI override logic of the unified server CLI (no server run)."""

    def _compile_options(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compile_options", PROJECT_ROOT / "openwam" / "model" / "compile_options.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _policy_server(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("server", PROJECT_ROOT / "openwam" / "deploy" / "server.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _blank_args(self):
        args = MagicMock()
        for attr in (
            "device",
            "host",
            "port",
            "denoise_steps",
            "denoise_mode",
            "lead_modality",
            "variance_shift_alpha",
            "linear_offset",
            "shift",
            "compile_enabled",
            "inference_mode",
            "inference_horizon",
            "inference_delay_steps",
        ):
            setattr(args, attr, None)
        return args



    def test_cli_compile_enabled_false_disables_compile(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()
        compile_options = self._compile_options()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.compile_enabled = False

        deploy._apply_compile_enabled_override(cfg, args.compile_enabled)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is False
        compile_cfg = OmegaConf.select(cfg, "optimization.compile")
        assert compile_options.compile_enabled(compile_cfg, strict=True) is False

    def test_cli_compile_enabled_true_keeps_architecture_selection(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()
        compile_options = self._compile_options()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.compile_enabled = True

        deploy._apply_compile_enabled_override(cfg, args.compile_enabled)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is True
        compile_cfg = OmegaConf.select(cfg, "optimization.compile")
        assert compile_options.compile_enabled(compile_cfg, strict=True) is True

    def test_cli_compile_enabled_accepts_bool_strings(self):
        import argparse

        deploy = self._policy_server()

        parser = argparse.ArgumentParser()
        parser.add_argument("--compile-enabled", type=deploy._normalize_compile_enabled_arg)

        assert parser.parse_args(["--compile-enabled", "true"]).compile_enabled is True
        assert parser.parse_args(["--compile-enabled", "false"]).compile_enabled is False
        with pytest.raises(SystemExit):
            parser.parse_args(["--compile-enabled", "auto"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--compile-enabled", "none"])

    def test_mode_only_compile_sections_use_fast_path_defaults(self):
        compile_options = self._compile_options()

        self_cfg = {"self_attn": {}}
        self_section = compile_options.self_attn_compile_cfg(self_cfg)
        assert compile_options.section_enabled(self_section, default=False) is True
        assert compile_options.torch_compile_kwargs(self_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

        cross_cfg = {"cross_attn": {}}
        cross_section = compile_options.cross_attn_compile_cfg(cross_cfg)
        assert compile_options.section_enabled(cross_section, default=False) is True
        assert compile_options.torch_compile_kwargs(cross_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

        idm_cfg = {"idm": {}}
        idm_section = compile_options.idm_compile_cfg(idm_cfg)
        assert compile_options.section_enabled(idm_section, default=False) is True
        assert compile_options.torch_compile_kwargs(idm_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }
        assert compile_options.section_enabled(idm_section.video_loop, default=False) is True
        assert compile_options.section_enabled(idm_section.action_cache, default=False) is True

        tri_cfg = {"tri_system": {}}
        tri_section = compile_options.tri_system_compile_cfg(tri_cfg)
        assert compile_options.section_enabled(tri_section, default=False) is True
        assert compile_options.torch_compile_kwargs(tri_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

    def test_compile_section_enabled_false_is_respected(self):
        compile_options = self._compile_options()

        self_cfg = {"self_attn": {"enabled": False}}
        self_section = compile_options.self_attn_compile_cfg(self_cfg)
        assert compile_options.section_enabled(self_section, default=True) is False

        cross_cfg = {"cross_attn": {"enabled": False}}
        cross_section = compile_options.cross_attn_compile_cfg(cross_cfg)
        assert compile_options.section_enabled(cross_section, default=True) is False

        idm_cfg = {"idm": {"enabled": False}}
        idm_section = compile_options.idm_compile_cfg(idm_cfg)
        assert compile_options.section_enabled(idm_section, default=True) is False
        assert compile_options.section_enabled(idm_section.video_loop, default=True) is False
        assert compile_options.section_enabled(idm_section.action_cache, default=True) is False

        tri_cfg = {"tri_system": {"enabled": False}}
        tri_section = compile_options.tri_system_compile_cfg(tri_cfg)
        assert compile_options.section_enabled(tri_section, default=True) is False

    def test_idm_compile_subsections_can_be_disabled_independently(self):
        compile_options = self._compile_options()

        idm_section = compile_options.idm_compile_cfg(
            {
                "idm": {
                    "video_loop": {"enabled": False},
                    "action_cache": {"enabled": True},
                }
            }
        )

        assert compile_options.section_enabled(idm_section, default=True) is True
        assert compile_options.section_enabled(idm_section.video_loop, default=True) is False
        assert compile_options.section_enabled(idm_section.action_cache, default=False) is True

    def test_compile_enabled_validation_rejects_non_bool_values(self):
        compile_options = self._compile_options()

        assert compile_options.normalize_compile_enabled("true") is True
        assert compile_options.normalize_compile_enabled("false") is False
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("default")
        with pytest.raises(ValueError, match="Unknown legacy compile mode"):
            compile_options.compile_enabled({"mode": "default"}, strict=True)
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("self_attn")
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("cross-attn")

    def test_policy_server_entrypoint_validates_compile_enabled(self):
        from omegaconf import OmegaConf

        policy_server = self._policy_server()

        cfg = OmegaConf.create({"optimization": {"compile": {"enabled": "false"}}})
        policy_server._normalize_compile_enabled_in_cfg(cfg)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is False

        true_cfg = OmegaConf.create({"optimization": {"compile": {"enabled": "true"}}})
        policy_server._normalize_compile_enabled_in_cfg(true_cfg)
        assert OmegaConf.select(true_cfg, "optimization.compile.enabled") is True

        legacy_cfg = OmegaConf.create({"optimization": {"compile": {"mode": "auto"}}})
        policy_server._normalize_compile_enabled_in_cfg(legacy_cfg)
        assert OmegaConf.select(legacy_cfg, "optimization.compile.enabled") is True

        bad_cfg = OmegaConf.create({"optimization": {"compile": {"enabled": "default"}}})
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            policy_server._normalize_compile_enabled_in_cfg(bad_cfg)

    def test_policy_server_compile_enabled_cli_override(self):
        from omegaconf import OmegaConf

        policy_server = self._policy_server()

        args = policy_server._build_argparser().parse_args(["--compile-enabled", "false"])
        assert args.compile_enabled is False

        cfg = OmegaConf.create({})
        policy_server._apply_compile_enabled_override(cfg, args.compile_enabled)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is False

        with pytest.raises(SystemExit):
            policy_server._build_argparser().parse_args(["--compile-enabled", "self-attn"])



    def test_cli_inference_horizon_override_supports_sync(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "sync"
        args.inference_horizon = 24

        cfg = deploy._apply_execution_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.inference_mode") == "sync"
        assert OmegaConf.select(cfg, "inference.inference_horizon") == 24
        assert OmegaConf.select(cfg, "inference.inference_delay_steps") is None

    def test_cli_async_numeric_overrides_fail_fast_on_invalid_ranges(self):
        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "async"
        args.inference_horizon = 0
        with pytest.raises(ValueError, match="inference_horizon must be positive"):
            deploy._apply_execution_cli_overrides(cfg, args)

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "async"
        args.inference_horizon = 4
        args.inference_delay_steps = 4
        with pytest.raises(ValueError, match="inference_delay_steps must be < inference_horizon"):
            deploy._apply_execution_cli_overrides(cfg, args)

    def test_none_args_do_not_override(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        original_steps = OmegaConf.select(cfg, "inference.denoise_steps")

        args = self._blank_args()
        # All None — nothing should change
        for attr in (
            "device",
            "host",
            "port",
            "denoise_steps",
            "denoise_mode",
            "shift",
            "inference_mode",
            "inference_horizon",
            "inference_delay_steps",
        ):
            setattr(args, attr, None)

        cfg = deploy._apply_execution_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.denoise_steps") == original_steps

    def test_merge_with_training_cfg_uses_dataloader_dims(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        policy_server = self._policy_server()
        training_cfg = OmegaConf.create({"dataloader": {"num_frames": 49, "height": 720, "width": 1280}})
        deploy_cfg = deploy._load_deploy_yaml()

        # Remove inference dims so they should be filled from dataloader
        OmegaConf.update(deploy_cfg, "inference.num_frames", None, merge=False)
        OmegaConf.update(deploy_cfg, "inference.height", None, merge=False)
        OmegaConf.update(deploy_cfg, "inference.width", None, merge=False)

        merged = policy_server.merge_deploy_cfg(training_cfg, deploy_cfg)
        assert OmegaConf.select(merged, "inference.num_frames") == 49
        assert OmegaConf.select(merged, "inference.height") == 720
        assert OmegaConf.select(merged, "inference.width") == 1280

    def test_deploy_cfg_wins_over_training_cfg_on_overlap(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        policy_server = self._policy_server()
        training_cfg = OmegaConf.create({"inference": {"denoise_steps": 99}})
        deploy_cfg = deploy._load_deploy_yaml()
        OmegaConf.update(deploy_cfg, "inference.denoise_steps", 10, merge=False)

        merged = policy_server.merge_deploy_cfg(training_cfg, deploy_cfg)
        assert OmegaConf.select(merged, "inference.denoise_steps") == 10

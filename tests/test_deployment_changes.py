"""Tests for deployment-related changes.

Covers mixed-precision save/load behavior, deploy.yaml inference/optimization
sections, deploy.py config loading, CLI override logic, attention backend
logging, and joint_engine compile flags.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


# ---------------------------------------------------------------------------
# 1. mixed_precision in accelerate yaml (single source of truth)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. deployment.yaml — inference + deploy sections
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. deploy.py — config loading and CLI override logic
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 5. joint_engine.py — compile flags parsed from cfg.optimization.compile
# ---------------------------------------------------------------------------


class TestJointEngineCompileFlags:
    """Verify compile mode routing without broad default compile side effects."""

    def _make_filter_engine(self, architecture):
        from openwam.deploy.engine import JointInferenceEngine

        engine = JointInferenceEngine.__new__(JointInferenceEngine)
        engine.architecture = architecture
        engine._architecture_generate_accepts_extra_kwargs = None
        engine._architecture_generate_kwarg_names = None
        engine._architecture_generate_warned_dropped_kwargs = set()
        return engine

    def _make_engine(self, compile_enabled=False, return_arch=False, prompt_cache_cfg=None):
        from omegaconf import OmegaConf

        from openwam.deploy.engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "denoise_mode": "sync",
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 384,
                    "width": 320,
                },
                "optimization": {
                    "decode_video": True,
                    "compile": {
                        "enabled": compile_enabled,
                        "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
                        "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
                        "tri_system": {"torch_mode": "reduce-overhead", "dynamic": False},
                    },
                    "dit_cache": {"enabled": False},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )
        if prompt_cache_cfg is not None:
            cfg.optimization.prompt_embed_cache = prompt_cache_cfg

        arch = MagicMock()
        with patch("torch.compile") as mock_compile:
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            engine._init_optimizations()
        if return_arch:
            return engine, mock_compile, arch
        return engine, mock_compile

    def test_compile_enabled_false_does_not_broad_compile(self):
        _engine, mock_compile, arch = self._make_engine(return_arch=True)
        mock_compile.assert_not_called()
        arch.apply_compile_optimizations.assert_called_once()

    def test_compile_enabled_true_is_passed_to_architecture(self):
        from omegaconf import OmegaConf

        _engine, mock_compile, arch = self._make_engine(True, return_arch=True)
        mock_compile.assert_not_called()
        compile_cfg = arch.apply_compile_optimizations.call_args.args[0]
        assert OmegaConf.select(compile_cfg, "enabled") is True

    def test_prompt_embed_cache_default_is_bounded_default(self):
        engine, _ = self._make_engine()
        from openwam.deploy.engine import DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE, _BoundedPromptEmbedCache

        assert isinstance(engine._prompt_embed_cache, _BoundedPromptEmbedCache)
        assert engine._prompt_embed_cache._maxsize == DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE

    def test_prompt_embed_cache_eviction_preserves_lru_order(self, caplog):
        from openwam.deploy.engine import _BoundedPromptEmbedCache

        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        cache = _BoundedPromptEmbedCache(maxsize=3)
        for key in ("probe", "a", "b", "c"):
            cache[key] = key

        assert list(cache) == ["a", "b", "c"]

        # Reads refresh recency, so "a" survives the next eviction.
        assert cache["a"] == "a"
        cache["d"] = "d"
        assert list(cache) == ["c", "a", "d"]

        cache["e"] = "e"
        assert list(cache) == ["a", "d", "e"]
        warnings = [record for record in caplog.records if "prompt_embed_cache exceeded maxsize" in record.getMessage()]
        assert len(warnings) == 1



    def test_generate_kwarg_filter_treats_disabled_cache_drop_as_noop(self, caplog):
        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())
        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        filtered = engine._filter_architecture_generate_kwargs(
            {"schedule": object(), "prompt": "pick up the cube", "prompt_embed_cache": None}
        )
        assert set(filtered) == {"schedule", "prompt"}
        assert not [r for r in caplog.records if "does not accept deploy kwarg" in r.getMessage()]

    def test_generate_kwarg_filter_warns_once_for_meaningful_drops(self, caplog):
        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        kwargs = {
            "schedule": object(),
            "prompt": "pick up the cube",
            "cfg_scale": 1.5,
            "cfg_merge": False,
            "prompt_embed_cache": object(),
        }

        filtered = engine._filter_architecture_generate_kwargs(kwargs)
        engine._filter_architecture_generate_kwargs(kwargs)

        assert set(filtered) == {"schedule", "prompt"}
        warning_messages = [
            record.getMessage() for record in caplog.records if "does not accept deploy kwarg" in record.getMessage()
        ]
        assert len(warning_messages) == 1
        assert "cfg_scale" in warning_messages[0]
        assert "prompt_embed_cache" in warning_messages[0]

    def test_generate_kwarg_filter_keeps_default_noop_drops_quiet(self, caplog):
        from openwam.deploy.engine import _BoundedPromptEmbedCache

        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        kwargs = {
            "schedule": object(),
            "prompt": "pick up the cube",
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "prompt_embed_cache": _BoundedPromptEmbedCache(),
        }

        filtered = engine._filter_architecture_generate_kwargs(kwargs)

        assert set(filtered) == {"schedule", "prompt"}
        assert not [record for record in caplog.records if "does not accept deploy kwarg" in record.getMessage()]

    def test_generate_kwarg_filter_warns_for_configured_prompt_cache_drop(self, caplog):
        from openwam.deploy.engine import _BoundedPromptEmbedCache

        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        filtered = engine._filter_architecture_generate_kwargs(
            {
                "schedule": object(),
                "prompt": "pick up the cube",
                "cfg_scale": 1.0,
                "prompt_embed_cache": _BoundedPromptEmbedCache(maxsize=64),
            }
        )

        assert set(filtered) == {"schedule", "prompt"}
        warning_messages = [
            record.getMessage() for record in caplog.records if "does not accept deploy kwarg" in record.getMessage()
        ]
        assert len(warning_messages) == 1
        assert "prompt_embed_cache" in warning_messages[0]

    def test_base_architecture_does_not_broad_compile_backbones(self):
        from omegaconf import OmegaConf

        from tests.train.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        cfg = OmegaConf.create(
            {
                "enabled": False,
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()

    def test_architecture_auto_compile_respects_disabled_fast_path(self):
        from omegaconf import OmegaConf

        from tests.train.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        cfg = OmegaConf.create({"enabled": True, "cross_attn": {"enabled": False}})

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()


# ---------------------------------------------------------------------------
# 6. model_loader.py — dtype applied to all pipeline modules
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# 7. server CLI — _log_attention_backends output format
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. architecture.generate() — cudagraph_mark_step_begin placement
# ---------------------------------------------------------------------------


class TestCudagraphMarkStepBegin:
    """Structural check: every architecture forward dispatch in generate()
    must be preceded by ``torch.compiler.cudagraph_mark_step_begin()`` to
    prevent CUDA Graph tree from raising 'tensor output overwritten by
    subsequent run' when fixed-shape compile paths are active.

    After the architecture refactor the dispatch site is in
    ``BaseWAMArchitecture.generate()`` in ``base.py``.
    """

    _DISPATCH_SUBSTRINGS = (
        "noise_pred, action_noise_pred = self.forward(",
        "noise_pred = vb.finalize(state)",
        # §15 — CFG forward dispatch sites inside _forward_with_cfg
        "merged_noise, merged_action = self.forward(",
        "cond_noise, cond_action = self.forward(",
        "uncond_noise, uncond_action = self.forward(",
    )

    def _source(self):
        # Only check generate() method, not compute_loss()
        src = (PROJECT_ROOT / "openwam" / "model" / "architectures" / "base.py").read_text()
        marker = "def generate("
        idx = src.index(marker)
        return src[idx:]

    def test_mark_count_equals_dispatch_sites(self):
        src = self._source()
        mark_count = src.count("torch.compiler.cudagraph_mark_step_begin()")
        expected = sum(src.count(sub) for sub in self._DISPATCH_SUBSTRINGS)
        assert mark_count == expected, (
            f"Expected {expected} cudagraph_mark_step_begin() call(s), found {mark_count}. "
            "Add torch.compiler.cudagraph_mark_step_begin() before any new "
            "architecture forward dispatch."
        )

    def test_mark_appears_before_not_after(self):
        """The mark must appear on the line immediately before each dispatch site."""
        src = self._source()
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if any(sub in line for sub in self._DISPATCH_SUBSTRINGS):
                prev = i - 1
                while prev >= 0 and (lines[prev].strip() == "" or lines[prev].lstrip().startswith("#")):
                    prev -= 1
                assert "cudagraph_mark_step_begin" in lines[prev], (
                    f"Line {i + 1}: dispatch not preceded by "
                    f"cudagraph_mark_step_begin(). Found instead: {lines[prev]!r}"
                )

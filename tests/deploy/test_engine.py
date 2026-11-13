"""Engine-level tests migrated from ``tests/test_deployment_changes.py``.

Covers ``openwam.deploy.engine`` compile-flag routing, the bounded
prompt-embed cache, architecture generate-kwarg filtering, and the static
``cudagraph_mark_step_begin`` placement pins over architecture dispatch sites.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


# ==== from test_deployment_changes.py ====


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

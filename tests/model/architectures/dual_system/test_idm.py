"""Smoke tests for DualSystem IDM (Inverse Dynamics Model) architecture.

Tests the IDM-specific features:
  - Teacher-forcing attention mask layout
  - IDM architecture construction and registration
  - ActionDiT pre/post_attn round-trip under IDM
  - IDMMoTDriver teacher-forcing mask correctness
"""

import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture, IDMMoTDriver
from openwam.model.video_backbone.base import VideoBackbone
from tests.model.video_backbone.cosmos3.test_joint_self_attn import _TinyActionBackbone
from tests.model.video_backbone.cosmos_predict25.test_joint_self_attn import _build_rich_wrapper


def _make_idm(dim=64, video_dim=64):
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": 7,
        "dim": dim,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": video_dim,
        "bridge_layers": (0, 2),
        "idm_video_cond_noise_prob": 0.5,
    }
    return build_architecture("dual_system_idm", cfg)


def test_idm_registered():
    """IDM variant should be in the supported architecture registry."""
    from openwam.model import list_supported_architectures

    supported = list_supported_architectures()
    assert "dual_system_idm" in supported


def test_idm_construction():
    """IDM architecture constructs with expected components."""
    arch = _make_idm()
    assert arch.action_backbone is not None
    assert arch.action_backbone.variant == "idm"
    assert hasattr(arch, "_mot_driver")
    assert hasattr(arch, "video_cond_noise_prob")
    assert arch.video_cond_noise_prob == 0.5


def test_idm_rejects_out_of_range_cond_noise_prob():
    """idm_video_cond_noise_prob outside [0, 1] silently breaks the Bernoulli gate."""
    import pytest

    from openwam.model import build_architecture

    base_cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 64,
        "bridge_layers": (0, 2),
    }
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="idm_video_cond_noise_prob must be in"):
            build_architecture("dual_system_idm", {**base_cfg, "idm_video_cond_noise_prob": bad})


def test_idm_legacy_cond_noise_key_still_works():
    """Keep the old unprefixed key working while the config migrates."""
    from openwam.model import build_architecture

    arch = build_architecture(
        "dual_system_idm",
        {
            "framework": "dual_system",
            "variant": "idm",
            "action_dim": 7,
            "dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "video_dim": 64,
            "bridge_layers": (0, 2),
            "video_cond_noise_prob": 0.25,
        },
    )
    assert arch.video_cond_noise_prob == 0.25


def test_idm_ignores_attention_mask_mode():
    """IDM owns its teacher-forcing / Stage-2 masks and ignores attention_mask_mode.

    A stale cross-modal mode (previously a hard error) must now construct fine
    and be dropped — never forwarded to the driver kwargs.
    """
    from openwam.model import build_architecture

    arch = build_architecture(
        "dual_system_idm",
        {
            "framework": "dual_system",
            "variant": "idm",
            "action_dim": 7,
            "dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "video_dim": 64,
            "bridge_layers": (0, 2),
            "attention_mask_mode": "mutual",
        },
    )
    assert "attention_mask_mode" not in arch._mot_driver_kwargs


def test_idm_action_dit_roundtrip():
    """ActionDiT pre/post_attn round-trip works under IDM architecture."""
    arch = _make_idm()
    ab = arch.action_backbone
    B, T_action = 2, 5

    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])
    context = torch.randn(B, 4, ab.text_dim)
    context_mask = torch.ones(B, 4, dtype=torch.bool)

    astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)

    for layer_id in range(ab.num_layers):
        q, k, v, post = ab.pre_attn_at_layer(layer_id, astate)
        assert q.shape == k.shape == v.shape == (B, T_action, ab.num_heads * ab.head_dim)
        attn_out = torch.randn_like(q)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_out, post)

    pred = ab.extract_prediction(astate)
    assert pred.shape == (B, T_action, 7)


def test_idm_teacher_forcing_mask():
    """Teacher-forcing mask has correct layout."""
    from openwam.model.architectures.dual_system.idm import IDMMoTDriver

    # Mock video backbone with a simple bidirectional v2v mask
    vb = MagicMock()
    vb.num_layers = 2
    vb.num_heads = 4
    vb.head_dim = 16
    vb.dim = 64

    def mock_v2v_mask(video_seq_len, video_tokens_per_frame, device):
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    vb.build_video_to_video_mask = mock_v2v_mask
    vb.video_attention_mask_mode = "bidirectional"

    ab = MagicMock()
    ab.num_layers = 2
    ab.num_heads = 4
    ab.head_dim = 16
    ab.training = False

    driver = IDMMoTDriver(vb, ab, attention_mask_mode="action_sees_video")

    s_noisy = 10
    s_cond = 10
    s_action = 5
    mask = driver._build_teacher_forcing_mask(
        s_noisy_video=s_noisy,
        s_cond_video=s_cond,
        s_action=s_action,
        video_tokens_per_frame=5,
        device=torch.device("cpu"),
    )

    total = s_noisy + s_cond + s_action
    assert mask.shape == (total, total)

    # Check noisy_video ↔ noisy_video (True for bidirectional)
    assert mask[:s_noisy, :s_noisy].all()
    # Check cond_video ↔ cond_video (True for bidirectional)
    assert mask[s_noisy : s_noisy + s_cond, s_noisy : s_noisy + s_cond].all()
    # Check action ↔ action (True)
    assert mask[s_noisy + s_cond :, s_noisy + s_cond :].all()
    # Check action → cond_video (True)
    assert mask[s_noisy + s_cond :, s_noisy : s_noisy + s_cond].all()

    # Check blocked paths
    # noisy_video → cond_video (False)
    assert not mask[:s_noisy, s_noisy : s_noisy + s_cond].any()
    # noisy_video → action (False)
    assert not mask[:s_noisy, s_noisy + s_cond :].any()
    # cond_video → noisy_video (False)
    assert not mask[s_noisy : s_noisy + s_cond, :s_noisy].any()
    # cond_video → action (False)
    assert not mask[s_noisy : s_noisy + s_cond, s_noisy + s_cond :].any()
    # action → noisy_video (False)
    assert not mask[s_noisy + s_cond :, :s_noisy].any()


def test_idm_resolve_from_framework_variant():
    """IDM can be resolved from (framework='dual_system', variant='idm')."""
    from openwam.model.architectures.registry import _FRAMEWORK_VARIANT_INDEX

    assert ("dual_system", "idm") in _FRAMEWORK_VARIANT_INDEX
    assert _FRAMEWORK_VARIANT_INDEX[("dual_system", "idm")] == "dual_system_idm"


class _CapturePrepareVideoBackbone(nn.Module):
    """Tiny video backbone that records per-prepare timesteps for IDM loss tests."""

    dim = 8
    num_layers = 1
    num_heads = 2
    head_dim = 4
    submodule_names = []

    def __init__(self):
        super().__init__()
        from tests.train.test_openwam_trainer import _MockScheduler

        self.scheduler = _MockScheduler()
        self.prepare_timesteps = []
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    @property
    def video_attention_mask_mode(self):
        return "bidirectional"

    @video_attention_mask_mode.setter
    def video_attention_mask_mode(self, mode):
        del mode

    def set_dtype_device(self, dtype, device):
        self._dtype = dtype
        self._device = device

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

    def prepare(self, **kw):
        from openwam.model.video_backbone.base import BlockLoopState

        latents = kw["latents"]
        timestep = kw["timestep"]
        self.prepare_timesteps.append(timestep.detach().clone())
        B, _, f, h, w = latents.shape
        s = f * h * w
        x = latents[:, :1].reshape(B, s, 1).expand(B, s, self.dim).contiguous()
        freqs = torch.polar(torch.ones(s, 1, self.head_dim // 2), torch.zeros(s, 1, self.head_dim // 2))
        t_mod = timestep.view(B, 1, 1, 1).expand(B, s, 6, self.dim).contiguous()
        context = torch.zeros(B, 2, self.dim)
        context_mask = torch.ones(B, 2, dtype=torch.bool)
        return BlockLoopState(
            hidden_states=x,
            time_mod=t_mod,
            rope_freqs=freqs,
            context=context,
            context_mask=context_mask,
            grid_frames=f,
            grid_height=h,
            grid_width=w,
        )

    def merge_idm_video_branches(self, noisy, cond):
        """Drive the *production* Wan merge (flat (B, L, D), 4D time_mod, rope_freqs,
        VACE hints) rather than a copy, so its validation stays covered. ``WanBase``'s
        implementation touches only the two states (never ``self``), so an unbound
        call on this fake is exact."""
        from openwam.model.video_backbone.wan_backbone import WanBase

        return WanBase.merge_idm_video_branches(self, noisy, cond)

    def split_idm_video_branches(self, merged, noisy, cond):
        from openwam.model.video_backbone.wan_backbone import WanBase

        return WanBase.split_idm_video_branches(self, merged, noisy, cond)

    def pre_attn_at_layer(self, layer_id, state):
        del layer_id
        return state.hidden_states, state.hidden_states, state.hidden_states, {"residual": state.hidden_states}

    def post_attn_at_layer(self, layer_id, state, attn_out, post_state):
        del layer_id
        state.hidden_states = post_state["residual"] + attn_out
        return state

    def run_block(self, block_id, state):
        del block_id
        state.hidden_states = state.hidden_states + 1
        return state

    def finalize(self, state):
        B = state.hidden_states.shape[0]
        return (
            state.hidden_states[:, :, :1]
            .transpose(1, 2)
            .reshape(B, 1, state.grid_frames, state.grid_height, state.grid_width)
        )

    def decode_video(self, latents, *, tiled=True):
        del latents, tiled
        return None


def _make_idm_with_video(vb, *, action_dim=3, text_dim=8):
    from openwam.model.architectures.dual_system.idm import DualSystemIDMArchitecture

    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": action_dim,
        "dim": 8,
        "ffn_dim": 16,
        "num_heads": 2,
        "attn_head_dim": 4,
        "video_dim": 8,
        "text_dim": text_dim,
        "bridge_layers": (0,),
        "idm_video_cond_noise_prob": 0.0,
        "mot_checkpoint_mixed_attn": False,
    }
    arch = DualSystemIDMArchitecture(cfg)
    arch.video_backbone = vb
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch.build_mot_driver()
    return arch


def test_idm_clean_cond_video_uses_zero_timestep():
    """Clean teacher-forcing cond video must use literal t=0, not scheduler index 0."""
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.init_training_schedulers(10)

    result = arch.compute_loss(
        input_latents=torch.randn(2, 1, 1, 1, 1),
        context=torch.randn(2, 2, arch.action_backbone.text_dim),
        context_mask=torch.ones(2, 2, dtype=torch.bool),
        actions=torch.randn(2, 3, 3),
        lambda_video=1.0,
        lambda_action=1.0,
    )

    assert torch.isfinite(result["loss"])
    assert len(vb.prepare_timesteps) >= 2
    cond_timesteps = vb.prepare_timesteps[1]
    assert torch.equal(cond_timesteps, torch.zeros_like(cond_timesteps))


def test_idm_training_requires_tokenwise_video_t_mod():
    """IDM should fail loudly if a backbone cannot represent noisy/cond timesteps in one sequence.

    The check now lives in the backbone's ``merge_idm_video_branches`` (the
    driver delegates branch concatenation to the backbone), so a non-4D
    ``time_mod`` must raise there and propagate through the driver loop.
    """
    from openwam.model.video_backbone.base import BlockLoopState

    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    driver = arch._mot_driver

    # 3D time_mod (not the token-wise 4D shape IDM requires).
    vstate = BlockLoopState(
        hidden_states=torch.zeros(1, 2, vb.dim),
        time_mod=torch.zeros(1, 6, vb.dim),
        rope_freqs=torch.zeros(2, 1, 2),
        context=torch.zeros(1, 1, vb.dim),
        grid_frames=2,
        grid_height=1,
        grid_width=1,
    )
    astate = MagicMock()
    astate.payload.x_action = torch.zeros(1, 1, vb.dim)

    import pytest

    with pytest.raises(ValueError, match="token-wise video t_mod"):
        driver.run_idm_training_loop(vstate, vstate, astate)


class _GenerateVideoBackbone(_CapturePrepareVideoBackbone):
    def __init__(self):
        super().__init__()
        self.prepare_count = 0
        self.prepare_context_lengths = []

    def preprocess_input_for_inference(self, *args, **kwargs):
        del args, kwargs
        return {
            "latents": torch.ones(1, 1, 1, 1, 1),
            "context": torch.randn(1, 2, self.dim),
            "context_mask": torch.ones(1, 2, dtype=torch.bool),
        }

    def prepare(self, **kw):
        self.prepare_count += 1
        self.prepare_context_lengths.append(int(kw["context"].shape[1]))
        return super().prepare(**kw)

    def finalize(self, state):
        B = state.hidden_states.shape[0]
        return torch.zeros(B, 1, state.grid_frames, state.grid_height, state.grid_width)


def test_idm_generate_honors_action_num_frames_and_prefills_video_once(monkeypatch):
    """OpenWAM deploy passes raw action_num_frames separately from video_num_frames."""
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()

    result = arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    assert result["video"] is None
    assert result["actions"].shape == (4, 3)
    # One prepare in stage 1 and one video-cache prefill prepare before action denoising.
    assert vb.prepare_count == 2


def test_idm_generate_runs_attached_normalizer_on_actions(monkeypatch):
    """IDM's own generate() must unnormalize the output via the attached normalizer.

    Regression: the action_normalizer -> normalizer rename left idm.py reading a
    stale getattr key, silently skipping unnormalization for IDM deploys.
    """
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    class _AddConstNormalizer:
        def unnormalize(self, x):
            return x + 100.0

    torch.manual_seed(0)
    arch = _make_idm_with_video(_GenerateVideoBackbone())
    arch.eval()
    gen_kwargs = dict(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    # Same global + per-call seed → identical denoised actions; only the normalizer differs.
    torch.manual_seed(42)
    raw = arch.generate(**gen_kwargs)["actions"]
    arch.normalizer = _AddConstNormalizer()
    torch.manual_seed(42)
    normalized = arch.generate(**gen_kwargs)["actions"]

    torch.testing.assert_close(torch.from_numpy(normalized), torch.from_numpy(raw) + 100.0)


def test_idm_generate_pins_inactive_unified_action_dims(monkeypatch):
    """IDM's stage-2 loop must keep unsupervised unified dims on the analytic
    sigma * eps0 path, exactly like BaseWAMArchitecture.generate."""
    import numpy as np

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    gen_kwargs = dict(
        schedule=[(1000.0, 1000.0), (0.0, 500.0), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    arch = _make_idm_with_video(_GenerateVideoBackbone())
    arch.eval()
    torch.manual_seed(42)
    baseline = arch.generate(**gen_kwargs)["actions"]
    # The randomly initialized ActionDiT emits nonzero flow on every channel,
    # so unpinned inactive dims do NOT land on 0.
    assert np.abs(baseline[:, 1:]).max() > 0.0

    torch.manual_seed(42)
    pinned = arch.generate(**gen_kwargs, active_action_mask=torch.tensor([True, False, False]))["actions"]
    # sigma_end == 0 -> pinned inactive dims land exactly on 0; the active dim
    # still integrates real flow.
    assert np.abs(pinned[:, 1:]).max() == 0.0
    assert np.abs(pinned[:, 0]).max() > 0.0


def test_idm_generate_with_proprio_appends_context_once(monkeypatch):
    """IDM generate appends proprio once even though stage 2 bypasses forward()."""
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch._use_proprioception_context = True
    arch.proprio_dim = 3
    arch.context_dim = arch.action_backbone.text_dim
    arch.proprio_encoder = nn.Linear(3, arch.context_dim)
    arch.eval()

    result = arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
        proprio=torch.randn(1, 3),
    )

    assert result["video"] is None
    assert result["actions"].shape == (4, 3)
    assert vb.prepare_count == 2
    assert vb.prepare_timesteps[0].shape == torch.Size([1])
    assert vb.prepare_context_lengths == [3, 3]


class _ReuseOnceVideoCache:
    """Tiny cache mock that recomputes twice, then reuses the previous velocity."""

    def __init__(self):
        self.should_calls = []
        self.update_calls = []
        self.get_cached_calls = 0
        self._cached = None

    def should_recompute(self, sigma):
        self.should_calls.append(float(sigma))
        return len(self.should_calls) < 3

    def update(self, velocity, sigma):
        self.update_calls.append(float(sigma))
        self._cached = velocity.detach()

    def get_cached(self):
        self.get_cached_calls += 1
        return self._cached


def test_idm_video_cache_matches_joint_loop():
    """The cached Stage-2 path must equal a full joint loop on the action output.

    Stage 2 reuses ``prefill_video_cache`` + ``run_action_with_video_cache``
    instead of running joint attention every action step. That's only valid
    when the joint mask blocks ``v→a`` — see
    ``DualSystemMoTDriver._build_attention_mask``. If that invariant ever regresses
    (or the cache layout drifts from what the joint K/V would be), the
    cached action output will diverge from the joint action output. This
    test pins the equivalence numerically.
    """
    import copy as _copy

    torch.manual_seed(0)
    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    B = 1
    latents = torch.randn(B, 1, 1, 1, 1)
    vstate_joint = vb.prepare(latents=latents, timestep=torch.zeros(B))
    vstate_cache = _copy.copy(vstate_joint)
    vstate_cache.hidden_states = vstate_joint.hidden_states.clone()

    action_latents = torch.randn(B, 3, arch.action_backbone.action_dim)
    a_timestep = torch.tensor([0.5])
    context = torch.randn(B, 2, arch.action_backbone.text_dim)
    context_mask = torch.ones(B, 2, dtype=torch.bool)

    astate_joint = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )
    astate_cache = arch.action_backbone.prepare_state(
        action_latents, a_timestep, context=context, context_mask=context_mask
    )

    _, astate_joint = driver.run_joint_loop(vstate_joint, astate_joint)
    pred_joint = arch.action_backbone.extract_prediction(astate_joint)

    video_kv_cache, _, _ = driver.prefill_video_cache(vstate_cache)
    astate_cache = driver.run_action_with_video_cache(
        astate_cache,
        video_kv_cache=video_kv_cache,
        video_seq_len=int(vstate_cache.hidden_states.shape[1]),
    )
    pred_cache = arch.action_backbone.extract_prediction(astate_cache)

    assert torch.allclose(pred_joint, pred_cache, atol=1e-5, rtol=1e-5)


def test_idm_generate_forwards_the_prefix_gate_to_stage_two(monkeypatch):
    """generate() must hand Stage 2 the gate ``prefill_video_cache`` returned.

    The driver-level tests cover ``build_cached_action_mask`` and
    ``run_action_with_video_cache``; this covers the caller that connects them.
    Setting either forwarding site in ``generate()`` to ``None`` restores the
    original und-padding leak with every other test green, because nothing else
    asserts the value ever leaves ``prefill_video_cache``.
    """
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    sentinel = torch.tensor([[True, True, False]])
    seen = {}
    real_prefill = driver.prefill_video_cache
    real_stage2 = driver.run_action_with_video_cache

    def spy_prefill(vstate):
        cache, key_len, _gate = real_prefill(vstate)
        seen["produced"] = sentinel  # stand in for a padded batch's gate
        return cache, key_len, sentinel

    def spy_stage2(astate, *, video_kv_cache, video_seq_len, prefix_kv_mask=None):
        seen["forwarded"] = prefix_kv_mask
        # The real Stage 2 would size its mask from the gate; the fake backbone
        # has no prefix, so just record and run without it.
        return real_stage2(astate, video_kv_cache=video_kv_cache, video_seq_len=video_seq_len)

    monkeypatch.setattr(driver, "prefill_video_cache", spy_prefill)
    monkeypatch.setattr(driver, "run_action_with_video_cache", spy_stage2)

    arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    assert "forwarded" in seen, "Stage 2 was never reached"
    assert seen["forwarded"] is sentinel, (
        "generate() dropped prefill_video_cache's prefix gate instead of forwarding it to Stage 2"
    )


def test_idm_generate_forwards_the_prefix_gate_on_the_compiled_path(monkeypatch):
    """Same contract for the compiled Stage-2 branch, which builds its own mask.

    The eager branch delegates to ``run_action_with_video_cache``; the compiled
    branch calls ``build_cached_action_mask`` directly, so it is a second,
    independently-mutable forwarding site.
    """
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    vb = _GenerateVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.eval()
    driver = arch._mot_driver

    sentinel = torch.tensor([[True, True, False]])
    calls = []
    real_prefill = driver.prefill_video_cache
    real_build = driver.build_cached_action_mask

    def spy_prefill(vstate):
        cache, key_len, _gate = real_prefill(vstate)
        return cache, key_len, sentinel

    def spy_build(*, s_action, video_key_len, device, prefix_kv_mask=None):
        # Record every call: the forced fallback re-enters via the eager branch,
        # which would otherwise overwrite the compiled branch's argument.
        calls.append(prefix_kv_mask)
        return real_build(s_action=s_action, video_key_len=video_key_len, device=device, prefix_kv_mask=None)

    def _boom(*_a, **_k):
        raise RuntimeError("force the documented eager fallback after the mask is built")

    monkeypatch.setattr(driver, "prefill_video_cache", spy_prefill)
    monkeypatch.setattr(driver, "build_cached_action_mask", spy_build)
    arch._compiled_idm_action_cache_loop = _boom

    arch.generate(
        schedule=[(1.0, 1.0), (0.0, 0.5), (0.0, 0.0)],
        prompt="",
        num_frames=3,
        action_num_frames=5,
        decode_video=False,
        seed=0,
    )

    assert calls, "the compiled Stage-2 branch was never entered"
    assert calls[0] is sentinel, "the compiled path built its action mask without prefill_video_cache's prefix gate"


# ==== from test_cosmos3_idm.py ====
# IDM teacher-forcing support for cosmos3_edge — CPU tests.
#
# Covers the backbone's branch merge/split contract and the prefix-KV mask
# widening the IDM driver needs, since Cosmos3's per-layer keys carry the cached
# und (text) stream that has no matching query rows.

pytest.importorskip("diffusers")

from openwam.model.architectures.utils.mask_modes import widen_mask_for_prefix_kv  # noqa: E402
from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone  # noqa: E402
from openwam.model.video_backbone.cosmos3 import dit_forward, idm_merge, text_pack  # noqa: E402
from openwam.model.video_backbone.cosmos3._vendor.transformer_cosmos3 import (  # noqa: E402
    Cosmos3OmniTransformer,
)

MINI = dict(
    attention_bias=False,
    head_dim=6,
    hidden_size=12,
    intermediate_size=24,
    latent_channel=2,
    latent_patch_size=1,
    num_attention_heads=2,
    num_hidden_layers=2,
    num_key_value_heads=1,
    patch_latent_dim=2,
    qk_norm_for_text=False,
    use_und_k_norm_for_gen=True,
    hidden_act="relu2",
    rms_norm_eps=1e-5,
    rope_axes_dim=[1, 1, 1],
    rope_theta=1e8,
    vocab_size=32,
)


def _net():
    torch.manual_seed(0)
    return Cosmos3OmniTransformer(**MINI).eval()


def _state(net, ids, lat, ncp=0):
    text_pos = text_pack.text_mrope_positions(ids.numel(), float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        ids.numel(), grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    und_mask = torch.ones(lat.shape[0], ids.numel(), dtype=torch.bool)
    ctx, und_kv = dit_forward.run_und_tower(net, ids.unsqueeze(0).expand(lat.shape[0], -1), und_mask, cos_u, sin_u)
    return dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((lat.shape[0],), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=ncp,
    )


def test_merge_concats_tokens_and_rotary():
    net = _net()
    ids = torch.tensor([1, 2, 3, 4])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))

    s_n_before, s_c_before = noisy.hidden_states.shape[1], cond.hidden_states.shape[1]
    merged, s_noisy, s_cond = noisy_merged = idm_merge.merge_branches(noisy, cond)
    assert (s_noisy, s_cond) == (s_n_before, s_c_before)
    assert merged.hidden_states.shape[1] == s_noisy + s_cond
    assert merged.extras["cos_gen"].shape[1] == s_noisy + s_cond
    assert merged.extras["sin_gen"].shape[1] == s_noisy + s_cond
    assert merged.grid_frames == noisy.grid_frames + cond.grid_frames
    # und cache is shared (same prompt) and the prefix declaration survives.
    assert merged.extras["und_kv"] is noisy.extras["und_kv"]
    assert merged.prefix_kv_len == noisy.prefix_kv_len
    del noisy_merged


def test_split_is_inverse_of_merge():
    net = _net()
    ids = torch.tensor([5, 6, 7])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    hn, hc = noisy.hidden_states.clone(), cond.hidden_states.clone()

    merged, _, _ = idm_merge.merge_branches(noisy, cond)
    noisy2, cond2 = idm_merge.split_branches(merged, noisy, cond)
    assert torch.equal(noisy2.hidden_states, hn)
    assert torch.equal(cond2.hidden_states, hc)


def test_merge_rejects_mismatched_spatial_layout():
    net = _net()
    ids = torch.tensor([1, 2])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 2, 4, 2))
    with pytest.raises(ValueError, match="spatial token layout"):
        idm_merge.merge_branches(noisy, cond)


def test_merged_state_runs_through_blocks():
    net = _net()
    ids = torch.tensor([1, 2, 3, 4])
    noisy = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    cond = _state(net, ids, torch.randn(1, MINI["latent_channel"], 3, 2, 2))
    with torch.no_grad():
        merged, s_noisy, s_cond = idm_merge.merge_branches(noisy, cond)
        for i in range(len(net.layers)):
            merged = dit_forward.run_block(net, i, merged)
        noisy2, cond2 = idm_merge.split_branches(merged, noisy, cond)
    assert noisy2.hidden_states.shape[1] == s_noisy
    assert cond2.hidden_states.shape[1] == s_cond
    assert torch.isfinite(noisy2.hidden_states).all() and torch.isfinite(cond2.hidden_states).all()


def test_prefix_widening_shapes_and_semantics():
    # Square mask + a declared prefix -> rectangular mask with visible prefix cols.
    st = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=None)
    m = torch.zeros((5, 5), dtype=torch.bool)
    w = widen_mask_for_prefix_kv(m, st)
    assert w.shape == (5, 8)
    assert w[:, :3].all() and not w[:, 3:].any()

    # Per-sample padding gate broadcasts to (B, 1, S, prefix + S).
    pm = torch.tensor([[True, True, False]])
    st2 = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=pm)
    w2 = widen_mask_for_prefix_kv(torch.ones((5, 5), dtype=torch.bool), st2)
    assert w2.shape == (1, 1, 5, 8)
    assert w2[0, 0, :, 2].sum() == 0  # padded und column masked for every query row

    # No prefix declared -> untouched (byte-identical for Wan / predict2.5).
    st3 = types.SimpleNamespace(prefix_kv_len=0, prefix_kv_mask=None)
    m3 = torch.ones((4, 4), dtype=torch.bool)
    assert widen_mask_for_prefix_kv(m3, st3) is m3


def test_backbone_overrides_idm_stubs():
    """``callable(...)`` is not a test: ``VideoBackbone`` defines both hooks as
    raising stubs, so it is true for every subclass. Assert the override."""
    for name in ("merge_idm_video_branches", "split_idm_video_branches"):
        base_fn = getattr(VideoBackbone, name)
        assert getattr(Cosmos3EdgeVideoBackbone, name) is not base_fn, f"{name} is still the base stub"
    # And the base really does raise — otherwise the check above proves nothing.
    with pytest.raises(NotImplementedError):
        VideoBackbone.merge_idm_video_branches(Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone), None, None)


def test_prefix_widening_preserves_rank_and_stays_shared():
    """Rank in == rank out, and ``None`` (no padding) stays batch-shared.

    The no-padding signal is ``prefix_kv_mask is None``, decided host-side at
    pack time — deliberately NOT an ``.all()`` test on the tensor, which would
    cost a device sync and a dynamo graph break inside the compiled MoT region.
    """
    # 4D mask with no per-sample gate used to crash (2D pad cat'd onto 4D).
    st = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=None)
    m4 = torch.ones((2, 1, 5, 5), dtype=torch.bool)
    w4 = widen_mask_for_prefix_kv(m4, st)
    assert w4.shape == (2, 1, 5, 8)

    # None keeps the cheap batch-shared 2D mask.
    w2 = widen_mask_for_prefix_kv(torch.ones((5, 5), dtype=torch.bool), st)
    assert w2.shape == (5, 8) and w2.all()

    # A tensor gate is honored per sample even when it happens to be all-True:
    # the helper never inspects its values, so callers must pass None to opt
    # into the shared path.
    st_tensor = types.SimpleNamespace(prefix_kv_len=3, prefix_kv_mask=torch.ones(4, 3, dtype=torch.bool))
    w_t = widen_mask_for_prefix_kv(torch.ones((5, 5), dtype=torch.bool), st_tensor)
    assert w_t.shape == (4, 1, 5, 8) and w_t.all()


def _b2_state_with_padding(net, pad_len=2):
    """A B=2 cosmos3 video state whose second prompt is right-padded.

    Sample 0 uses all 5 und tokens; sample 1 uses 3 and pads the tail, which is
    the only configuration in which the Stage-2 prefix gate does any work.
    """
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    und_mask = torch.tensor([[True] * 5, [True] * (5 - pad_len) + [False] * pad_len])
    lat = torch.randn(2, MINI["latent_channel"], 2, 2, 2)
    text_pos = text_pack.text_mrope_positions(ids.shape[1], float_positions=True)
    grid = text_pack.patch_grid(*lat.shape[2:], int(net.config.latent_patch_size))
    _, vis_pos = text_pack.build_joint_positions(
        ids.shape[1], grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    cos_u, sin_u = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), lat.device, lat.dtype)
    ctx, und_kv = dit_forward.run_und_tower(net, ids, und_mask, cos_u, sin_u)
    state = dit_forward.prepare_block_loop(
        net,
        latents=lat,
        timestep=torch.full((2,), 500.0),
        context=ctx,
        und_mask=und_mask,
        und_kv=und_kv,
        vision_positions=vis_pos.unsqueeze(1),
        num_clean_prefix_frames=0,
    )
    return state, und_mask


def _idm_driver(net):
    vb = Cosmos3EdgeVideoBackbone(
        net=net,
        vae=None,
        tokenizer=None,
        dim=MINI["hidden_size"],
        num_layers=MINI["num_hidden_layers"],
        num_heads=MINI["num_attention_heads"],
        head_dim=MINI["head_dim"],
        context_dim=MINI["hidden_size"],
    ).eval()
    ab = _TinyActionBackbone(
        MINI["hidden_size"], MINI["num_hidden_layers"], MINI["num_attention_heads"], MINI["head_dim"]
    ).eval()
    return IDMMoTDriver(vb, ab, attention_mask_mode="action_sees_video")


def test_idm_stage2_wiring_keeps_padded_und_keys_out():
    """End-to-end: prefill_video_cache -> run_action_with_video_cache on a padded batch.

    Drives the actual plumbing rather than ``build_cached_action_mask`` in
    isolation, by poisoning the cached K/V at the padded und columns: if the
    gate is dropped anywhere along the way (prefill not returning it, Stage 2
    ignoring it), the poison reaches the action stream and the output moves.
    """
    torch.manual_seed(0)
    net = _net()
    driver = _idm_driver(net)
    pad_len = 2

    def stage2(poison: bool, gate_override="keep"):
        torch.manual_seed(7)  # identical latents across calls; only the poison varies
        state, und_mask = _b2_state_with_padding(net, pad_len=pad_len)
        with torch.no_grad():
            kv_cache, key_len, gate = driver.prefill_video_cache(state)
            assert gate is not None, "prefill must hand Stage 2 the prefix gate for a padded batch"
            assert torch.equal(gate, und_mask)
            if poison:
                # Sample 1's padded und slots only; every other key is untouched.
                for layer in kv_cache:
                    for t in (layer["k"], layer["v"]):
                        t[1, und_mask.shape[1] - pad_len : und_mask.shape[1]] = 1e4
            astate = types.SimpleNamespace(
                payload=types.SimpleNamespace(x_action=torch.zeros(2, 3, MINI["hidden_size"]))
            )
            astate.payload.x_action.copy_(torch.arange(2 * 3 * MINI["hidden_size"]).float().view(2, 3, -1) * 1e-2)
            out = driver.run_action_with_video_cache(
                astate,
                video_kv_cache=kv_cache,
                video_seq_len=key_len,
                prefix_kv_mask=gate if gate_override == "keep" else gate_override,
            )
        return out.payload.x_action

    clean = stage2(poison=False)
    poisoned = stage2(poison=True)
    assert torch.allclose(clean, poisoned, atol=1e-6), (
        f"padded und keys leaked into the action stream: max diff {(clean - poisoned).abs().max().item():.3e}"
    )

    # Positive control: without the gate the same poison must move the output,
    # otherwise the assertion above would pass vacuously.
    leaked = stage2(poison=True, gate_override=None)
    assert (clean - leaked).abs().max().item() > 1e-3, "poison is inert; the test cannot detect a leak"


def _stage2_action_mask(driver, s_action, key_len, prefix_mask):
    return driver.build_cached_action_mask(
        s_action=s_action, video_key_len=key_len, device=torch.device("cpu"), prefix_kv_mask=prefix_mask
    )


def test_stage2_action_mask_gates_padded_und_columns():
    """Regression: the cached Stage-2 mask must close padded und columns.

    An all-ones mask over the cached key length would open every und slot,
    including right-padding that the joint loop masks out — a silent wrong
    answer the moment IDM inference is batched with unequal prompt lengths.
    """
    driver = IDMMoTDriver.__new__(IDMMoTDriver)
    prefix, s_video, s_action = 4, 10, 3
    prefix_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    mask = _stage2_action_mask(driver, s_action=s_action, key_len=prefix + s_video, prefix_mask=prefix_mask)

    # Key axis is [prefix | video | action]; queries are the action rows only.
    assert mask.shape == (2, 1, s_action, prefix + s_video + s_action)
    # Sample 0: all four und columns open. Sample 1: last two are padding.
    assert mask[0, 0, :, :prefix].all()
    assert mask[1, 0, :, :2].all() and not mask[1, 0, :, 2:prefix].any()
    # Video + action columns stay fully visible for both samples.
    assert mask[:, :, :, prefix:].all()


def test_stage2_action_mask_matches_joint_loop_action_rows():
    """The cached mask must equal the action-query row slice of the joint mask."""
    driver = IDMMoTDriver.__new__(IDMMoTDriver)
    prefix, s_video, s_action = 4, 10, 3
    prefix_mask = torch.tensor([[True, True, False, False]])

    cached = _stage2_action_mask(driver, s_action, prefix + s_video, prefix_mask)
    # Joint loop: square [video, action] mask (action rows all-True under
    # action_sees_video), then widened by the same prefix gate.
    joint = torch.ones((s_video + s_action, s_video + s_action), dtype=torch.bool)
    joint = widen_mask_for_prefix_kv(joint, types.SimpleNamespace(prefix_kv_len=prefix, prefix_kv_mask=prefix_mask))
    assert torch.equal(cached, joint[:, :, s_video:, :])


def test_stage2_action_mask_prefix_free_is_all_ones():
    """Wan / predict2.5 (no prefix) keep the byte-identical 2D all-ones mask."""
    driver = IDMMoTDriver.__new__(IDMMoTDriver)
    mask = _stage2_action_mask(driver, s_action=3, key_len=10, prefix_mask=None)
    assert mask.shape == (3, 13) and mask.all()


# ==== from test_cosmos_predict25_idm.py ====
# IDM (dual_system/idm) support for the CosmosPredict25 video backbone.
#
# IDM teacher-forcing merges a noisy + cond video branch into one MoT pass. On
# Wan the branches are flat ``(B, L, D)`` sequences; on Cosmos they are 5D grids
# ``(B, T, H, W, D)`` with per-frame modulation in ``state.extras``. These CPU
# tests use the ``_RichCosmosBlock`` fakes (shared with the joint_self_attn
# suite) to validate:
#
# * ``force_per_token_t_mod`` yields a per-frame ``t_embedding_B_T_D``;
# * the backbone-owned branch merge/split round-trips and reports token counts
#   (``T·H·W``), not ``hidden_states.shape[1]`` (``T``);
# * the 3-branch ``run_idm_training_loop`` runs end-to-end on the 5D state;
# * the teacher-forcing mask is token-granular (action↔cond, not action↔noisy);
# * train (3-branch) and deploy (Stage-2 frozen-video KV cache) action
#   predictions match — the FastWAM-IDM core invariant, on Cosmos.


def _make_cosmos_idm(num_blocks=2, *, action_dim=3, text_dim=12):
    """Build a DualSystemIDMArchitecture on a CosmosPredict25 rich-fake backbone."""
    backbone = _build_rich_wrapper(num_blocks=num_blocks)
    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": action_dim,
        "dim": 16,
        "ffn_dim": 32,
        "num_heads": 4,
        "attn_head_dim": 4,
        "video_dim": 16,
        "text_dim": text_dim,
        "bridge_layers": tuple(range(num_blocks)),
        "idm_video_cond_noise_prob": 0.0,
        "mot_checkpoint_mixed_attn": False,
    }
    arch = DualSystemIDMArchitecture(cfg)
    arch.video_backbone = backbone
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch.build_mot_driver()
    return arch, backbone


def _prep(backbone, *, timestep, force_per_token_t_mod, seed=0):
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 16, 2, 4, 4, generator=g)
    context = torch.randn(1, 4, 12, generator=g)
    return backbone.prepare(
        input_latents=latents,
        context=context,
        timestep=timestep,
        force_per_token_t_mod=force_per_token_t_mod,
    )


# ----------------------------------------------------------------------
# force_per_token_t_mod → per-frame t_embedding
# ----------------------------------------------------------------------


def test_force_per_token_t_mod_yields_per_frame_t_embedding():
    _, backbone = _make_cosmos_idm()
    # latents (1,16,2,4,4) → grid frames f=2 (temporal patch 1).
    state_pf = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True)
    assert state_pf.grid_frames == 2
    assert state_pf.extras["t_embedding_B_T_D"].shape[1] == 2  # per-frame
    assert state_pf.extras["adaln_lora_B_T_3D"].shape[1] == 2

    state_default = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=False)
    assert state_default.extras["t_embedding_B_T_D"].shape[1] == 1  # broadcast


# ----------------------------------------------------------------------
# merge / split round-trip + token-count contract
# ----------------------------------------------------------------------


def test_merge_branches_reports_token_counts_and_concatenates_extras():
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)

    H, W = noisy.grid_height, noisy.grid_width
    merged, s_noisy, s_cond = backbone.merge_idm_video_branches(noisy, cond)

    # token counts are T·H·W, NOT hidden_states.shape[1] (== T for the 5D grid)
    assert s_noisy == noisy.grid_frames * H * W
    assert s_cond == cond.grid_frames * H * W
    assert s_noisy != noisy.hidden_states.shape[1]  # guard: shape[1] is frames

    # hidden state + frame extras concat along the frame axis
    assert merged.hidden_states.shape[1] == noisy.grid_frames + cond.grid_frames
    assert merged.grid_frames == noisy.grid_frames + cond.grid_frames
    assert merged.extras["t_embedding_B_T_D"].shape[1] == noisy.grid_frames + cond.grid_frames
    # per-token RoPE concat along the token axis (dim=0)
    assert merged.extras["rope_emb_L_1_1_D"].shape[0] == s_noisy + s_cond


def test_merge_split_round_trips():
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)
    noisy_ref = noisy.hidden_states.clone()
    cond_ref = cond.hidden_states.clone()
    # Per-frame modulation extras must survive the merge/split untouched (they feed
    # finalize per branch); split_branches deliberately does NOT write them back.
    t_emb_noisy_ref = noisy.extras["t_embedding_B_T_D"].clone()
    t_emb_cond_ref = cond.extras["t_embedding_B_T_D"].clone()

    merged, _, _ = backbone.merge_idm_video_branches(noisy, cond)
    # No block loop runs the merged state changes here; splitting must recover both.
    noisy_out, cond_out = backbone.split_idm_video_branches(merged, noisy, cond)
    assert torch.allclose(noisy_out.hidden_states, noisy_ref)
    assert torch.allclose(cond_out.hidden_states, cond_ref)
    assert torch.allclose(noisy_out.extras["t_embedding_B_T_D"], t_emb_noisy_ref)
    assert torch.allclose(cond_out.extras["t_embedding_B_T_D"], t_emb_cond_ref)


def test_merge_rejects_mismatched_spatial_layout__predict25():
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True)
    cond = backbone.prepare(
        input_latents=torch.randn(1, 16, 2, 4, 6),  # different W
        context=torch.randn(1, 4, 12),
        timestep=torch.zeros(1),
        force_per_token_t_mod=True,
    )
    with pytest.raises(ValueError, match="spatial token layout"):
        backbone.merge_idm_video_branches(noisy, cond)


def test_merge_rejects_broadcast_t_mod():
    # The module docstring requires both branches prepared with
    # force_per_token_t_mod=True; a broadcast (B, 1, D) t_embedding must be
    # rejected up front rather than silently producing a (B, 2, D) emb for the
    # 2T-frame grid (mirrors the Wan-side ndim==4 guard).
    _, backbone = _make_cosmos_idm()
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=False, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)
    assert noisy.grid_frames == 2  # but t_embedding_B_T_D is broadcast (B, 1, D)
    with pytest.raises(ValueError, match="per-frame modulation"):
        backbone.merge_idm_video_branches(noisy, cond)


# ----------------------------------------------------------------------
# teacher-forcing mask is token-granular
# ----------------------------------------------------------------------


def test_teacher_forcing_mask_token_granularity():
    arch, backbone = _make_cosmos_idm()
    driver = arch._mot_driver
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)

    _, s_noisy, s_cond = backbone.merge_idm_video_branches(noisy, cond)
    s_action = 3
    tpf = driver._video_tokens_per_frame(noisy)
    mask = driver._build_teacher_forcing_mask(
        s_noisy_video=s_noisy,
        s_cond_video=s_cond,
        s_action=s_action,
        video_tokens_per_frame=tpf,
        device=torch.device("cpu"),
    )
    total = s_noisy + s_cond + s_action
    assert mask.shape == (total, total)
    # action (tail rows) attends to cond block but NOT the noisy block
    a0 = s_noisy + s_cond
    assert mask[a0:, s_noisy:a0].all()  # action → cond
    assert not mask[a0:, :s_noisy].any()  # action → noisy blocked
    # noisy and cond never attend across to each other
    assert not mask[:s_noisy, s_noisy:a0].any()


# ----------------------------------------------------------------------
# 3-branch training loop runs on the 5D state
# ----------------------------------------------------------------------


def test_idm_training_loop_runs_and_preserves_5d_state():
    arch, backbone = _make_cosmos_idm()
    arch.eval()
    driver = arch._mot_driver

    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=1)
    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=2)
    actions = torch.randn(1, 3, arch.action_backbone.action_dim)
    astate = arch.action_backbone.prepare_state(
        actions, torch.tensor([0.5]), context=torch.randn(1, 4, 12), context_mask=torch.ones(1, 4, dtype=torch.bool)
    )

    vstate_noisy, vstate_cond, astate = driver.run_idm_training_loop(noisy, cond, astate)
    pred = arch.action_backbone.extract_prediction(astate)

    assert vstate_noisy.hidden_states.shape == (1, 2, 2, 2, 16)  # 5D preserved
    assert torch.isfinite(vstate_noisy.hidden_states).all()
    assert torch.isfinite(pred).all()


# ----------------------------------------------------------------------
# train (3-branch) ↔ deploy (Stage-2 frozen-video KV cache) consistency
# ----------------------------------------------------------------------


def test_idm_train_action_matches_deploy_stage2_cosmos():
    """3-branch train action_pred must equal Stage-2 deploy action_pred on Cosmos.

    Train cond branch uses per-frame t=0 (force_per_token_t_mod); deploy prefill
    uses the scalar t=0 broadcast path. Both yield identical per-frame modulation
    (t=0 is uniform across frames), so the cached video K/V — and the action
    prediction — must match.
    """
    arch, backbone = _make_cosmos_idm()
    arch.eval()
    driver = arch._mot_driver

    cond = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=True, seed=5)
    noisy = _prep(backbone, timestep=torch.tensor([0.7]), force_per_token_t_mod=True, seed=6)
    cond_deploy = _prep(backbone, timestep=torch.zeros(1), force_per_token_t_mod=False, seed=5)

    actions = torch.randn(1, 3, arch.action_backbone.action_dim)
    a_timestep = torch.tensor([0.5])
    context = torch.randn(1, 4, 12)
    context_mask = torch.ones(1, 4, dtype=torch.bool)

    astate_train = arch.action_backbone.prepare_state(actions, a_timestep, context=context, context_mask=context_mask)
    astate_deploy = arch.action_backbone.prepare_state(actions, a_timestep, context=context, context_mask=context_mask)

    _, _, astate_train = driver.run_idm_training_loop(noisy, cond, astate_train)
    pred_train = arch.action_backbone.extract_prediction(astate_train)

    video_seq_len = int(cond_deploy.grid_frames) * driver._video_tokens_per_frame(cond_deploy)
    kv_cache, _, _ = driver.prefill_video_cache(cond_deploy)
    astate_deploy = driver.run_action_with_video_cache(
        astate_deploy, video_kv_cache=kv_cache, video_seq_len=video_seq_len
    )
    pred_deploy = arch.action_backbone.extract_prediction(astate_deploy)

    assert torch.allclose(pred_train, pred_deploy, atol=1e-5, rtol=1e-5), (
        "Cosmos IDM train↔deploy action_pred diverged.\n"
        f"  train: {pred_train.flatten()[:8]}\n"
        f"  deploy: {pred_deploy.flatten()[:8]}"
    )

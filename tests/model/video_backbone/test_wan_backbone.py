"""Unit tests for vace_cache / prompt_embed_cache in
WanBase.preprocess_input_for_inference.

The deploy method builds every conditioning signal with explicit helpers (no
WanVideoPipeline unit-runner). The only cached quantity is the text embedding
``(context, seq_lens)``, produced by ``_encode_text`` and reused via either
cache. These tests spy on ``_encode_text`` to verify:

1. Cold start: both caches miss → text is encoded.
2. Same prompt (vace_cache hit): text encode skipped.
3. New prompt (vace_cache key mismatch): text re-encoded.
4. Seen prompt via prompt_embed_cache (vace cleared): text encode skipped.
5. Unseen prompt: both miss → text encoded.
6. prompt_A → prompt_B → prompt_A: prompt_A embed still cached.
7. vace_cache records prompt_key + populated.
8. No caches: runs cleanly every call.

Plus the I2V deploy first-frame unwrap path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from openwam.model.video_backbone.wan_backbone import WanBase

# ---------------------------------------------------------------------------
# Minimal mock infrastructure
# ---------------------------------------------------------------------------


class _MockScheduler:
    def set_timesteps(self, num_inference_steps, shift):
        pass


class _MockPipe:
    def __init__(self):
        self.scheduler = _MockScheduler()


class _MockTokenizer:
    def __call__(self, prompts, **kw):
        n = len(prompts)
        return torch.zeros(n, 4, dtype=torch.long), torch.ones(n, 4, dtype=torch.long)


class _MockTextEncoder:
    """Records each call so a test can count how often text was encoded."""

    def __init__(self, calls):
        self._calls = calls

    def __call__(self, ids, mask):
        self._calls.append(list(range(ids.shape[0])))
        return torch.zeros(ids.shape[0], 4, 8)


class _MockWanVB:
    """WanVideoBackbone-like object exercising the deploy cache seam.

    The encode seams (text / noise / I2V clip+y) are stubbed so the test needs
    no real weights or GPU; ``_encode_text`` records its calls so a test can
    assert whether the text path was hit or served from a cache.
    """

    def __init__(self, pipe=None):
        self._pipe = pipe if pipe is not None else _MockPipe()
        self._is_ti2v = False
        self._has_vace = False
        self._device = "cpu"
        self._dtype = torch.float32
        self._encode_text_calls: list[list] = []
        # Encode + conditioning seams now live in wan.encode / wan.conditioning
        # free functions; provide the state they read explicitly.
        self._tokenizer = _MockTokenizer()
        self.text_encoder = _MockTextEncoder(self._encode_text_calls)
        self._height_division_factor = 16
        self._width_division_factor = 16
        self._time_division_factor = 4
        self._time_division_remainder = 1
        self._latent_spec = SimpleNamespace(
            z_dim=4, spatial_compression=8, temporal_compression=4, causal_temporal=True
        )
        self._dit = SimpleNamespace(has_image_input=False)
        self.video_encoder = None
        # preprocess passes vae= unconditionally, but build_deploy_noise only
        # dereferences it when _latent_spec is None (never here). The I2V tests
        # override this with a real encode stub via _make_i2v_mock.
        self.vae = None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def scheduler(self):
        return self._pipe.scheduler

    @property
    def dit(self):
        # backbone build_deploy_i2v_clip/y read self.dit; mirror self._dit.
        return self._dit

    # --- real method under test (conditioning seams now in wan.conditioning) ---
    def preprocess_input_for_inference(self, **kw):
        return WanBase.preprocess_input_for_inference(self, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prep(vb, prompt, vace_cache=None, prompt_embed_cache=None, seed=0):
    return vb.preprocess_input_for_inference(
        prompt=prompt,
        vace_video=None,
        first_frame_image=None,
        num_frames=17,
        height=32,
        width=32,
        seed=seed,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        vace_cache=vace_cache,
        prompt_embed_cache=prompt_embed_cache,
    )


def _text_calls(vb) -> int:
    return len(vb._encode_text_calls)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cold_start_encodes_text():
    """No cache provided — text is encoded."""
    vb = _MockWanVB()
    _prep(vb, "prompt_A")
    assert _text_calls(vb) == 1


def test_vace_cache_hit_skips_text_encode():
    """After first call, same prompt → vace_cache hit → text encode skipped."""
    vb = _MockWanVB()
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache)
    assert vace_cache.get("populated")
    assert vace_cache.get("prompt_key") == "prompt_A"
    assert _text_calls(vb) == 1

    _prep(vb, "prompt_A", vace_cache=vace_cache)
    assert _text_calls(vb) == 1, "text should be served from vace_cache on a hit"


def test_vace_cache_prompt_change_causes_miss():
    """Prompt change → vace_cache key mismatch → text re-encoded."""
    vb = _MockWanVB()
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache)
    assert vace_cache["prompt_key"] == "prompt_A"

    _prep(vb, "prompt_B", vace_cache=vace_cache)
    assert _text_calls(vb) == 2, "expected text re-encode on prompt change"
    assert vace_cache["prompt_key"] == "prompt_B"


def test_prompt_embed_cache_hit_skips_text_encode():
    """After first call with prompt_A, a second episode (vace_cache cleared)
    still skips text encode via prompt_embed_cache."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache
    assert _text_calls(vb) == 1

    vace_cache.clear()
    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == 1, "text should be served from prompt_embed_cache"


def test_prompt_embed_cache_miss_on_new_prompt():
    """First time seeing prompt_B: both caches miss → text encoded."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}

    _prep(vb, "prompt_A", prompt_embed_cache=prompt_embed_cache)
    assert "prompt_A" in prompt_embed_cache

    _prep(vb, "prompt_B", prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == 2, "expected text encode for an unseen prompt"
    assert "prompt_B" in prompt_embed_cache


def test_prompt_embed_cache_survives_vace_overwrite():
    """After prompt_A → prompt_B → prompt_A: prompt_A embed is still cached."""
    vb = _MockWanVB()
    prompt_embed_cache: dict = {}
    vace_cache: dict = {}

    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    _prep(vb, "prompt_B", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert vace_cache["prompt_key"] == "prompt_B"
    calls_before = _text_calls(vb)

    _prep(vb, "prompt_A", vace_cache=vace_cache, prompt_embed_cache=prompt_embed_cache)
    assert _text_calls(vb) == calls_before, "prompt_A embed should still be cached"
    assert vace_cache["prompt_key"] == "prompt_A"


def test_vace_cache_stores_prompt_key():
    """vace_cache must record prompt_key + the cached embedding."""
    vb = _MockWanVB()
    vace_cache: dict = {}

    _prep(vb, "unique_prompt_xyz", vace_cache=vace_cache)
    assert vace_cache.get("prompt_key") == "unique_prompt_xyz"
    assert vace_cache.get("populated") is True
    assert vace_cache.get("context") is not None
    assert vace_cache.get("seq_lens") is not None


def test_both_caches_none_does_not_crash():
    """With no caches, the function encodes text every call."""
    vb = _MockWanVB()
    for i in range(3):
        _prep(vb, f"prompt_{i}")
    assert _text_calls(vb) == 3


def _make_i2v_mock():
    """Mock satisfying the I2V predicate of ``resolve_i2v_input_image`` plus the
    clip+y conditioning builders (image_encoder + vae stubs)."""
    mock_vb = _MockWanVB()
    mock_vb._dit = SimpleNamespace(has_image_input=True, require_clip_embedding=True, require_vae_embedding=True)
    mock_vb._is_ti2v = False
    mock_vb._has_vace = False
    mock_vb.image_encoder = SimpleNamespace(encode_image=lambda imgs: torch.zeros(1, 1, 8))
    mock_vb.vae = SimpleNamespace(encode=lambda vids, device, **kw: [torch.zeros(16, 5, 4, 4)])
    return mock_vb


def test_i2v_deploy_list_unwrap():
    """Deploy passes first_frame_image=[PIL]; the I2V path must write a single
    PIL (not a list) into inputs_shared["input_image"], keep
    vace_reference_image None (so no phantom prefix latent), and populate the
    clip_feature + y conditioning slots."""
    from PIL import Image

    mock_vb = _make_i2v_mock()
    img = Image.new("RGB", (320, 384))
    inputs = mock_vb.preprocess_input_for_inference(
        prompt="prompt_I2V",
        first_frame_image=[img],
        num_frames=17,
        height=32,
        width=32,
        seed=0,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
    )
    assert isinstance(inputs["input_image"], Image.Image), (
        f"I2V deploy should unwrap [PIL] to a single PIL, got {type(inputs['input_image'])}"
    )
    assert inputs["vace_reference_image"] is None
    assert inputs.get("clip_feature") is not None
    assert inputs.get("y") is not None


def test_i2v_deploy_unwrap_stable_across_cache():
    """Across a vace_cache-populated call, the I2V unwrap + conditioning are
    rebuilt identically (only the text embed is cached)."""
    from PIL import Image

    mock_vb = _make_i2v_mock()
    img = Image.new("RGB", (320, 384))
    vace_cache: dict = {}

    mock_vb.preprocess_input_for_inference(
        prompt="prompt_I2V",
        first_frame_image=[img],
        num_frames=17,
        height=32,
        width=32,
        seed=0,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        vace_cache=vace_cache,
    )
    assert vace_cache.get("populated")

    inputs = mock_vb.preprocess_input_for_inference(
        prompt="prompt_I2V",
        first_frame_image=[img],
        num_frames=17,
        height=32,
        width=32,
        seed=1,
        tiled=False,
        num_inference_steps=2,
        shift=5.0,
        vace_cache=vace_cache,
    )
    assert isinstance(inputs["input_image"], Image.Image)
    assert inputs["vace_reference_image"] is None
    assert inputs.get("clip_feature") is not None
    assert inputs.get("y") is not None


# ==== from test_video_backbone_smoke.py ====
# Smoke tests for the video backbone module.
#
# Verifies that the video backbone can be imported and basic model
# instantiation works without requiring GPU or model weights.


def test_import_pipeline():
    """The Wan construction entry (loader) + golden-reference forward import cleanly."""
    from openwam.model.video_backbone.wan._reference import model_fn_wan_video
    from openwam.model.video_backbone.wan.loader import load_wan_components, new_components

    assert callable(load_wan_components)
    assert callable(new_components)
    assert callable(model_fn_wan_video)


def test_wan_video_backbone_adapter_freq_helpers():
    """extend_freqs_with_action_tokens appends 1D action RoPE by default."""
    from openwam.model.video_backbone.wan import action_tokens

    freqs = torch.polar(torch.ones(4, 1, 6), torch.zeros(4, 1, 6))
    extended = action_tokens.extend_freqs_with_action_tokens(freqs, 2)
    assert extended.shape == (6, 1, 6)
    assert torch.allclose(extended[-2], torch.ones_like(extended[-2]))
    assert not torch.allclose(extended[-1], torch.ones_like(extended[-1]))

    # n_action_tokens=0 is a passthrough.
    assert action_tokens.extend_freqs_with_action_tokens(freqs, 0) is freqs


def test_wan_video_backbone_is_ti2v():
    """_is_ti2v returns True when fuse_vae_embedding_in_latents is set."""
    from types import SimpleNamespace

    from openwam.model.video_backbone.wan_backbone import Wan21

    pipe_a = SimpleNamespace(
        dit=SimpleNamespace(seperated_timestep=True, fuse_vae_embedding_in_latents=True),
        use_unified_sequence_parallel=False,
    )
    adapter_a = Wan21(pipe_a)
    assert adapter_a._is_ti2v is True

    pipe_b = SimpleNamespace(
        dit=SimpleNamespace(seperated_timestep=False),
        use_unified_sequence_parallel=False,
    )
    adapter_b = Wan21(pipe_b)
    assert adapter_b._is_ti2v is False


def test_wan_needs_first_frame_skip_truth_table():
    """``needs_first_frame_skip`` is True only for Wan configs where
    ``latent[0]`` is unconditionally a clean conditioning frame the loss
    must skip: TI2V (``fuse_vae_embedding_in_latents``) is the only such
    variant today.

    I2V is *not* on the skip list. Its first-frame reference rides on the
    ``y`` side channel; ``latent[0]`` itself is fully noised on both train
    and deploy, and deploy must denoise it from pure noise into the
    predicted frame 0. Skipping it during training leaves frame 0
    unsupervised → garbage at inference (train/deploy divergence). Same
    rationale as VACE — both fully supervise ``latent[0]``.

    VACE is *not* on the skip list: its first-frame condition flows
    through the ``vace_context`` bypass while the video latent path stays
    fully noised + fully supervised (matching native ``WanVideoUnit_VACE``
    semantics).

    Future T2V (none of the three) returns False so ``latent[0]`` enters
    the loss as a predicted frame.
    """
    from types import SimpleNamespace

    from openwam.model.video_backbone.wan_backbone import Wan21

    def _make_adapter(*, ti2v=False, vace=False, image_input=False):
        pipe = SimpleNamespace(
            dit=SimpleNamespace(
                seperated_timestep=ti2v,
                fuse_vae_embedding_in_latents=ti2v,
                has_image_input=image_input,
            ),
            vace=object() if vace else None,
            use_unified_sequence_parallel=False,
        )
        return Wan21(pipe)

    assert _make_adapter(ti2v=True).needs_first_frame_skip is True
    # I2V: y side-channel conveys frame 0; latent[0] is fully noised on
    # train and deploy and must be supervised — skip stays off.
    assert _make_adapter(image_input=True).needs_first_frame_skip is False
    # VACE: native convention noises every frame; loss covers latent[0].
    assert _make_adapter(vace=True).needs_first_frame_skip is False
    # Future Wan T2V: no TI2V / no VACE / no image input → skip stays off.
    assert _make_adapter().needs_first_frame_skip is False


def test_video_backbone_abc_needs_first_frame_skip_default_false():
    """The ABC default keeps every backbone that doesn't opt in OFF, so cosmos_predict25
    T2V (no override) treats ``latent[0]`` as a predicted frame in the loss."""
    from openwam.model.video_backbone.base import VideoBackbone

    # Property is defined on the ABC so we can read it off the class without
    # instantiating (constructor needs subclass-specific kwargs).
    descriptor = vars(VideoBackbone).get("needs_first_frame_skip")
    assert descriptor is not None, "needs_first_frame_skip must be defined on VideoBackbone"

    class _FakeBackbone:
        # Reuse the descriptor through a minimal stand-in to verify the default
        # without paying the ABC ``__init_subclass__`` machinery.
        needs_first_frame_skip = descriptor

    assert _FakeBackbone().needs_first_frame_skip is False


def _build_tiny_wan_backbone(*, ti2v: bool):
    """Construct a minimal real WanVideoBackbone wrapping a CPU WanModel.

    ``ti2v=True`` flips the ``seperated_timestep`` + ``fuse_vae_embedding_in_latents``
    pair so the TI2V branch of ``prepare()`` fires (independent of
    ``force_per_token_t_mod``). ``ti2v=False`` exercises the non-TI2V path that
    falls through to the ``force_per_token_t_mod`` elif.
    """
    from types import SimpleNamespace

    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan21

    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=ti2v,
        fuse_vae_embedding_in_latents=ti2v,
    )
    pipe = SimpleNamespace(
        dit=model,
        motion_controller=None,
        vace=None,
        use_unified_sequence_parallel=False,
    )
    return Wan21(pipe), model


def test_force_per_token_t_mod_broadcast_shape():
    """`force_per_token_t_mod=True` on a non-TI2V DiT must produce 4D t_mod with
    every token sharing the broadcasted ``time_embedding(timestep)`` value."""
    backbone, model = _build_tiny_wan_backbone(ti2v=False)
    B, F, H, W = 1, 2, 4, 4
    latents = torch.randn(B, model.in_dim, F, H, W)
    timestep = torch.tensor([500.0])
    context = torch.randn(B, 3, 32)

    state = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        force_per_token_t_mod=True,
    )
    assert state.time_mod.dim() == 4, "force_per_token_t_mod must yield a 4D t_mod"

    # Every token's row in t_mod should equal the broadcasted single-timestep
    # projection — proves the broadcast path didn't accidentally vary by token.
    # ``time_projection`` is a Linear over an (B, L, dim) tensor whose L rows
    # are identical inputs, so the outputs are mathematically equal; we still
    # need a small atol because batched matmul (CPU MKL in particular) is not
    # guaranteed to use the same reduction order across rows (~3e-8 in float32).
    first = state.time_mod[:, 0]
    assert torch.allclose(state.time_mod, first.unsqueeze(1).expand_as(state.time_mod), atol=1e-6, rtol=1e-6)


def test_zero_clean_prefix_t_mod_overwrites_first_frame():
    """`zero_clean_prefix_t_mod=True` + a clean ref must overwrite the first
    frame's t_mod with the ``time_embedding(timestep=0)`` projection, leaving
    later frames at the sampled timestep — mirrors TI2V's first-frame zeroing."""
    backbone, model = _build_tiny_wan_backbone(ti2v=False)
    B, F, H, W = 1, 3, 4, 4
    latents = torch.randn(B, model.in_dim, F, H, W)
    timestep = torch.tensor([500.0])
    context = torch.randn(B, 3, 32)
    # ``first_frame_latents`` non-None is the trigger that mirrors how
    # ``WanVideoBackbone.preprocess_input_for_train`` flags VACE in production.
    first_frame_latents = latents[:, :, 0:1].clone()

    state = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        force_per_token_t_mod=True,
        zero_clean_prefix_t_mod=True,
        first_frame_latents=first_frame_latents,
    )
    assert state.time_mod.dim() == 4

    # ``tokens_per_frame`` reflects the after-patchify token layout
    # (H//patch_h * W//patch_w with patch=(1,2,2) → 2*2=4 tokens/frame).
    tokens_per_frame = (H * W) // 4

    # First-frame tokens must all share the same t_mod row (the t=0 projection),
    # different from the sampled-t row used by later frames. atol/rtol cover
    # the same ~3e-8 batched-matmul roundoff documented above.
    first_frame_rows = state.time_mod[:, :tokens_per_frame]
    later_frame_rows = state.time_mod[:, tokens_per_frame:]
    assert torch.allclose(first_frame_rows, first_frame_rows[:, :1].expand_as(first_frame_rows), atol=1e-6, rtol=1e-6)
    assert torch.allclose(later_frame_rows, later_frame_rows[:, :1].expand_as(later_frame_rows), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first_frame_rows[:, 0], later_frame_rows[:, 0]), (
        "first-frame t_mod row should differ from the sampled-timestep row"
    )

    # Sanity: the first-frame row must match the manually computed t=0 projection.
    from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d

    zero_ts = torch.zeros_like(timestep)
    t_zero = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, zero_ts).to(latents.dtype))
    expected = model.time_projection(t_zero).unflatten(1, (6, model.dim))  # (B, 6, dim)
    assert torch.allclose(first_frame_rows[:, 0], expected, atol=1e-5, rtol=1e-5)


def test_zero_clean_prefix_t_mod_inactive_without_trigger():
    """Without ``first_frame_latents`` AND ``num_clean_prefix_frames=0``, the
    ``zero_clean_prefix_t_mod=True`` flag is a no-op — every token retains the
    sampled-timestep projection. This guards I2V (which has no
    ``first_frame_latents``) from accidentally getting first-frame zeroing."""
    backbone, _ = _build_tiny_wan_backbone(ti2v=False)
    latents = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 32)

    state = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        force_per_token_t_mod=True,
        zero_clean_prefix_t_mod=True,
        # NOTE: no first_frame_latents kwarg.
    )
    assert state.time_mod.dim() == 4
    first = state.time_mod[:, 0]
    # Same batched-matmul roundoff caveat as
    # ``test_force_per_token_t_mod_broadcast_shape``.
    assert torch.allclose(state.time_mod, first.unsqueeze(1).expand_as(state.time_mod), atol=1e-6, rtol=1e-6)


def test_ti2v_branch_ignores_zero_clean_prefix_t_mod_kwarg():
    """TI2V's ``seperated_timestep + fuse_vae_embedding_in_latents`` branch
    fires before the ``force_per_token_t_mod`` elif and handles first-frame
    zeroing on its own. Passing ``zero_clean_prefix_t_mod=True`` to a TI2V
    backbone must be a structurally inert no-op."""
    backbone, _ = _build_tiny_wan_backbone(ti2v=True)
    latents = torch.randn(1, 4, 2, 4, 4)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 32)

    state_a = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=1,
    )
    state_b = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=1,
        force_per_token_t_mod=True,
        zero_clean_prefix_t_mod=True,
    )
    assert state_a.time_mod.dim() == 4
    assert state_b.time_mod.dim() == 4
    assert torch.allclose(state_a.time_mod, state_b.time_mod), (
        "TI2V branch must be unaffected by the broadcast-path kwargs"
    )


def test_license_exists():
    """Apache 2.0 LICENSE file must exist in the extracted Wan license directory."""
    from pathlib import Path

    license_path = (
        Path(__file__).resolve().parents[3] / "openwam" / "model" / "video_backbone" / "wan" / "license" / "LICENSE"
    )
    assert license_path.exists(), f"LICENSE not found at {license_path}"
    content = license_path.read_text()
    assert "Apache License" in content

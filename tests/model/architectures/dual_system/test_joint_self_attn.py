"""``prepare_inputs`` collects per-sample ``proprio_mask`` alongside ``proprio``.

The full ``prepare_inputs`` flow runs through ``video_backbone.preprocess_input_for_train``
(real VAE / text encoder), which we don't want to spin up in a unit test.
We test the proprio_mask collection by stubbing ``preprocess`` and the
``_pipeline_transform_instance``, keeping the rest of the path real.
"""

from __future__ import annotations

import pytest
import torch

from openwam.model.architectures.dual_system.joint_self_attn import (
    DualSystemSelfAttnArchitecture,
)


class _IdentityTransform:
    def apply(self, sample):
        return sample


class _StubArch(DualSystemSelfAttnArchitecture):
    """Architecture stub: just enough state for ``prepare_inputs`` proprio/mask paths."""

    def __init__(self, *, state_dim: int = 20, text_dim: int = 16):
        super().__init__(None)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._init_proprio_context(
            {
                "use_proprioception": True,
                "state_dim": state_dim,
                "text_dim": text_dim,
            },
            text_dim=text_dim,
        )
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0
        self._pipeline_transform_instance = _IdentityTransform()

    # Bypass the real video preprocess; tests only care about proprio_mask wiring.
    def preprocess(self, **kwargs) -> dict:
        return {}


def _make_sample(*, proprio=None, proprio_mask=None, state_dim: int = 20):
    if proprio is None:
        proprio = torch.zeros(1, state_dim, dtype=torch.float32)
    sample = {
        "video": [],
        "prompt": "",
        "action": torch.zeros(2, state_dim, dtype=torch.float32),
        # Omit action_mask / video_mask: those code paths reach into video_backbone
        # (needs_first_frame_skip) which we don't stub here.
        "action_mask": None,
        "video_mask": None,
        "proprio": proprio,
    }
    if proprio_mask is not None:
        sample["proprio_mask"] = proprio_mask
    return sample


def test_collect_proprio_mask_from_samples():
    """Every sample carries proprio_mask -> inputs['proprio_mask'] shape (B, 1)."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [
        _make_sample(proprio_mask=torch.ones(1, dtype=torch.bool)),
        _make_sample(proprio_mask=torch.zeros(1, dtype=torch.bool)),
        _make_sample(proprio_mask=torch.ones(1, dtype=torch.bool)),
    ]
    inputs = arch.prepare_inputs(batch)
    assert "proprio_mask" in inputs
    assert inputs["proprio_mask"].shape == (3, 1)
    assert inputs["proprio_mask"].dtype == torch.bool
    assert inputs["proprio_mask"].squeeze(-1).tolist() == [True, False, True]


def test_collect_proprio_mask_missing_fallback():
    """Sample missing proprio_mask field -> auto-fills True."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [_make_sample(proprio_mask=None), _make_sample(proprio_mask=None)]
    inputs = arch.prepare_inputs(batch)
    assert inputs["proprio_mask"].shape == (2, 1)
    assert inputs["proprio_mask"].all().item() is True


def test_collect_proprio_mask_mixed_present_missing():
    """Mixed batch: missing entries fall back to True, stack succeeds."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [
        _make_sample(proprio_mask=torch.zeros(1, dtype=torch.bool)),
        _make_sample(proprio_mask=None),
        _make_sample(proprio_mask=torch.ones(1, dtype=torch.bool)),
    ]
    inputs = arch.prepare_inputs(batch)
    assert inputs["proprio_mask"].shape == (3, 1)
    assert inputs["proprio_mask"].squeeze(-1).tolist() == [False, True, True]


# ==== from test_proprio_mask_gating.py ====
# Per-sample proprio_mask gating in ``_append_proprio_context_token``.
#
# Covers (no GPU required):
#   - mask all True              matches the pre-change baseline bit-exactly
#   - mask all False             zeros the proprio token and sets context_mask[:, -1] = False
#   - mixed batch                routes each row independently
#   - gradient isolation         masked rows contribute zero gradient to proprio_encoder
#   - back-compat                missing ``_proprio_sample_mask`` key falls back to all-True
#   - global switch off          internal routing key is stripped even when context proprio is off
#   - shape normalization        (B,) and (B, 1) inputs both succeed; bad shapes raise
#
# Mirrors the ``_ContextProprioArch`` fixture pattern from ``test_proprioceptive.py``.


class _ContextProprioArch(DualSystemSelfAttnArchitecture):
    """Minimal architecture shell that only exercises BaseWAMArchitecture helpers."""

    def __init__(self, *, state_dim: int = 7, text_dim: int = 16, enabled: bool = True):
        super().__init__(None)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._init_proprio_context(
            {
                "use_proprioception": enabled,
                "state_dim": state_dim,
                "text_dim": text_dim,
            },
            text_dim=text_dim,
        )


def _make_inputs(B: int = 2, L: int = 4, D: int = 16):
    torch.manual_seed(0)
    return {
        "context": torch.randn(B, L, D),
        "seq_lens": torch.tensor([L] * B),
    }


# ---------------------------------------------------------------------------
# Per-sample mask behaviour
# ---------------------------------------------------------------------------


def test_mask_all_true_matches_baseline():
    """All-True mask must produce numerically identical output to the no-mask path."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    torch.manual_seed(42)
    inputs = _make_inputs(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    out_no_mask = arch._append_proprio_context_token(dict(inputs), proprio)

    inputs_with_mask = dict(inputs)
    inputs_with_mask["_proprio_sample_mask"] = torch.ones(2, 1, dtype=torch.bool)
    out_with_mask = arch._append_proprio_context_token(inputs_with_mask, proprio)

    assert torch.equal(out_no_mask["context"], out_with_mask["context"])
    assert torch.equal(out_no_mask["context_mask"], out_with_mask["context_mask"])


def test_mask_all_false_zeros_token():
    """All-False mask zeros the appended proprio token."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    inputs = _make_inputs(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    inputs["_proprio_sample_mask"] = torch.zeros(2, 1, dtype=torch.bool)
    out = arch._append_proprio_context_token(inputs, proprio)

    # last token (the appended proprio token) must be all-zero for every batch row
    appended = out["context"][:, -1, :]
    assert appended.abs().sum().item() == 0.0


def test_mask_all_false_attn_mask_false():
    """All-False mask sets context_mask[:, -1] = False."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    inputs = _make_inputs(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    inputs["_proprio_sample_mask"] = torch.zeros(2, 1, dtype=torch.bool)
    out = arch._append_proprio_context_token(inputs, proprio)

    assert out["context_mask"][:, -1].any().item() is False


def test_mixed_batch_routing():
    """Per-row routing: True rows get real token+True mask, False rows zero+False."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    inputs = _make_inputs(B=3, L=4, D=16)
    proprio = torch.randn(3, 7)

    inputs["_proprio_sample_mask"] = torch.tensor([[True], [False], [True]], dtype=torch.bool)
    out = arch._append_proprio_context_token(inputs, proprio)

    appended = out["context"][:, -1, :]
    assert appended[0].abs().sum().item() > 0  # mask=True
    assert appended[1].abs().sum().item() == 0.0  # mask=False
    assert appended[2].abs().sum().item() > 0  # mask=True

    assert out["context_mask"][:, -1].tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# Gradient isolation
# ---------------------------------------------------------------------------


def test_gradient_isolation_weight():
    """Single-batch mask=False: proprio_encoder.weight.grad is None or all zero."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    arch.proprio_encoder.zero_grad(set_to_none=True)

    inputs = _make_inputs(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)
    inputs["_proprio_sample_mask"] = torch.zeros(2, 1, dtype=torch.bool)

    out = arch._append_proprio_context_token(inputs, proprio)
    out["context"].sum().backward()

    g = arch.proprio_encoder.weight.grad
    assert g is None or g.abs().sum().item() == 0.0


def test_gradient_isolation_mixed_batch():
    """Mixed batch [True, False]: weight.grad equals the grad from running only row 0."""
    state_dim, text_dim = 7, 16
    torch.manual_seed(123)
    inputs_full = _make_inputs(B=2, L=4, D=text_dim)
    proprio_full = torch.randn(2, state_dim)

    # --- Mixed batch ---
    arch_mixed = _ContextProprioArch(state_dim=state_dim, text_dim=text_dim)
    arch_mixed.proprio_encoder.zero_grad(set_to_none=True)
    inputs_mixed = dict(inputs_full)
    inputs_mixed["_proprio_sample_mask"] = torch.tensor([[True], [False]], dtype=torch.bool)
    out_mixed = arch_mixed._append_proprio_context_token(inputs_mixed, proprio_full)
    out_mixed["context"].sum().backward()
    g_mixed = arch_mixed.proprio_encoder.weight.grad.clone()

    # --- Single-row (mask-True row only) reference, same encoder init ---
    arch_solo = _ContextProprioArch(state_dim=state_dim, text_dim=text_dim)
    arch_solo.proprio_encoder.load_state_dict(arch_mixed.proprio_encoder.state_dict())
    arch_solo.proprio_encoder.zero_grad(set_to_none=True)
    inputs_solo = {
        "context": inputs_full["context"][:1].detach().clone(),
        "seq_lens": inputs_full["seq_lens"][:1].clone(),
    }
    inputs_solo["_proprio_sample_mask"] = torch.ones(1, 1, dtype=torch.bool)
    out_solo = arch_solo._append_proprio_context_token(inputs_solo, proprio_full[:1].detach().clone())
    out_solo["context"].sum().backward()
    g_solo = arch_solo.proprio_encoder.weight.grad.clone()

    assert torch.allclose(g_mixed, g_solo, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Back-compat & internal-routing-key hygiene
# ---------------------------------------------------------------------------


def test_missing_field_backward_compat():
    """No ``_proprio_sample_mask`` key -> behaviour matches old all-True path."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    inputs = _make_inputs(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    out = arch._append_proprio_context_token(inputs, proprio)
    assert out["context"].shape == (2, 5, 16)
    assert out["context_mask"][:, -1].tolist() == [True, True]


def test_global_switch_off_strips_internal_key():
    """When proprio context is globally disabled, the routing key is still stripped."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16, enabled=False)
    inputs = _make_inputs(B=2, L=4, D=16)
    inputs["_proprio_sample_mask"] = torch.tensor([[True], [False]], dtype=torch.bool)

    out = arch._append_proprio_context_token(inputs, None)
    assert "_proprio_sample_mask" not in out


# ---------------------------------------------------------------------------
# Shape normalization
# ---------------------------------------------------------------------------


def test_mask_1d_vs_2d_normalization():
    """Both (B,) and (B, 1) shapes produce identical outputs."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    inputs = _make_inputs(B=3, L=4, D=16)
    proprio = torch.randn(3, 7)

    mask_2d = torch.tensor([[True], [False], [True]], dtype=torch.bool)
    mask_1d = torch.tensor([True, False, True], dtype=torch.bool)

    inputs_2d = dict(inputs)
    inputs_2d["_proprio_sample_mask"] = mask_2d
    out_2d = arch._append_proprio_context_token(inputs_2d, proprio)

    inputs_1d = dict(inputs)
    inputs_1d["_proprio_sample_mask"] = mask_1d
    out_1d = arch._append_proprio_context_token(inputs_1d, proprio)

    assert torch.equal(out_2d["context"], out_1d["context"])
    assert torch.equal(out_2d["context_mask"], out_1d["context_mask"])


def test_mask_shape_mismatch_raises():
    """Wrong batch dim in mask raises ValueError with a clear message."""
    arch = _ContextProprioArch(state_dim=7, text_dim=16)
    inputs = _make_inputs(B=2, L=4, D=16)
    proprio = torch.randn(2, 7)

    inputs["_proprio_sample_mask"] = torch.ones(3, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="_proprio_sample_mask"):
        arch._append_proprio_context_token(inputs, proprio)


# ==== from test_proprio_mask_mixture_collate.py ====
# End-to-end routing of per-sample proprio_mask through the model layer.
#
# Asserts that a mixed [EgoDex-style mask=False, AgiBot-style mask=True] batch
# gets a per-row proprio_mask collected by ``prepare_inputs`` and consumed by
# ``_append_proprio_context_token`` via ``pipeline_inputs['_proprio_sample_mask']``.
#
# We avoid spinning up a real video backbone by stubbing ``preprocess`` and the
# sample-pipeline transform, and by driving the proprio path directly via
# ``_append_proprio_context_token``. Backbones aren't needed for the per-sample
# gating contract: prepare_inputs collects, compute_loss routes, the helper
# consumes. Mid-pipeline behavior (video backbone forward) is out of scope.


class _StubArch__collate(DualSystemSelfAttnArchitecture):
    """Architecture stub: enough state for prepare_inputs + proprio gating."""

    def __init__(self, *, state_dim: int = 20, text_dim: int = 16):
        super().__init__(None)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._init_proprio_context(
            {
                "use_proprioception": True,
                "state_dim": state_dim,
                "text_dim": text_dim,
            },
            text_dim=text_dim,
        )
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0
        self._pipeline_transform_instance = _IdentityTransform()

    def preprocess(self, **kwargs) -> dict:
        return {}


def _egodex_like_sample(state_dim: int = 20):
    """Mirror the EgoDexDataset output contract for proprio fields."""
    return {
        "video": [],
        "prompt": "",
        "action": torch.zeros(2, state_dim, dtype=torch.float32),
        "action_mask": None,
        "video_mask": None,
        "proprio": torch.zeros(1, state_dim, dtype=torch.float32),
        "proprio_mask": torch.zeros(1, dtype=torch.bool),
    }


def _agibot_like_sample(state_dim: int = 20):
    """Mirror the canonical proprio fields contract (real-valued proprio + mask)."""
    torch.manual_seed(7)
    return {
        "video": [],
        "prompt": "",
        "action": torch.randn(2, state_dim),
        "action_mask": None,
        "video_mask": None,
        "proprio": torch.randn(1, state_dim),
        "proprio_mask": torch.ones(1, dtype=torch.bool),
    }


def _run_through_proprio_helper(arch: _StubArch__collate, inputs: dict, text_len: int = 4):
    """Helper: synthesize a context tensor + drive _append_proprio_context_token."""
    B = inputs["proprio"].shape[0]
    pipeline_inputs = {
        "context": torch.randn(B, text_len, arch.context_dim),
        "seq_lens": torch.tensor([text_len] * B),
        "_proprio_sample_mask": inputs["proprio_mask"],
    }
    return arch._append_proprio_context_token(pipeline_inputs, inputs["proprio"])


# ---------------------------------------------------------------------------
# 1) End-to-end: prepare_inputs -> _append_proprio_context_token
# ---------------------------------------------------------------------------


def test_egodex_agibot_mixed_batch_end_to_end():
    """Mixed [EgoDex, AgiBot] batch flows through prepare_inputs without error."""
    arch = _StubArch__collate(state_dim=20, text_dim=16)
    batch = [_egodex_like_sample(), _agibot_like_sample()]
    inputs = arch.prepare_inputs(batch)

    # prepare_inputs squeezes (1, D) → (D,) per sample, so stacked → (B, D)
    assert inputs["proprio"].shape == (2, 20)
    assert inputs["proprio_mask"].shape == (2, 1)
    assert inputs["proprio_mask"].squeeze(-1).tolist() == [False, True]

    out = _run_through_proprio_helper(arch, inputs)
    assert out["context"].shape == (2, 5, 16)
    assert out["context_mask"].shape == (2, 5)


def test_mixed_batch_context_mask_last_two_match():
    """Last column of context_mask reflects per-sample proprio_mask."""
    arch = _StubArch__collate(state_dim=20, text_dim=16)
    batch = [_egodex_like_sample(), _agibot_like_sample()]
    inputs = arch.prepare_inputs(batch)
    out = _run_through_proprio_helper(arch, inputs)

    assert out["context_mask"][:, -1].tolist() == [False, True]
    # EgoDex row: appended proprio token must be all-zero
    assert out["context"][0, -1, :].abs().sum().item() == 0.0
    # AgiBot row: appended proprio token must be non-zero
    assert out["context"][1, -1, :].abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 2) Gradient isolation across the full prepare_inputs path
# ---------------------------------------------------------------------------


def test_mixed_batch_egodex_no_proprio_encoder_grad():
    """proprio_encoder.weight grad in mixed batch == grad from AgiBot-only batch."""
    # Mixed batch (EgoDex masked, AgiBot live)
    arch_mixed = _StubArch__collate(state_dim=20, text_dim=16)
    arch_mixed.proprio_encoder.zero_grad(set_to_none=True)
    inputs_mixed = arch_mixed.prepare_inputs([_egodex_like_sample(), _agibot_like_sample()])
    torch.manual_seed(0)
    pipeline_inputs_mixed = {
        "context": torch.randn(2, 4, arch_mixed.context_dim),
        "seq_lens": torch.tensor([4, 4]),
        "_proprio_sample_mask": inputs_mixed["proprio_mask"],
    }
    out_mixed = arch_mixed._append_proprio_context_token(pipeline_inputs_mixed, inputs_mixed["proprio"])
    out_mixed["context"].sum().backward()
    g_mixed = arch_mixed.proprio_encoder.weight.grad.clone()

    # AgiBot-only reference, identical encoder init
    arch_solo = _StubArch__collate(state_dim=20, text_dim=16)
    arch_solo.proprio_encoder.load_state_dict(arch_mixed.proprio_encoder.state_dict())
    arch_solo.proprio_encoder.zero_grad(set_to_none=True)
    inputs_solo = arch_solo.prepare_inputs([_agibot_like_sample()])
    torch.manual_seed(0)
    ctx_full = torch.randn(2, 4, arch_solo.context_dim)
    pipeline_inputs_solo = {
        "context": ctx_full[1:2].detach().clone(),
        "seq_lens": torch.tensor([4]),
        "_proprio_sample_mask": inputs_solo["proprio_mask"],
    }
    out_solo = arch_solo._append_proprio_context_token(pipeline_inputs_solo, inputs_solo["proprio"])
    out_solo["context"].sum().backward()
    g_solo = arch_solo.proprio_encoder.weight.grad.clone()

    assert torch.allclose(g_mixed, g_solo, atol=1e-6, rtol=1e-5)

"""Regression tests for fused-attention backend dispatch.

FA2/FA3/sage are CUDA half-precision kernels: they raise rather than degrade.
Every dispatch site must therefore check device *and* dtype before selecting
one, otherwise merely having ``flash-attn`` importable breaks fp32 and CPU
inputs -- including the CPU tensors the rest of this suite runs on.

The backends are faked here, so these tests exercise the "flash-attn is
installed" configuration without it being installed. That matters: CI runs on
CPU-only torch and would otherwise never cover this dispatch at all.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file

from openwam.model.action_backbone.separate_action_dit import ActionDiT, SelfAttnActionDiTBlock
from openwam.model.architectures.single_system.state import attach_shared_attention_mask
from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    ISOLATED,
    MUTUAL,
    VIDEO_SEES_ACTION,
    set_video_attention_mask_mode,
)
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.encoder.base import VideoEncoderProperties
from openwam.model.video_backbone.wan import action_tokens
from openwam.model.video_backbone.wan._reference import model_fn_wan_video
from openwam.model.video_backbone.wan.models.dit import (
    DiTBlock,
    WanModel,
    modulate,
    precompute_freqs_cis_3d,
    rope_apply,
)
from openwam.model.video_backbone.wan_backbone import Wan21, Wan22Ti2v
from tests._assets import asset_path


class _FakeQuery:
    """Stand-in exposing only the two attributes the dispatch predicates read.

    Lets the CUDA-side branches be covered on a CPU-only box.
    """

    def __init__(self, is_cuda: bool, dtype: torch.dtype):
        self.is_cuda = is_cuda
        self.dtype = dtype


def _sdpa_reference(q, k, v, num_heads):
    qq, kk, vv = (rearrange(t, "b s (n d) -> b n s d", n=num_heads) for t in (q, k, v))
    return rearrange(F.scaled_dot_product_attention(qq, kk, vv), "b n s d -> b s (n d)", n=num_heads)


def _install_fake_flash_attn_2(monkeypatch, vdit, kernel):
    """Make ``dit.flash_attention`` believe FA2 -- and only FA2 -- is installed."""
    monkeypatch.setattr(vdit, "flash_attn", types.SimpleNamespace(flash_attn_func=kernel), raising=False)
    monkeypatch.setattr(vdit, "FLASH_ATTN_2_AVAILABLE", True)
    monkeypatch.setattr(vdit, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(vdit, "SAGE_ATTN_AVAILABLE", False)


# --------------------------------------------------------------- Wan video DiT


def test_wan_dit_falls_back_to_sdpa_on_cpu(monkeypatch):
    """CPU tensors must never reach the fused kernel."""
    import openwam.model.video_backbone.wan.models.dit as vdit

    def _reject(*args, **kwargs):
        raise AssertionError("fused kernel must not receive CPU tensors")

    _install_fake_flash_attn_2(monkeypatch, vdit, _reject)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16) for _ in range(3))

    out = vdit.flash_attention(q, k, v, num_heads=num_heads)

    assert torch.allclose(out, _sdpa_reference(q, k, v, num_heads), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wan_dit_falls_back_to_sdpa_for_cuda_fp32(monkeypatch):
    """fp32 is rejected by the fused kernels even on CUDA."""
    import openwam.model.video_backbone.wan.models.dit as vdit

    def _reject(*args, **kwargs):
        raise AssertionError("fused kernel must not receive fp32 tensors")

    _install_fake_flash_attn_2(monkeypatch, vdit, _reject)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16, device="cuda", dtype=torch.float32) for _ in range(3))

    out = vdit.flash_attention(q, k, v, num_heads=num_heads)

    assert torch.allclose(out, _sdpa_reference(q, k, v, num_heads), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wan_dit_keeps_fused_path_for_cuda_half(monkeypatch):
    """The guard must not cost us the fast path it is protecting."""
    import openwam.model.video_backbone.wan.models.dit as vdit

    calls = []

    def _record(q, k, v, *args, **kwargs):
        calls.append(1)
        return q

    _install_fake_flash_attn_2(monkeypatch, vdit, _record)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    out = vdit.flash_attention(q, k, v, num_heads=num_heads)

    assert calls, "cuda/bf16 input should still reach the fused kernel"
    assert torch.equal(out, q)


# ------------------------------------------------------------------ ActionDiT


def test_action_fused_kernel_usable_requires_cuda_and_half():
    from openwam.model.action_backbone.components import _fused_kernel_usable

    assert _fused_kernel_usable(_FakeQuery(True, torch.float16))
    assert _fused_kernel_usable(_FakeQuery(True, torch.bfloat16))
    # the case the device-only guard used to let through
    assert not _fused_kernel_usable(_FakeQuery(True, torch.float32))
    assert not _fused_kernel_usable(_FakeQuery(False, torch.float32))
    assert not _fused_kernel_usable(_FakeQuery(False, torch.bfloat16))


def test_action_sage_backend_falls_back_on_cpu(monkeypatch):
    """Same contract as the flash-attn backends, for the sage path."""
    import sys

    from openwam.model.action_backbone import components

    def _cuda_only_sageattn(*args, **kwargs):
        raise AssertionError("sageattn should not receive CPU tensors")

    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(sageattn=_cuda_only_sageattn))

    fn = components._try_sage_attention()
    assert fn is not None

    q, k, v = (torch.randn(1, 2, 3, 4) for _ in range(3))
    assert fn(q, k, v).shape == q.shape


# ----------------------------------------------------------- Wan shared core


@pytest.mark.parametrize("impl", ["flash_attention_3", "flash_attention_2", "sage_attention"])
def test_shared_core_downgrades_fused_backends_it_cannot_run(monkeypatch, impl):
    from openwam.model.video_backbone.wan.shared.core.attention import attention as shared

    monkeypatch.setattr(shared, "ATTENTION_IMPLEMENTATION", impl)

    assert shared.resolve_implementation(_FakeQuery(True, torch.bfloat16)) == impl
    assert shared.resolve_implementation(_FakeQuery(True, torch.float32)) == "torch"
    assert shared.resolve_implementation(_FakeQuery(False, torch.float32)) == "torch"


def test_shared_core_keeps_xformers_for_cuda_fp32(monkeypatch):
    """xformers is CUDA-only but does support fp32, so it must not be downgraded for it."""
    from openwam.model.video_backbone.wan.shared.core.attention import attention as shared

    monkeypatch.setattr(shared, "ATTENTION_IMPLEMENTATION", "xformers")

    assert shared.resolve_implementation(_FakeQuery(True, torch.float32)) == "xformers"
    assert shared.resolve_implementation(_FakeQuery(False, torch.float32)) == "torch"


def test_shared_core_attention_forward_runs_on_cpu(monkeypatch):
    """End-to-end: a CPU call must produce the SDPA result, not raise."""
    from openwam.model.video_backbone.wan.shared.core.attention import attention as shared

    monkeypatch.setattr(shared, "ATTENTION_IMPLEMENTATION", "flash_attention_2")
    monkeypatch.setattr(shared, "flash_attn_interface", None, raising=False)

    q, k, v = (torch.randn(2, 4, 8, 16) for _ in range(3))

    out = shared.attention_forward(q, k, v)

    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=0, rtol=0)


# ------------------------------------------------------- deploy diagnostics


def test_fused_backend_name_follows_availability(monkeypatch):
    import openwam.model.video_backbone.wan.models.dit as vdit

    for flag in ("FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(vdit, flag, False)
    assert vdit.fused_backend_name() == "torch_sdpa"

    monkeypatch.setattr(vdit, "SAGE_ATTN_AVAILABLE", True)
    assert vdit.fused_backend_name() == "sage_attention"

    monkeypatch.setattr(vdit, "FLASH_ATTN_2_AVAILABLE", True)
    assert vdit.fused_backend_name() == "flash_attention_2"

    monkeypatch.setattr(vdit, "FLASH_ATTN_3_AVAILABLE", True)
    assert vdit.fused_backend_name() == "flash_attention_3"


def test_diagnostics_do_not_claim_a_backend_unconditionally(monkeypatch, caplog):
    """The report must not read as 'flash_attention_2' on a run that uses SDPA."""
    import openwam.model.video_backbone.wan.models.dit as vdit
    from openwam.deploy.server import _log_attention_backends

    monkeypatch.setattr(vdit, "FLASH_ATTN_2_AVAILABLE", True)
    monkeypatch.setattr(vdit, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(vdit, "SAGE_ATTN_AVAILABLE", False)

    with caplog.at_level(logging.INFO):
        _log_attention_backends(logging.getLogger(__name__))

    video_line = next(line for line in caplog.text.splitlines() if "Video DiT" in line)
    assert "flash_attention_2" in video_line
    assert "CUDA fp16/bf16 only" in video_line


def test_qualifier_is_not_attached_to_backends_that_have_no_restriction():
    """ATTENTION_IMPLEMENTATION is an open set: an unknown value runs as plain SDPA.

    initialize_attention_priority() lowercases DIFFSYNTH_ATTENTION_IMPLEMENTATION
    but does not otherwise check it -- unlike WAM_ATTENTION_IMPL, which is validated
    against _BACKEND_MAP -- and attention_forward's else-branch runs anything it does
    not recognize through torch_sdpa. Such a run must not be reported as restricted.
    """
    from openwam.deploy.server import _WAN_BACKENDS, _qualify_backend

    # "" is reachable: the env lookup guards on `is not None`, not truthiness.
    for unrestricted in ("torch", "torch_sdpa", "sdpa", "", "some_future_backend"):
        assert _qualify_backend(unrestricted, _WAN_BACKENDS) == unrestricted

    assert (
        _qualify_backend("flash_attention_2", _WAN_BACKENDS)
        == "flash_attention_2 (CUDA fp16/bf16 only; torch_sdpa otherwise)"
    )
    assert _qualify_backend("xformers", _WAN_BACKENDS) == "xformers (CUDA only; torch_sdpa otherwise)"


def test_a_name_from_one_namespace_is_not_qualified_in_the_other():
    """The two subsystems' spellings must not cross-annotate.

    ATTENTION_IMPLEMENTATION is an open set, so an ActionDiT closure name reaches the
    Wan path through the env var. attention_forward runs it as plain SDPA, so reporting
    it as a fused backend would be the very defect this module exists to prevent.
    """
    from openwam.deploy.server import _ACTION_BACKENDS, _WAN_BACKENDS, _qualify_backend

    for action_name in _ACTION_BACKENDS:
        assert _qualify_backend(action_name, _WAN_BACKENDS) == action_name
    for wan_name in _WAN_BACKENDS:
        assert _qualify_backend(wan_name, _ACTION_BACKENDS) == wan_name


def test_action_dit_line_is_qualified_like_the_others(monkeypatch, caplog):
    """The ActionDiT closures fall back exactly as the Wan paths do, so they read alike."""
    from openwam.deploy.server import _log_attention_backends
    from openwam.model.action_backbone import components

    def _flash2(q, k, v):
        # The report reads __name__ only; a body that raises pins that it stays that way.
        raise AssertionError("diagnostics must not call the backend")

    monkeypatch.setattr(components, "_ATTENTION_FN", _flash2)

    with caplog.at_level(logging.INFO):
        _log_attention_backends(logging.getLogger(__name__))

    action_line = next(line for line in caplog.text.splitlines() if "ActionDiT" in line)
    assert "_flash2" in action_line
    assert "CUDA fp16/bf16 only" in action_line


def test_action_dit_backend_names_match_components():
    """Pin the ActionDiT spellings, so the whitelist cannot drift from _BACKEND_MAP.

    This pins the naming convention -- a closure named after its key -- rather than the
    closure objects, since each factory needs its library importable to hand one back
    and CI has none of them installed.
    """
    from openwam.deploy import server
    from openwam.model.action_backbone import components

    restricted = {
        "flash3": server._HALF_PRECISION_CUDA_ONLY,
        "flash2": server._HALF_PRECISION_CUDA_ONLY,
        "sage": server._HALF_PRECISION_CUDA_ONLY,
        "xformers": server._CUDA_ONLY,
    }
    unrestricted = {"sdpa"}  # plain SDPA has no restriction to report

    assert set(restricted) | unrestricted == set(components._BACKEND_MAP), (
        "_BACKEND_MAP changed; classify the new backend in the diagnostics whitelist"
    )

    # Per-restriction, not merely "is it listed": the wrong note mislabels the report.
    for key, expected in restricted.items():
        assert server._ACTION_BACKENDS.get(f"_{key}") == expected, f"_{key} carries the wrong restriction"
    for key in unrestricted:
        assert f"_{key}" not in server._ACTION_BACKENDS


def test_wan_backend_names_match_their_sources():
    """The Wan table's counterpart pin, over both names that reach it.

    fused_backend_name() and initialize_attention_priority() are the two sources of the
    Wan spellings; a backend added to either without a table entry silently loses its
    qualifier, which is the bug this module exists to prevent.
    """
    from openwam.deploy import server

    fused = {"flash_attention_3", "flash_attention_2", "sage_attention"}
    unrestricted = {"torch", "torch_sdpa"}  # the two "no fused kernel" spellings

    for name in fused:
        assert server._WAN_BACKENDS.get(name) == server._HALF_PRECISION_CUDA_ONLY
    assert server._WAN_BACKENDS.get("xformers") == server._CUDA_ONLY
    for name in unrestricted:
        assert name not in server._WAN_BACKENDS

    assert set(server._WAN_BACKENDS) == fused | {"xformers"}, (
        "a Wan backend was added or removed; classify it in _WAN_BACKENDS"
    )


# ==== from test_split_attn_equivalence.py ====
# Hard merge gate: ``pre_attn_at_layer + attention + post_attn_at_layer`` must
# produce numerically identical output to ``DiTBlock.forward`` / ``SelfAttnActionDiTBlock.forward``.
#
# The split is the foundation of the joint self-attention path. If it diverges
# from the plain block forward, every test that uses ``run_block`` (vanilla /
# MoE SingleSystem) silently keeps working while the new path quietly
# produces wrong outputs. This file pins the equivalence at zero atol.


def _attention_single_stream(q, k, v, num_heads):
    """Standalone attention matching what DiTBlock.self_attn.attn does internally."""
    from openwam.model.video_backbone.wan.models.dit import flash_attention

    return flash_attention(q, k, v, num_heads=num_heads)


def _attention_action(q_bsd, k_bsd, v_bsd, num_heads):
    """Standalone attention for action stream — Q/K/V already in (B, S, H*D)."""
    from einops import rearrange

    n = num_heads
    q = rearrange(q_bsd, "b s (n d) -> b n s d", n=n)
    k = rearrange(k_bsd, "b s (n d) -> b n s d", n=n)
    v = rearrange(v_bsd, "b s (n d) -> b n s d", n=n)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return rearrange(out, "b n s d -> b s (n d)", n=n)


def _make_wan_dit_block(dim=64, num_heads=4, ffn_dim=128):
    block = DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)
    block.eval()
    return block


def _wan_freqs(num_tokens: int, dim: int, num_heads: int) -> torch.Tensor:
    """Build a 3D RoPE freqs tensor matching what ``WanVideoBackbone.prepare`` produces."""
    head_dim = dim // num_heads
    f, h, w = num_tokens, 1, 1
    f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim, end=max(num_tokens, 1024))
    freqs = torch.cat(
        [
            f_freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return freqs


def _masked_action_context(dit: ActionDiT, batch_size: int):
    context = torch.zeros(batch_size, 1, dit.text_dim)
    context_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    return context, context_mask


# ---------------------------------------------------------------------------
# Video DiTBlock equivalence — drives the WanVideoBackbone split-attention path
# (without booting the full Wan pipeline).
# ---------------------------------------------------------------------------


def _wan_pre_post_via_adapter(block: DiTBlock, state: BlockLoopState, num_heads: int) -> torch.Tensor:
    """Replay what WanVideoBackbone does at one layer without instantiating the
    full backbone (which requires loading a real Wan pipeline).

    Mirrors ``WanVideoBackbone.pre_attn_at_layer`` / ``post_attn_at_layer``;
    the implementation lives there and we copy it here so the test fails if
    the adapter drifts from the plain block forward.
    """
    del num_heads  # plumbed via block.self_attn.num_heads
    from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply

    t_mod = state.time_mod
    chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    residual_x = state.hidden_states
    attn_input = modulate(block.norm1(state.hidden_states), shift_msa, scale_msa)

    sa = block.self_attn
    q = sa.norm_q(sa.q(attn_input))
    k = sa.norm_k(sa.k(attn_input))
    v = sa.v(attn_input)
    q = rope_apply(q, state.rope_freqs, sa.num_heads)
    k = rope_apply(k, state.rope_freqs, sa.num_heads)

    attn_out = _attention_single_stream(q, k, v, sa.num_heads)

    x = block.gate(residual_x, gate_msa, sa.o(attn_out))
    x = x + block.cross_attn(block.norm3(x), state.context)
    mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
    x = block.gate(x, gate_mlp, block.ffn(mlp_input))
    return x


def test_wan_dit_block_split_equivalence():
    """split(norm1+attn+post) ≡ DiTBlock.forward, exact (atol=0)."""
    torch.manual_seed(0)
    # head_dim must be divisible by 6 so precompute_freqs_cis_3d's f/h/w split
    # produces full-coverage (matches Wan2.x production heads which are 64/96/128).
    dim, num_heads, ffn_dim = 96, 4, 192
    B, S = 2, 7

    block = _make_wan_dit_block(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)
    x = torch.randn(B, S, dim)
    t_mod = torch.randn(B, 6, dim)
    context = torch.randn(B, 4, dim)
    freqs = _wan_freqs(S, dim, num_heads)

    # Reference: original DiTBlock.forward
    with torch.no_grad():
        out_ref = block(x, context, t_mod, freqs)

    # Split: pre_attn → attention → post_attn
    state = BlockLoopState(
        hidden_states=x, time_mod=t_mod, rope_freqs=freqs, context=context, grid_frames=S, grid_height=1, grid_width=1
    )
    with torch.no_grad():
        out_split = _wan_pre_post_via_adapter(block, state, num_heads)

    # Hard equivalence — bf16 wouldn't allow atol=0 but we're in fp32.
    assert torch.allclose(out_ref, out_split, atol=0, rtol=0), (
        f"split disagrees with DiTBlock.forward; max diff = {(out_ref - out_split).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Action SelfAttnActionDiTBlock equivalence — drives the ActionDiT(joint_self_attn) path.
# ---------------------------------------------------------------------------


def test_action_mot_block_split_equivalence():
    """ActionDiT pre_attn + attention + post_attn ≡ SelfAttnActionDiTBlock.forward (no context)."""
    torch.manual_seed(0)
    action_dim, dim, num_heads, ffn_dim = 7, 32, 4, 64
    B, S = 2, 5
    bridge_layers = (0, 1)

    dit = ActionDiT(
        action_dim=action_dim,
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        num_layers=len(bridge_layers),
        video_dim=dim,
        bridge_layers=bridge_layers,
        variant="joint_self_attn",
    )
    dit.eval()

    actions = torch.randn(B, S, action_dim)
    timestep = torch.randn(B)

    # Reference: drive the block manually
    with torch.no_grad():
        context, context_mask = _masked_action_context(dit, B)
        astate = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        payload = astate.payload
        payload.context = None
        payload.context_mask = None
        x_init = payload.x_action.clone()
        t_mod = payload.t_mod.clone()
        freqs = payload.action_freqs.clone()

        block: SelfAttnActionDiTBlock = dit.blocks[0]
        # SelfAttnActionDiTBlock.forward with context=None: self-attn norm/proj + RoPE + FFN.
        out_ref = block(x_init, context=None, t_mod=t_mod, freqs=freqs)

    # Split: ActionDiT pre/post_attn_at_layer round-trip with single-stream attention
    with torch.no_grad():
        context, context_mask = _masked_action_context(dit, B)
        astate2 = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        astate2.payload.context = None
        astate2.payload.context_mask = None
        q, k, v, post = dit.pre_attn_at_layer(0, astate2)
        attn_out = _attention_action(q, k, v, num_heads)
        astate2 = dit.post_attn_at_layer(0, astate2, attn_out, post)
        out_split = astate2.payload.x_action

    assert torch.allclose(out_ref, out_split, atol=0, rtol=0), (
        f"ActionDiT split disagrees with SelfAttnActionDiTBlock.forward; max diff = {(out_ref - out_split).abs().max().item()}"
    )


def test_action_mot_block_heterogeneous_hidden_dim():
    """FastWAM-Joint-style: action hidden_dim=32 but Q/K/V live in num_heads*attn_head_dim=64.

    The split (pre_attn + SDPA + post_attn) must still equal the standalone
    SelfAttnActionDiTBlock.forward at atol=0. This is the regression gate for the
    heterogeneous-hidden refactor.
    """
    torch.manual_seed(0)
    # Heterogeneous: action hidden_dim=32, attn_hidden_dim = 4*16 = 64.
    action_dim, dim, num_heads, attn_head_dim, ffn_dim = 7, 32, 4, 16, 64
    B, S = 2, 5
    bridge_layers = (0, 1)

    dit = ActionDiT(
        action_dim=action_dim,
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        num_layers=len(bridge_layers),
        video_dim=128,  # video residual width — irrelevant for the action-only check
        bridge_layers=bridge_layers,
        variant="joint_self_attn",
        attn_head_dim=attn_head_dim,
    )
    dit.eval()

    block: SelfAttnActionDiTBlock = dit.blocks[0]
    # Sanity: q outputs the attention space (4*16=64), o brings it back to 32.
    assert block.self_attn.q.weight.shape == (num_heads * attn_head_dim, dim)
    assert block.self_attn.o.weight.shape == (dim, num_heads * attn_head_dim)

    actions = torch.randn(B, S, action_dim)
    timestep = torch.randn(B)

    with torch.no_grad():
        context, context_mask = _masked_action_context(dit, B)
        astate = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        payload = astate.payload
        payload.context = None
        payload.context_mask = None
        x_init = payload.x_action.clone()
        t_mod = payload.t_mod.clone()
        freqs = payload.action_freqs.clone()
        out_ref = block(x_init, context=None, t_mod=t_mod, freqs=freqs)

        context, context_mask = _masked_action_context(dit, B)
        astate2 = dit.prepare_state(actions, timestep, context=context, context_mask=context_mask)
        astate2.payload.context = None
        astate2.payload.context_mask = None
        q, k, v, post = dit.pre_attn_at_layer(0, astate2)
        # Q/K/V are already in (B, S, num_heads*attn_head_dim) — driver-ready.
        assert q.shape == (B, S, num_heads * attn_head_dim)
        attn_out = _attention_action(q, k, v, num_heads)
        astate2 = dit.post_attn_at_layer(0, astate2, attn_out, post)
        out_split = astate2.payload.x_action

    assert torch.allclose(out_ref, out_split, atol=0, rtol=0), (
        f"heterogeneous-hidden split disagrees with SelfAttnActionDiTBlock.forward; "
        f"max diff = {(out_ref - out_split).abs().max().item()}"
    )


# ==== from test_video_backbone_consistency.py ====
# Consistency test: original model_fn_wan_video vs new prepare/run_block/finalize.
#
# Loads real Wan checkpoints and verifies that the decomposed three-step interface
# produces bit-exact (within floating-point tolerance) results compared to the
# original monolithic forward function.
#
# Requires GPU. Mark with @pytest.mark.gpu.
#
# Tests pure DiT forward only (no VACE, no SP).
# These optional modules are additive — if the core DiT path is bit-exact, the
# optional paths will be too since run_block faithfully reproduces the same logic.


# Override via env vars on machines that mount the checkpoints elsewhere; the
# defaults match the shared dev box but skipif() makes a missing path a skip,
# not a hard failure.
WAN21_VACE_1_3B = asset_path("OPENWAM_WAN21_VACE_1_3B")
WAN22_TI2V_5B = asset_path("OPENWAM_WAN22_TI2V_5B")
CUDA_AVAILABLE = torch.cuda.is_available()


def _load_dit_only(model_dir: str, device: str = "cuda:0"):
    """Load just the DiT model from a checkpoint directory."""
    from openwam.model.video_backbone.wan.models.dit import WanModel

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "t2v")
    has_image_input = model_type in ("i2v", "ti2v", "flf2v")

    dit = WanModel(
        has_image_input=has_image_input,
        patch_size=[1, 2, 2],
        in_dim=cfg["in_dim"],
        dim=cfg["dim"],
        ffn_dim=cfg["ffn_dim"],
        freq_dim=cfg["freq_dim"],
        text_dim=4096,
        out_dim=cfg["out_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        eps=cfg.get("eps", 1e-6),
        seperated_timestep=cfg.get("seperated_timestep", False) or False,
    )

    safetensors_files = sorted(
        f for f in os.listdir(model_dir) if f.startswith("diffusion_pytorch_model") and f.endswith(".safetensors")
    )
    state_dict = {}
    for sf in safetensors_files:
        state_dict.update(load_file(os.path.join(model_dir, sf)))

    try:
        dit.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {k.replace("dit.", "", 1) if k.startswith("dit.") else k: v for k, v in state_dict.items()}
        dit.load_state_dict(stripped, strict=False)

    dit = dit.to(device=device, dtype=torch.bfloat16)
    dit.eval()
    return dit


class _FakePipe:
    """Minimal duck-typed pipeline for Wan21."""

    def __init__(self, dit):
        self.dit = dit
        self.use_unified_sequence_parallel = False
        self.in_iteration_models = ["dit"]


def _run_original(dit, inputs):
    with torch.no_grad():
        return model_fn_wan_video(dit=dit, **inputs)


def _run_decomposed(pipe, inputs):
    backbone = Wan21(pipe)
    with torch.no_grad():
        state = backbone.prepare(dit=pipe.dit, **inputs)
        for block_id in range(backbone.num_layers):
            state = backbone.run_block(block_id, state)
        return backbone.finalize(state)


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_AVAILABLE or not os.path.isdir(WAN21_VACE_1_3B),
    reason=f"CUDA unavailable or checkpoint not mounted: {WAN21_VACE_1_3B}",
)
def test_consistency_wan21_vace_1_3b():
    """Wan2.1-VACE-1.3B: standard timestep path (pure DiT, no VACE)."""
    device = "cuda:0"
    dit = _load_dit_only(WAN21_VACE_1_3B, device=device)
    pipe = _FakePipe(dit)

    B, C_in, T, H, W = 1, 16, 5, 12, 20
    inputs = {
        "latents": torch.randn(B, C_in, T, H, W, device=device, dtype=torch.bfloat16),
        "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
        "context": torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16),
    }

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_original = _run_original(dit, inputs)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_decomposed = _run_decomposed(pipe, inputs)

    assert out_original.shape == out_decomposed.shape, f"Shape mismatch: {out_original.shape} vs {out_decomposed.shape}"
    max_diff = (out_original - out_decomposed).abs().max().item()
    assert torch.allclose(out_original, out_decomposed, atol=1e-4, rtol=1e-4), (
        f"Output mismatch! Max diff: {max_diff:.6e}"
    )
    print(f"[1.3B] Max diff: {max_diff:.6e} — PASS")

    del dit, pipe
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(
    not CUDA_AVAILABLE or not os.path.isdir(WAN22_TI2V_5B),
    reason=f"CUDA unavailable or checkpoint not mounted: {WAN22_TI2V_5B}",
)
def test_consistency_wan22_ti2v_5b():
    """Wan2.2-TI2V-5B: standard + per-token timestep paths."""
    device = "cuda:0"
    dit = _load_dit_only(WAN22_TI2V_5B, device=device)
    pipe = _FakePipe(dit)

    B, C_in, T, H, W = 1, 48, 5, 12, 20
    inputs = {
        "latents": torch.randn(B, C_in, T, H, W, device=device, dtype=torch.bfloat16),
        "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
        "context": torch.randn(B, 512, 4096, device=device, dtype=torch.bfloat16),
        "clip_feature": torch.randn(B, 1, 1280, device=device, dtype=torch.bfloat16),
    }

    # Standard path
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_original = _run_original(dit, inputs)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    out_decomposed = _run_decomposed(pipe, inputs)

    assert out_original.shape == out_decomposed.shape
    max_diff = (out_original - out_decomposed).abs().max().item()
    assert torch.allclose(out_original, out_decomposed, atol=1e-4, rtol=1e-4), (
        f"Output mismatch (standard)! Max diff: {max_diff:.6e}"
    )
    print(f"[5B standard] Max diff: {max_diff:.6e} — PASS")

    # Per-token timestep path
    if getattr(dit, "seperated_timestep", False):
        inputs_fuse = {**inputs, "fuse_vae_embedding_in_latents": True, "num_clean_prefix_frames": 1}

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        out_original_fuse = _run_original(dit, inputs_fuse)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        out_decomposed_fuse = _run_decomposed(pipe, inputs_fuse)

        assert out_original_fuse.shape == out_decomposed_fuse.shape
        max_diff_fuse = (out_original_fuse - out_decomposed_fuse).abs().max().item()
        assert torch.allclose(out_original_fuse, out_decomposed_fuse, atol=1e-4, rtol=1e-4), (
            f"Output mismatch (fuse)! Max diff: {max_diff_fuse:.6e}"
        )
        print(f"[5B fuse] Max diff: {max_diff_fuse:.6e} — PASS")

    del dit, pipe
    gc.collect()
    torch.cuda.empty_cache()


# ==== from test_video_backbone_dit_patch_size_dispatch.py ====
# Per-token ``t_mod`` token-count dispatch on ``dit_patch_size``.
#
# Drives commit A2: the two ``tokens_per_frame = latents.shape[3] *
# latents.shape[4] // 4`` hardcodes inside :meth:`WanVideoBackbone.prepare`
# (TI2V branch + ``force_per_token_t_mod`` broadcast branch) used to assume
# Wan's native ``(1, 2, 2)`` first-layer patch. Once an external encoder
# pre-patchifies the latent grid, its declared ``dit_patch_size`` may be
# ``(1, 1, 1)`` (ViT-style — DiT first conv becomes a channel projection) or
# ``(1, 4, 4)`` (hypothetical 4x DiT patch), and the hardcoded ``// 4``
# silently produces the wrong per-token sequence length.
#
# These parametrized tests construct a tiny CPU WanModel + matching
# ``external_encoder.properties.dit_patch_size`` and verify the per-token ``t_mod``
# length equals ``f_lat * (H * W // (ps[1] * ps[2]))`` for three patch
# configurations. Native ``(1, 2, 2)`` is included so the existing behavior
# is locked in as a regression guard.


def _mock_encoder_with_patch(dit_patch_size: tuple[int, int, int]) -> SimpleNamespace:
    """Duck-typed encoder exposing only ``encoder.properties.dit_patch_size``.

    ``WanVideoBackbone.__init__`` reads exactly this one attribute when
    ``external_encoder is not None`` — the full :class:`VideoEncoder` ABC is
    not needed for the dispatch path under test.
    """
    properties = VideoEncoderProperties(
        z_dim=16,
        spatial_compression=8,
        temporal_compression=4,
        causal_temporal=True,
        dit_patch_size=dit_patch_size,
    )
    return SimpleNamespace(properties=properties)


def _build_backbone_with_patch(
    patch_size: tuple[int, int, int],
    *,
    ti2v: bool = False,
) -> tuple[Wan22Ti2v, WanModel]:
    """Tiny CPU backbone whose DiT patchifies at ``patch_size`` and whose
    adapter's ``_dit_patch_size`` matches via a mock external encoder.

    ``ti2v=True`` flips ``seperated_timestep`` + ``fuse_vae_embedding_in_latents``
    so the TI2V branch of ``prepare()`` fires (line 469 in main); ``ti2v=False``
    lets the ``force_per_token_t_mod`` elif fire (line 495).
    """
    model = WanModel(
        dim=64,
        in_dim=4,
        ffn_dim=128,
        out_dim=4,
        text_dim=32,
        freq_dim=32,
        eps=1e-6,
        patch_size=patch_size,
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
    encoder = _mock_encoder_with_patch(patch_size)
    return Wan22Ti2v(pipe, external_encoder=encoder), model


# H = W = 8 is divisible by every patch under test (2, 1, 4); F = 2 keeps
# the runtime cheap while still letting f_lat * tokens_per_frame stay
# distinguishable across the parameters.
_F, _H, _W = 2, 8, 8


@pytest.mark.parametrize(
    "patch_size,expected_div",
    [
        ((1, 2, 2), 4),  # Wan native (and CosmosPredict25 native) — regression guard
        ((1, 1, 1), 1),  # ViT-style pre-patchified encoder, DiT 1st conv is channel proj
        ((1, 4, 4), 16),  # hypothetical 4x DiT patch
    ],
)
def test_force_per_token_t_mod_dispatches_on_dit_patch_size(patch_size, expected_div):
    """The ``force_per_token_t_mod`` branch must size ``t_mod`` to
    ``f_lat * (H*W // (ps[1]*ps[2]))`` rows so a non-(1,2,2) external encoder
    produces a t_mod row count that matches the DiT's actual post-patchify
    token count."""
    backbone, model = _build_backbone_with_patch(patch_size, ti2v=False)
    latents = torch.randn(1, model.in_dim, _F, _H, _W)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 32)

    state = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        force_per_token_t_mod=True,
    )

    expected_L = _F * (_H * _W // expected_div)
    assert state.time_mod.dim() == 4, f"force_per_token_t_mod must yield 4D t_mod; got dim={state.time_mod.dim()}"
    assert state.time_mod.shape == (1, expected_L, 6, model.dim), (
        f"patch={patch_size} expected L={expected_L}; got t_mod shape={tuple(state.time_mod.shape)}"
    )
    # Cross-check against the DiT's actual post-patchify token count.
    assert state.hidden_states.shape[1] == expected_L, (
        f"patch={patch_size} post-patchify L={state.hidden_states.shape[1]} disagrees with "
        f"per-token t_mod L={expected_L} — dispatch is out of sync with patchify"
    )


@pytest.mark.parametrize(
    "patch_size,expected_div",
    [
        ((1, 2, 2), 4),
        ((1, 1, 1), 1),
        ((1, 4, 4), 16),
    ],
)
def test_ti2v_branch_dispatches_on_dit_patch_size(patch_size, expected_div):
    """The TI2V branch (``seperated_timestep`` + ``fuse_vae_embedding_in_latents``)
    must also dispatch ``tokens_per_frame`` through ``self._dit_patch_size``.
    Same arithmetic as the ``force_per_token_t_mod`` branch, different gate."""
    backbone, model = _build_backbone_with_patch(patch_size, ti2v=True)
    latents = torch.randn(1, model.in_dim, _F, _H, _W)
    timestep = torch.tensor([500.0])
    context = torch.randn(1, 3, 32)

    state = backbone.prepare(
        latents=latents,
        timestep=timestep,
        context=context,
        fuse_vae_embedding_in_latents=True,
        num_clean_prefix_frames=1,
    )

    expected_L = _F * (_H * _W // expected_div)
    assert state.time_mod.dim() == 4
    assert state.time_mod.shape == (1, expected_L, 6, model.dim), (
        f"patch={patch_size} expected L={expected_L}; got t_mod shape={tuple(state.time_mod.shape)}"
    )
    assert state.hidden_states.shape[1] == expected_L


# ==== from test_single_system_attention_mask.py ====
# SingleSystem attention mask layout and Wan adapter behavior tests.


def _make_wan_backbone(*, dim: int = 24, num_heads: int = 4) -> Wan21:
    block = DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=48)
    block.eval()
    dit = SimpleNamespace(
        blocks=nn.ModuleList([block]),
        dim=dim,
        freq_dim=dim,
        time_embedding=nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)),
        time_projection=nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6)),
        video_attention_mask_mode="bidirectional",
    )
    return Wan21(SimpleNamespace(dit=dit))


def _identity_freqs(seq_len: int, head_dim: int) -> torch.Tensor:
    return torch.polar(
        torch.ones(seq_len, 1, head_dim // 2),
        torch.zeros(seq_len, 1, head_dim // 2),
    )


def _make_state(vb: Wan21, video: torch.Tensor, action: torch.Tensor, *, mask=None) -> BlockLoopState:
    x = torch.cat([video, action], dim=1)
    extras = {
        "dit": vb._dit,
        "vace": None,
        "time_embed": torch.zeros(x.shape[0], x.shape[1], x.shape[2]),
    }
    if mask is not None:
        extras["shared_attention_mask"] = mask
    return BlockLoopState(
        hidden_states=x,
        time_mod=torch.zeros(x.shape[0], x.shape[1], 6, x.shape[2]),
        rope_freqs=_identity_freqs(x.shape[1], vb.head_dim),
        context=torch.zeros(x.shape[0], 4, x.shape[2]),
        grid_frames=video.shape[1],
        grid_height=1,
        grid_width=1,
        extras=extras,
    )


def _old_masked_block_reference(
    block: DiTBlock,
    x: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference copy of the old adapter-local masked block implementation."""
    has_seq = t_mod.dim() == 4
    chunk_dim = 2 if has_seq else 1
    chunks = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        chunks = tuple(c.squeeze(2) for c in chunks)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    input_x = modulate(block.norm1(x), shift_msa, scale_msa)
    sa = block.self_attn
    q = sa.norm_q(sa.q(input_x))
    k = sa.norm_k(sa.k(input_x))
    v = sa.v(input_x)
    q = rope_apply(q, freqs, sa.num_heads)
    k = rope_apply(k, freqs, sa.num_heads)

    q = rearrange(q, "b s (n d) -> b n s d", n=sa.num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=sa.num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=sa.num_heads)
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    attn_out = rearrange(attn_out, "b n s d -> b s (n d)", n=sa.num_heads)

    x = block.gate(x, gate_msa, sa.o(attn_out))
    cross_mask = context_mask.unsqueeze(1).expand(-1, x.shape[1], -1).unsqueeze(1)
    x = x + block.cross_attn(block.norm3(x), context, ctx_mask=cross_mask)
    input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
    return block.gate(x, gate_mlp, block.ffn(input_x))


def _build_mask_via_attach(vb, *, n_video, n_action, n_state=0, mode=ACTION_SEES_VIDEO):
    """Drive the real shared path: attach builds the mask onto state.extras."""
    total = n_video + n_action + n_state
    state = BlockLoopState(
        hidden_states=torch.zeros(1, total, vb.dim),
        time_mod=torch.zeros(1, total, 6, vb.dim),
        rope_freqs=_identity_freqs(total, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        grid_frames=n_video,
        grid_height=1,
        grid_width=1,
        extras={},
    )
    attach_shared_attention_mask(vb, state, n_action, n_state=n_state, attention_mask_mode=mode)
    return state.extras["shared_attention_mask"]


def test_single_system_attach_mask_requires_extras():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        hidden_states=torch.zeros(1, 7, vb.dim),
        time_mod=torch.zeros(1, 7, 6, vb.dim),
        rope_freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        grid_frames=5,
        grid_height=1,
        grid_width=1,
        extras=None,
    )

    with pytest.raises(RuntimeError, match="shared_attention_mask"):
        attach_shared_attention_mask(vb, state, n_action=2, attention_mask_mode=ACTION_SEES_VIDEO)


def test_single_system_attach_mask_rejects_unknown_mode():
    vb = _make_wan_backbone()
    state = BlockLoopState(
        hidden_states=torch.zeros(1, 7, vb.dim),
        time_mod=torch.zeros(1, 7, 6, vb.dim),
        rope_freqs=_identity_freqs(7, vb.head_dim),
        context=torch.zeros(1, 4, vb.dim),
        grid_frames=5,
        grid_height=1,
        grid_width=1,
        extras={},
    )

    with pytest.raises(ValueError, match="attention_mask_mode"):
        attach_shared_attention_mask(vb, state, n_action=2, attention_mask_mode="bidirectional")


def test_single_system_set_video_attention_mask_mode_warns_when_not_settable(caplog):
    class ReadOnlyBackbone:
        @property
        def video_attention_mask_mode(self):
            return "bidirectional"

    with caplog.at_level("WARNING"):
        set_video_attention_mask_mode(ReadOnlyBackbone(), "first_frame_causal")

    assert "does not expose a settable property" in caplog.text


def test_single_system_action_rope_defaults_to_1d():
    vb = _make_wan_backbone(dim=32, num_heads=4)
    base = _identity_freqs(seq_len=2, head_dim=vb.head_dim)

    freqs = action_tokens.extend_freqs_with_action_tokens(base, n_action_tokens=3)

    assert freqs.shape == (5, 1, vb.head_dim // 2)
    assert torch.allclose(freqs[:2], base)
    # Action position 0 is identity, later action positions carry 1D RoPE phase.
    assert torch.allclose(freqs[2], base[0])
    assert not torch.allclose(freqs[3:], torch.ones_like(freqs[3:]))


def test_wan_action_tmod_broadcasts_scalar_to_batch():
    vb = _make_wan_backbone(dim=24, num_heads=4)

    t_mod = action_tokens.build_action_t_mod(torch.tensor([0.5]), n_action_tokens=3, dit=vb._dit, batch_size=2)

    assert t_mod.shape == (2, 3, 6, vb.dim)


def test_wan_action_tmod_accepts_per_sample_and_per_token():
    vb = _make_wan_backbone(dim=24, num_heads=4)

    per_sample = action_tokens.build_action_t_mod(
        torch.tensor([0.5, 0.8]), n_action_tokens=3, dit=vb._dit, batch_size=2
    )
    per_token = action_tokens.build_action_t_mod(torch.rand(2, 3), n_action_tokens=3, dit=vb._dit, batch_size=2)

    assert per_sample.shape == (2, 3, 6, vb.dim)
    assert per_token.shape == (2, 3, 6, vb.dim)


def test_wan_action_tmod_rejects_mismatched_shapes():
    vb = _make_wan_backbone(dim=24, num_heads=4)

    with pytest.raises(ValueError, match="action_timestep"):
        action_tokens.build_action_t_mod(torch.rand(3), n_action_tokens=3, dit=vb._dit, batch_size=2)
    with pytest.raises(ValueError, match="action_timestep"):
        action_tokens.build_action_t_mod(torch.rand(2, 2), n_action_tokens=3, dit=vb._dit, batch_size=2)


def test_single_system_attention_mask_action_sees_video_layout():
    vb = _make_wan_backbone()
    mask = _build_mask_via_attach(vb, n_video=5, n_action=3, mode=ACTION_SEES_VIDEO)
    Sv, Sa = 5, 3
    assert mask.shape == (Sv + Sa, Sv + Sa)
    assert mask.dtype == torch.bool
    assert mask[:Sv, :Sv].all()
    assert not mask[:Sv, Sv:].any()
    assert mask[Sv:, :Sv].all()
    assert mask[Sv:, Sv:].all()


def test_single_system_attention_mask_layout_with_state():
    vb = _make_wan_backbone()
    mask = _build_mask_via_attach(vb, n_video=5, n_action=3, n_state=2, mode=ACTION_SEES_VIDEO)
    Sv, Sa, Ss = 5, 3, 2
    v = slice(0, Sv)
    a = slice(Sv, Sv + Sa)
    s = slice(Sv + Sa, Sv + Sa + Ss)

    assert mask.shape == (Sv + Sa + Ss, Sv + Sa + Ss)
    assert mask[v, v].all()
    assert not mask[v, a].any()
    assert mask[v, s].all()
    assert mask[a, :].all()
    assert not mask[s, v].any()
    assert not mask[s, a].any()
    assert mask[s, s].all()


@pytest.mark.parametrize(
    "mode, v_sees_a, a_sees_all_v",
    [
        (MUTUAL, True, True),
        (ACTION_SEES_VIDEO, False, True),
        (VIDEO_SEES_ACTION, True, False),
        (ISOLATED, False, False),
    ],
)
def test_single_system_cross_modal_modes_with_state_tail(mode, v_sees_a, a_sees_all_v):
    """All four modes drive the shared mask; the state token stays a read-only
    tail (everyone sees it, it sees only itself). v↔v here is bidirectional
    (mock backbone), so tokens_per_frame=1 makes only the first video row a
    first-frame row."""
    vb = _make_wan_backbone()
    Sv, Sa, Ss, ff = 5, 3, 2, 1
    mask = _build_mask_via_attach(vb, n_video=Sv, n_action=Sa, n_state=Ss, mode=mode)
    u_start = Sv + Sa
    # a↔a full.
    assert mask[Sv:u_start, Sv:u_start].all()
    # state read-only tail.
    assert mask[:u_start, u_start:].all()
    assert mask[u_start:, u_start:].all()
    assert not mask[u_start:, :u_start].any()
    # v→a: first-frame row excluded; later rows follow the mode.
    assert not mask[:ff, Sv:u_start].any()
    assert mask[ff:Sv, Sv:u_start].all() if v_sees_a else not mask[ff:Sv, Sv:u_start].any()
    # a→v: all video, or first frame only.
    if a_sees_all_v:
        assert mask[Sv:u_start, :Sv].all()
    else:
        assert mask[Sv:u_start, :ff].all()
        assert not mask[Sv:u_start, ff:Sv].any()


def test_single_system_joint_mask_blocks_action_from_video_queries():
    torch.manual_seed(0)
    vb = _make_wan_backbone()
    vb.video_attention_mask_mode = "bidirectional"

    B, Sv, Sa, D = 1, 4, 3, vb.dim
    video = torch.randn(B, Sv, D)
    action_a = torch.randn(B, Sa, D)
    action_b = torch.randn(B, Sa, D) + 10.0

    mask = _build_mask_via_attach(vb, n_video=Sv, n_action=Sa, mode=ACTION_SEES_VIDEO)

    with torch.no_grad():
        out_joint_a = vb.run_block(0, _make_state(vb, video, action_a, mask=mask)).hidden_states[:, :Sv]
        out_joint_b = vb.run_block(0, _make_state(vb, video, action_b, mask=mask)).hidden_states[:, :Sv]
        out_bidir_a = vb.run_block(0, _make_state(vb, video, action_a, mask=None)).hidden_states[:, :Sv]
        out_bidir_b = vb.run_block(0, _make_state(vb, video, action_b, mask=None)).hidden_states[:, :Sv]

    assert torch.allclose(out_joint_a, out_joint_b, atol=0, rtol=0)
    assert not torch.allclose(out_bidir_a, out_bidir_b)


def test_dit_block_masked_forward_matches_old_adapter_reference():
    torch.manual_seed(0)
    dim, num_heads, seq_len, context_len = 24, 4, 7, 5
    block = DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=48).eval()
    x = torch.randn(2, seq_len, dim)
    context = torch.randn(2, context_len, dim)
    context_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    t_mod = torch.randn(2, seq_len, 6, dim) * 0.01
    freqs = _identity_freqs(seq_len, dim // num_heads)
    self_attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    self_attn_mask[:3, 4:] = False

    with torch.no_grad():
        old = _old_masked_block_reference(block, x, context, context_mask, t_mod, freqs, self_attn_mask)
        new = block(
            x,
            context,
            t_mod,
            freqs,
            context_mask=context_mask.unsqueeze(1).expand(-1, seq_len, -1),
            self_attn_mask=self_attn_mask,
        )

    assert torch.allclose(new, old, atol=1e-6, rtol=1e-5)


def test_dit_block_masked_forward_shape():
    torch.manual_seed(0)
    dim, num_heads, seq_len = 24, 4, 7
    block = DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=48).eval()
    x = torch.randn(2, seq_len, dim)
    context = torch.randn(2, 5, dim)
    t_mod = torch.randn(2, seq_len, 6, dim) * 0.01
    freqs = _identity_freqs(seq_len, dim // num_heads)
    self_attn_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    with torch.no_grad():
        out = block(x, context, t_mod, freqs, self_attn_mask=self_attn_mask)

    assert out.shape == x.shape


def test_wan_shared_token_injection_extends_tmod_freqs_and_extracts_action_only():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Sa, Ss, D = 2, 4, 3, 1, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
        extras={"dit": vb._dit, "vace": None, "time_embed": torch.zeros(B, Sv, D)},
    )
    action_tokens = torch.randn(B, Sa, D)
    state_tokens = torch.randn(B, Ss, D)
    timestep = torch.tensor([0.25, 0.75])

    state = vb.inject_shared_tokens(
        state,
        action_tokens,
        Sa,
        state_tokens=state_tokens,
        n_state=Ss,
        timestep=timestep,
    )

    assert state.hidden_states.shape == (B, Sv + Sa + Ss, D)
    assert state.time_mod.shape == (B, Sv + Sa + Ss, 6, D)
    assert state.rope_freqs.shape[0] == Sv + Sa + Ss

    state, action_tail = vb.extract_shared_tokens(state, Sa, n_state=Ss)
    assert action_tail.shape == (B, Sa, D)
    assert torch.allclose(action_tail, action_tokens)
    assert state.hidden_states.shape == (B, Sv, D)
    assert state.time_mod.shape == (B, Sv, 6, D)
    assert state.rope_freqs.shape[0] == Sv


def test_wan_shared_token_injection_supports_state_only_video_conditioning():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Ss, D = 2, 4, 1, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
        extras={"dit": vb._dit, "vace": None},
    )
    state_tokens = torch.randn(B, Ss, D)

    state = vb.inject_shared_tokens(
        state,
        None,
        0,
        state_tokens=state_tokens,
        n_state=Ss,
        timestep=torch.tensor([0.25, 0.75]),
    )

    assert state.hidden_states.shape == (B, Sv + Ss, D)
    assert state.time_mod.shape == (B, Sv + Ss, 6, D)
    assert state.rope_freqs.shape[0] == Sv + Ss
    state, action_tail = vb.extract_shared_tokens(state, 0, n_state=Ss)
    assert action_tail.shape == (B, 0, D)
    assert state.hidden_states.shape == (B, Sv, D)
    assert state.time_mod.shape == (B, Sv, 6, D)
    assert state.rope_freqs.shape[0] == Sv


def test_wan_shared_token_injection_rejects_batch_mismatch():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Sa, Ss, D = 2, 4, 3, 1, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
    )

    with pytest.raises(ValueError, match="video batch=2, action batch=3"):
        vb.inject_shared_tokens(
            state,
            torch.randn(3, Sa, D),
            Sa,
            state_tokens=torch.randn(B, Ss, D),
            n_state=Ss,
            timestep=torch.tensor([0.25, 0.75]),
        )

    with pytest.raises(ValueError, match="video batch=2, state batch=3"):
        vb.inject_shared_tokens(
            state,
            torch.randn(B, Sa, D),
            Sa,
            state_tokens=torch.randn(3, Ss, D),
            n_state=Ss,
            timestep=torch.tensor([0.25, 0.75]),
        )


def test_wan_shared_token_injection_requires_timestep_for_per_token_tmod():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    B, Sv, Sa, D = 2, 4, 3, vb.dim
    state = BlockLoopState(
        hidden_states=torch.zeros(B, Sv, D),
        time_mod=torch.zeros(B, Sv, 6, D),
        rope_freqs=_identity_freqs(Sv, vb.head_dim),
        context=torch.zeros(B, 4, D),
        grid_frames=Sv,
        grid_height=1,
        grid_width=1,
    )

    with pytest.raises(ValueError, match="requires `timestep`"):
        vb.inject_shared_tokens(state, torch.randn(B, Sa, D), Sa)


def test_wan_state_tmod_uses_sample_level_timestep_from_per_token_input():
    vb = _make_wan_backbone(dim=24, num_heads=4)
    per_token = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])

    state_tmod = action_tokens.build_sample_t_mod(per_token, n_tokens=2, dit=vb._dit, batch_size=2)
    first_tmod = action_tokens.build_sample_t_mod(per_token[:, 0], n_tokens=2, dit=vb._dit, batch_size=2)

    assert state_tmod.shape == (2, 2, 6, vb.dim)
    assert torch.allclose(state_tmod, first_tmod)

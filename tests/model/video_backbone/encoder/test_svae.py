"""Unit tests for the S-VAE feature reducer core module."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from openwam.model.video_backbone.encoder.svae import (
    _CHECKPOINT_FORMAT_VERSION,
    SVAE,
    DiagonalGaussian,
    build_svae,
    load_svae,
    reducer,
    svae_loss,
)
from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder


def _mk(**kw) -> SVAE:
    cfg = dict(input_dim=16, latent_dim=4, num_heads=2, num_layers=1, intermediate_size=16, dropout=0.0)
    cfg.update(kw)
    return SVAE(**cfg)


def _inp(B=2, C=16, T=3, H=2, W=2) -> torch.Tensor:
    return torch.randn(B, C, T, H, W)


def test_encode_eval_shape():
    mu = _mk().eval().encode(_inp())
    assert mu.shape == (2, 4, 3, 2, 2)


def test_encode_train_returns_tuple():
    z, mu, logvar = _mk().train().encode(_inp())
    assert z.shape == mu.shape == logvar.shape == (2, 4, 3, 2, 2)


def test_decode_shape():
    rec = _mk().eval().decode(torch.randn(2, 4, 3, 2, 2))
    assert rec.shape == (2, 16, 3, 2, 2)


def test_forward_keys_and_shapes():
    out = _mk().train()(_inp())
    assert {"recon", "target", "mu", "logvar"} <= set(out)
    assert out["recon"].shape == out["target"].shape == (2, 16, 3, 2, 2)
    assert out["mu"].shape == out["logvar"].shape == (2, 4, 3, 2, 2)


def test_loss_is_finite_and_nonneg():
    out = _mk().train()(_inp())
    loss, stats = svae_loss(out, beta=1e-4, free_bits_per_dim=0.05)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    for k in ("mse", "cos", "kl", "loss"):
        assert torch.isfinite(stats[k])


def test_free_bits_floor():
    # With a collapsed posterior (mu≈0, logvar≈0), per-dim KL≈0; free bits
    # raise the reported KL to the floor.
    out = _mk().train()(_inp())
    out = {**out, "mu": torch.zeros_like(out["mu"]), "logvar": torch.zeros_like(out["logvar"])}
    _, stats = svae_loss(out, beta=1.0, free_bits_per_dim=0.05)
    assert stats["kl"].item() == pytest.approx(0.05, abs=1e-6)


def test_eval_is_deterministic():
    m = _mk().eval()
    z = _inp()
    assert torch.allclose(m.encode(z), m.encode(z))


def test_train_is_stochastic():
    torch.manual_seed(0)
    m = _mk().train()
    z = _inp()
    assert not torch.allclose(m.encode(z)[0], m.encode(z)[0])


def test_set_input_stats():
    m = _mk().eval()
    mean = torch.arange(16).float()
    std = torch.full((16,), 2.0)
    m.set_input_stats(mean, std)
    assert torch.allclose(m.input_mean, mean)
    assert torch.allclose(m.input_std, std)


def test_set_input_stats_clamps_zero_std():
    m = _mk().eval()
    m.set_input_stats(torch.zeros(16), torch.zeros(16))
    assert (m.input_std > 0).all()


def test_standardisation_stats_stay_fp32_across_dtype_cast():
    # The deploy host casts the encoder to bf16; the standardisation
    # buffers must stay fp32 (they are dataset stats, not compute tensors) so
    # the standardisation matches standalone fp32-buffer training.
    m = _mk().eval()
    m.set_input_stats(torch.arange(16).float(), torch.full((16,), 2.0))
    m.to(torch.bfloat16)
    assert m.input_mean.dtype == torch.float32
    assert m.input_std.dtype == torch.float32
    assert m.enc_proj.weight.dtype == torch.bfloat16  # params did cast


def test_input_dim_head_divisibility():
    with pytest.raises(ValueError):
        SVAE(input_dim=15, latent_dim=4, num_heads=2)


def test_diagonal_gaussian_eval_returns_mu():
    g = DiagonalGaussian().eval()
    z, mu, _ = g(torch.randn(5, 8))
    assert torch.allclose(z, mu)


def test_config_dict_and_checkpoint_roundtrip(tmp_path):
    m = _mk()
    m.set_input_stats(torch.arange(16).float(), torch.full((16,), 2.0))  # non-trivial buffers
    cfg = m.config_dict()
    assert build_svae(cfg).config_dict() == cfg

    p = tmp_path / "svae.pt"
    torch.save({"format_version": 2, "model_config": cfg, "state_dict": m.state_dict()}, p)
    m2 = load_svae(str(p))

    # persistent standardisation buffers must survive the round-trip
    assert torch.allclose(m.input_mean, m2.input_mean)
    assert torch.allclose(m.input_std, m2.input_std)

    m.eval()
    m2.eval()
    z = _inp()
    assert torch.allclose(m.encode(z), m2.encode(z))


def test_load_svae_rejects_bad_checkpoint(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"foo": 1}, p)
    with pytest.raises(ValueError):
        load_svae(str(p))


def test_load_svae_rejects_unsupported_format_version(tmp_path):
    # A checkpoint carrying valid model_config/state_dict but a mismatched (or
    # missing) format_version must fail fast in load_svae, not deeper inside
    # load_state_dict.
    m = _mk()
    for bad in ({"format_version": 999}, {}):  # wrong version, and missing entirely
        p = tmp_path / f"v_{bad.get('format_version', 'none')}.pt"
        torch.save({**bad, "model_config": m.config_dict(), "state_dict": m.state_dict()}, p)
        with pytest.raises(ValueError, match="format_version"):
            load_svae(str(p))


def test_encode_mean_deterministic_even_in_train_mode():
    # The host model.train() can recursively put the frozen reducer in train
    # mode; encode_mean must still be deterministic and equal to eval-mode mu.
    m = _mk().train()
    z = _inp()
    a = m.encode_mean(z)
    b = m.encode_mean(z)
    assert torch.allclose(a, b)
    m.eval()
    assert torch.allclose(a, m.encode(z))


def test_encode_mean_deterministic_with_dropout_in_train_mode():
    # encode_mean must bypass Transformer-block dropout too (not only Gaussian
    # sampling): with a non-zero dropout config and the module left in train
    # mode, two calls must be identical, and the prior (train) mode is restored.
    m = SVAE(input_dim=16, latent_dim=4, num_layers=1, num_heads=2, intermediate_size=16, dropout=0.5)
    m.train()
    z = _inp()
    a = m.encode_mean(z)
    b = m.encode_mean(z)
    assert torch.allclose(a, b)
    assert m.training  # train mode restored after the temporary eval()


def test_forward_recon_target_in_standardized_space():
    m = _mk().eval()
    mean = torch.arange(16).float()
    std = torch.full((16,), 3.0)
    m.set_input_stats(mean, std)
    z = _inp()
    out = m(z)
    expected = (z - mean.view(1, 16, 1, 1, 1)) / std.view(1, 16, 1, 1, 1)
    assert torch.allclose(out["target"], expected, atol=1e-5)


# ==== from test_vjepa_svae.py ====
# Integration tests for the S-VAE reducer inside VJEPA21VideoEncoder.
#
# Self-contained CPU mock ViT (mirrors the one in test_external_encoder.py) so
# these tests need neither the third_party submodule nor real weights.


class _MockVJEPAViT(nn.Module):
    """Tiny CPU stand-in: (B, C, T, H, W) -> (B, L, D), tubelet=2 token count."""

    def __init__(self, embed_dim: int, patch: int = 16, tubelet: int = 2):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch = int(patch)
        self.tubelet = int(tubelet)
        self.img_temporal_dim_size = 1
        self._proj = nn.Linear(1, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, T, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        L = h * w if T == 1 else (T // self.tubelet) * h * w
        seed = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)
        return self._proj(seed)


def _svae_cfg(input_dim: int = 1408, latent_dim: int = 48) -> dict:
    return dict(
        input_dim=input_dim, latent_dim=latent_dim, num_heads=2, num_layers=1, intermediate_size=16, dropout=0.0
    )


def _enc(embed_dim: int = 1408, svae: bool = False, latent_dim: int = 48, svae_target_dim=None) -> VJEPA21VideoEncoder:
    kw = {}
    if svae:
        kw["svae_config"] = _svae_cfg(embed_dim, latent_dim)
    if svae_target_dim is not None:
        kw["svae_target_dim"] = svae_target_dim
    return VJEPA21VideoEncoder(_MockVJEPAViT(embed_dim), embed_dim=embed_dim, variant="t", **kw)


def test_S1_spec_zdim_and_feature_norm_track_reducer():
    raw = _enc(1408, svae=False)
    assert raw.properties.z_dim == 1408
    assert raw.feature_norm.normalized_shape == (1408,)

    red = _enc(1408, svae=True, latent_dim=48)
    assert red.properties.z_dim == 48
    assert red.feature_norm.normalized_shape == (48,)


def test_S2_batch_encode_output_is_reduced():
    red = _enc(1408, svae=True, latent_dim=48)
    z = red.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape == (1, 48, 3, 2, 2)  # (B, latent, T_lat=3, H/16, W/16)


def test_S3_svae_runs_after_pool_on_raw_dim():
    # Spy on the reducer input: it must receive the POST-POOL (T_lat=3),
    # RAW-dim (1408) tensor — not pre-pool (T=5) nor a reduced one.
    red = _enc(1408, svae=True)
    captured = {}
    orig = red._svae.encode_mean

    def spy(z):
        captured["shape"] = tuple(z.shape)
        return orig(z)

    red._svae.encode_mean = spy
    red.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert captured["shape"] == (1, 1408, 3, 2, 2)


def test_S4_pooled_for_training_raw_vs_reduced():
    raw = _enc(1408, svae=False)
    out = raw.batch_encode_pooled_for_svae_training(torch.randn(1, 3, 9, 32, 32))
    assert out.shape == (1, 1408, 3, 2, 2)  # raw dim, post-pool

    red = _enc(1408, svae=True)
    with pytest.raises(RuntimeError, match="raw encoder"):
        red.batch_encode_pooled_for_svae_training(torch.randn(1, 3, 9, 32, 32))


def test_S5_svae_target_dim_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        _enc(1408, svae=True, latent_dim=48, svae_target_dim=24)


def test_S6_reducer_is_frozen_eval():
    red = _enc(1408, svae=True)
    assert red._svae.training is False
    assert all(not p.requires_grad for p in red._svae.parameters())


def test_S7_deploy_sidecar_write_and_read(tmp_path):
    red = _enc(1408, svae=True)
    out = tmp_path / "ckpt"
    out.mkdir()
    # save_deploy_assets is strictly self-contained: it copies manifest.json
    # from encoder.model_path (hard error if absent) and writes the S-VAE
    # sidecar alongside. Provide a dummy manifest so the call succeeds; this
    # test asserts on the S-VAE sidecar half.
    src = tmp_path / "enc_src"
    src.mkdir()
    (src / "manifest.json").write_text("{}")
    cfg = SimpleNamespace(
        model=SimpleNamespace(video_backbone=SimpleNamespace(encoder=SimpleNamespace(model_path=str(src))))
    )
    red.save_deploy_assets(str(out), cfg)

    sidecar = out / "svae_config.json"
    assert sidecar.exists()
    # Sidecar is versioned ({format_version, model_config}); the reader unpacks
    # and returns the inner model_config (== config_dict()).
    payload = json.loads(sidecar.read_text())
    assert payload["format_version"] == _CHECKPOINT_FORMAT_VERSION
    assert payload["model_config"] == red._svae.config_dict()
    assert reducer.read_sidecar(str(out)) == red._svae.config_dict()
    assert reducer.read_sidecar(str(tmp_path / "absent")) is None


def test_S7b_deploy_sidecar_rejects_legacy_unversioned(tmp_path):
    # A legacy raw-kwargs sidecar (no format_version) must fail fast with a clear
    # message, not crash later inside build_svae with an unexpected keyword.
    out = tmp_path / "ckpt"
    out.mkdir()
    (out / "svae_config.json").write_text(
        json.dumps({"input_dim": 1408, "latent_dim": 48, "hidden_dim": 16, "num_layers": 1, "num_heads": 2})
    )
    with pytest.raises(ValueError, match="format_version"):
        reducer.read_sidecar(str(out))


def test_S8_deploy_skeleton_strict_load_roundtrip():
    enc1 = _enc(1408, svae=True)
    # non-trivial reducer buffers so the round-trip actually exercises them
    enc1._svae.set_input_stats(torch.randn(1408).abs() + 0.1, torch.rand(1408) + 0.5)

    # rebuild a zero-weight shell from the sidecar config, then strict-load
    skeleton = VJEPA21VideoEncoder(
        _MockVJEPAViT(1408), embed_dim=1408, variant="skel", svae_config=enc1._svae.config_dict()
    )
    missing_unexpected = skeleton.load_state_dict(enc1.state_dict(), strict=True)
    assert not missing_unexpected.missing_keys and not missing_unexpected.unexpected_keys

    enc1.eval()
    skeleton.eval()
    v9 = torch.randn(1, 3, 9, 32, 32)
    assert torch.allclose(enc1.batch_encode(v9), skeleton.batch_encode(v9), atol=1e-5)

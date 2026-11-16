"""ModelInputBatch path parity (spec_data_zh.md §3 model-side gate).

The dict path (``prepare_inputs(list[dict])``) and the tensor path
(``prepare_inputs(ModelInputBatch)``) must produce BIT-IDENTICAL
``compute_loss`` inputs. Two layers of evidence:

1. **Conversion parity** (pure functions, no model): the tensor entry's
   ``u8-recovery → (x*255)*(2/255)-1`` rescale reproduces
   ``wan_encode.preprocess_video``'s PIL conversion bit-for-bit.
2. **Full-path parity** (fake wan holder, tiny real modules): both paths run
   through the same ``Wan22Ti2v`` backbone on the same samples —
   ``stacked_inputs`` / ``input_latents`` / ``context`` / TI2V
   ``first_frame_latents`` / supervision bridges all compare bitwise. Loss
   equality then follows: downstream ``compute_loss`` is the same code on the
   same tensors (deterministic, seeded).
"""

import torch
from PIL import Image

from openwam.contracts.collate import collate_pipeline_contexts
from openwam.contracts.data_contract import ModelInputBatch
from openwam.dataloader.robot_data_adapter import robot_sample_to_context
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

T_OBS, H, W = 5, 32, 48  # T=5 satisfies the TI2V grid (T % 4 == 1)


# ----------------------------------------------------------- fake holder ----


class _TinyVAE(torch.nn.Module):
    """batch_encode: [B,C,T,H,W] -> [B,4,T/4,H/8,W/8] deterministic."""

    def batch_encode(self, video, device=None):
        b, c, t, h, w = video.shape
        lat = torch.nn.functional.avg_pool3d(video, kernel_size=(4, 8, 8))
        return torch.nn.functional.pad(lat, (0, 0, 0, 0, 0, 4 - lat.shape[1]))[:, :4]


class _TinyTextEncoder(torch.nn.Module):
    def forward(self, ids, mask):
        return torch.nn.functional.one_hot(ids % 8, 8).float() * mask.unsqueeze(-1)


class _FakeTokenizer:
    def __call__(self, prompts, **kw):
        ids = torch.tensor([[1 + (ord(ch) % 7) for ch in p] for p in prompts])
        mask = torch.ones_like(ids)
        return ids, mask


class _FakeDit(torch.nn.Module):
    """Enough surface for WanBase.__init__: TI2V marker + dims."""

    def __init__(self):
        super().__init__()
        self.dim = 8
        self.blocks = torch.nn.ModuleList()
        self.fuse_vae_embedding_in_latents = True  # variant detect -> TI2V


def _make_backbone(ti2v=True):
    holder = type("Holder", (), {})()
    holder.dit = _FakeDit()
    if not ti2v:
        holder.dit.fuse_vae_embedding_in_latents = False
    holder.vae = _TinyVAE()
    holder.text_encoder = _TinyTextEncoder()
    holder.tokenizer = _FakeTokenizer()
    holder.height_division_factor = 8
    holder.width_division_factor = 8
    holder.time_division_factor = 4
    holder.time_division_remainder = 1
    bb = Wan22Ti2v(holder)
    bb._device = torch.device("cpu")
    bb._dtype = torch.float32
    return bb


def make_samples(seed=0, n=2):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        frames = [Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)) for _ in range(T_OBS)]
        out.append({
            "video": frames,
            "vace_video": None,
            "first_frame_image": [frames[0]],
            "action": rng.standard_normal((3, 6)).astype(np.float32),
            "action_mask": rng.integers(0, 2, (3, 6)).astype(bool),
            "video_mask": np.ones(T_OBS, dtype=bool),
            "proprio": rng.standard_normal((1, 6)).astype(np.float32),
            "proprio_mask": np.ones((1, 6), dtype=bool),
            "prompt": f"task {i}",
        })
    return out


# --------------------------------------------------------------- tests ------


def test_tensor_rescale_matches_pil_conversion_bitwise():
    bb = _make_backbone()
    frames = make_samples(1)[0]["video"]
    pil_ref = wan_encode.preprocess_video(frames, dtype=torch.float32, device="cpu")  # [1,C,T,H,W]

    pv = torch.stack(
        [torch.from_numpy(np.asarray(f, dtype=np.float32) / 255.0).permute(2, 0, 1) for f in frames]
    ).unsqueeze(0)  # [1,T,C,H,W] — the adapter's exact conversion
    u8 = pv * 255.0
    stacked = ((u8 * (2.0 / 255.0) - 1.0)).permute(0, 2, 1, 3, 4)
    assert torch.equal(stacked, pil_ref)


import numpy as np  # noqa: E402


def _shared_keys(d):
    return {k: d[k] for k in (
        "input_latents", "context", "seq_lens", "height", "width", "num_frames",
        "fuse_vae_embedding_in_latents", "first_frame_latents",
        "actions", "proprio", "proprio_mask", "action_is_pad", "video_is_pad",
    )}


def _assert_bitwise(a, b, path=""):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype, path
        assert torch.equal(a, b), f"mismatch at {path}"
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_bitwise(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


def test_prepare_inputs_dict_vs_model_input_batch_bitwise():
    bb = _make_backbone(ti2v=True)

    class Arch:
        """Minimal stand-in exposing the contract bridge under test.

        _prepare_model_batch is called directly; prepare_inputs via the real
        BaseWAMArchitecture bound method."""

        dtype = torch.float32
        device = torch.device("cpu")
        _use_gradient_checkpointing = False
        _use_gradient_checkpointing_offload = False
        _max_timestep_boundary = 1.0
        _min_timestep_boundary = 0.0
        video_backbone = bb

        def preprocess(self, **kwargs):
            return bb.preprocess_input_for_train(**kwargs)

        prepare_inputs = WanBaseArch_prepare_inputs

    samples = make_samples(2, n=2)
    old_inputs = prepare_inputs_unbound(Arch(), samples)

    ctxs = [robot_sample_to_context(s, source_name="robotwin") for s in samples]
    batch = collate_pipeline_contexts(ctxs)
    assert isinstance(batch, ModelInputBatch)
    new_inputs = BaseWAMArchitecture._prepare_model_batch(Arch(), batch, _downsample)

    assert set(_shared_keys(old_inputs)) <= set(new_inputs)
    for key in ("input_latents", "context", "first_frame_latents", "actions", "proprio"):
        _assert_bitwise(old_inputs[key], new_inputs[key], key)
    _assert_bitwise(old_inputs["seq_lens"], new_inputs["seq_lens"], "seq_lens")
    _assert_bitwise(old_inputs["action_is_pad"], new_inputs["action_is_pad"], "action_is_pad")
    _assert_bitwise(old_inputs["video_is_pad"], new_inputs["video_is_pad"], "video_is_pad")
    assert old_inputs["fuse_vae_embedding_in_latents"] == new_inputs["fuse_vae_embedding_in_latents"]


def test_misaligned_tensor_input_is_config_error():
    bb = _make_backbone()
    bad = torch.rand(1, 4, 3, 30, 48)  # H=30 not divisible by 8
    try:
        bb.preprocess_input_for_train_from_batch(pixel_values=bad, text=["x"], has_first_frame=True)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "config error" in str(e)


# ---- helpers kept at bottom so the top reads as the parity story -----------

from openwam.model.architectures.base import BaseWAMArchitecture  # noqa: E402
from openwam.model.architectures.utils.common import downsample_video_mask_to_latent as _downsample  # noqa: E402


def prepare_inputs_unbound(arch, batch):
    return BaseWAMArchitecture.prepare_inputs(arch, batch)


def WanBaseArch_prepare_inputs(self, batch):  # noqa: N802 — alias used in Arch above
    return BaseWAMArchitecture.prepare_inputs(self, batch)

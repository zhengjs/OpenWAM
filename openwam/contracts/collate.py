"""Collation: a list of PipelineContext -> one ModelInputBatch (spec_data_zh.md §1.2).

Collation is owned exclusively by this module — no architecture may collate.

Rules (normative):

- Visual layouts are mutually exclusive; exactly one per batch: ``wam_canvas``
  (uniform stacked tensor), ``standard_tensor`` (per-cam dict), ``qwen_vl_patch``
  (packed patches + grids). An all-empty visual section is allowed only for
  synthetic/state-only batches.
- ``pixel_values_by_cam`` stacks PER CAMERA: iteration order = ``camera_order``
  argument, falling back to the first sample's insertion order. A missing
  camera on any sample is a hard error, never a zero-fill.
- ``image_grid_thw`` concatenates along dim 0 across frames/cams/batch (HF
  Qwen-VL convention): no batch dim, no padding; per-image boundaries recover
  via ``cumsum(t*h*w)``.
- Language paths are mutually exclusive: WAM raw ``prompt`` strings XOR VLA
  ``input_ids``. ``input_ids``/``attention_mask``/``vlm_labels`` right-pad to
  the max sequence length (pad 0 / mask 0 / labels -100).
- Optional tensor fields are all-or-none; mixed presence is a hard error.
- ``trajectory_mask`` rank reconciliation: all-1D -> ``[B, T]``; all-2D ->
  ``[B, T, D]``; mixed ranks -> hard error.
- ``ego_state_tensor`` promotes ``[D]`` to ``[1, D]`` before stacking.
- ``sample_weights`` is always emitted (``[B]`` float32, default 1.0).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from openwam.contracts.data_contract import ModelInputBatch, PipelineContext

__all__ = ["collate_pipeline_contexts"]


def _all_or_none(name: str, values: Sequence[Optional[object]]) -> bool:
    """Return True when present on every sample; raise on mixed presence."""
    present = [v is not None for v in values]
    if any(present) and not all(present):
        raise ValueError(f"collate: field {name!r} present on some samples and missing on others")
    return bool(present and all(present))


def _stack_equal_shapes(tensors: List[Tensor], name: str) -> Tensor:
    """Stack along a new batch dim; every tensor must share one shape."""
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"collate: {name!r} shapes differ across batch: {sorted(shapes)}")
    return torch.stack(tensors, dim=0)


def _pad_1d(tensors: List[Tensor], pad_value: float, name: str) -> Tensor:
    """Right-pad per-sample 1-D tensors to the max length -> [B, L_max]."""
    for t in tensors:
        if t.dim() != 1:
            raise ValueError(f"collate: {name!r} must be per-sample 1-D [L], got {tuple(t.shape)}")
    max_len = max(t.shape[0] for t in tensors)
    pad_vals = {"attention_mask": 0.0, "vlm_labels": -100.0}
    rows = []
    for t in tensors:
        pad_len = max_len - t.shape[0]
        pad = torch.full((pad_len,), pad_vals.get(name, pad_value), dtype=t.dtype, device=t.device)
        rows.append(torch.cat([t, pad], dim=0) if pad_len else t)
    return torch.stack(rows, dim=0)


def _collate_visuals(ctxs: List[PipelineContext], camera_order: Optional[List[str]]) -> dict:
    """Resolve the exactly-one-visual-layout rule; returns batch kwargs."""
    pixel_values = [c.pixel_values for c in ctxs]
    by_cam = [c.pixel_values_by_cam for c in ctxs]
    grids = [c.image_grid_thw for c in ctxs]
    has_pv = _all_or_none("pixel_values", pixel_values)
    has_by_cam = _all_or_none("pixel_values_by_cam", by_cam)
    has_grids = _all_or_none("image_grid_thw", grids)

    # standard_tensor excludes the other two.
    if has_by_cam:
        if has_pv:
            raise ValueError("collate: pixel_values and pixel_values_by_cam are mutually exclusive")
        if has_grids:
            raise ValueError("collate: pixel_values_by_cam and image_grid_thw are mutually exclusive")

        cam_sets = [set(d.keys()) for d in by_cam]
        reference = cam_sets[0]
        for i, cams in enumerate(cam_sets):
            if cams != reference:
                raise ValueError(
                    f"collate: camera set mismatch at sample {i}: "
                    f"{sorted(cams)} vs reference {sorted(reference)}"
                )
        order = list(camera_order) if camera_order is not None else list(by_cam[0].keys())
        if set(order) != reference:
            raise ValueError(f"collate: camera_order {order} does not match sample cameras {sorted(reference)}")
        merged: Dict[str, Tensor] = {}
        for cam in order:
            per_cam = [d[cam] for d in by_cam]  # each [T_obs, C, H_cam, W_cam]
            merged[cam] = _stack_equal_shapes(per_cam, f"pixel_values_by_cam[{cam!r}]")
        return {"pixel_values_by_cam": merged, "pixel_values": None, "image_grid_thw": None}

    # qwen_vl_patch: packed patches + grids, both concatenated (HF convention).
    if has_grids:
        if not has_pv:
            raise ValueError("collate: image_grid_thw requires packed pixel_values (qwen_vl_patch mode)")
        for i, t in enumerate(pixel_values):
            if t.dim() != 2:
                raise ValueError(
                    f"collate: qwen_vl_patch pixel_values must be [L_patches, patch_dim], "
                    f"sample {i} got {tuple(t.shape)}"
                )
        widths = {t.shape[1] for t in pixel_values}
        if len(widths) != 1:
            raise ValueError(f"collate: qwen patch widths differ across batch: {sorted(widths)}")
        for i, g in enumerate(grids):
            if g.dim() != 2 or g.shape[1] != 3:
                raise ValueError(f"collate: image_grid_thw must be [N_img, 3], sample {i} got {tuple(g.shape)}")
        return {
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(grids, dim=0),
            "pixel_values_by_cam": None,
        }

    # wam_canvas (or no visuals at all).
    if has_pv:
        stacked = _stack_equal_shapes(pixel_values, "pixel_values")
        if stacked.dim() != 5:
            raise ValueError(
                f"collate: wam_canvas pixel_values must be [T_obs, C, H, W] per sample, got rank {stacked.dim() - 1}"
            )
        return {"pixel_values": stacked, "pixel_values_by_cam": None, "image_grid_thw": None}
    return {"pixel_values": None, "pixel_values_by_cam": None, "image_grid_thw": None}


def _collate_language(ctxs: List[PipelineContext]) -> dict:
    """Resolve the exactly-one-language-path rule; returns batch kwargs."""
    prompts = [c.prompt for c in ctxs]
    ids = [c.input_ids for c in ctxs]
    masks = [c.attention_mask for c in ctxs]
    labels = [c.vlm_labels for c in ctxs]
    has_prompt = _all_or_none("prompt", prompts)
    has_ids = _all_or_none("input_ids", ids)
    if has_prompt and has_ids:
        raise ValueError("collate: prompt (WAM) and input_ids (VLA) are mutually exclusive language paths")

    if has_prompt:
        return {"prompt": list(prompts), "input_ids": None, "attention_mask": None, "vlm_labels": None}
    if has_ids:
        attention = masks[0] is not None
        _all_or_none("attention_mask", masks)
        vlm_labels = labels[0] is not None
        _all_or_none("vlm_labels", labels)
        return {
            "prompt": None,
            "input_ids": _pad_1d(list(ids), 0.0, "input_ids"),
            "attention_mask": _pad_1d(list(masks), 0.0, "attention_mask") if attention else None,
            "vlm_labels": _pad_1d(list(labels), -100.0, "vlm_labels") if vlm_labels else None,
        }
    return {"prompt": None, "input_ids": None, "attention_mask": None, "vlm_labels": None}


def _collate_state_action(ctxs: List[PipelineContext]) -> dict:
    """State / supervision stacking with mask-rank reconciliation."""
    egos = [c.ego_state_tensor for c in ctxs]
    kwargs: dict = {
        "ego_state_tensor": None,
        "ego_state_mask": None,
        "gt_trajectory_norm": None,
        "trajectory_mask": None,
    }
    if _all_or_none("ego_state_tensor", egos):
        promoted = [e if e.dim() == 2 else e.unsqueeze(0) for e in egos]  # [D] -> [1, D]
        kwargs["ego_state_tensor"] = _stack_equal_shapes(promoted, "ego_state_tensor")  # [B, T_hist, D]

    state_masks = [c.ego_state_mask for c in ctxs]
    if _all_or_none("ego_state_mask", state_masks):
        # Same 1-D/2-D reconciliation as BaseArchitecture.prepare_inputs: promote
        # a per-dim [D] flag to [1, D] so mixed-rank batches stack.
        promoted_masks = [m if m.dim() == 2 else m.unsqueeze(0) for m in state_masks]
        kwargs["ego_state_mask"] = _stack_equal_shapes(promoted_masks, "ego_state_mask")  # [B, T_hist, D]

    trajs = [c.gt_trajectory_norm for c in ctxs]
    if _all_or_none("gt_trajectory_norm", trajs):
        kwargs["gt_trajectory_norm"] = _stack_equal_shapes(list(trajs), "gt_trajectory_norm")  # [B, T_pred, D]

    masks = [c.trajectory_mask for c in ctxs]
    if _all_or_none("trajectory_mask", masks):
        ranks = {m.dim() for m in masks}
        if len(ranks) != 1:
            raise ValueError(f"collate: trajectory_mask ranks differ across batch: {sorted(ranks)}")
        kwargs["trajectory_mask"] = _stack_equal_shapes(list(masks), "trajectory_mask")  # [B, T] | [B, T, D]
    return kwargs


def collate_pipeline_contexts(
    ctxs: Sequence[PipelineContext],
    *,
    camera_order: Optional[List[str]] = None,
) -> ModelInputBatch:
    """Collate N PipelineContext into one ModelInputBatch (see module rules)."""
    if not ctxs:
        raise ValueError("collate: cannot collate an empty list")
    ctxs = list(ctxs)

    videos = [c.video_mask for c in ctxs]
    video_mask = (
        _stack_equal_shapes([torch.as_tensor(v) for v in videos], "video_mask")
        if _all_or_none("video_mask", videos)
        else None
    )

    return ModelInputBatch(
        **_collate_visuals(ctxs, camera_order),
        video_mask=video_mask,
        **_collate_language(ctxs),
        **_collate_state_action(ctxs),
        sample_weights=torch.tensor([c.sample_weight for c in ctxs], dtype=torch.float32),
    )

"""Multi-view image processor: raw camera frames → one of three visual layouts
(spec_data_zh.md §5).

``mode`` selects the mutually-exclusive layout (contracts §1.2):

- ``wam_canvas``: composite the camera views into ONE canvas per frame
  (reuses ``transforms/multiview.py::assemble_multiview_layout``), output
  ``pixel_values`` ``[T_obs, C, H, W]``.
- ``standard_tensor``: per-camera tensors, NO upsampling (only optional
  ``per_cam_resolution`` downscale), output ``pixel_values_by_cam``.
- ``qwen_vl_patch``: dynamic-resolution patchification per frame (HF
  Qwen-VL convention), output packed ``pixel_values`` ``[L_patches, C*p*p]``
  plus ``image_grid_thw`` ``[N_images, 3]``; token merging happens model-side.

Merging views into a single token stream is the model's job — this processor
only produces pixels + layout metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.preprocessors.base import BasePreprocessor
from openwam.dataloader.transforms.multiview import assemble_multiview_layout

__all__ = ["MultiViewImageProcessor"]

_VISUAL_MODES = ("wam_canvas", "standard_tensor", "qwen_vl_patch")


def _to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(frame, dtype=np.uint8))


def _patchify(img: Image.Image, patch_size: int) -> np.ndarray:
    """HF Qwen-VL patchification: ``[H, W, 3]`` uint8 → ``[N, 3*p*p]`` float32
    patches in row-major order (N = (H/p)*(W/p); token merging is model-side)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    h, w = arr.shape[:2]
    arr = arr.reshape(h // patch_size, patch_size, w // patch_size, patch_size, 3)
    arr = arr.transpose(0, 2, 1, 3, 4)  # [Hp, Wp, p, p, 3]
    return arr.reshape(-1, 3 * patch_size * patch_size)


class MultiViewImageProcessor(BasePreprocessor):
    def __init__(
        self,
        mode: str = "wam_canvas",
        camera_layout: Optional[List[str]] = None,
        canvas_size: Optional[List[int]] = None,
        top_height_ratio: float = 2.0 / 3.0,
        per_cam_resolution: Optional[Dict[str, List[int]]] = None,
        patch_size: int = 14,
        merge_size: int = 2,
    ):
        if mode not in _VISUAL_MODES:
            raise ValueError(f"MultiViewImageProcessor: unknown mode {mode!r}; expected {_VISUAL_MODES}")
        if mode == "wam_canvas" and (camera_layout is None or canvas_size is None):
            raise ValueError("MultiViewImageProcessor: wam_canvas requires camera_layout + canvas_size")
        self.mode = mode
        self.camera_layout = camera_layout
        self.canvas_size = tuple(canvas_size) if canvas_size else None
        self.top_height_ratio = top_height_ratio
        self.per_cam_resolution = per_cam_resolution
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.raw_images is None:
            raise ValueError("MultiViewImageProcessor: ctx.raw_images is empty (run multicam_decode first)")

        if self.mode == "wam_canvas":
            ctx.pixel_values = self._compose_canvas(ctx.raw_images)
        elif self.mode == "standard_tensor":
            ctx.pixel_values_by_cam = self._per_cam_tensors(ctx.raw_images)
        else:
            self._pack_qwen(ctx, ctx.raw_images)
        ctx.raw_images = None  # consumed
        return ctx

    # ----- layouts ---------------------------------------------------------------

    def _compose_canvas(self, raw_images: Dict[str, np.ndarray]) -> torch.Tensor:
        out_h, out_w = self.canvas_size
        first = raw_images[self.camera_layout[0]]
        num_frames = first.shape[0]
        frames = []
        for t in range(num_frames):
            pil_by_cam = {cam: _to_pil(frames_[t]) for cam, frames_ in raw_images.items()}
            canvas = assemble_multiview_layout(
                pil_by_cam,
                list(self.camera_layout),
                out_h,
                out_w,
                top_height_ratio=self.top_height_ratio,
            )
            arr = np.asarray(canvas, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # [C, H, W]
        return torch.stack(frames, dim=0)  # [T, C, H, W]

    def _per_cam_tensors(self, raw_images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for cam, frames in raw_images.items():
            resized = frames
            target = (self.per_cam_resolution or {}).get(cam)
            if target is not None:
                th, tw = int(target[0]), int(target[1])
                if frames.shape[1] != th or frames.shape[2] != tw:
                    pil = [_to_pil(f).resize((tw, th), Image.BILINEAR) for f in frames]
                    resized = np.stack([np.asarray(p) for p in pil], axis=0)
            tensor = torch.from_numpy(np.ascontiguousarray(resized, dtype=np.float32) / 255.0)
            out[cam] = tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        return out

    def _pack_qwen(self, ctx: PipelineContext, raw_images: Dict[str, np.ndarray]) -> None:
        p, m = self.patch_size, self.merge_size
        factor = p * m
        all_patches: List[np.ndarray] = []
        grids: List[List[int]] = []
        for cam in sorted(raw_images):
            for frame in raw_images[cam]:
                h, w = frame.shape[:2]
                th, tw = max(1, round(h / factor)) * factor, max(1, round(w / factor)) * factor
                img = _to_pil(frame)
                if (th, tw) != (h, w):
                    img = img.resize((tw, th), Image.BILINEAR)
                all_patches.append(_patchify(img, p))
                grids.append([1, th // p, tw // p])
        ctx.pixel_values = torch.from_numpy(np.concatenate(all_patches, axis=0))
        ctx.image_grid_thw = torch.tensor(grids, dtype=torch.long)

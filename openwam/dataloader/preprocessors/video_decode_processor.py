"""Multi-camera video decoder: clip_uri + cam_frame_offsets → raw frames
(spec_data_zh.md §5, training-only).

Runs inside DataLoader workers (decode is the expensive stage and must not
leak into deployment). Produces:

- ``raw_images``: ``{cam: [T_obs, H, W, 3] uint8}`` (camera-native resolution)
- ``frame_timestamps_ns``: PTS-derived per-frame timestamps of the first
  camera in ``camera_order`` (sync reference for ``TopicTimeSynchronizer``)

Color jitter (training-only, time-consistent: factors sampled ONCE per sample,
applied across all frames of all cameras) moved here from
``utils/video_augment.py`` — decode-stage jitter moved to the decode processor.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageEnhance

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.clip_reader import ClipReader
from openwam.dataloader.preprocessors.base import BasePreprocessor

__all__ = ["MultiCamVideoDecoder"]


class MultiCamVideoDecoder(BasePreprocessor):
    is_training_only = True

    def __init__(
        self,
        color_jitter: Optional[Dict[str, float]] = None,
        lru_size: int = 4,
    ):
        self._jitter = dict(color_jitter) if color_jitter else None
        self._lru_size = lru_size

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.cam_frame_offsets or ctx.clip_uri == "":
            raise ValueError("multicam_decode: requires clip_uri + cam_frame_offsets (Lance index block)")
        reader = ClipReader.instance(lru_size=self._lru_size)
        handle = reader.handle(ctx.clip_uri)

        factors = None
        if self._jitter:
            factors = tuple(
                1.0 + random.uniform(-self._jitter.get(k, 0.0), self._jitter.get(k, 0.0))
                for k in ("brightness", "contrast", "saturation")
            )

        raw_images: Dict[str, np.ndarray] = {}
        for cam, offsets in ctx.cam_frame_offsets.items():
            frames = handle.read_frames(cam, offsets)  # list of HWC uint8
            if factors is not None:
                frames = [self._jitter_frame(f, factors) for f in frames]
            raw_images[cam] = np.stack(frames, axis=0)

        ctx.raw_images = raw_images
        # Per-read-frame PTS aligned with cam_frame_offsets (contract: [T_obs]).
        ref_cam = next(iter(ctx.cam_frame_offsets))
        all_pts = handle.frame_timestamps_ns(ref_cam)
        ctx.frame_timestamps_ns = all_pts[list(ctx.cam_frame_offsets[ref_cam])]
        return ctx

    @staticmethod
    def _jitter_frame(frame: np.ndarray, factors) -> np.ndarray:
        img = Image.fromarray(frame)
        b, c, s = factors
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        img = ImageEnhance.Color(img).enhance(s)
        return np.asarray(img, dtype=np.uint8)

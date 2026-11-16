"""Robot-domain (Track A) data adapter: LeRobot v3 canonical sample dict -> PipelineContext (spec_data_zh.md §3).

The four pretraining readers stay untouched (spec_zh.md §5.1): they already
decode frames, assemble the multiview canvas, and normalize actions internally.
This adapter is a pure field mapping into the contract's PROCESSED slots —
Track A runs an EMPTY preprocessor chain (待确认 A). Unmapped canonical keys:
``vace_video`` (always None today) and ``first_frame_image`` (derivable
model-side from ``pixel_values[:, 0]`` — first-frame conditioning stays a
model-side concern).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch

from openwam.contracts.data_contract import PipelineContext

__all__ = ["robot_sample_to_context"]


def _video_to_canvas(video: list) -> torch.Tensor:
    """PIL frames -> [T_obs, C, H, W] float32 in [0, 1] (to_tensor semantics).

    Matches ``torchvision.transforms.functional.to_tensor`` for RGB PIL input;
    bit-parity with the transforms path is pinned by the P2 regression gate.
    """
    if not video:
        raise ValueError("robot_data_adapter: sample['video'] is empty")
    frames = []
    for img in video:
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, C]
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # [C, H, W]
    return torch.stack(frames, dim=0)


def robot_sample_to_context(
    sample: Mapping,
    *,
    source_name: str,
    clip_uri: str = "",
    sample_id: str = "",
) -> PipelineContext:
    """Map one canonical reader sample into a PipelineContext (spec §3 table)."""
    prompt = sample.get("prompt", "")

    def _tensor(value):
        if value is None or isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value)

    return PipelineContext(
        # 1. Index metadata — synthetic bookkeeping; real decoding is reader-internal.
        sample_id=sample_id,
        clip_uri=clip_uri,
        anchor_timestamp_us=0,
        cam_frame_offsets={},
        source_name=source_name,
        nav_instruction=prompt,
        # 2/3. Optional metadata, calibration & timing — Track A leaves them unset.
        # 4. Raw slots — already consumed inside the reader; stay None.
        # 5. Processed slots — direct field mapping (contract: processed slots are tensors).
        pixel_values=_video_to_canvas(sample["video"]),
        video_mask=_tensor(sample["video_mask"]),
        prompt=prompt,
        ego_state_tensor=_tensor(sample["proprio"]),
        ego_state_mask=_tensor(sample["proprio_mask"]),
        gt_trajectory_norm=_tensor(sample["action"]),
        trajectory_mask=_tensor(sample["action_mask"]),
    )

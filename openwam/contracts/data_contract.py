"""Data-side contracts: PipelineContext and ModelInputBatch (spec_data_zh.md §1.1).

Normative points:

- Exactly ONE context class serves both data tracks (LeRobot robot domain and
  the Lance driving store). Domain differences are expressed by *who fills a
  slot and how*, never by extra types.
- ``ModelInputBatch`` is frozen and pure-tensor; only these fields cross into
  ``BaseArchitecture.forward()``. VAE latents, text embeddings, VACE
  conditioning and other model-side GPU artifacts are deliberately absent.
- Multi-camera resolution tiers (front > side > rear) survive THROUGH the
  visual encoder: per-cam encode -> flatten-to-tokens -> concat in config
  camera order, merged MODEL-SIDE — never in the preprocessor chain.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from torch import Tensor


@dataclass
class PipelineContext:
    """Mutable, chain-passed context. Exactly one training slice (one window of one clip)."""

    # ── 1. Index metadata — SHARED, always required ──────────────────────────
    #    Filled by the dataset layer (Lance row / LeRobot window adapter).

    # Join key for logs / eval / resume auditing.
    sample_id: str
    # Storage locator of the source clip. Only the decoder/reader dereferences it.
    clip_uri: str
    # Sync origin. Driving: anchor camera exposure time; robot: window start.
    anchor_timestamp_us: int
    # Per-camera frame indices to read — the complete read plan. Driving:
    # precomputed at ingest into Lance. Robot: synthesized by the adapter. The
    # decoder performs ZERO index math.
    cam_frame_offsets: Dict[str, List[int]]
    # Which dataset/table this slice came from (mixture logging, per-source metrics).
    source_name: str
    # Task instruction. "" only for pure world-model samples.
    nav_instruction: str = ""

    # ── 2. Optional metadata — SHARED ─────────────────────────────────────────
    # From weighters (constant / success_rate / progress_delta); 1.0 = plain BC.
    sample_weight: float = 1.0
    # Filtering / curriculum / mixture stratification.
    scene_tags: Optional[List[str]] = None
    # Chain-of-thought target. Present ⇒ prompt processor builds vlm_labels.
    cot_text: Optional[str] = None

    # ── 3. Calibration & timing — DRIVING-ONLY ────────────────────────────────
    # {cam: {"extrinsics": [4,4] cam->ego-rig, "intrinsics": [3,3]}}.
    calib: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    # Real per-cam exposure stamps; consumed by the sync tolerance gate.
    frame_timestamps_ns: Optional[Dict[str, List[int]]] = None

    # ── 4. Raw decoded physical quantities — SHARED slots, per-track fillers ──
    # {cam: [T_obs, H_cam, W_cam, 3] uint8} at NATIVE per-cam resolution.
    raw_images: Optional[Dict[str, np.ndarray]] = None
    # [T_hist, D_state] float32, ego-rig frame. Robot: T_hist=1 proprio.
    raw_ego_history: Optional[np.ndarray] = None
    # [T_pred, D_action] float32, ego-rig frame. Driving: future ego plan.
    raw_gt_trajectory: Optional[np.ndarray] = None
    # [T_pred] or [T_pred, D_action] bool; required whenever raw_gt_trajectory set.
    raw_trajectory_valid_mask: Optional[np.ndarray] = None
    # [T_obs] bool — observation-window frame validity (WAM loss masking). Track A
    # readers emit it today ("video_mask"); driving windows are always full ⇒ None.
    video_mask: Optional[np.ndarray] = None

    # ── 5. Processed tensors — SHARED, produced by the preprocessor chain ─────
    # wam_canvas or qwen_vl_patch layout (see ModelInputBatch).
    pixel_values: Optional[Tensor] = None
    # standard_tensor mode, {cam: [T_obs, C, H_cam, W_cam]} at tier resolution.
    pixel_values_by_cam: Optional[Dict[str, Tensor]] = None
    # qwen_vl_patch only, [N_img, 3].
    image_grid_thw: Optional[Tensor] = None
    # WAM path: raw string; model-side text encode.
    prompt: Optional[str] = None
    # VLA path (tokenized in the worker).
    input_ids: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    # -100 on prompt tokens when cot_text present.
    vlm_labels: Optional[Tensor] = None
    # [T_hist, D_enc] (T_hist=1 for robot proprio).
    ego_state_tensor: Optional[Tensor] = None
    # [T_pred, D_action].
    gt_trajectory_norm: Optional[Tensor] = None
    # [T_pred] or [T_pred, D_action] bool.
    trajectory_mask: Optional[Tensor] = None


@dataclass(frozen=True)
class ModelInputBatch:
    """Pure-tensor batch. Only these cross into ``BaseArchitecture.forward()``.

    Resolution heterogeneity survives THROUGH the visual encoder: each camera is
    encoded at its own tier; the streams merge only after per-cam flatten-to-
    tokens, model-side. Three layouts, exactly one per batch:

      wam_canvas       pixel_values [B, T_obs, C, H_canvas, W_canvas]
      standard_tensor  pixel_values_by_cam {cam: [B, T_obs, C, H_cam, W_cam]}
      qwen_vl_patch    pixel_values packed per-image patches + image_grid_thw
    """

    # Visuals (exactly one of the three layouts per preprocessor chain).
    pixel_values: Optional[Tensor] = None
    pixel_values_by_cam: Optional[Dict[str, Tensor]] = None
    image_grid_thw: Optional[Tensor] = None

    # [B, T_obs] bool, frame validity (WAM).
    video_mask: Optional[Tensor] = None

    # Language (exactly one path: raw WAM strings XOR tokenized VLA ids).
    prompt: Optional[List[str]] = None
    input_ids: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    vlm_labels: Optional[Tensor] = None

    # State / action supervision.
    ego_state_tensor: Optional[Tensor] = None
    gt_trajectory_norm: Optional[Tensor] = None
    # [B, T_pred] bool, or [B, T_pred, D_action] for per-dim validity (robot
    # 80-D reserved dims). Subsumes action_is_pad.
    trajectory_mask: Optional[Tensor] = None

    # [B] float32.
    sample_weights: Optional[Tensor] = None
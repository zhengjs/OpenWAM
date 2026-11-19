"""Ego state / trajectory normalization processor (spec_data_zh.md §5).

Consumes the raw slots produced by ``TopicTimeSynchronizer`` (history) and the
Lance index (``gt_future_traj_ego``), normalizes them with the frozen
``dataset_statistics.json`` (§2 contract), and emits the processed slots
``ego_state_tensor`` / ``gt_trajectory_norm`` / ``trajectory_mask``.

Also hosts the deploy-side inverses on the same stats —
``denormalize_trajectory`` (model action output → ego-rig meters) and
π0.5-style ``state2str`` (state readout for prompt building) — so train and
deploy share one normalization instance.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.preprocessors.base import BasePreprocessor
from openwam.dataloader.utils.driving_stats import _MODE_TO_INTERNAL, load_dataset_statistics
from openwam.dataloader.utils.normalization import apply_normalization

__all__ = ["EgoStateProcessor"]


def _fourier_expand(arr: np.ndarray, num_bands: int) -> np.ndarray:
    """Sin/cos Fourier embedding at geometric frequencies 2^k·π (k < bands).

    ``[..., D]`` → ``[..., D * 2 * num_bands]`` (per dim: [sin(x), cos(x), ...]).
    """
    feats = []
    for k in range(num_bands):
        scaled = arr * (np.pi * (2**k))
        feats.append(np.sin(scaled))
        feats.append(np.cos(scaled))
    return np.concatenate(feats, axis=-1)


class EgoStateProcessor(BasePreprocessor):
    """``raw_ego_history``/``raw_gt_trajectory`` → normalized processed slots.

    Args:
        stats_path: ``dataset_statistics.json`` path (mode + per-block vectors).
        fourier_num_bands: optional Fourier embedding of the NORMALIZED ego
            state (0 = off). Expands ``ego_state_tensor``'s last dim.
        state_dim_names: optional names for ``state2str``; defaults to
            ``state_0..D-1`` (D_state semantics are dataset-specific).
    """

    def __init__(
        self,
        stats_path: str,
        fourier_num_bands: int = 0,
        state_dim_names: Optional[List[str]] = None,
    ):
        self.stats = load_dataset_statistics(stats_path)
        self._internal_mode = _MODE_TO_INTERNAL[self.stats["mode"]]
        self._fourier_num_bands = int(fourier_num_bands)
        self._state_dim_names = state_dim_names

    # ----- training path -------------------------------------------------------

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.raw_ego_history is not None:
            hist = apply_normalization(
                np.asarray(ctx.raw_ego_history, dtype=np.float32),
                self.stats["ego_state"],
                self._internal_mode,
            )
            if self._fourier_num_bands > 0:
                hist = _fourier_expand(hist, self._fourier_num_bands)
            ctx.ego_state_tensor = torch.from_numpy(np.ascontiguousarray(hist))

        if ctx.raw_gt_trajectory is not None:
            traj = apply_normalization(
                np.asarray(ctx.raw_gt_trajectory, dtype=np.float32),
                self.stats["trajectory"],
                self._internal_mode,
            )
            ctx.gt_trajectory_norm = torch.from_numpy(np.ascontiguousarray(traj))

        if ctx.raw_trajectory_valid_mask is not None:
            ctx.trajectory_mask = torch.as_tensor(
                np.asarray(ctx.raw_trajectory_valid_mask), dtype=torch.bool
            )
        # Raw slots are chain-internal: cleared once consumed (same hygiene as
        # MultiViewImageProcessor clearing raw_images).
        ctx.raw_ego_history = None
        ctx.raw_gt_trajectory = None
        ctx.raw_trajectory_valid_mask = None
        return ctx

    # ----- deploy-side inverses (shared instance, spec §5 dual mode) -----------

    def denormalize_trajectory(self, traj_norm: np.ndarray) -> np.ndarray:
        """Invert the trajectory normalization: normalized ``[..., D_action]``
        → ego-rig units. Clipped extremes cannot be recovered (saturating map)."""
        stats = self.stats["trajectory"]
        mode = self.stats["mode"]
        if mode == "q99":
            scale = np.maximum(stats["q99"] - stats["q01"], 1e-6)
            return (traj_norm + 1.0) / 2.0 * scale + stats["q01"]
        if mode == "min_max":
            scale = np.maximum(stats["max"] - stats["min"], 1e-6)
            return (traj_norm + 1.0) / 2.0 * scale + stats["min"]
        return traj_norm * np.maximum(stats["std"], 1e-6) + stats["mean"]

    def state2str(self, ego_state: np.ndarray) -> str:
        """π0.5-style state readout for prompt building: ``"speed=1.2, yaw_rate=0.1"``."""
        names = self._state_dim_names or [f"state_{i}" for i in range(ego_state.shape[-1])]
        return ", ".join(
            f"{name}={val:.3f}" for name, val in zip(names, np.asarray(ego_state).reshape(-1))
        )

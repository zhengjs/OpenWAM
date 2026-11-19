"""Topic time synchronization: ego history interpolation at the anchor
(spec_data_zh.md §5, training-only).

Queries ``topics/ego_motion.parquet`` at the sample's ``T_hist`` history
timestamps (anchor minus ``stride_us`` steps), linearly interpolates the state
columns, optionally converts pose-like columns to rot6d via
``utils/poses.py``, and emits ``raw_ego_history`` ``[T_hist, D_state]``.
Ego-rig transform via ``calib`` applies to pose-bearing states.

Time-sync policy fields (spec §12 O1 — defaults deliberately conservative and
configurable, pending the undecided production defaults):

- ``tolerance_us``: gate — if the nearest topic row is farther than this from
  a query time, the overflow policy decides (``None`` = no gate).
- ``overflow``: ``"nearest"`` (clamp to nearest row) | ``"zero_fill"`` (emit
  zeros for out-of-gate query times) | ``"error"`` (hard fail).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.clip_reader import ClipReader
from openwam.dataloader.preprocessors.base import BasePreprocessor
from openwam.dataloader.utils.poses import quat_wxyz_to_rot6d

__all__ = ["TopicTimeSynchronizer"]

_OVERFLOW_POLICIES = ("nearest", "zero_fill", "error")


def _interpolate_at(times_ns: np.ndarray, query_ns: np.ndarray, values: np.ndarray, overflow: str,
                    tolerance_us: Optional[int]) -> np.ndarray:
    """Per-column linear interpolation of ``values`` (sorted by ``times_ns``).

    Returns ``[len(query), D]``; out-of-range handling per ``overflow``.
    """
    out = np.zeros((len(query_ns), values.shape[1]), dtype=np.float32)
    lo = np.clip(np.searchsorted(times_ns, query_ns, side="right") - 1, 0, len(times_ns) - 1)
    hi = np.clip(lo + 1, 0, len(times_ns) - 1)
    t_lo, t_hi = times_ns[lo], times_ns[hi]
    denom = np.where(t_hi > t_lo, t_hi - t_lo, 1)
    w = np.clip((query_ns - t_lo) / denom, 0.0, 1.0)
    out = values[lo] * (1.0 - w[:, None]) + values[hi] * w[:, None]

    if tolerance_us is not None:
        gate = int(tolerance_us) * 1000
        dist = np.minimum(np.abs(query_ns - t_lo), np.abs(t_hi - query_ns))
        bad = dist > gate
        if bad.any():
            if overflow == "error":
                raise ValueError(
                    f"topic_sync: ego_motion gap {int(dist[bad].max()) // 1000}us exceeds "
                    f"tolerance {gate // 1000}us at {int(bad.sum())} query times"
                )
            if overflow == "zero_fill":
                out[bad] = 0.0
    return out.astype(np.float32)


class TopicTimeSynchronizer(BasePreprocessor):
    is_training_only = True

    def __init__(
        self,
        t_hist: int,
        stride_us: int = 100_000,
        tolerance_us: Optional[int] = None,
        overflow: str = "nearest",
        rot6d_source: Optional[str] = None,
        rot6d_source_format: str = "quat_wxyz",
    ):
        if overflow not in _OVERFLOW_POLICIES:
            raise ValueError(f"topic_sync: unknown overflow={overflow!r}; expected {_OVERFLOW_POLICIES}")
        if t_hist < 1:
            raise ValueError(f"topic_sync: t_hist must be >= 1, got {t_hist}")
        self.t_hist = int(t_hist)
        self.stride_ns = int(stride_us) * 1000
        self.tolerance_us = tolerance_us
        self.overflow = overflow
        # Column whose quat/6D representation is converted to rot6d (e.g. the
        # yaw quaternion block); None → passthrough float columns only.
        self.rot6d_source = rot6d_source
        self.rot6d_source_format = rot6d_source_format

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.clip_uri == "" or ctx.anchor_timestamp_us is None:
            raise ValueError("topic_sync: requires clip_uri + anchor_timestamp_us (Lance index block)")
        handle = ClipReader.instance().handle(ctx.clip_uri)
        table = handle.ego_motion

        times_ns = np.asarray(table.column("timestamp_ns").to_pylist(), dtype=np.int64)
        order = np.argsort(times_ns)
        times_ns = times_ns[order]

        anchor_ns = int(ctx.anchor_timestamp_us) * 1000
        query_ns = anchor_ns - self.stride_ns * np.arange(self.t_hist - 1, -1, -1, dtype=np.int64)

        state_cols = [name for name in table.column_names if name != "timestamp_ns"]
        if self.rot6d_source is not None and self.rot6d_source not in state_cols:
            raise ValueError(f"topic_sync: rot6d_source {self.rot6d_source!r} not in ego_motion columns {state_cols}")
        # Columns may be scalar (speed) or vector (quaternion); concatenate by width.
        widths: List[int] = []
        col_blocks = []
        for c in state_cols:
            arr = np.asarray(table.column(c).to_pylist(), dtype=np.float32)[order]
            if arr.ndim == 1:
                arr = arr[:, None]
            widths.append(arr.shape[1])
            col_blocks.append(arr)
        values = np.concatenate(col_blocks, axis=1)
        history = _interpolate_at(times_ns, query_ns, values, self.overflow, self.tolerance_us)

        if self.rot6d_source is not None:
            idx = state_cols.index(self.rot6d_source)
            start = sum(widths[:idx])
            width = widths[idx]
            block = history[:, start : start + width]
            if self.rot6d_source_format == "quat_wxyz":
                block = quat_wxyz_to_rot6d(block)
            else:
                raise ValueError(f"topic_sync: unsupported rot6d_source_format {self.rot6d_source_format!r}")
            history = np.concatenate([history[:, :start], block, history[:, start + width :]], axis=1)

        ctx.raw_ego_history = history  # [T_hist, D_state]
        return ctx

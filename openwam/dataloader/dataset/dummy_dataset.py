"""Dummy autonomous-driving dataset (Track B debug): synthetic index rows in
memory + the SAME on-disk clip layout as the real suite — no ``lance`` needed.

``DummyDrivingDataset`` mirrors :class:`~openwam.dataloader.dataset.autodriving_dataset.AutoDrivingDataset`
field-for-field (sample_id / clip_uri / anchor / cam_frame_offsets / nav /
cot / scene_tags / sample_weight / raw_gt_trajectory + mask, then the same
preprocessor chain), but the "index" is synthesized deterministically over a
synthetic clip built by ``utils/synthetic_clip.build_synthetic_clip``. This is
the debugging twin of the real reader: everything downstream (collate, model
bridge, trainer) cannot tell them apart, while the data layer needs no lance
package and no real ingest. Use via registry type ``dummy_clip``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from openwam.contracts.config_schema import PreprocessorChainConfig
from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.dataset import BaseDataset
from openwam.dataloader.preprocessors import PreProcessorPipeline
from openwam.dataloader.utils.driving_stats import load_dataset_statistics

logger = logging.getLogger(__name__)

__all__ = ["DummyDrivingDataset"]


class DummyDrivingDataset(BaseDataset):
    """In-memory synthetic index over one generated clip (see module docstring).

    Args:
        dataset_dir:        storage root; auto-generates ``raw_clips/clipA`` +
                            ``dataset_statistics.json`` when missing.
        stats_path:         ``dataset_statistics.json`` (T_hist/T_pred/D_state/D_action).
        preprocessor_chain: chain config run per sample in ``__getitem__``.
        num_samples:        synthetic index rows (all pointing at the one clip,
                            anchors spread deterministically across the clip).
        scene_tags:         tag list stamped on every row (kept for config parity).
        name:               ``source_name`` stamped on every sample.
    """

    def __init__(
        self,
        dataset_dir: str,
        stats_path: str,
        preprocessor_chain: Optional[PreprocessorChainConfig] = None,
        num_samples: int = 4,
        scene_tags: Optional[Sequence[str]] = None,
        name: str = "dummy_clip",
    ):
        from openwam.dataloader.utils.synthetic_clip import build_synthetic_clip

        self._name = name
        self._root = Path(dataset_dir)
        if not (self._root / "raw_clips").is_dir():
            logger.info("[dummy_clip] generating synthetic storage at %s", self._root)
            build_synthetic_clip(self._root)

        self._stats = load_dataset_statistics(stats_path)
        self._t_pred = self._stats["T_pred"]
        self._d_action = self._stats["D_action"]

        self._rows = self._synthesize_rows(int(num_samples), list(scene_tags or []))
        self._chain = (
            PreProcessorPipeline.from_config(preprocessor_chain) if preprocessor_chain else None
        )

    # -- index synthesis -----------------------------------------------------

    def _synthesize_rows(self, num_samples: int, scene_tags: List[str]) -> List[dict]:
        """Rows matching the real ingest schema (tools/ingest_lance.py:268),
        anchors spread over the clip with room for T_obs history and T_pred."""
        clip_uri = str(self._root / "raw_clips" / "clipA")
        # The synthetic clip's timeline: probe it once for the real PTS grid.
        from openwam.dataloader.utils.clip_reader import ClipHandle

        handle = ClipHandle(clip_uri)
        ts_ns = handle.frame_timestamps_ns("cam_front")
        t0_us = int(ts_ns[0]) // 1000
        step_us = int((ts_ns[1] - ts_ns[0]) // 1000)

        rows = []
        for i in range(num_samples):
            anchor_us = t0_us + step_us * (6 + 8 * i) + 500
            offsets = {cam: [6 + 8 * i + k for k in range(5)] for cam in ("cam_front", "cam_left", "cam_rear")}
            rows.append({
                "sample_id": f"dummy_{i:04d}",
                "clip_uri": clip_uri,
                "anchor_timestamp_us": anchor_us,
                "cam_frame_offsets": offsets,
                "nav_instruction": "go straight",
                "cot_text": "",
                "scene_tags": scene_tags,
                "sample_weight": 1.0,
                "gt_future_traj_ego": np.zeros((self._t_pred * self._d_action,), dtype=np.float32),
                "valid_mask": np.ones((self._t_pred,), dtype=np.int8),
            })
        return rows

    # -- BaseDataset ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> PipelineContext:
        row = self._rows[int(idx)]
        ctx = PipelineContext(
            sample_id=row["sample_id"],
            clip_uri=row["clip_uri"],
            anchor_timestamp_us=row["anchor_timestamp_us"],
            cam_frame_offsets=dict(row["cam_frame_offsets"]),
            source_name=self._name,
            nav_instruction=row["nav_instruction"],
            prompt=row["nav_instruction"],  # raw-string language path (Wan text encode)
            cot_text=row["cot_text"],
            scene_tags=list(row["scene_tags"]),
            sample_weight=float(row["sample_weight"]),
            calib=json.loads((Path(row["clip_uri"]) / "calib.json").read_text(encoding="utf-8")),
            raw_gt_trajectory=np.asarray(row["gt_future_traj_ego"], dtype=np.float32).reshape(
                self._t_pred, self._d_action
            ),
            raw_trajectory_valid_mask=np.asarray(row["valid_mask"], dtype=np.int8),
        )
        if self._chain is not None:
            ctx = self._chain.process(ctx)
        return ctx

    @property
    def action_dim(self) -> int:
        return self._d_action

    @property
    def normalization_stats(self) -> Optional[dict]:
        return None

    @property
    def names(self) -> List[str]:
        return [self._name]

    @classmethod
    def from_config(cls, config, split: str = "train") -> "DummyDrivingDataset":
        from openwam.dataloader.utils import get_cfg

        chain_cfg = None
        if get_cfg(config, "preprocessor_chain", None) is not None:
            chain_cfg = PreprocessorChainConfig(**dict(get_cfg(config, "preprocessor_chain")))
        return cls(
            dataset_dir=get_cfg(config, "dataset_dir"),
            stats_path=get_cfg(config, "stats_path"),
            preprocessor_chain=chain_cfg,
            num_samples=int(get_cfg(config, "num_samples", 4)),
            scene_tags=get_cfg(config, "scene_tags", None),
            name=get_cfg(config, "dataset_name", "dummy_clip"),
        )

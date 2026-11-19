"""Autonomous-driving (Track B) dataset over Lance sample indices (spec_data_zh.md §4.4).

``AutoDrivingDataset`` is a map-style dataset: each ``__getitem__`` resolves one
Lance row into a ``PipelineContext`` (index block + optional metadata + raw
trajectory slots + calib), then runs the configured preprocessor chain
worker-side (decode → sync → ego_state → image → prompt). Output contexts go
straight to ``contracts.collate`` — or into the iter-based mixture
(``IterMixtureDataset`` passes ``PipelineContext`` through untouched; the
Lance row already carries ``source_name`` / ``sample_weight``).

``lance`` is imported lazily: this module loads on robot-only hosts that do
not install it (no driving configs in play there). For billion-row indices the
offline iter-based path (§6.2) loads only the rank's strided plan slice; this
in-memory filtered table is the per-suite baseline reader.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from openwam.contracts.config_schema import PreprocessorChainConfig
from openwam.contracts.data_contract import PipelineContext
from openwam.dataloader.dataset import BaseDataset
from openwam.dataloader.preprocessors import PreProcessorPipeline
from openwam.dataloader.utils.driving_stats import load_dataset_statistics

logger = logging.getLogger(__name__)

__all__ = ["AutoDrivingDataset"]

_PROJECT_COLS = [
    "sample_id",
    "clip_uri",
    "anchor_timestamp_us",
    "cam_frame_offsets",
    "nav_instruction",
    "cot_text",
    "scene_tags",
    "sample_weight",
    "gt_future_traj_ego",
    "valid_mask",
]


@lru_cache(maxsize=64)
def _load_calib(clip_uri: str) -> str:
    """Per-process calib cache keyed by clip dir (spec §4.4); returns the JSON
    string so the cached value is immutable."""
    import json
    from pathlib import Path

    return (Path(clip_uri) / "calib.json").read_text(encoding="utf-8")


class AutoDrivingDataset(BaseDataset):
    """One ``<dataset_name>.lance`` index; samples are Lance rows + raw clips.

    Args:
        lance_uri:      Path to the ``.lance`` dataset (or ``dataset_dir`` +
                        ``dataset_name`` via ``from_config``).
        stats_path:     ``dataset_statistics.json`` (drives ``D_action``/``T_pred``).
        preprocessor_chain: chain config run per sample in ``__getitem__``.
        scene_tags:     keep rows whose ``scene_tags`` contain ANY of these.
        name:           ``source_name`` stamped on every sample.
    """

    def __init__(
        self,
        lance_uri: str,
        stats_path: str,
        preprocessor_chain: Optional[PreprocessorChainConfig] = None,
        scene_tags: Optional[Sequence[str]] = None,
        name: str = "lance_clip",
    ):
        import lance  # lazy: robot-only hosts never install it

        self._name = name
        self._lance_dataset = lance.dataset(lance_uri)
        self._table = self._lance_dataset.to_table(columns=_PROJECT_COLS)

        if scene_tags:
            wanted = set(scene_tags)

            def _keep(tags) -> bool:
                return bool(wanted.intersection(tags or []))

            mask = [_keep(t) for t in self._table.column("scene_tags").to_pylist()]
            if not any(mask):
                raise ValueError(f"AutoDrivingDataset({name}: scene_tags filter {sorted(wanted)} matches no rows")
            self._table = self._table.filter(np.asarray(mask))

        self._stats = load_dataset_statistics(stats_path)
        self._t_pred = self._stats["T_pred"]
        self._d_action = self._stats["D_action"]
        gt_width = len(self._table.column("gt_future_traj_ego")[0].as_py())
        if gt_width != self._t_pred * self._d_action:
            raise ValueError(
                f"AutoDrivingDataset({name}): gt_future_traj_ego width {gt_width} != "
                f"T_pred*D_action {self._t_pred}*{self._d_action} from {stats_path}"
            )
        self._chain = (
            PreProcessorPipeline.from_config(preprocessor_chain) if preprocessor_chain else None
        )

    def __len__(self) -> int:
        return self._table.num_rows

    def __getitem__(self, idx: int) -> PipelineContext:
        row = self._table.slice(int(idx), 1).to_pylist()[0]
        ctx = PipelineContext(
            sample_id=row["sample_id"],
            clip_uri=row["clip_uri"],
            anchor_timestamp_us=row["anchor_timestamp_us"],
            cam_frame_offsets=dict(row["cam_frame_offsets"] or {}),
            source_name=self._name,
            nav_instruction=row["nav_instruction"] or "",
            cot_text=row["cot_text"],
            scene_tags=list(row["scene_tags"] or []),
            sample_weight=float(row["sample_weight"] or 1.0),
            calib=json.loads(_load_calib(row["clip_uri"])),
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
    def from_config(cls, config, split: str = "train") -> "AutoDrivingDataset":
        from openwam.dataloader.utils import get_cfg

        lance_uri = get_cfg(config, "lance_uri", None)
        if lance_uri is None:
            lance_uri = str(
                Path(get_cfg(config, "dataset_dir"))
                / "lance_indices"
                / f"{get_cfg(config, 'dataset_name')}.lance"
            )
        chain_cfg = None
        if get_cfg(config, "preprocessor_chain", None) is not None:
            chain_cfg = PreprocessorChainConfig(**dict(get_cfg(config, "preprocessor_chain")))
        return cls(
            lance_uri=lance_uri,
            stats_path=get_cfg(config, "stats_path"),
            preprocessor_chain=chain_cfg,
            scene_tags=get_cfg(config, "scene_tags", None),
            name=get_cfg(config, "dataset_name", "lance_clip"),
        )

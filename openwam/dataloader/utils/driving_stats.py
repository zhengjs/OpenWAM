"""``dataset_statistics.json`` loader — the driving-side stats contract (spec_data_zh.md §2).

The legacy robot path (``normalization_stats.npy``, ``{joint, eef}`` schema)
keeps loading through :func:`load_stats_file`; this module is the NEW driving
schema consumed by ``EgoStateProcessor`` and ``tools/ingest_lance.py``:

```json
{
  "mode": "q99",                     // q99 | min_max | mean_std  (NormMode strings)
  "ego_state":  {"min": [...], "max": [...], "q01": [...], "q99": [...], "mean": [...], "std": [...]},
  "trajectory": {"min": [...], "max": [...], "q01": [...], "q99": [...], "mean": [...], "std": [...]},
  "T_hist": 1, "D_state": 6, "T_pred": 40, "D_action": 3
}
```

Normalization itself is NOT re-implemented here: consumers feed the returned
stat vectors into the same ``apply_normalization`` / ``Normalizer`` code paths
the robot domain uses (one train/deploy parity guarantee). Validation reuses
``materialize_eef_stats`` (shape/finiteness/ordering checks + neutral fill of
vectors the active mode doesn't consume); the rot6d pinning logic is inert at
these widths (6 / 3-D).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from openwam.dataloader.utils.normalization import materialize_eef_stats

__all__ = ["load_dataset_statistics"]

# JSON ``mode`` strings are NormMode values; translate to the internal
# apply_normalization/Normalizer mode names for materialize_eef_stats.
_MODE_TO_INTERNAL = {"q99": "quantile", "min_max": "min-max", "mean_std": "z-score"}
_DIM_KEYS = ("T_hist", "D_state", "T_pred", "D_action")
_BLOCK_DIM = {"ego_state": "D_state", "trajectory": "D_action"}


def load_dataset_statistics(path: Union[str, Path]) -> dict:
    """Load and validate a ``dataset_statistics.json`` file.

    Returns::

        {
          "mode": <str, unchanged>,
          "ego_state":  {six float32 vectors, shape (D_state,)},
          "trajectory": {six float32 vectors, shape (D_action,)},
          "T_hist": int, "D_state": int, "T_pred": int, "D_action": int,
        }

    Raises ``ValueError``/``KeyError`` on unknown mode, missing blocks, or
    malformed vectors (wrong shape, non-finite, min>max, std<0).
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    mode = data.get("mode")
    if mode not in _MODE_TO_INTERNAL:
        raise ValueError(
            f"{path}: unknown mode {mode!r}; expected one of {sorted(_MODE_TO_INTERNAL)}"
        )
    internal_mode = _MODE_TO_INTERNAL[mode]

    dims = {}
    for key in _DIM_KEYS:
        if key not in data:
            raise ValueError(f"{path}: missing {key!r}")
        dims[key] = int(data[key])

    out = {"mode": mode, **dims}
    for block, dim_key in _BLOCK_DIM.items():
        if block not in data:
            raise ValueError(f"{path}: missing stats block {block!r}")
        out[block] = materialize_eef_stats(
            dict(data[block]),
            internal_mode,
            dim=dims[dim_key],
            strict_minmax=False,
            source_hint=str(path),
        )
    return out

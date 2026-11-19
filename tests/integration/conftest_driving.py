"""Shared synthetic Track B storage fixture (mp4 + parquet topics + calib).

Thin re-export: the builder lives in production code now
(``openwam/dataloader/utils/synthetic_clip.py``) so the dummy dataset and the
smoke entry share it; this shim keeps the historical test imports working.
"""

from openwam.dataloader.utils.synthetic_clip import build_synthetic_clip  # noqa: F401

"""Concrete dataset implementations (``xxx_dataset.py`` — one file per data
source) plus the shared ``BaseDataset`` root.

The registry (``openwam/dataloader/registry.py``) lazily imports from this
package; cross-file references go through
``from openwam.dataloader.dataset.base_dataset import BaseDataset``.
Shared reader scaffolding (LeRobot v3) lives in ``dataloader/utils/``.
"""

from openwam.dataloader.dataset.base_dataset import BaseDataset

__all__ = ["BaseDataset"]

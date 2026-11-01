"""VLA action head ABC: the contract a VLA action head exposes to its host
VLA architecture.

Unlike WAM action backbones (:class:`SharedActionBackbone` /
:class:`ActionDiTBackbone`, which expose encode/decode contracts consumed by
the WAM denoising loop), a VLA head is a self-contained predictor: the host
VLA architecture (:class:`BaseVLAArchitecture`) extracts VLM features and
delegates loss / prediction to the head via the two methods below.

Like the WAM action backbones there is no registry — each VLA architecture
builds its head via :func:`build_vla_action_head`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


class VLAActionHead(nn.Module, ABC):
    """Swappable action head for VLA architectures.

    Subclasses MUST implement ``compute_loss`` / ``predict_action`` plus the
    ``action_dim`` / ``action_horizon`` properties. The shape convention of
    ``encoder_hidden_states`` is head-specific:

    - :class:`L1RegressionActionHead` — gathered action-token queries (B, chunk, H)
    - :class:`FlowMatchingActionHead` — full hidden sequence (B, L, H)
    - :class:`LayerwiseFlowMatchingActionHead` — list of per-layer (B, L, H)
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """DoF of the predicted action vector."""

    @property
    @abstractmethod
    def action_horizon(self) -> int:
        """Number of future action steps predicted per chunk."""

    @abstractmethod
    def compute_loss(
        self,
        encoder_hidden_states: Union[Tensor, list[Tensor]],
        actions: Tensor,
        *,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Loss over (context features, ground-truth ``actions``).

        ``state`` (proprioception) and the two masks are consumed by the heads
        that support them; heads without that capability ignore them.
        """

    @abstractmethod
    def predict_action(
        self,
        encoder_hidden_states: Union[Tensor, list[Tensor]],
        *,
        state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict a normalized action chunk (B, action_horizon, action_dim)."""

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Match the lightweight backbone interface scanned by ``BaseArchitecture``."""
        self.to(dtype=dtype, device=device)

    def save_deploy_assets(self, output_dir: str, cfg) -> None:
        """Same hook as backbones (default no-op — heads have no side files)."""

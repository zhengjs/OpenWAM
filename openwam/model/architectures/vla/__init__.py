"""VLA architecture family (``framework=vla``).

Adapted from starVLA (starVLA/model/framework/VLM4A/): QwenOFT / QwenPI /
QwenGR00T — VLM-conditioned action prediction over multi-view images plus a
language instruction, sharing the family-agnostic :class:`BaseArchitecture`
root with the WAM families (no video stream).

    from openwam.model import build_architecture
    arch = build_architecture("vla_qwen_oft", cfg)
"""

from openwam.model.architectures.vla.base_vla import BaseVLAArchitecture
from openwam.model.architectures.vla.qwen_groot import QwenGR00TArchitecture
from openwam.model.architectures.vla.qwen_oft import QwenOFTArchitecture
from openwam.model.architectures.vla.qwen_pi import QwenPIArchitecture

__all__ = [
    "BaseVLAArchitecture",
    "QwenOFTArchitecture",
    "QwenPIArchitecture",
    "QwenGR00TArchitecture",
]

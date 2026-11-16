"""First-frame conditioning derivation (formerly ``transforms/pipeline.py``).

Reads a sample dict's ``video`` and fills the first-frame conditioning fields
the Wan video backbones consume (TI2V reference frame / VACE spatial reference).
"""


class FirstFrameConditioningTransform:
    """Add first-frame conditioning fields to a sample.

    - ``first_frame_image``: first frame, used as TI2V first-frame condition on
      Wan2.2-TI2V backbones or VACE spatial reference on Wan2.1-VACE backbones.
      The OpenWAM layer stays backend-agnostic; the mapping to the Wan
      pipeline's ``vace_reference_image`` input happens at the pipeline boundary.
    - ``vace_video``: None (inactive conditioning by default).
    """

    def __init__(self, use_first_frame_as_reference: bool = True):
        self.use_first_frame_as_reference = use_first_frame_as_reference

    def apply(self, data: dict) -> dict:
        if "vace_video" not in data:
            data["vace_video"] = None

        if "first_frame_image" not in data:
            if self.use_first_frame_as_reference and "video" in data and data["video"]:
                data["first_frame_image"] = [data["video"][0]]
            else:
                data["first_frame_image"] = None

        return data

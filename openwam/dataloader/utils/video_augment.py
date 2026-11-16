"""Training-time video color jitter (formerly ``transforms/video.py``).

Pure PIL helpers operating on ``List[PIL.Image]``; all random parameters are
drawn once per sample and applied across all frames (temporal consistency).
"""

import random

from PIL import Image, ImageEnhance, ImageOps, ImageStat


def color_jitter_enabled(color_jitter) -> bool:
    """Whether a yaml ``color_jitter`` value asks for jitter.

    ``None`` / ``False`` / ``{}`` → disabled. A dict may also disable itself
    explicitly with ``enabled: false`` while keeping its strength knobs.
    """
    if not color_jitter:
        return False
    if hasattr(color_jitter, "get") and not color_jitter.get("enabled", True):
        return False
    return True


class VideoColorJitter:
    """Random color jitter applied consistently across all frames.

    Only active during training. Callers may provide one PIL ``L`` mask per
    frame under ``video_jitter_exclusion_masks``; white regions are excluded
    from contrast statistics and restored after augmentation.

    Args:
        brightness: Max brightness change factor.
        contrast: Max contrast change factor.
        saturation: Max saturation change factor.
        hue: Max hue change (not applied in PIL-only mode).
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.0,
        training: bool = True,
    ):
        self.training = training
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(self, data: dict) -> dict:
        """Same dict interface the readers already call; kept verbatim."""
        if not self.training:
            return data

        if "video" not in data or not data["video"]:
            return data

        exclusion_masks = data.get("video_jitter_exclusion_masks")
        if exclusion_masks is not None and len(exclusion_masks) != len(data["video"]):
            raise ValueError(
                "video_jitter_exclusion_masks must have the same length as video: "
                f"{len(exclusion_masks)} != {len(data['video'])}"
            )

        # Sample random factors (same for all frames)
        b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        s_factor = 1.0 + random.uniform(-self.saturation, self.saturation)

        augmented = []
        for index, frame in enumerate(data["video"]):
            original = frame
            exclusion_mask = exclusion_masks[index] if exclusion_masks is not None else None
            if exclusion_mask is not None and exclusion_mask.size != frame.size:
                raise ValueError(
                    f"video jitter exclusion mask must match its frame size: {exclusion_mask.size} != {frame.size}"
                )

            frame = ImageEnhance.Brightness(frame).enhance(b_factor)
            if exclusion_mask is None:
                frame = ImageEnhance.Contrast(frame).enhance(c_factor)
            else:
                # PIL contrast blends against the image's grayscale mean. Compute
                # that mean from valid camera slots only, so structural black
                # padding neither changes nor biases the visible views.
                valid_mask = ImageOps.invert(exclusion_mask.convert("L"))
                stats = ImageStat.Stat(frame.convert("L"), mask=valid_mask)
                if stats.count[0] > 0:
                    mean = int(stats.mean[0] + 0.5)
                    baseline = Image.new("L", frame.size, mean).convert(frame.mode)
                    frame = Image.blend(baseline, frame, c_factor)
            frame = ImageEnhance.Color(frame).enhance(s_factor)
            if exclusion_mask is not None:
                # Restore excluded slots after all operations. Valid-view black
                # pixels are intentionally unmasked and still receive jitter.
                frame.paste(original, mask=exclusion_mask.convert("L"))
            augmented.append(frame)

        data["video"] = augmented
        return data

import math

import torch
import torch.nn.functional as F


def _normalized_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point torch.Tensor in [0, 1].")
    # Work in float32 even for half/bfloat16 inputs, without modifying the source.
    return torch.nan_to_num(value.detach().float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


def _validate_number(name, value, minimum, maximum):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


class AdvancedAlphaProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "invert_alpha": (["enable", "disable"], {"default": "enable"}),
                "midrange_cut": (["disable", "enable"], {"default": "disable"}),
                "cut_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma_correction": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 3.0, "step": 0.1}),
                "remove_black_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_grayscale": (["enable", "disable"], {"default": "enable"}),
            },
            "optional": {
                "alpha_mode": (["legacy", "straight"], {
                    "default": "legacy",
                    "tooltip": "legacy preserves the original RGB calculation. Use straight for PNG export and compositing without dark fringes.",
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional transparency mask, e.g. Load Image MASK: 1 = transparent, 0 = opaque. Multiplied with the generated alpha and any input RGBA alpha.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Output", "Black_Removed")
    CATEGORY = "Image Processing"
    FUNCTION = "convert"
    DESCRIPTION = "Generate RGBA images from luminance, optionally removing black areas. Preserves existing transparency."

    def convert(self, image, invert_alpha, midrange_cut, cut_threshold,
                gamma_correction, remove_black_threshold, force_grayscale,
                alpha_mode="legacy", mask=None):
        for name, value in (("invert_alpha", invert_alpha), ("midrange_cut", midrange_cut),
                            ("force_grayscale", force_grayscale)):
            if value not in ("enable", "disable"):
                raise ValueError(f"{name} must be 'enable' or 'disable'.")
        if alpha_mode not in ("legacy", "straight"):
            raise ValueError("alpha_mode must be 'legacy' or 'straight'.")
        _validate_number("cut_threshold", cut_threshold, 0.0, 1.0)
        _validate_number("remove_black_threshold", remove_black_threshold, 0.0, 1.0)
        _validate_number("gamma_correction", gamma_correction, 1.0, 3.0)
        if (not isinstance(image, torch.Tensor) or image.ndim != 4
                or image.shape[-1] not in (1, 3, 4) or any(size == 0 for size in image.shape)):
            raise ValueError("image must have non-empty shape [batch, height, width, channels] with 1, 3 or 4 channels.")

        batch = _normalized_tensor(image, "image")
        source_rgb = batch[..., :3]
        if batch.shape[-1] == 1:
            source_rgb = source_rgb.expand(-1, -1, -1, 3)

        source_alpha = batch[..., 3] if batch.shape[-1] == 4 else None
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
                raise ValueError("mask must have shape [height, width] or [batch, height, width].")
            mask = _normalized_tensor(mask, "mask").to(batch.device)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if any(size == 0 for size in mask.shape) or mask.shape[0] not in (1, batch.shape[0]):
                raise ValueError("mask must be non-empty and have one frame or the same batch size as image.")
            if mask.shape[1:] != batch.shape[1:3]:
                # Also handles Load Image's 64x64 empty mask for opaque images.
                mask = F.interpolate(mask.unsqueeze(1), size=batch.shape[1:3],
                                     mode="bilinear", align_corners=False).squeeze(1)
            opacity = 1.0 - mask
            source_alpha = opacity if source_alpha is None else source_alpha * opacity

        linear_rgb = source_rgb.pow(gamma_correction)
        if force_grayscale == "enable":
            gray = (linear_rgb * batch.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1, keepdim=True)
            linear_rgb = gray.clamp(0, 1).expand(-1, -1, -1, 3)

        # Keep the original luminance coefficients and threshold order for RGB workflows.
        alpha = (source_rgb * batch.new_tensor([0.299, 0.587, 0.114])).sum(dim=-1).clamp(0, 1)
        if invert_alpha == "enable":
            alpha = 1.0 - alpha
        if midrange_cut == "enable":
            alpha = (alpha >= cut_threshold).float()
        # Apply source transparency AFTER thresholding so hidden pixels stay hidden.
        if source_alpha is not None:
            alpha = alpha * source_alpha
        black_mask = (source_rgb < remove_black_threshold).all(dim=-1)
        processed_alpha = alpha.masked_fill(black_mask, 0.0)

        straight_rgb = linear_rgb.pow(1.0 / gamma_correction) if alpha_mode == "straight" else None

        def compose_rgba(opacity):
            if alpha_mode == "legacy":
                rgb = (linear_rgb * opacity.unsqueeze(-1)).pow(1.0 / gamma_correction)
            else:
                # Straight alpha is the representation expected by PNG/compositors.
                rgb = torch.where(opacity.unsqueeze(-1) > 0, straight_rgb, 0.0)
            return torch.cat((rgb, opacity.unsqueeze(-1)), dim=-1).clamp(0, 1)

        return (compose_rgba(alpha), compose_rgba(processed_alpha))


NODE_CLASS_MAPPINGS = {
    "AdvancedAlphaProcessor": AdvancedAlphaProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAlphaProcessor": "Advanced Alpha Processor with Black Removal",
}

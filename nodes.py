import torch

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
                "force_grayscale": (["enable", "disable"], {"default": "enable"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Output", "Black_Removed")
    CATEGORY = "Image Processing"
    FUNCTION = "convert"

    def compute_alpha(self, rgb: torch.Tensor, invert: str) -> torch.Tensor:
        # RGB to Luminance conversion
        alpha = torch.matmul(rgb, torch.tensor([0.299, 0.587, 0.114], device=rgb.device))
        return torch.clamp(1.0 - alpha, 0, 1) if invert == "enable" else alpha

    def apply_gamma(self, tensor: torch.Tensor, gamma: float) -> torch.Tensor:
        return torch.clamp(tensor.pow(gamma), 0.0, 1.0)

    def convert(self, image: torch.Tensor, invert_alpha: str, midrange_cut: str, 
                cut_threshold: float, gamma_correction: float, 
                remove_black_threshold: float, force_grayscale: str) -> tuple:
        
        device = image.device
        B, H, W, C = image.shape
        
        # Gamma correction
        linear_image = self.apply_gamma(image, gamma_correction)

        # Grayscale conversion
        if force_grayscale == "enable":
            gray = torch.matmul(linear_image[..., :3], 
                              torch.tensor([0.2126, 0.7152, 0.0722], device=device))
            rgb = gray.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            rgb = linear_image[..., :3]

        # Alpha channel processing
        alpha = self.compute_alpha(image[..., :3], invert_alpha)
        
        # Midrange cutting
        if midrange_cut == "enable":
            processed_alpha = (alpha > cut_threshold).float()
        else:
            processed_alpha = alpha.clone()

        # Black removal mask
        black_mask = torch.all(image[..., :3] < remove_black_threshold, dim=-1)
        final_alpha = torch.where(black_mask, 0.0, processed_alpha)

        # Compose output images
        def compose_output(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            premult = rgb * alpha.unsqueeze(-1)
            srgb = torch.clamp(premult.pow(1.0/gamma_correction), 0.0, 1.0)
            return torch.cat([srgb, alpha.unsqueeze(-1)], dim=-1)

        orig_output = compose_output(rgb, alpha)
        processed_output = compose_output(rgb, final_alpha)

        return (orig_output, processed_output)
    
NODE_CLASS_MAPPINGS = {
    "AdvancedAlphaProcessor": AdvancedAlphaProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAlphaProcessor": "Advanced Alpha Processor with Black Removal"
}

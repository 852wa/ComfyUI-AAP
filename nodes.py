import numpy as np
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
                "remove_black_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "force_grayscale": (["enable", "disable"], {"default": "enable"})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Output", "Black_Removed")
    CATEGORY = "Image Processing"
    FUNCTION = "convert"

    def convert(self, image, invert_alpha, midrange_cut, cut_threshold, gamma_correction, remove_black_threshold, force_grayscale):
        batch = image.cpu().numpy()
        results_orig = []
        results_processed = []
        
        for img in batch:
            # ガンマ補正適用
            img_linear = np.power(img, gamma_correction)
            
            # グレースケール変換
            if force_grayscale == "enable":
                gray = np.dot(img_linear[..., :3], [0.2126, 0.7152, 0.0722])
                gray = np.clip(gray, 0, 1)
                rgb = np.stack([gray]*3, axis=-1)
            else:
                rgb = img_linear[..., :3]
            
            # アルファチャンネル生成
            alpha = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            if invert_alpha == "enable":
                alpha = 1.0 - alpha
                
            # 中間色カット処理
            if midrange_cut == "enable":
                processed_alpha = np.where(alpha >= cut_threshold, 1.0, 0.0)
            else:
                processed_alpha = alpha.copy()
            
            # 元のアルファを保持
            original_alpha = processed_alpha.copy()
            
            # 黒領域除去処理
            black_mask = np.all(img[..., :3] < remove_black_threshold, axis=-1)
            processed_alpha[black_mask] = 0.0
            
            # 事前乗算アルファ処理
            def compose_rgba(rgb, alpha):
                premult_rgb = rgb * alpha[..., np.newaxis]
                rgba = np.concatenate([
                    np.power(premult_rgb, 1.0/gamma_correction),
                    alpha[..., np.newaxis]
                ], axis=-1)
                return rgba
            
            # オリジナルと処理済みを生成
            rgba_orig = compose_rgba(rgb, original_alpha)
            rgba_processed = compose_rgba(rgb, processed_alpha)
            
            results_orig.append(rgba_orig)
            results_processed.append(rgba_processed)
        
        tensor_orig = torch.from_numpy(np.array(results_orig)).float()
        tensor_processed = torch.from_numpy(np.array(results_processed)).float()
        
        return (tensor_orig, tensor_processed)

NODE_CLASS_MAPPINGS = {
    "AdvancedAlphaProcessor": AdvancedAlphaProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAlphaProcessor": "Advanced Alpha Processor with Black Removal"
}

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
        """
        RGB画像から輝度ベースのアルファチャンネルを計算。

        Args:
            rgb (torch.Tensor): 入力RGB画像テンソル。形状は[..., 3]である必要がある。
            invert (str): アルファチャンネルを反転するかどうかを指定する。"enable"の場合は輝度の逆数を、"disable"の場合は輝度をアルファ値とする。

        Returns:
            torch.Tensor: 計算されたアルファチャンネル。形状は入力rgbと同じですが、最後の次元が1になる。
        """
        # RGBから輝度を計算
        alpha = torch.matmul(rgb, torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device)) 
        # invertが"enable"の場合、輝度の逆数をアルファ値とする
        return torch.clamp(1.0 - alpha, 0, 1) if invert == "enable" else torch.clamp(alpha, 0, 1)

    def apply_gamma(self, tensor: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        テンソルにガンマ補正を適用する。

        Args:
            tensor (torch.Tensor): 入力テンソル。
            gamma (float): ガンマ値。

        Returns:
            torch.Tensor: ガンマ補正されたテンソル。クランプされて0から1の範囲に収められる。
        """
        return torch.clamp(tensor.pow(1.0 / gamma), 0.0, 1.0)

    def convert(self, image: torch.Tensor, invert_alpha: str, midrange_cut: str, 
                cut_threshold: float, gamma_correction: float, 
                remove_black_threshold: float, force_grayscale: str) -> tuple:
        """
        高度なアルファチャンネル処理を実行し、オプションで黒領域を除去。

        Args:
            image (torch.Tensor): 入力画像テンソル。形状は[B, H, W, C]である必要がある。
            invert_alpha (str): アルファチャンネルを反転するかどうかを指定する。"enable"または"disable"。
            midrange_cut (str): 中間色カットを適用するかどうかを指定する。"enable"または"disable"。
            cut_threshold (float): 中間色カットの閾値。
            gamma_correction (float): ガンマ補正値。
            remove_black_threshold (float): 黒領域除去の閾値。
            force_grayscale (str): 画像を強制的にグレースケールに変換するかどうかを指定する。"enable"または"disable"。

        Returns:
            tuple: 処理された画像と黒領域が除去された画像のタプル。
        """
        
        device = image.device
        _B, _H, _W, _C = image.shape
        
        # ガンマ補正を適用してリニアRGB空間に変換する
        linear_image = self.apply_gamma(image, gamma_correction)

        # グレースケール変換
        if force_grayscale == "enable":
            # RGBをグレースケールに変換
            gray = torch.matmul(linear_image[..., :3], 
                                torch.tensor([0.2126, 0.7152, 0.0722], device=device))
            # グレースケール画像を3チャンネルに複製してRGB画像を作成
            rgb = gray.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            # 入力画像のRGBチャンネルをそのまま使用
            rgb = linear_image[..., :3]

        # アルファチャンネル生成
        base_alpha = self.compute_alpha(image[..., :3], invert_alpha)
        
        # 中間色カット処理
        if midrange_cut == "enable":
            # 閾値以上のアルファ値を1、それ以外を0に設定し、中間色を削除
            processed_alpha = (base_alpha >= cut_threshold).float()
        else:
            # 中間色カットを適用しない場合は、元のアルファ値をそのまま使用
            processed_alpha = base_alpha.clone()

        # 黒領域除去処理
        # RGB値がすべて閾値未満のピクセルを黒と判定
        black_mask = torch.all(image[..., :3] < remove_black_threshold, dim=-1)
        # 黒領域のアルファ値を0に設定し、それ以外の領域はprocessed_alphaの値を使用
        final_alpha = torch.where(black_mask, 0.0, processed_alpha)

        # 事前乗算アルファ処理
        def compose_output(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
            """
            RGB画像とアルファチャンネルを合成し、事前乗算アルファを適用する。

            Args:
                rgb (torch.Tensor): RGB画像テンソル。
                alpha (torch.Tensor): アルファチャンネルテンソル。

            Returns:
                torch.Tensor: 合成された画像テンソル。
            """
            # RGBにアルファを乗算して事前乗算アルファを適用
            premult = rgb * alpha.unsqueeze(-1)
            # ガンマ補正を適用してsRGB空間に戻す
            srgb = torch.clamp(premult.pow(gamma_correction), 0.0, 1.0)
            # sRGB画像とアルファチャンネルを結合して最終的な画像を作成
            return torch.cat([srgb, alpha.unsqueeze(-1)], dim=-1)

        # 中間色カット処理および黒除去処理前の画像と、処理後の画像を生成
        orig_output = compose_output(rgb, processed_alpha)
        processed_output = compose_output(rgb, final_alpha)

        return (orig_output, processed_output)

NODE_CLASS_MAPPINGS = {
    "AdvancedAlphaProcessor": AdvancedAlphaProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedAlphaProcessor": "Advanced Alpha Processor with Black Removal"
}
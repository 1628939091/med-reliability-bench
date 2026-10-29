"""扰动套件：这是论文贡献①"模态依赖性"的实验材料。

identity  : 原图（对照组）
blank     : 全灰图 —— 检测"模态坍缩"：若模型答对率不降，说明它根本没看图
noise     : 加随机噪声 —— 鲁棒性
flip      : 上下翻转 —— 空间理解（医学影像方向性很重要）
swap      : 换成别的题的图 —— 对抗性检查：答案不该变对
low_res   : 降采样 —— 分辨率敏感度
text_only : 不给图 —— 语言捷径探测的对照组

实现说明：优先用 PIL 做像素级扰动；PIL 缺失时 blank/text_only/swap 仍可用
（标准库即可实现），noise/flip/low_res 会降级为 identity 并打警告。
"""

from __future__ import annotations

import os
import random

from .data import write_png

ALL_PERTURBATIONS = ["identity", "blank", "noise", "flip", "swap", "low_res", "text_only"]

try:
    from PIL import Image  # noqa: F401
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def build_perturbations(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    return PerturbationSuite(cache_dir)


class PerturbationSuite:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._degraded: list[str] = []
        if not HAS_PIL:
            self._degraded = ["noise", "flip", "low_res"]

    def apply(self, name: str, image_path: str, sample_id: str,
              other_image: str | None = None) -> str:
        """返回扰动后的图片路径；text_only 返回空串表示不提供图。"""
        if name == "text_only":
            return ""
        if name == "identity" or name not in ALL_PERTURBATIONS:
            return image_path
        if name == "blank":
            out = os.path.join(self.cache_dir, f"{sample_id}_blank.png")
            write_png(out, 64, 64, lambda x, y: (128, 128, 128))
            return out
        if name == "swap":
            return other_image or image_path
        if not HAS_PIL:
            # 降级：无法做像素操作时退回原图（报告里会标注）
            return image_path

        from PIL import Image, ImageFilter
        img = Image.open(image_path).convert("RGB")
        if name == "noise":
            rng = random.Random(f"noise|{sample_id}")
            px = img.load()
            for yy in range(img.height):
                for xx in range(img.width):
                    r, g, b = px[xx, yy]
                    n = rng.randint(-60, 60)
                    px[xx, yy] = (max(0, min(255, r + n)),
                                  max(0, min(255, g + n)),
                                  max(0, min(255, b + n)))
            out = os.path.join(self.cache_dir, f"{sample_id}_noise.png")
            img.save(out)
            return out
        if name == "flip":
            out = os.path.join(self.cache_dir, f"{sample_id}_flip.png")
            img.transpose(Image.FLIP_TOP_BOTTOM).save(out)
            return out
        if name == "low_res":
            small = img.resize((img.width // 8, img.height // 8),
                               Image.BILINEAR)
            out = os.path.join(self.cache_dir, f"{sample_id}_lowres.png")
            small.resize(img.size, Image.NEAREST).save(out)
            return out
        return image_path

    @property
    def degraded(self) -> list[str]:
        return list(self._degraded)

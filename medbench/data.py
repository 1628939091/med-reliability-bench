"""数据层：样例数据集生成 + JSONL 加载。

真实实验时，把 SLAKE / VQA-RAD / PathVQA 转成同样的 JSONL 格式即可，
格式：{"id": ..., "question": ..., "options": ["A","B"], "answer": "A",
       "image": "images/xxx.png", "category": "image_dependent|text_prior"}
"""

from __future__ import annotations

import json
import os
import random
import struct
import zlib
from dataclasses import dataclass, field


@dataclass
class Sample:
    sid: str
    question: str
    options: list[str]
    answer: str
    image: str          # 图片相对路径（text_only 场景仍会保留图片字段）
    category: str       # image_dependent=看图才能答 / text_prior=靠语言先验即可猜
    # 分层统计用的元数据（器官/模态/语言/答案类型/同一图像 id），
    # 真实数据集由 adapters.py 写入；演示集为空 dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.sid, "question": self.question,
            "options": self.options, "answer": self.answer,
            "image": self.image, "category": self.category,
            "meta": self.meta,
        }


def load_dataset(path: str) -> list[Sample]:
    """从 JSONL 文件加载评测集。"""
    samples = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            img = d.get("image")
            if img and not os.path.isabs(img) and not os.path.exists(
                os.path.join(base, img)
            ):
                img = os.path.join(base, img) if os.path.exists(
                    os.path.join(base, img)) else img
            samples.append(Sample(
                sid=d["id"], question=d["question"], options=d["options"],
                answer=d["answer"], image=d.get("image", ""),
                category=d.get("category", "image_dependent"),
                meta=d.get("meta") or {},
            ))
    return samples


# --------------------------------------------------------------------------
# 纯标准库的最小 PNG 写出器（避免强制依赖 PIL，样例图可直接生成）
# --------------------------------------------------------------------------
def write_png(path: str, width: int, height: int, pixel_fn) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (struct.pack(">I", len(data)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            r, g, b = pixel_fn(x, y)
            raw.extend((r & 0xFF, g & 0xFF, b & 0xFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw))))
        f.write(chunk(b"IEND", b""))


# --------------------------------------------------------------------------
# 生成一个 40 题的演示数据集（结构对齐真实医学 VQA：二元选择 + 影像图）
# --------------------------------------------------------------------------
_TPL_IMG = [
    ("胸部CT显示右肺下叶结节边缘毛刺征", "影像边界是否模糊不清？"),
    ("眼底彩照显示视盘边界清晰", "视盘是否存在水肿表现？"),
    ("病理切片HE染色，腺体结构异常", "是否提示腺体结构异型？"),
    ("MRI T2加权像显示高信号病灶", "病灶信号是否高于周围组织？"),
]

_TPL_TEXT = [
    ("阿司匹林的常见不良反应是胃肠道刺激", "阿司匹林是否属于抗血小板药物？"),
    ("2型糖尿病患者的空腹血糖通常升高", "1型糖尿病以胰岛素抵抗为主要病因？"),
    ("高血压诊断阈值一般为140/90 mmHg", "正常血压是否定义为低于120/80 mmHg？"),
    ("细菌性感染常引起白细胞升高", "抗生素对病毒性感冒是否直接有效？"),
]


def generate_sample_dataset(out_dir: str, n_per_cat: int = 20, seed: int = 42) -> str:
    """生成演示数据集：n_per_cat*2 道二元选择题 + 每题一张 64x64 的合成"医学影像"。

    图片内容与答案有对应关系（模拟真实 VQA 中"图里才有的信息"），
    这样接入真实模型后，扰动实验才具备可解释性。
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    rng = random.Random(seed)
    samples = []
    pid = 0

    # ---- 影像依赖题：答案由"图中病灶模式"决定 ----
    for i in range(n_per_cat):
        desc, q = _TPL_IMG[i % len(_TPL_IMG)]
        answer = rng.choice(["A", "B"])
        pid += 1
        img_rel = f"images/img_{pid:03d}.png"
        img_abs = os.path.join(out_dir, img_rel)
        _write_lesion_png(img_abs, malignant=(answer == "A"))
        samples.append({
            "id": f"imgdep-{pid:03d}", "question": q,
            "options": ["A", "B"], "answer": answer,
            "image": img_rel, "category": "image_dependent",
        })

    # ---- 文本先验题：不需要图也能答 ----
    for i in range(n_per_cat):
        desc, q = _TPL_TEXT[i % len(_TPL_TEXT)]
        answer = rng.choice(["A", "B"])
        pid += 1
        img_rel = f"images/img_{pid:03d}.png"
        _write_lesion_png(os.path.join(out_dir, img_rel), malignant=False)
        samples.append({
            "id": f"txtpri-{pid:03d}", "question": q,
            "options": ["A", "B"], "answer": answer,
            "image": img_rel, "category": "text_prior",
        })

    jsonl_path = os.path.join(out_dir, "dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return jsonl_path


def _write_lesion_png(path: str, malignant: bool) -> None:
    """合成一张简单"影像"：良性=均匀灰底+亮斑；恶性=不均匀纹理+毛刺状暗纹。"""
    def px(x, y):
        base = 160
        if malignant:
            stripe = 40 if (x * 7 + y * 3) % 16 < 4 else 0
            blob = 60 if (x - 32) ** 2 + (y - 32) ** 2 < 120 else 0
            return (base - stripe + blob, base - stripe + blob // 2, base - stripe)
        blob = 80 if (x - 32) ** 2 + (y - 32) ** 2 < 150 else 0
        return (base + blob, base + blob, base + blob)

    write_png(path, 64, 64, px)

"""
medbench — 医学多模态 VLM 可靠性评测框架
========================================
四个评测维度（对应论文的四个实验）：
  1. 模态依赖性  : 干净图 vs 扰动图（空白/噪声/翻转/换图）的准确率差 Δ
  2. 语言捷径    : 纯文本输入 与 图+文输入 的答案一致率
  3. 答案稳定性  : 同题重复采样 k 次的 pass^k / 答案熵
  4. 校准性      : 模型自报置信度 vs 实际正确率 (ECE / 可靠性图数据)

设计原则：所有模型都通过同一个 answer() 接口调用，真实模型走 API，
Mock 模型用于本地无 API 调试 —— 二者跑的是同一套评测代码。
"""

from .data import Sample, load_dataset, generate_sample_dataset
from .models import build_model
from .perturb import build_perturbations
from .runner import run_eval
from .report import render_report

__all__ = [
    "Sample", "load_dataset", "generate_sample_dataset",
    "build_model", "build_perturbations",
    "run_eval", "render_report",
]

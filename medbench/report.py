"""报告生成：把 summary 渲染成 Markdown（论文表格的直接底稿）。"""

from __future__ import annotations

import os
from datetime import datetime


def render_report(result: dict, out_path: str) -> str:
    s = result["summary"]
    k = result["k"]
    lines = [
        "# MedReliability-Bench 评测报告",
        f"\n生成时间：{datetime.now():%Y-%m-%d %H:%M} | 重复采样次数 k={k}",
        "\n## 一、各模型核心指标总览\n",
        "| 模型 | 准确率(干净图) | 95%CI | 坍缩差 Δ | 语言捷径一致率 | pass^k | 多数投票acc | 回答熵 | ECE | 拒答率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in s.items():
        lo, hi = m["ci_identity"]
        lines.append(
            f"| {name} | {m['acc_identity']:.3f} | [{lo:.3f},{hi:.3f}] "
            f"| {m['collapse_delta']:+.3f} | {m['shortcut_agreement']:.3f} "
            f"| {m['pass_k']:.3f} | {m['majority_acc']:.3f} "
            f"| {m['answer_entropy']:.3f} | {m['ece']:.3f} "
            f"| {m.get('refusal_rate', 0.0):.1%} |")

    lines += [
        "\n**指标解读（写给第一次看的你）**",
        "- **坍缩差 Δ**：干净图准确率 − 空白图准确率。模型真的在看图时 Δ 显著>0；"
        "Δ≈0 说明它只靠文字先验在猜（模态坍缩），这在多篇 2025–2026 医学 VQA 研究中被证实存在。",
        "- **语言捷径一致率**：不给图和给图，答案一样吗？一致率≈1 的模型大概率在做语言捷径。",
        "- **pass^k**：同一道题问 k 次全部答对的概率。医疗场景要的是稳定正确，不是碰运气。",
        "- **ECE**：校准误差。模型说 95% 置信时是否真有 95% 正确？医疗部署的前置条件。",
        "- **拒答率**：平台内容审核（HTTP 451）把医学影像判为敏感内容而拒绝作答的比例。"
        "拒答按答错计，但它是独立指标——医疗 AI 走第三方 API 时的真实部署摩擦。",
    ]

    lines += ["\n## 二、模态依赖性：各扰动下的准确率\n",
              "| 模型 | 干净图 | 空白图 | 噪声 | 翻转 | 换图 | 低分辨率 | 纯文本 |",
              "|---|---|---|---|---|---|---|---|"]
    for name, m in s.items():
        ap = m["acc_by_pert"]
        lines.append("| {n} | {identity:.3f} | {blank:.3f} | {noise:.3f} "
                     "| {flip:.3f} | {swap:.3f} | {low_res:.3f} "
                     "| {text_only:.3f} |".format(
                         n=name, **{p: ap.get(p, float("nan")) for p in
                                    ["identity", "blank", "noise", "flip",
                                     "swap", "low_res", "text_only"]}))

    # 自动结论（演示用；真实实验中这部分就是论文的 Analysis 小节素材）
    lines += ["\n## 三、自动诊断\n"]
    for name, m in s.items():
        diag = []
        if m["collapse_delta"] < 0.05:
            diag.append("**疑似模态坍缩**（Δ≈0）：模型对图像扰动不敏感")
        if m["shortcut_agreement"] > 0.9:
            diag.append("**强语言捷径**：图文输入与纯文本输入答案几乎一致")
        if m["pass_k"] < m["acc_identity"] - 0.15:
            diag.append("**答案不稳定**：pass^k 显著低于 pass@1，重复提问会翻车")
        if m["ece"] > 0.15:
            diag.append("**校准差**（ECE>0.15）：置信度不可信，临床部署风险高")
        lines.append(f"- **{name}**：{'；'.join(diag) if diag else '各项指标正常'}")

    if result.get("degraded_perturbations"):
        lines.append("\n> ⚠️ 注意：本机缺少 Pillow，以下扰动被降级为原图："
                     + ", ".join(result["degraded_perturbations"])
                     + "。`pip install pillow` 后重跑即可获得完整结果。")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path

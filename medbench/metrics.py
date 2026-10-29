"""指标层 —— 论文的核心统计武器库。

 accuracy / bootstrap CI     : 基础准确率 + 95% 置信区间
 collapse_delta              : acc(干净图) - acc(空白图)，≈0 即"模态坍缩"
 shortcut_agreement          : 纯文本答案 与 图+文答案 的一致率（语言捷径强度）
 pass_at_k                   : k 次重复全部答对的题占比（tau-bench 式可靠性）
 answer_entropy              : 同一题 k 次回答的分散程度
 ece + reliability_bins      : 期望校准误差 + 可靠性图数据
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(correct_flags, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0):
    """对准确率做 Bootstrap 置信区间 —— 论文里所有准确率都要带 CI。"""
    flags = list(correct_flags)
    n = len(flags)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [flags[rng.randrange(n)] for _ in range(n)]
        stats.append(mean(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean(flags), lo, hi


def collapse_delta(acc_by_pert: dict) -> float:
    """模态坍缩指标：干净图准确率减去空白图准确率。正常模型应显著 > 0。"""
    return acc_by_pert.get("identity", 0.0) - acc_by_pert.get("blank", 0.0)


def shortcut_agreement(records) -> float:
    """语言捷径强度：同一题在 identity 与 text_only 下答案一致的比例。"""
    ans = defaultdict(dict)
    for r in records:
        if r["perturbation"] in ("identity", "text_only"):
            ans[r["sid"]][r["perturbation"]] = r["pred"]
    both = [v for v in ans.values() if "identity" in v and "text_only" in v]
    if not both:
        return 0.0
    return mean(1.0 if v["identity"] == v["text_only"] else 0.0 for v in both)


def pass_at_k(records, k: int) -> float:
    """pass^k：每道题 k 次重复全部答对的比例。暴露"单次pass@1高但反复问会翻车"的问题。"""
    by_sample = defaultdict(list)
    for r in records:
        if r["perturbation"] == "identity":
            by_sample[r["sid"]].append(bool(r["correct"]))
    eligible = {s: v for s, v in by_sample.items() if len(v) >= k}
    if not eligible:
        return 0.0
    return mean(1.0 if all(v[:k]) else 0.0 for v in eligible.values())


def answer_entropy(records) -> float:
    """k 次回答的平均熵：熵越高 = 越不稳定。"""
    by_sample = defaultdict(list)
    for r in records:
        if r["perturbation"] == "identity":
            by_sample[r["sid"]].append(r["pred"])
    ents = []
    for preds in by_sample.values():
        cnt = Counter(preds)
        total = sum(cnt.values())
        ents.append(-sum((c / total) * math.log2(c / total) for c in cnt.values()))
    return mean(ents) if ents else 0.0


def ece(confidences, correct_flags, n_bins: int = 10) -> float:
    """期望校准误差（Expected Calibration Error）。
    ECE 大 = 模型说"我 95% 确定"时实际远没有 95% 对 —— 医疗场景的致命缺陷。"""
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct_flags):
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, ok))
    total = len(confidences)
    err = 0.0
    for b in bins:
        if b:
            err += (len(b) / total) * abs(mean(ok for _, ok in b) - mean(c for c, _ in b))
    return err


def reliability_bins(confidences, correct_flags, n_bins: int = 10):
    """可靠性图数据：每个 bin 的 (平均置信度, 实际准确率, 样本占比)。"""
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct_flags):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    out = []
    for b in bins:
        if b:
            out.append({"conf": mean(c for c, _ in b),
                        "acc": mean(ok for _, ok in b),
                        "frac": len(b) / len(confidences)})
    return out


def majority_vote_acc(records, k: int) -> float:
    """多数投票准确率：k 次采样取众数 —— 便宜的自洽性提升手段。"""
    by_sample = defaultdict(list)
    for r in records:
        if r["perturbation"] == "identity":
            by_sample[r["sid"]].append((r["pred"], r["gold"]))
    res = []
    for pairs in by_sample.values():
        k_p = pairs[:k]
        if not k_p:
            continue
        top = Counter(p for p, _ in k_p).most_common(1)[0][0]
        gold = k_p[0][1]          # 记录自带的 gold 标签（不是第一次的预测）
        res.append(1.0 if top == gold else 0.0)
    return mean(res) if res else 0.0

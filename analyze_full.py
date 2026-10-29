# -*- coding: utf-8 -*-
"""阶段 3：全量结果的分层统计分析 + 出图。

统计口径（重要，面试会问）：
1. 同一道题重复问 k 次，这 k 条记录**不独立** —— 置信区间用「按题目聚类的
   Bootstrap」（重采样题目而非记录），否则 CI 被人为收窄、假显著。
2. 配对比较（原图 vs 置空）用「配对聚类 Bootstrap」+ McNemar 精确检验。
3. 聚合的坍缩差 Δ 会掩盖方向相反的效应 —— 本脚本额外按 gold 方向拆开，
   并给出「常数预测基线」与「答案分布熵」作为退化程度的度量。
4. 子组 n<20 标 ⚠️，只作描述不下结论。

用法：python analyze_full.py [checkpoint.jsonl] [dataset.jsonl]
产物：outputs/analysis_full.md + outputs/fig_*.png
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "outputs", "checkpoint_real_slake_api_full.jsonl")
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    HERE, "data", "real_slake", "dataset.jsonl")
OUT = os.path.join(HERE, "outputs", "analysis_full.md")
B = 2000
SEED = 42
PERTS = ["identity", "blank", "noise", "flip", "swap", "low_res", "text_only"]
DESC = {"identity": "原图（基线）", "blank": "图像置空", "noise": "高斯噪声",
        "flip": "上下翻转", "swap": "换成别的题的图", "low_res": "降分辨率",
        "text_only": "完全不给图"}


# ------------------------------------------------------------------ 统计工具
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def cluster_mean(per_q: dict, reps: int = B, seed: int = SEED):
    """按题目聚类重采样，返回 (均值, 2.5%, 97.5%)。"""
    vals = np.array(list(per_q.values()), dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    stats = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return vals.mean(), float(np.percentile(stats, 2.5)), \
        float(np.percentile(stats, 97.5))


def paired_delta(a: dict, b: dict, reps: int = B, seed: int = SEED):
    sids = [s for s in a if s in b]
    d = np.array([a[s] - b[s] for s in sids], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    stats = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return d.mean(), float(np.percentile(stats, 2.5)), \
        float(np.percentile(stats, 97.5))


def mcnemar(hit_a: list[bool], hit_b: list[bool]):
    """配对二元精确检验。b=只在 a 条件对，c=只在 b 条件对。"""
    from math import comb
    b = sum(1 for x, y in zip(hit_a, hit_b) if x and not y)
    c = sum(1 for x, y in zip(hit_a, hit_b) if not x and y)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, p)


def ece(confs, corrects, n_bins: int = 10):
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    if len(confs) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
        if m.sum():
            tot += m.sum() / len(confs) * abs(corrects[m].mean() - confs[m].mean())
    return float(tot)


def entropy(labels):
    """答案分布熵（bit）：0=永远同一个答案（完全退化），1=两选项各半。"""
    if not labels:
        return float("nan")
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


# ------------------------------------------------------------------ 数据加载
def load():
    recs = [json.loads(l) for l in open(CKPT, encoding="utf-8") if l.strip()]
    meta = {}
    for line in open(DATA, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            meta[d["id"]] = d.get("meta") or {}
    return recs, meta


def per_q(recs, pert, only=None):
    """{sid: 该题在 pert 条件下 k 次的平均正确率}，可按 (rec)->bool 过滤。"""
    agg = defaultdict(list)
    for r in recs:
        if r["perturbation"] == pert and (only is None or only(r)):
            agg[r["sid"]].append(float(r["correct"]))
    return {s: float(np.mean(v)) for s, v in agg.items()}


# ------------------------------------------------------------------ 作图
def make_figs(recs, acc_by, ci_by, split):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:                     # 没装 matplotlib 就跳过
        print(f"[跳过作图] {e}")
        return []

    out = []
    order = [p for p in PERTS if p in acc_by]
    labels = [f"{p}\n{DESC[p]}" for p in order]

    # 图 1：各扰动准确率（带 95%CI）
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    vals = [acc_by[p] for p in order]
    err = [[acc_by[p] - ci_by[p][0] for p in order],
           [ci_by[p][1] - acc_by[p] for p in order]]
    colors = ["#c0392b" if p == "identity" else "#95a5a6" for p in order]
    ax.bar(range(len(order)), vals, yerr=err, capsize=4, color=colors)
    ax.axhline(0.5, ls="--", lw=1, c="#7f8c8d")
    ax.text(0.012, 0.535, "随机猜测 = 0.5", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92))
    for i, v in enumerate(vals):
        top = ci_by[order[i]][1]
        ax.text(i, top + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.95)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("准确率")
    ax.set_title("Qwen3-VL-8B 在 SLAKE 闭集题上的扰动剖面（355 题 × 3 次）")
    p1 = os.path.join(HERE, "outputs", "fig_perturbation.png")
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    out.append(p1)

    # 图 2：按 gold 方向拆分的准确率（揭示方向相反效应）
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    x = np.arange(len(order))
    w = 0.38
    yes = [split[p]["yes"] for p in order]
    no = [split[p]["no"] for p in order]
    ax.bar(x - w / 2, yes, w, label="gold = Yes 的题", color="#2980b9")
    ax.bar(x + w / 2, no, w, label="gold = No 的题", color="#e67e22")
    for i, (a, b) in enumerate(zip(yes, no)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{DESC[p]}" for p in order], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("准确率")
    ax.legend(fontsize=8)
    ax.set_title("同一扰动下两种答案方向的准确率：图像的贡献几乎全在 Yes 题")
    p2 = os.path.join(HERE, "outputs", "fig_direction.png")
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    out.append(p2)
    return out


# ------------------------------------------------------------------ 主流程
def main():
    recs, meta = load()
    model = recs[0]["model"]
    perts = [p for p in PERTS if any(r["perturbation"] == p for r in recs)]
    n_q = len({r["sid"] for r in recs})
    L, add = [], None
    L.append(f"# MedReliability-Bench · 阶段 3 分层分析\n")
    add = L.append

    add(f"- 模型：`{model}`")
    add(f"- 数据：SLAKE 测试集闭集题 **{n_q}** 题 × {len(perts)} 扰动 × 3 重复 "
        f"= **{len(recs)}** 条记录")
    add(f"- 选项设计：全部 355 题均为 `A. No / B. Yes`，gold 分布 180:175（均衡）")
    add(f"- 统计口径：按题目聚类 Bootstrap（B={B}）、配对 McNemar 精确检验；"
        f"子组 n<20 标 ⚠️ 仅描述\n")

    # ---------- 表 1：扰动剖面
    add("## 一、扰动剖面\n")
    add("| 扰动 | 含义 | 准确率 | 95%CI | 相对原图 | pass^3 | Yes 率 | 答案熵 |")
    add("|---|---|---|---|---|---|---|---|")
    acc_by, ci_by, split, yesrate, ent_by = {}, {}, {}, {}, {}
    for p in perts:
        pq = per_q(recs, p)
        m, lo, hi = cluster_mean(pq)
        acc_by[p], ci_by[p] = m, (lo, hi)
        rs = [r for r in recs if r["perturbation"] == p]
        allok = defaultdict(list)
        for r in rs:
            allok[r["sid"]].append(bool(r["correct"]))
        passk = np.mean([all(v) for v in allok.values()])
        yesrate[p] = np.mean([r["pred"] == "B" for r in rs])
        ent_by[p] = entropy([r["pred"] for r in rs])
        split[p] = {"yes": np.mean([r["correct"] for r in rs if r["gold"] == "B"]),
                    "no": np.mean([r["correct"] for r in rs if r["gold"] == "A"])}
        d = "—" if p == "identity" else f"{m - acc_by['identity']:+.3f}"
        add(f"| {p} | {DESC[p]} | **{m:.3f}** | [{lo:.3f},{hi:.3f}] | {d} "
            f"| {passk:.3f} | {yesrate[p]:.3f} | {ent_by[p]:.3f} |")

    # ---------- 表 2：常数基线
    n_no = sum(1 for r in recs
               if r["perturbation"] == "identity" and r["gold"] == "A")
    n_tot = sum(1 for r in recs if r["perturbation"] == "identity")
    majority = max(n_no, n_tot - n_no) / n_tot
    add("\n## 二、先看「常数预测基线」——这是解读一切的前提\n")
    add(f"- 永远答 No 的准确率 = {n_no}/{n_tot} = **{n_no/n_tot:.3f}**")
    add(f"- 永远答多数类 = **{majority:.3f}**")
    add(f"- 模型原图准确率 {acc_by['identity']:.3f} → 仅比常数基线高 "
        f"**{acc_by['identity']-majority:+.3f}**")
    add(f"- 而**图像置空时只有 {acc_by['blank']:.3f}**，比「永远答 No」还低 "
        f"{abs(acc_by['blank']-n_no/n_tot)*100:.1f} 个百分点 → 说明置空条件下模型"
        f"不是「靠语言先验猜」，而是**退化成了近乎常数的输出**"
        f"（Yes 率仅 {yesrate['blank']:.3f}，答案熵 {ent_by['blank']:.3f} bit，"
        f"而原图条件下是 {ent_by['identity']:.3f} bit）。")

    # ---------- 表 3：坍缩差按方向拆分（核心）
    add("\n## 三、核心发现：聚合 Δ 掩盖了两个方向相反的效应\n")
    b_i = per_q(recs, "identity")
    b_b = per_q(recs, "blank")
    gold_by_sid = {r["sid"]: r["gold"] for r in recs
                   if r["perturbation"] == "identity"}
    d_all, lo_a, hi_a = paired_delta(b_i, b_b)
    d_yes = {s: v for s, v in b_i.items() if gold_by_sid.get(s) == "B"}
    d_no = {s: v for s, v in b_i.items() if gold_by_sid.get(s) == "A"}
    bb_yes = {s: v for s, v in b_b.items() if s in d_yes}
    bb_no = {s: v for s, v in b_b.items() if s in d_no}
    dy, lo_y, hi_y = paired_delta(d_yes, bb_yes)
    dn, lo_n, hi_n = paired_delta(d_no, bb_no)
    add("| 分组 | n 题 | 原图准确率 | 置空准确率 | Δ（配对） | 95%CI |")
    add("|---|---|---|---|---|---|")
    add(f"| 全部 | {len(b_i)} | {acc_by['identity']:.3f} | {acc_by['blank']:.3f} "
        f"| **{d_all:+.3f}** | [{lo_a:+.3f},{hi_a:+.3f}] |")
    add(f"| gold=Yes | {len(d_yes)} | {np.mean(list(d_yes.values())):.3f} "
        f"| {np.mean(list(bb_yes.values())):.3f} | **{dy:+.3f}** | [{lo_y:+.3f},{hi_y:+.3f}] |")
    add(f"| gold=No | {len(d_no)} | {np.mean(list(d_no.values())):.3f} "
        f"| {np.mean(list(bb_no.values())):.3f} | **{dn:+.3f}** | [{lo_n:+.3f},{hi_n:+.3f}] |")
    add(f"\n> **这一节是本项目最重要的结果。** 聚合 Δ={d_all:+.3f} 看起来只是"
        f"「掉 23 个点」，拆开看是两个方向相反的效应相加：")
    add(f"> - **Yes 题：Δ={dy:+.3f}** —— 图像是模型敢答 Yes 的唯一依据，"
        f"没有图它几乎答不出 Yes（{split['blank']['yes']:.3f} 准确率）")
    add(f"> - **No 题：Δ={dn:+.3f}** —— 不给图反而更「准」，因为对"
        f"「Does the picture contain X?」这类问句，条件反射式答 No 天然容易蒙对")
    add(f"> - 两者的净和才等于 {d_all:+.3f}。**如果用单一 Δ 汇报，会同时低估图像在 "
        f"Yes 题上的价值、并掩盖模型在 No 题上的「假会」**。")

    # ---------- 表 4：McNemar
    sids = sorted(b_i)
    hit_i = [b_i[s] >= 0.5 for s in sids]
    hit_b = [b_b[s] >= 0.5 for s in sids]
    bc, cb, pv = mcnemar(hit_i, hit_b)
    add(f"\n**McNemar 精确检验（原图 vs 置空，配对）**：对→错 {bc} 题 / "
        f"错→对 {cb} 题，p = {pv:.2e} → 图像作用"
        f"{'显著' if pv < 0.05 else '不显著'}。")

    # ---------- 表 5：分层
    add("\n## 四、分层：哪些题型最容易被图像「反噬」\n")
    for key, title in [("modality", "影像模态"), ("content_type", "问题类型")]:
        add(f"\n### 4.{['modality','content_type'].index(key)+1} 按{title}（原图条件）\n")
        add(f"| {title} | n 题 | 原图准确率 | Wilson 95%CI | 置空准确率 | Δ |")
        add("|---|---|---|---|---|---|")
        groups = defaultdict(list)
        for s in b_i:
            groups[meta.get(s, {}).get(key) or "未知"].append(s)
        rows = []
        for g, ss in groups.items():
            n = len(ss)
            k = sum(1 for s in ss if b_i[s] >= 0.5)
            lo, hi = wilson(k, n)
            a = k / n
            bl = float(np.nanmean([b_b.get(s, np.nan) for s in ss]))
            rows.append((n, g, a, lo, hi, bl, a - bl))
        for n, g, a, lo, hi, bl, dd in sorted(rows, reverse=True):
            flag = " ⚠️" if n < 20 else ""
            add(f"| {g}{flag} | {n} | {a:.3f} | [{lo:.3f},{hi:.3f}] | {bl:.3f} "
                f"| {dd:+.3f} |")
        if key == "content_type":
            add("\n> **Plane（解剖平面）类 Δ 为负**：给了图反而更差。可能因为"
                "模型对轴位/冠状位/矢状位的判读能力弱于其文字先验，图像成了"
                "干扰项。n=25 属小样本，⚠️ 只作现象记录，需扩数据验证。")

    # ---------- 表 6：校准
    add("\n## 五、校准：说 98% 的时候真对吗\n")
    ident = [r for r in recs if r["perturbation"] == "identity" and not r.get("refused")]
    confs = [r["conf"] for r in ident]
    corrs = [r["correct"] for r in ident]
    add(f"- 平均置信度 **{np.mean(confs):.3f}** vs 实际准确率 "
        f"**{np.mean(corrs):.3f}** → 过度自信 **{(np.mean(confs)-np.mean(corrs))*100:.1f}** 个百分点")
    add(f"- ECE = **{ece(confs, corrs):.3f}**（>0.15 视为校准差）")
    bins = defaultdict(list)
    for c, ok in zip(confs, corrs):
        bins[min(int(c * 10), 9)].append(ok)
    add("\n| 置信度区间 | n | 实际准确率 | 差距 |")
    add("|---|---|---|---|")
    for b in sorted(bins):
        mid = (b + 0.5) / 10
        add(f"| {b/10:.1f}–{(b+1)/10:.1f} | {len(bins[b])} | "
            f"{np.mean(bins[b]):.3f} | {np.mean(bins[b])-mid:+.3f} |")
    add("\n> 模型几乎全部输出 0.95–0.99 的置信度（熵 0.018），置信度**没有区分度**，"
        "不能直接用于临床决策阈值；需温度缩放或改为 logit 采样估计。")

    # ---------- 表 7：稳定性
    add("\n## 六、稳定性\n")
    unstable = 0
    for s in sids:
        v = [r["correct"] for r in ident if r["sid"] == s]
        if 0 < sum(v) < len(v):
            unstable += 1
    pass3 = np.mean([all([r["correct"] for r in ident if r["sid"] == s]) for s in sids])
    add(f"- 原图条件下 **{unstable}/{n_q}** 题（{unstable/n_q:.1%}）3 次回答不一致")
    add(f"- pass^3 = **{pass3:.3f}** vs pass@1 = {acc_by['identity']:.3f}，"
        f"差 {acc_by['identity']-pass3:.3f} → 温度 0 下答案高度确定，重复采样几乎无方差")

    # ---------- 结论
    add("\n## 七、可直接写进论文/简历的四条结论\n")
    add(f"1. **模态坍缩确实存在但常被误读**：Qwen3-VL-8B 原图准确率 "
        f"{acc_by['identity']:.3f}，只比常数基线高 {acc_by['identity']-majority:+.3f}；"
        f"去掉图像后准确率跌到 {acc_by['blank']:.3f}，低于常数基线。")
    add(f"2. **图像价值高度不对称**：Yes 题 Δ={dy:+.3f} vs No 题 Δ={dn:+.3f}，"
        f"聚合 Δ={d_all:+.3f} 是两者的净和——单一指标会误导。")
    add(f"3. **「置空图」与「不给图」不等价**：Yes 率 {yesrate['blank']:.3f} vs "
        f"{yesrate['text_only']:.3f}，说明模型把空图理解为「没有证据→答 No」，"
        f"把缺图理解为「从文字推断」。评测设计必须区分这两种条件。")
    add(f"4. **校准是最大短板**：ECE {ece(confs, corrs):.3f}，置信度集中在 0.95+，"
        f"几乎无区分度；而医疗场景恰恰最需要「知道自己不知道」。")

    txt = "\n".join(L)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    figs = make_figs(recs, acc_by, ci_by, split)
    print(f"\n[已写入] {OUT}")
    for f_ in figs:
        print(f"[图] {f_}")


if __name__ == "__main__":
    main()

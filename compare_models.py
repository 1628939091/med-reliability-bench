# -*- coding: utf-8 -*-
"""阶段 4：跨模型 / 跨数据集复现检验（横向对比报告）。

回答一个问题：阶段 3 的三条发现，是「Qwen3-VL-8B 在 SLAKE 上的特性」，
还是「这一类 VLM 在这个任务上的共性」？

判定逻辑（每条发现都写成可证伪的谓词，跑完自动判 True/False）：
    H1 模态坍缩存在   : 原图准确率显著高于常数基线，且置空后跌破常数基线
    H2 图像价值不对称 : gold=Yes 的 Δ > 0 且显著；gold=No 的 Δ ≤ 0
    H3 置空≠缺图      : |Yes率(blank) − Yes率(text_only)| > 0.10
    H4 校准崩塌       : ECE > 0.15 且 平均置信度 − 准确率 > 0.15

用法：python compare_models.py
      （自动扫描 outputs/checkpoint_*.jsonl，只处理能配对到数据集的）
产物：outputs/compare_models.md + outputs/fig_forest.png
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "compare_models.md")
B = 1000
SEED = 42

# (显示名, checkpoint, 数据集 jsonl) —— 数据集必须与 checkpoint 对应，
# 否则 meta 里的 same_image / gold 会错位，聚类全乱。
RUNS = [
    ("Qwen3-VL-8B · SLAKE",
     "outputs/checkpoint_real_slake_api_full.jsonl", "data/real_slake/dataset.jsonl"),
    ("GLM-4.5V · SLAKE",
     "outputs/checkpoint_real_slake_glm45v_full.jsonl", "data/real_slake/dataset.jsonl"),
    ("Qwen3-VL-8B · VQA-RAD",
     "outputs/checkpoint_real_vqarad_api_full.jsonl",
     "data/real_vqarad_subset/dataset.jsonl"),
    ("GLM-4.5V · VQA-RAD",
     "outputs/checkpoint_real_vqarad_glm45v_full.jsonl",
     "data/real_vqarad_subset/dataset.jsonl"),
]


# ------------------------------------------------------------------ 统计
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def boot_delta(a: dict, b: dict, reps=B, seed=SEED):
    keys = [k for k in a if k in b]
    if not keys:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(keys)
    st = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, n, n)
        va, vb = [], []
        for j in pick:
            va.extend(a[keys[j]])
            vb.extend(b[keys[j]])
        st[i] = np.mean(va) - np.mean(vb)
    da = np.mean([v for vs in a.values() for v in vs])
    db = np.mean([v for vs in b.values() for v in vs])
    return da - db, float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def ece(confs, corrs, n_bins=10):
    confs, corrs = np.asarray(confs, float), np.asarray(corrs, float)
    if len(confs) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
        if m.sum():
            tot += m.sum() / len(confs) * abs(corrs[m].mean() - confs[m].mean())
    return float(tot)


def entropy(labels):
    if not labels:
        return float("nan")
    _, cnt = np.unique(labels, return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())


# ------------------------------------------------------------------ 单次运行
def analyse(ckpt, data):
    recs = [json.loads(l) for l in open(ckpt, encoding="utf-8") if l.strip()]
    meta = {}
    for line in open(data, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            meta[d["id"]] = d.get("meta") or {}

    def img(sid):
        return (meta.get(sid) or {}).get("same_image") or sid

    per = defaultdict(lambda: defaultdict(list))
    for r in recs:
        per[r["perturbation"]][img(r["sid"])].append(float(r["correct"]))
    perts = [p for p in per]
    has = lambda p: p in per and sum(len(v) for v in per[p].values()) > 0

    model = recs[0]["model"]
    n_q = len({r["sid"] for r in recs})
    ident = [r for r in recs if r["perturbation"] == "identity"]
    m = {"model": model, "n_q": n_q, "n_rec": len(recs), "perts": perts}

    m["acc"] = float(np.mean([r["correct"] for r in ident])) if ident else float("nan")
    k = int(sum(r["correct"] >= 0.5 for r in ident))
    m["acc_ci"] = wilson(k, len(ident))

    gold = {r["sid"]: r["gold"] for r in ident}
    n_no = sum(1 for r in ident if r["gold"] == "A")
    n_yes = sum(1 for r in ident if r["gold"] == "B")
    m["baseline"] = n_no / len(ident) if ident else float("nan")
    m["n_yes"], m["n_no"] = n_yes, n_no

    if has("identity") and has("blank"):
        d, lo, hi = boot_delta(dict(per["identity"]), dict(per["blank"]))
        m["delta"], m["delta_ci"] = d, (lo, hi)
        # 按 gold 方向拆开
        for tag, g in [("yes", "B"), ("no", "A")]:
            sids = {s for s, gg in gold.items() if gg == g}
            # 用图像作键时需按是否属于该组重建
            ia, ib = defaultdict(list), defaultdict(list)
            for r in recs:
                if r["sid"] in sids:
                    if r["perturbation"] == "identity":
                        ia[img(r["sid"])].append(float(r["correct"]))
                    elif r["perturbation"] == "blank":
                        ib[img(r["sid"])].append(float(r["correct"]))
            dd, ll, hh = boot_delta(dict(ia), dict(ib))
            m[f"d_{tag}"], m[f"d_{tag}_ci"] = dd, (ll, hh)
    else:
        m["delta"], m["delta_ci"] = float("nan"), (float("nan"), float("nan"))
        m["d_yes"] = m["d_no"] = float("nan")
        m["d_yes_ci"] = m["d_no_ci"] = (float("nan"), float("nan"))

    # pass^k（identity）
    if ident:
        kk = max(r["rep"] for r in ident) + 1
        allok = defaultdict(list)
        for r in ident:
            allok[r["sid"]].append(bool(r["correct"]))
        m["pass_k"] = float(np.mean([len(v) >= kk and all(v) for v in allok.values()]))
        m["k"] = kk
    else:
        m["pass_k"], m["k"] = float("nan"), 0

    for p in ["blank", "text_only"]:
        rs = [r for r in recs if r["perturbation"] == p]
        m[f"yes_{p}"] = float(np.mean([r["pred"] == "B" for r in rs])) if rs else float("nan")
        m[f"ent_{p}"] = entropy([r["pred"] for r in rs]) if rs else float("nan")
    m["acc_blank"] = float(np.mean([r["correct"] for r in recs
                                    if r["perturbation"] == "blank"])) if has("blank") else float("nan")

    oc = [r for r in ident if not r.get("refused")]
    m["conf"] = float(np.mean([r["conf"] for r in oc])) if oc else float("nan")
    m["ece"] = ece([r["conf"] for r in oc], [r["correct"] for r in oc]) if oc else float("nan")
    m["refusal"] = float(np.mean([bool(r.get("refused")) for r in recs])) if recs else 0.0
    return m


def hyp(m):
    """四条假设的可证伪判定。"""
    h = {}
    h["H1 模态坍缩"] = bool(
        not math.isnan(m["acc"]) and m["acc"] > m["baseline"]
        and not math.isnan(m["acc_blank"]) and m["acc_blank"] < m["baseline"])
    dy, dn = m.get("d_yes", float("nan")), m.get("d_no", float("nan"))
    h["H2 价值不对称"] = bool((not math.isnan(dy)) and (not math.isnan(dn))
                          and dy > 0.1 and dn < 0)
    h["H3 置空≠缺图"] = bool((not math.isnan(m["yes_blank"]))
                          and (not math.isnan(m["yes_text_only"]))
                          and abs(m["yes_blank"] - m["yes_text_only"]) > 0.10)
    h["H4 校准崩塌"] = bool(not math.isnan(m["ece"]) and m["ece"] > 0.15
                         and (m["conf"] - m["acc"]) > 0.15)
    return h


def forest(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"[跳过作图] {e}")
        return None
    fig, ax = plt.subplots(figsize=(8.2, 0.75 * len(rows) + 1.8), dpi=150)
    for i, (name, m) in enumerate(rows):
        y = len(rows) - i
        lo, hi = m["delta_ci"]
        ax.plot([lo, hi], [y, y], lw=2.2, color="#c0392b")
        ax.plot([m["delta"]], [y], "o", ms=7, color="#c0392b")
        ax.text(hi + 0.02, y, f"{m['delta']:+.3f}  [{lo:+.3f},{hi:+.3f}]",
                va="center", fontsize=8, color="#2c3e50")
        ax.text(-0.02, y, name, ha="right", va="center", fontsize=8.5,
                transform=ax.get_yaxis_transform())
    ax.axvline(0, ls="--", lw=1.2, c="#34495e")
    ax.set_yticks([])
    ax.set_xlabel("坍缩差 Δ = 原图准确率 − 置空准确率（95% 图像聚类 CI）")
    ax.set_title("跨模型 / 跨数据集：图像到底值多少分", fontsize=11)
    ax.set_xlim(-0.05, max(m["delta_ci"][1] for _, m in rows) + 0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    rows, skipped = [], []
    for name, ck, data in RUNS:
        p1, p2 = os.path.join(HERE, ck), os.path.join(HERE, data)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            skipped.append(name)
            continue
        print(f"[分析] {name}")
        rows.append((name, analyse(p1, p2)))

    L = ["# 跨模型 / 跨数据集复现检验\n"]
    add = L.append
    if skipped:
        add(f"> 未跑（缺 checkpoint）：{', '.join(skipped)}\n")

    add("## 一、核心指标一览\n")
    add("| 运行 | 题数 | 原图 acc | 常数基线 | 置空 acc | Δ [95%CI] "
        "| Δ(Yes题) | Δ(No题) | pass^k | ECE | 置信度 | 拒答率 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in rows:
        dy = m.get("d_yes", float("nan"))
        dn = m.get("d_no", float("nan"))
        add(f"| {name} | {m['n_q']} | {m['acc']:.3f} | {m['baseline']:.3f} "
            f"| {m['acc_blank']:.3f} | **{m['delta']:+.3f}** "
            f"[{m['delta_ci'][0]:+.3f},{m['delta_ci'][1]:+.3f}] "
            f"| {dy:+.3f} | {dn:+.3f} | {m['pass_k']:.3f} | {m['ece']:.3f} "
            f"| {m['conf']:.3f} | {m['refusal']:.1%} |")

    add("\n## 二、四条假设的复现矩阵\n")
    names = [n for n, _ in rows]
    hs = [hyp(m) for _, m in rows]
    keys = ["H1 模态坍缩", "H2 价值不对称", "H3 置空≠缺图", "H4 校准崩塌"]
    add("| 假设 | " + " | ".join(names) + " | 复现 |")
    add("|---" * (len(names) + 2) + "|")
    for k in keys:
        vals = ["✅" if h[k] else "❌" for h in hs]
        cnt = sum(h[k] for h in hs)
        add(f"| {k} | " + " | ".join(vals) + f" | {cnt}/{len(hs)} |")

    if rows:
        n_full = len(rows)
        line = []
        for k in keys:
            cnt = sum(h[k] for h in hs)
            line.append(f"{k} {cnt}/{n_full}")
        add(f"\n**复现率**：" + "；".join(line))

        add("\n## 三、怎么解读\n")
        h1 = sum(h["H1 模态坍缩"] for h in hs)
        add(f"- H1（坍缩存在）复现 {h1}/{len(hs)}："
            f"{'跨模型稳健，可以作为主结论' if h1 == len(hs) else '存在不一致，主结论需限定条件'}。")
        h3 = sum(h["H3 置空≠缺图"] for h in hs)
        add(f"- H3（置空≠缺图）复现 {h3}/{len(hs)}：这是最容易被忽略但最影响评测设计的一条——"
            f"如果换模型后也成立，说明「用灰图代理无视觉信息」是**普遍错误**，"
            f"这条可以直接写进论文的方法论批评部分。")
        h4 = sum(h["H4 校准崩塌"] for h in hs)
        add(f"- H4（校准崩塌）复现 {h4}/{len(hs)}："
            f"{'所有模型都过度自信，说明是 VLM 通病而非个例' if h4 == len(hs) else '并非所有模型都如此，需分模型讨论'}。")
        add(f"\n> 写简历/论文时的口径：不要写「我们发现 X」，"
            f"写「我们在 {len(hs)} 个模型×数据集组合上检验了 X，复现 {h1}/{len(hs)}」。"
            f"前者是轶事，后者是证据。")

    txt = "\n".join(L)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    ok = [(n, m) for n, m in rows if not math.isnan(m["delta"])]
    if ok:
        fig = forest(ok, os.path.join(HERE, "outputs", "fig_forest.png"))
        if fig:
            print(f"[图] {fig}")
    print(f"[已写入] {OUT}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""阶段 4：跨模型 / 跨数据集复现检验（横向对比报告）。

回答一个问题：阶段 3 的三条发现，是「Qwen3-VL-8B 在 SLAKE 上的特性」，
还是「这一类 VLM 在这个任务上的共性」？

判定逻辑（每条发现都写成可证伪的谓词，跑完自动判 True/False）：
    H1 模态坍缩存在   : Δ 的 95%CI 下界 > 0（图像确有贡献），
                        且置空后准确率不高于常数基线 + 0.05（退化为近似常数预测）
    H2 图像价值不对称 : gold=Yes 的 Δ CI 下界 > 0，且 ΔYes − ΔNo > 0.15
                        （只要求「不对称」，不强求 No 题 Δ 为负——负号是模型特性）
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
from math import comb

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


def mcnemar_p(n01, n10):
    """配对 McNemar 精确检验（双侧，二项精确）。

    n01 = 条件 A 错、条件 B 对；n10 = 条件 A 对、条件 B 错。
    只看「不一致对」，一致对不提供信息——这是配对设计的全部意义：
    题目难度、模型整体水平都被消掉了，剩下的是「两个条件谁更有用」。
    """
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def binom_vs_base(k, n, base):
    """单侧二项检验：观察到的正确数 k/n 是否显著高于常数基线 base。

    H0：该条件下的真实准确率 = 「永远答多数类」的准确率。
    这是判断「模型到底有没有用上图像」的正确参照——不能拿两个坏条件互相比。
    """
    k, n = int(round(k)), int(n)
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, i) * base ** i * (1 - base) ** (n - i)
                        for i in range(k, n + 1)))


# ------------------------------------------------------------------ 单次运行
def analyse(ckpt, data):
    recs = [json.loads(l) for l in open(ckpt, encoding="utf-8") if l.strip()]
    meta, qtext = {}, {}
    for line in open(data, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            meta[d["id"]] = d.get("meta") or {}
            qtext[d["id"]] = d.get("question") or ""

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

    for p in ["identity", "blank", "swap", "text_only"]:
        rs = [r for r in recs if r["perturbation"] == p]
        m[f"yes_{p}"] = float(np.mean([r["pred"] == "B" for r in rs])) if rs else float("nan")
        m[f"ent_{p}"] = entropy([r["pred"] for r in rs]) if rs else float("nan")
    for p in ["blank", "swap"]:
        rs = [r for r in recs if r["perturbation"] == p]
        m[f"acc_{p}"] = float(np.mean([r["correct"] for r in rs])) if rs else float("nan")

    # 配对比较 blank vs swap：同一题同一重复下「有没有图」vs「图对不对」
    # 只有题级配对，才能把「题目难度」这个混杂因素消掉
    bykey = defaultdict(dict)
    for r in recs:
        if r["perturbation"] in ("blank", "swap"):
            bykey[(r["sid"], r["rep"])][r["perturbation"]] = bool(r["correct"])
    pr = [(v["swap"], v["blank"]) for v in bykey.values()
          if "swap" in v and "blank" in v]
    if pr:
        n01 = sum(1 for s, b in pr if not s and b)   # 有图(错图)错、没图对
        n10 = sum(1 for s, b in pr if s and not b)   # 有图(错图)对、没图错
        m["swap_blank"] = {"n": len(pr), "n01": n01, "n10": n10,
                           "p": mcnemar_p(n01, n10)}

    # 换错图 vs 置空：按题型分层——聚合差值可能是题型构成造成的假象
    topic_field = next((f for f in ("content_type", "question_type")
                        if any((meta.get(r["sid"]) or {}).get(f) for r in recs)), None)
    m["topic_field"] = topic_field
    topics = {}
    if topic_field:
        grp = defaultdict(list)
        for (sid, rep), v in bykey.items():
            if "swap" in v and "blank" in v:
                q = (meta.get(sid) or {}).get(topic_field)
                if q:
                    grp[q].append((sid, v["swap"], v["blank"]))
        for q, lst in grp.items():
            nq = len({s for s, _, _ in lst})
            if nq < 10:            # 少于 10 题的组只描述不检验
                continue
            a = sum(s for _, s, _ in lst) / len(lst)
            b = sum(x for _, _, x in lst) / len(lst)
            c01 = sum(1 for _, s, x in lst if not s and x)
            c10 = sum(1 for _, s, x in lst if s and not x)
            # 该题型的常数基线：永远答多数类
            sids_q = {s for s, _, _ in lst}
            gq = [r["gold"] for r in ident if r["sid"] in sids_q]
            base = (sum(1 for g in gq if g == "B") / len(gq)) if gq else float("nan")
            base = max(base, 1 - base) if gq else float("nan")
            # 原图 / 换错图 是否显著超过该题型的常数基线
            idq = [float(r["correct"]) for r in ident if r["sid"] in sids_q]
            k_id = sum(idq)
            k_sw = sum(s for _, s, _ in lst)
            topics[q] = {
                "n_q": nq, "swap": a, "blank": b, "gap": a - b,
                "p": mcnemar_p(c01, c10), "base": base,
                "acc_id": (k_id / len(idq)) if idq else float("nan"),
                "p_id": binom_vs_base(k_id, len(idq), base) if idq else float("nan"),
                "p_sw": binom_vs_base(k_sw, len(lst), base),
            }
    m["topics"] = topics

    # 否定式提问陷阱：只在 SLAKE 这类含 Abnormality 题型的运行上做
    # 「Does the liver look normal?」答 Yes 表示「正常」，与「Is there abnormality?」
    # 的极性相反。这类题会把「极性理解错误」伪装成「视觉判读失败」。
    neg = {}
    if meta and any((v or {}).get("content_type") == "Abnormality" for v in meta.values()):
        import re as _re
        pat = _re.compile(r"\bnormal\b|\babnormal", _re.I)
        for group, fn in [("否定式（含 normal/abnormal）", lambda q: bool(pat.search(q))),
                          ("直接式（其余）", lambda q: not pat.search(q))]:
            v = [r for r in ident if (meta.get(r["sid"]) or {}).get("content_type")
                 == "Abnormality" and fn(qtext.get(r["sid"], ""))]
            if len(v) < 30:
                continue
            acc = float(np.mean([r["correct"] for r in v]))
            yes = float(np.mean([r["pred"] == "B" for r in v]))
            gold = float(np.mean([r["gold"] == "B" for r in v]))
            # 按自身 Yes 率作答的期望准确率：模型的「分布」是对的、只是「谁对谁」错了
            exp = yes * gold + (1 - yes) * (1 - gold)
            # 临床口径：Yes = 正常，No = 异常 → 敏感度 = 异常被判出、特异度 = 正常被判对
            tp = sum(1 for r in v if r["gold"] == "A" and r["pred"] == "A")   # 异常→异常
            fn_ = sum(1 for r in v if r["gold"] == "A" and r["pred"] == "B")  # 异常→正常（漏诊）
            tn = sum(1 for r in v if r["gold"] == "B" and r["pred"] == "B")   # 正常→正常
            fp = sum(1 for r in v if r["gold"] == "B" and r["pred"] == "A")   # 正常→异常
            neg[group] = {"n": len(v), "acc": acc, "flip": 1 - acc,
                          "exp": exp, "yes": yes, "gold": gold,
                          "sens": tp / (tp + fn_) if (tp + fn_) else float("nan"),
                          "spec": tn / (tn + fp) if (tn + fp) else float("nan"),
                          "tp": tp, "fn": fn_, "tn": tn, "fp": fp}
    m["negation"] = neg

    # 模态分层：必须用「子集自己的常数基线」校正，否则答案比例会冒充坍缩幅度
    mods = {}
    names = {(meta.get(r["sid"]) or {}).get("modality") for r in ident} - {None}
    for mod in sorted(names):
        sids = {r["sid"] for r in ident
                if (meta.get(r["sid"]) or {}).get("modality") == mod}
        gs = [r["gold"] for r in ident if r["sid"] in sids]
        bl = [float(r["correct"]) for r in recs
              if r["perturbation"] == "blank" and r["sid"] in sids]
        a_m = float(np.mean([r["correct"] for r in ident if r["sid"] in sids]))
        b_m = float(np.mean(bl)) if bl else float("nan")
        mods[mod] = {"n": len(sids), "acc": a_m, "blank": b_m,
                     "base": sum(1 for g in gs if g == "A") / len(gs),
                     "delta": a_m - b_m}
    m["modal"] = mods

    oc = [r for r in ident if not r.get("refused")]
    m["conf"] = float(np.mean([r["conf"] for r in oc])) if oc else float("nan")
    m["ece"] = ece([r["conf"] for r in oc], [r["correct"] for r in oc]) if oc else float("nan")
    # 置信度的区分度：ECE 只衡量「水平差」，不衡量「有没有信息」
    if oc:
        cc = np.array([r["conf"] for r in oc], float)
        yy = np.array([r["correct"] for r in oc], float)
        pos, neg = cc[yy >= 0.5], cc[yy < 0.5]
        m["conf_nuniq"] = len(set(cc.tolist()))
        if len(pos) and len(neg):
            m["conf_auc"] = float(np.mean(
                (pos[:, None] > neg[None, :]) * 1.0
                + (pos[:, None] == neg[None, :]) * 0.5))
            m["conf_gap"] = float(pos.mean() - neg.mean())
        else:
            m["conf_auc"] = m["conf_gap"] = float("nan")
    m["refusal"] = float(np.mean([bool(r.get("refused")) for r in recs])) if recs else 0.0
    return m


def hyp(m):
    """四条假设的可证伪判定。所有阈值只依赖 CI 与固定容差，不看「哪个模型」。"""
    h = {}
    d_lo = m.get("delta_ci", (float("nan"), float("nan")))[0]
    h["H1 模态坍缩"] = bool(
        (not math.isnan(d_lo)) and d_lo > 0
        and not math.isnan(m["acc_blank"])
        and m["acc_blank"] <= m["baseline"] + 0.05)
    dy, dn = m.get("d_yes", float("nan")), m.get("d_no", float("nan"))
    dy_lo = m.get("d_yes_ci", (float("nan"), float("nan")))[0]
    h["H2 价值不对称"] = bool((not math.isnan(dy)) and (not math.isnan(dn))
                          and (not math.isnan(dy_lo)) and dy_lo > 0
                          and (dy - dn) > 0.15)
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
        h2 = sum(h["H2 价值不对称"] for h in hs)
        add(f"- H1（坍缩存在）复现 {h1}/{len(hs)}："
            f"{'跨模型稳健' if h1 == len(hs) else '存在不一致，主结论需限定条件'}。"
            f"**但要小心这条几乎是同义反复**——灰图会让模型恒答 No，"
            f"准确率自然落回「全答 No」基线。真正有信息的问题是"
            f"「原图是否显著高于基线」，答案在第四节。")
        add(f"- H2（价值不对称）复现 {h2}/{len(hs)}："
            f"{'跨模型跨数据集稳健，可以当主结论' if h2 == len(hs) else '不稳定，需限定条件'}。"
            f"注意「Δ(No) 为负」不是这条的必需部分——它只是 Qwen 的特性（GLM 上是 +0.050）。")
        h3 = sum(h["H3 置空≠缺图"] for h in hs)
        add(f"- H3（置空≠缺图）复现 {h3}/{len(hs)}：这是最容易被忽略但最影响评测设计的一条——"
            f"如果换模型后也成立，说明「用灰图代理无视觉信息」是**普遍错误**，"
            f"这条可以直接写进论文的方法论批评部分。")
        h4 = sum(h["H4 校准崩塌"] for h in hs)
        add(f"- H4（校准崩塌）复现 {h4}/{len(hs)}："
            f"{'所有模型都过度自信，说明是 VLM 通病而非个例' if h4 == len(hs) else '并非所有模型都如此，需分模型讨论——把它当「VLM 通病」写进结论会翻车'}。")

        # 四：blank vs swap —— 有边界条件的一条
        pairs = [(n, m) for n, m in rows
                 if not (math.isnan(m.get("acc_blank", float("nan")))
                         or math.isnan(m.get("acc_swap", float("nan"))))]
        if pairs:
            add("\n## 四、视觉接地有多深：按题型扣掉常数基线\n")
            add("「置空」= 喂一张 64×64 灰图；「换错图」= 喂**另一道题**的真实影像"
                "（按题号随机抽，见 `runner.py`）。两者都在破坏「图像内容」，"
                "区别在于错图里还留着模态、部位这些**真实影像才有的粗属性**。\n")
            add("| 运行 | 置空 acc | 换错图 acc | 差值 | 配对 n | 只有换错图对 | 只有置空对 | McNemar p |")
            add("|---|---|---|---|---|---|---|---|")
            sig = []
            for n, m in pairs:
                sb = m.get("swap_blank") or {}
                gap = m["acc_swap"] - m["acc_blank"]
                pv = sb.get("p", float("nan"))
                if (not math.isnan(pv)) and pv <= 0.05:
                    sig.append(n)
                add(f"| {n} | {m['acc_blank']:.3f} | {m['acc_swap']:.3f} "
                    f"| {gap:+.3f} | {sb.get('n', 0)} | {sb.get('n10', 0)} "
                    f"| {sb.get('n01', 0)} | **{pv:.4f}** |" if pv <= 0.05 else
                    f"| {n} | {m['acc_blank']:.3f} | {m['acc_swap']:.3f} "
                    f"| {gap:+.3f} | {sb.get('n', 0)} | {sb.get('n10', 0)} "
                    f"| {sb.get('n01', 0)} | {pv:.3f} |")

            if not sig:
                add(f"\n> {len(pairs)}/{len(pairs)} 个运行的配对检验都**不显著**："
                    "换错图与置空没有区别。")
            else:
                add(f"\n> **{len(pairs) - len(sig)}/{len(pairs)} 个运行上两者无区别，"
                    f"但 {'、'.join(sig)} 上换错图显著更高。**")
            add("\n**但这个比较本身是个陷阱**：两个「坏输入」互相比较，"
                "只能说明谁更坏，不能说明模型到底有没有用上图像。"
                "正确的参照物是**该条件自己的常数基线**（永远答多数类）。"
                "下面两节把这个拆开。")

            # 4.1 为什么 VQA-RAD 上换错图更高：灰图不是「无信息」，是「恒答 No」
            witht = [(n, m) for n, m in rows if m.get("topics")]
            add("\n### 4.1 灰图不是「零信息」，而是「钉死在 No 上」\n")
            add("看置空条件下的 Yes 率：SLAKE 的多数题型 Yes 率是 **0.000**——"
                "模型看到灰图就几乎一律答 No。所以置空准确率会**精确等于该子集的"
                "「全答 No」基线**，而一旦某个题型 gold 偏 Yes，"
                "「恒答 No」就会变成**主动答错**，准确率远低于基线。\n")
            add("| 运行 | 题型 | gold Yes 占比 | 常数基线 | 置空 acc | 置空−基线 | 换错图 acc | 换错图−基线 |")
            add("|---|---|---|---|---|---|---|---|")
            for n, m in witht:
                for q, t in sorted(m["topics"].items(),
                                   key=lambda x: -abs(x[1]["blank"] - x[1]["base"]))[:3]:
                    if math.isnan(t.get("base", float("nan"))):
                        continue
                    add(f"| {n} | {q} | {1 - t['base']:.3f} | {t['base']:.3f} "
                        f"| {t['blank']:.3f} | {t['blank'] - t['base']:+.3f} "
                        f"| {t['swap']:.3f} | {t['swap'] - t['base']:+.3f} |")
            add("\n> 这就是 VQA-RAD 上「换错图 > 置空」的全部来源："
                "**不是错图带来了信息，而是灰图把模型钉在 No 上、在 gold 偏 Yes 的题型上"
                "表现成负分。** 换个错图只是让它回到「按图作答」，于是回到基线附近。")

            # 4.2 用基线当尺子：谁真的超过了「永远答多数类」
            add("\n### 4.2 用常数基线当尺子：图像到底被用在哪一层\n")
            add("`p(原图)` / `p(换错图)` 是单侧二项检验（H0：真实准确率 = 该题型常数基线）。"
                "p<0.05 才算「真的用上了图像」。\n")
            add("| 运行 | 题型 | 题数 | 基线 | 原图 | p(原图) | 换错图 | p(换错图) | 读法 |")
            add("|---|---|---|---|---|---|---|---|---|")
            for n, m in witht:
                for q, t in sorted(m["topics"].items(),
                                   key=lambda x: -(x[1].get("acc_id", 0)
                                                   - x[1].get("base", 0))):
                    if math.isnan(t.get("p_id", float("nan"))):
                        continue
                    if t["p_id"] <= 0.05:
                        verdict = "**✔ 图像有效**"
                    elif t["acc_id"] < t["base"] - 0.05:
                        verdict = "**✘ 看图更差**"
                    else:
                        verdict = "✘ 等效常数预测器"
                    if t["p_sw"] <= 0.05:
                        verdict += "；错图也超基线"
                    add(f"| {n} | {q} | {t['n_q']} | {t['base']:.3f} | {t['acc_id']:.3f} "
                        f"| {t['p_id']:.4f} | {t['swap']:.3f} | {t['p_sw']:.4f} | {verdict} |")

            add("\n> 这张表是本项目最重要的一张。它做了一件前面所有指标都做不到的事："
                "**把「真看图」和「靠先验猜」在同一张表上分开**——"
                "判据是「原图准确率是否显著高于该题型的常数基线」。")
            add(">")
            add("> **Qwen3-VL-8B 的视觉接地只到粗属性**：器官有无（Organ 0.818 vs 基线 0.506，"
                "p<1e−5）、模态（0.797 vs 0.522）、位置（0.800 vs 0.533）、"
                "尺寸（0.709 vs 0.538）、病灶有无（0.675 vs 0.578）都显著超基线；"
                "但**异常判读（Abnormality 0.520 vs 0.551，p=0.87）和平面判断"
                "（Plane 0.440 vs 0.680，p≈1.0）完全不超基线，Plane 上甚至比不看图还差**。")
            add(">")
            add("> **GLM-4.5V 在全部 6 个题型上都显著超基线**，包括异常判读"
                "（Abnormality 0.646 vs 0.551，**p=0.0006**）和平面判断"
                "（Plane 0.933 vs 0.680，p<1e−5）。")
            add(">")
            add("> 但 Qwen 在 Abnormality 上的失败还有一个更具体的原因，见 4.3——"
                "**不是没看图，是把否定式提问的极性搞反了。**")
            add(">")
            add("> 换错图这一列几乎从不显著超过基线（唯一例外：VQA-RAD 的 PRES 题型 "
                "0.650 vs 0.578，p=0.0007——因为 VQA-RAD 只有头/胸/腹三个部位，"
                "随机换图有 1/3 概率同部位，问「有没有肝」时换到腹部片仍能答对），"
                "而原图有 6 个题型显著超过。**所以「错图 ≈ 没图」这个直觉本身没错，"
                "错的是把它解释成「模型只感知图像存在性」。**")

            # 4.3 否定式提问陷阱
            negs = [(n, m) for n, m in rows if m.get("negation")]
            if negs:
                add("\n### 4.3 否定式提问陷阱：把「极性理解错误」伪装成「视觉判读失败」\n")
                add("SLAKE 的 Abnormality 题型里混着两种问法，**极性相反**：\n")
                add("- 直接式：`Is there an abnormality in the liver?` → 答 Yes 表示**有异常**")
                add("- 否定式：`Does the liver look normal?` → 答 Yes 表示**正常**")
                add("")
                add("「随机期望」= 按模型自己的 Yes 率去随机作答能得到的准确率。"
                    "**若实际准确率低于随机期望，说明模型的回答与金标负相关——典型的极性反转。**\n")
                add("| 运行 | 问法 | n | 准确率 | 随机期望 | 差 | 翻转预测后 | 敏感度(异常判出) | 特异度(正常判对) |")
                add("|---|---|---|---|---|---|---|---|---|")
                for n, m in negs:
                    for g, t in m["negation"].items():
                        add(f"| {n} | {g} | {t['n']} | {t['acc']:.3f} | {t['exp']:.3f} "
                            f"| {t['acc'] - t['exp']:+.3f} | {t['flip']:.3f} "
                            f"| **{t['sens']:.3f}** | {t['spec']:.3f} |")
                add("\n> 读法（敏感度 = 真异常被判出；漏诊率 = 1 − 敏感度）：")
                for n, m in negs:
                    d = m["negation"]
                    ng = next((v for k, v in d.items() if k.startswith("否定式")), None)
                    di = next((v for k, v in d.items() if k.startswith("直接式")), None)
                    if not ng or not di:
                        continue
                    parts = [f"> - **{n}**：同一批影像、同一判断任务，"
                             f"换成「Does it look normal?」的否定式提问后，"
                             f"敏感度 {di['sens']:.3f} → **{ng['sens']:.3f}**"
                             f"（漏诊率 {1-ng['sens']:.0%}），"
                             f"特异度 {di['spec']:.3f} → {ng['spec']:.3f}。"]
                    if ng["sens"] < di["sens"] - 0.15:
                        parts.append(f"敏感度掉了 {di['sens']-ng['sens']:.2f}，"
                                     "**措辞本身就让异常变得更容易被漏掉**。")
                    else:
                        parts.append("两种问法下敏感度接近，没有被措辞带偏。")
                    if ng["acc"] < ng["exp"] - 0.03:
                        parts.append(f"而且准确率 {ng['acc']:.3f} **低于**按自身 Yes 率"
                                     f"随机作答的期望 {ng['exp']:.3f}——"
                                     "**回答与金标负相关，是极性理解反了，不是看不见病灶。**")
                    else:
                        parts.append(f"准确率 {ng['acc']:.3f} 高于随机期望 {ng['exp']:.3f}，"
                                     "极性没有反，只是判读能力有限。")
                    add("".join(parts))
                add(">")
                add("> **这条的评测学意义**：只看 Abnormality 的聚合准确率，"
                    "你会得出「模型看不见异常」的结论——对 Qwen 来说那是错的，"
                    "真相是**否定式提问把它的极性带反了**，两种效应在聚合指标里互相抵消。"
                    "更普遍的是：**两个模型的敏感度都在否定式提问下明显下降**"
                    "（Qwen 0.606→0.273，GLM 0.803→0.591），"
                    "而聚合准确率把这个下降完全掩盖了（Qwen 甚至看起来只是「比 GLM 差一点」）。")
                add(">")
                add("> 临床场景里「这个结节正常吗」「有没有异常」本来就是混着问的，"
                    "所以这个缺陷真实存在、且会被标准评测完全掩盖。"
                    "**报告医学 VLM 的异常判读能力时，必须同时给敏感度、并按提问极性分层**——"
                    "只报 accuracy 是失职。这条可以直接写进论文方法论部分，"
                    "也可以作为对「医学 VLM 已达放射科医生水平」这类说法的具体反驳。")

            # 4.3 Yes 率变化
            add("\n### 4.4 视觉特征确实进来了，只是没换来准确率\n")
            add("| 运行 | Yes 率·原图 | Yes 率·置空 | Yes 率·换错图 | 换错图−置空 |")
            add("|---|---|---|---|---|")
            for n, m in pairs:
                yb, ys = m.get("yes_blank", float("nan")), m.get("yes_swap", float("nan"))
                add(f"| {n} | {m.get('yes_identity', float('nan')):.3f} | {yb:.3f} "
                    f"| {ys:.3f} | {ys - yb:+.3f} |")
            seen = [m.get("yes_swap", 0) - m.get("yes_blank", 0) for _, m in pairs]
            if any(abs(v) > 0.10 for v in seen):
                add("\n> 有的模型换错图后 Yes 率大幅上升（它确实读了错图里的东西），"
                    "有的几乎不动。「分布变了、对错没变」这个组合恰好说明："
                    "视觉特征进来了，但与题目无关。")

        # 五：模态分层必须扣掉子集自己的基线
        withm = [(n, m) for n, m in rows if m.get("modal")]
        if withm:
            add("\n## 五、模态分层：先扣掉「子集自己的常数基线」\n")
            add("各模态的 Yes/No 比例不同，「永远答 No」在三个子集上的准确率本来就不同"
                "（X-Ray 的 Yes 金标占 55.2%，所以全答 No 只有 0.448）。"
                "不扣这个差就横比 Δ，会把「答案比例」误读成「坍缩幅度」。\n")
            add("| 运行 | 模态 | n | 原图 acc | 置空 acc | 子集常数基线 | 置空−基线 | Δ |")
            add("|---|---|---|---|---|---|---|---|")
            for n_, m in withm:
                for mod in ["CT", "X-Ray", "MRI"]:
                    s = m["modal"].get(mod)
                    if not s:
                        continue
                    gap = s["blank"] - s["base"]
                    add(f"| {n_} | {mod} | {s['n']} | {s['acc']:.3f} | {s['blank']:.3f} "
                        f"| {s['base']:.3f} | {gap:+.3f} | {s['delta']:+.3f} |")
            add("\n> 看最后一列「置空−基线」而不是 Δ：它回答的是「没有图时，模型是否"
                "退化成常数预测器」。"
                "若该值在 ±0.05 内，说明完全退化；若显著为负，说明模型在没有图时"
                "**比常数预测器更差**（主动答错），这才是真正的病理信号。"
                "Δ 在不同模态间的排序会随答案比例漂移，不能直接横比。")

        # 六：置信度有没有信息量
        cc = [(n, m) for n, m in rows
              if not math.isnan(m.get("conf_auc", float("nan")))]
        if cc:
            add("\n## 六、校准：ECE 差很多，但置信度在两个模型上都没用\n")
            add("| 运行 | 平均置信度 | 原图 acc | ECE | conf(correct)−conf(incorrect) "
                "| 区分度 AUC | 唯一置信度取值数 |")
            add("|---|---|---|---|---|---|---|")
            for n_, m in cc:
                add(f"| {n_} | {m['conf']:.3f} | {m['acc']:.3f} | {m['ece']:.3f} "
                    f"| {m['conf_gap']:+.3f} | {m['conf_auc']:.3f} | {m['conf_nuniq']} |")
            aucs = " / ".join(f"{m['conf_auc']:.3f}" for _, m in cc)
            uniqs = " / ".join(str(m["conf_nuniq"]) for _, m in cc)
            add(f"\n> ECE 只衡量「说 90% 时是否真对 90%」，它**不衡量置信度有没有信息量**。"
                f"{len(cc)} 个运行的区分度 AUC 为 {aucs}（0.5 = 完全无区分力），"
                f"唯一取值数只有 {uniqs} 个——"
                "置信度是被量化成两三档的口头禅，不是概率。"
                "所以正确的结论不是「某个模型校准差」，而是"
                "**各家的 confidence 字段都不能用于临床决策阈值**；"
                "ECE 高低只是「口头禅偏乐观多少」的差别。")

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

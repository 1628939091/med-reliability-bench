"""评测编排器：遍历 模型 × 扰动 × 题目 × 重复次数，产出结构化记录。

真实实验只需在 config 里把 mock 模型换成 api 模型，其余代码零改动 ——
这也是你论文可复现性声明（reproducibility statement）的底气。
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from .data import Sample, load_dataset, generate_sample_dataset
from .models import build_model, RefusalError, APITransientError
from .perturb import build_perturbations


def run_eval(config: dict, data_dir: str | None = None) -> dict:
    """config 结构见 configs/demo_config.json。返回 {"records": [...], "summary": ...}"""
    # 1. 数据
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data", "demo")
    jsonl = os.path.join(data_dir, "dataset.jsonl")
    if not os.path.exists(jsonl):
        generate_sample_dataset(data_dir,
                                n_per_cat=config.get("n_per_cat", 20))
    samples = load_dataset(jsonl)
    if config.get("max_samples"):
        samples = samples[: config["max_samples"]]

    # 2. 扰动套件
    suite = build_perturbations(
        os.path.join(os.path.dirname(jsonl), "perturb_cache"))

    k = config.get("k_repeats", 5)
    pert_names = config.get("perturbations",
                            ["identity", "blank", "noise", "flip",
                             "swap", "low_res", "text_only"])

    # 2.5 断点续跑：逐条落盘，崩溃后重跑自动跳过已完成的组合
    ckpt = config.get("checkpoint")
    records = []
    done_keys = set()
    if ckpt and os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                records.append(r)
                done_keys.add((r["model"], r["perturbation"],
                               r["sid"], r["rep"]))
        if records:
            print(f"      [checkpoint] 已有 {len(records)} 条记录，续跑中……")
    ckpt_f = open(ckpt, "a", encoding="utf-8") if ckpt else None

    records_n = len(records)
    total = len(config["models"]) * len(pert_names) * len(samples) * k
    done = records_n
    n_skip = 0

    # 3. 主循环
    for mcfg in config["models"]:
        model = build_model(mcfg)
        for pert in pert_names:
            for s in samples:
                other = None
                if pert == "swap":  # 换成另一道题的图
                    # 旧实现用 next() 取「第一张不同的图」→ 所有题换到同一张固定图，
                    # 使 swap 退化成 fixed-image 条件（VQA-RAD 上 swap 反而高于 blank 即此因）。
                    # 现按题随机抽一张不同的图，种子绑定 sid 保证断点续跑可复现。
                    _imgs = sorted({x.image for x in samples})
                    _cands = [x for x in _imgs if x != s.image]
                    other_img = random.Random(f"swap|{s.sid}").choice(_cands) \
                        if _cands else s.image
                    other = os.path.join(os.path.dirname(jsonl), other_img) \
                        if not os.path.isabs(other_img) else other_img
                img = suite.apply(
                    pert,
                    s.image if os.path.isabs(s.image)
                    else os.path.join(os.path.dirname(jsonl), s.image),
                    s.sid, other)
                for rep in range(k):
                    key = (model.name, pert, s.sid, rep)
                    if key in done_keys:      # 已跑过，跳过
                        continue
                    try:
                        pred, conf = model.answer(s.question, s.options,
                                                  s.answer, img, pert, rep)
                        refused = False
                    except RefusalError as e:
                        # 拒答 = 一次观测：记为答错，conf 记 0，单独统计拒答率
                        pred, conf, refused = "<refused>", 0.0, True
                    except APITransientError as e:
                        # 重试全失败：跳过本题继续跑（不写入 checkpoint，
                        # 下次续跑会自动补测），绝不让整跑崩掉
                        n_skip += 1
                        print(f"      [skip] {s.sid} rep{rep} 服务端持续失败，"
                              f"已跳过（累计跳过 {n_skip}）: {e}", flush=True)
                        continue
                    rec = {
                        "model": model.name, "perturbation": pert,
                        "sid": s.sid, "category": s.category, "rep": rep,
                        "pred": pred, "gold": s.answer, "conf": conf,
                        "correct": float(pred == s.answer),
                        "refused": refused,
                    }
                    records.append(rec)
                    if ckpt_f:
                        ckpt_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        ckpt_f.flush()
                    done += 1
                    if done % 50 == 0 or done == total:
                        ok = sum(r["correct"] for r in records)
                        n_ref = sum(1 for r in records if r.get("refused"))
                        print(f"      进度 {done}/{total} "
                              f"({done / total:.0%})  累计正确率 "
                              f"{ok / len(records):.3f}  拒答 {n_ref}"
                              f"  跳过 {n_skip}",
                              flush=True)
    if ckpt_f:
        ckpt_f.close()

    summary = _summarize(records, k)
    return {"records": records, "summary": summary,
            "degraded_perturbations": suite.degraded, "k": k,
            "n_skipped": n_skip}


def _summarize(records, k: int) -> dict:
    """按模型聚合所有指标。"""
    out = {}
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)

    for name, recs in by_model.items():
        # 每种扰动的准确率
        acc_by_pert = {}
        ci_by_pert = {}
        for pert in {r["perturbation"] for r in recs}:
            flags = [r["correct"] for r in recs
                     if r["perturbation"] == pert]
            m, lo, hi = _bootstrap(flags)
            acc_by_pert[pert] = m
            ci_by_pert[pert] = (lo, hi)

        ident = [r for r in recs if r["perturbation"] == "identity"]
        out[name] = {
            "acc_identity": acc_by_pert.get("identity", 0.0),
            "acc_by_pert": acc_by_pert,
            "ci_identity": ci_by_pert.get("identity", (0.0, 0.0)),
            "collapse_delta": acc_by_pert.get("identity", 0.0)
                              - acc_by_pert.get("blank", 0.0),
            "shortcut_agreement": _shortcut(recs),
            "pass_k": _passk(recs, k),
            "majority_acc": _majority(recs, k),
            "answer_entropy": _entropy(recs),
            "ece": _ece_of(ident),
            "reliability": _reliability_of(ident),
            "refusal_rate": (sum(1 for r in recs if r.get("refused"))
                             / len(recs)) if recs else 0.0,
        }
    return out


# 把 metrics.py 的函数在这里复用（避免循环依赖就直接内联调用）
import statistics  # noqa: E402
from .metrics import (bootstrap_ci, shortcut_agreement, pass_at_k,  # noqa: E402
                      answer_entropy, ece, reliability_bins, majority_vote_acc)


def _bootstrap(flags):
    return bootstrap_ci(flags)


def _shortcut(recs):
    return shortcut_agreement(recs)


def _passk(recs, k):
    return pass_at_k(recs, k)


def _majority(recs, k):
    return majority_vote_acc(recs, k)


def _entropy(recs):
    return answer_entropy(recs)


def _ece_of(ident):
    return ece([r["conf"] for r in ident], [r["correct"] for r in ident])


def _reliability_of(ident):
    return reliability_bins([r["conf"] for r in ident],
                            [r["correct"] for r in ident])

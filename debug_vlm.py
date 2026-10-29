# -*- coding: utf-8 -*-
"""调试脚本 v2：逐题打印 identity / blank 两种扰动下的预测，定位异常来源。

用法（先设 $env:MRB_API_KEY）：
    python debug_vlm.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medbench.models import OpenAICompatVLM
from medbench.perturb import build_perturbations

HERE = os.path.dirname(os.path.abspath(__file__))

model = OpenAICompatVLM(
    "debug", "https://api.siliconflow.cn/v1",
    os.environ.get("MRB_API_KEY"),
    "Qwen/Qwen3-VL-8B-Instruct", 0.0)

recs = [json.loads(l) for l in open(
    os.path.join(HERE, "data", "real_slake", "dataset.jsonl"),
    encoding="utf-8")][:10]

suite = build_perturbations(os.path.join(HERE, "data", "real_slake",
                                         "perturb_cache"))

for pert in ("identity", "blank"):
    print(f"===== 扰动: {pert} =====")
    n_ok = 0
    for r in recs:
        img = suite.apply(pert, r["image"], r["id"])
        pred, conf = model.answer(r["question"], r["options"], r["answer"],
                                  img, pert, 0)
        ok = pred == r["answer"]
        n_ok += ok
        mark = "✓" if ok else "✗"
        print(f"[{mark}] {r['id']}  gold={r['answer']} pred={pred} "
              f"conf={conf}  raw={model.last_raw[:80]!r}")
    print(f"--> {pert} acc = {n_ok}/{len(recs)}")
    print()

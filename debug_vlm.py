# -*- coding: utf-8 -*-
"""调试脚本 v3：逐题打印 identity / blank 两种扰动下的预测，定位异常来源。

用法（先在终端设置 $env:MRB_API_KEY）：
    python debug_vlm.py                                     # 默认 Qwen3-VL-8B + SLAKE 前 10 题
    python debug_vlm.py configs/real_slake_glm45v_sanity.json 5
    python debug_vlm.py configs/real_vqarad_api_sanity.json 5

换模型最容易出的问题就是「解析层抠不出答案，悄悄全走兜底」——
表现是 acc≈0.5 且坍缩差≈0。这个脚本把 raw 原文打出来，一眼能看出是模型没答对
还是代码没读出来。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medbench.models import OpenAICompatVLM
from medbench.perturb import build_perturbations

HERE = os.path.dirname(os.path.abspath(__file__))

cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 10

if cfg_path:
    with open(os.path.join(HERE, cfg_path), encoding="utf-8") as f:
        cfg = json.load(f)
    mc = cfg["models"][0]
    if mc.get("backend") != "api":
        raise SystemExit("debug_vlm 只支持 backend=api 的配置")
    model = OpenAICompatVLM(mc["name"], mc["base_url"],
                            mc.get("api_key") or os.environ.get("MRB_API_KEY"),
                            mc["model"], mc.get("temperature", 0.0))
    data_dir = cfg.get("data_dir", os.path.join("data", "real_slake"))
    # 没显式给题数时，沿用配置里的 max_samples（sanity 配置通常是 8~10）
    cfg_n = cfg.get("max_samples") or 0
    if len(sys.argv) <= 2 and 0 < cfg_n <= 50:
        n_show = cfg_n
    print(f"[配置] {cfg_path} → 模型 {mc['model']} / 数据 {data_dir}")
else:
    model = OpenAICompatVLM(
        "debug", "https://api.siliconflow.cn/v1",
        os.environ.get("MRB_API_KEY"),
        "Qwen/Qwen3-VL-8B-Instruct", 0.0)
    data_dir = os.path.join("data", "real_slake")
    print("[默认] Qwen3-VL-8B / SLAKE")

ds_path = os.path.join(HERE, data_dir, "dataset.jsonl")
recs = [json.loads(l) for l in open(ds_path, encoding="utf-8")][:n_show]

suite = build_perturbations(os.path.join(HERE, data_dir, "perturb_cache"))

for pert in ("identity", "blank"):
    print(f"===== 扰动: {pert} =====")
    n_ok = 0
    for r in recs:
        # 数据集里的 image 是相对 dataset.jsonl 所在目录的路径，先折成绝对路径
        img = os.path.abspath(os.path.join(HERE, data_dir, r["image"]))
        img = suite.apply(pert, img, r["id"])
        pred, conf = model.answer(r["question"], r["options"], r["answer"],
                                  img, pert, 0)
        ok = pred == r["answer"]
        n_ok += ok
        mark = "✓" if ok else "✗"
        print(f"[{mark}] {r['id']}  gold={r['answer']} pred={pred} "
              f"conf={conf}  raw={model.last_raw[:90]!r}")
    print(f"--> {pert} acc = {n_ok}/{len(recs)}")
    print()

fails = getattr(model, "parse_fail", 0)
if fails:
    print(f"⚠️ 解析兜底 {fails} 次（回复里没有可识别的答案标记）——"
          f"说明该模型的输出格式没被解析层覆盖，看上面 raw 的实际样式")

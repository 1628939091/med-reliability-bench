# -*- coding: utf-8 -*-
"""项目一入口：一键跑完整个评测并生成报告。

用法：
    python run_demo.py                  # 用 Mock 模型演示（零成本，无需API）
    python run_demo.py configs/api.json # 接真实模型
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medbench.runner import run_eval
from medbench.report import render_report

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "configs", "demo_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)

    print(f"[1/3] 加载配置: {cfg_path}")
    # 数据目录由配置决定（config 里写 "data_dir": "data/real_slake" 等），
    # 缺省回退到合成演示数据。2026-10-29 修复：此前写死 data/demo，
    # 导致接真实数据后评测实际仍在跑演示集（两次不同数据集指标完全相同即征兆）。
    data_dir = os.path.join(HERE, config.get("data_dir",
                                             os.path.join("data", "demo")))
    # 断点续跑：每个配置一个 checkpoint，崩溃/中断后重跑自动跳过已完成项
    if "checkpoint" not in config:
        stem = os.path.splitext(os.path.basename(cfg_path))[0]
        os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
        config["checkpoint"] = os.path.join(HERE, "outputs",
                                            f"checkpoint_{stem}.jsonl")
    print("[2/3] 运行评测（模型 × 扰动 × 题目 × k 次重复）……")
    result = run_eval(config, data_dir=data_dir)
    print(f"      共产生 {len(result['records'])} 条评测记录")

    out = os.path.join(HERE, "outputs", "report.md")
    render_report(result, out)
    print(f"[3/3] 报告已生成: {out}")
    print("\n===== 核心指标速览 =====")
    for name, m in result["summary"].items():
        print(f"  {name}: acc={m['acc_identity']:.3f}  坍缩差={m['collapse_delta']:+.3f}"
              f"  pass^{result['k']}={m['pass_k']:.3f}  ECE={m['ece']:.3f}"
              f"  拒答率={m.get('refusal_rate', 0):.1%}")


if __name__ == "__main__":
    main()

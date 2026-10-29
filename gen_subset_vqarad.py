# -*- coding: utf-8 -*-
"""为 VQA-RAD 生成分层抽样测试子集（跨数据集复现用，控制 API 成本）。

为什么抽样而不是全跑：
    VQA-RAD 有 1193 道闭集题，×7 扰动×3 重复 = 25053 次调用，
    单模型要跑十几小时。论文里做跨数据集复现，300 题的分层样本足够把
    Δ 的 CI 收到 ±0.06 以内，成本降到 1/4。

为什么分层而不是随机抽：
    VQA-RAD 的题目按器官（CHEST/ABD/HEAD）和题型（PRES/SIZE/ABN/...）分布
    很不均匀。纯随机抽 300 题，小器官（如 HEAD 里的某些题型）可能只剩个位数，
    分层后每个器官的样本量可控，跨数据集比较才不会被构成差异污染。

分层口径：(answer ∈ {A,B}) × (organ ∈ {CHEST, ABD, HEAD})，共 6 层，
按比例分配 + 最大余数法分配剩余名额，固定种子可复现。

用法：python gen_subset_vqarad.py [N]
产物：data/real_vqarad_subset/dataset.jsonl（与源目录同层，相对图片路径不变）
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from math import floor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "real_vqarad", "dataset.jsonl")
DST_DIR = os.path.join(HERE, "data", "real_vqarad_subset")
DST = os.path.join(DST_DIR, "dataset.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 42


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    strata = defaultdict(list)
    for r in rows:
        organ = (r.get("meta") or {}).get("organ") or "OTHER"
        strata[(r["answer"], organ)].append(r)

    total = len(rows)
    # 1) 按比例算配额（向下取整）
    quota, rema = {}, {}
    for k, v in strata.items():
        exact = N * len(v) / total
        quota[k] = floor(exact)
        rema[k] = exact - quota[k]
    # 2) 剩余名额按余数从大到小分配（最大余数法）
    left = N - sum(quota.values())
    for k, _ in sorted(rema.items(), key=lambda kv: -kv[1])[:left]:
        quota[k] += 1

    rng = random.Random(SEED)
    picked = []
    for k, v in sorted(strata.items()):
        n = min(quota[k], len(v))
        picked.extend(rng.sample(v, n))

    # 极端情况下配额仍不足（层太小），从剩余池随机补
    if len(picked) < N:
        got = {r["id"] for r in picked}
        pool = [r for r in rows if r["id"] not in got]
        picked.extend(rng.sample(pool, N - len(picked)))

    picked.sort(key=lambda r: r["id"])
    os.makedirs(DST_DIR, exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[写入] {DST}")
    print(f"  总题数 {len(picked)} / 源 {total}  唯一图像 "
          f"{len({r['meta']['same_image'] for r in picked})}")
    print(f"  答案分布 {dict(Counter(r['answer'] for r in picked))}")
    print(f"  器官分布 {dict(Counter(r['meta']['organ'] for r in picked))}")
    print(f"  题型分布 {dict(Counter(r['meta']['question_type'] for r in picked))}")
    print(f"  预计调用量 {len(picked)}×7 扰动×3 重复 = {len(picked)*21}")


if __name__ == "__main__":
    main()

"""把公开医学多模态数据集转成 medbench 的 JSONL 格式。

设计决定（重要，写论文时要说明）：

1. **只取闭集（yes/no 或有限候选）题**。开放式题（"这是什么器官"）无法用
   选项准确率评估，混进来会让"坍缩差"变成生成质量差异，混淆结论。丢弃的
   开放题数量会被打印出来，论文里作为数据筛选口径报告。

2. **options 只放标签（A/B/C/D），选项文本嵌进 question**。原因：真实 VLM
   调用时选项文本必须让模型看到，而 MockVLM 用 options 当标签做配对实验。
   把文本放 question、标签放 options，两套后端都不用改。

3. **保留 meta**（器官/模态/语言/答案类型/内容类型）。毕业论文的分层统计
   （混合效应模型、按器官分组）全靠这几个字段，丢了就没法做。

用法：
    python -m medbench.adapters --dataset slake --src /path/to/SLAKE --out data/slake
    python -m medbench.adapters --selftest          # 用假数据验证三个转换器
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

# 每个数据集的"闭集"判定：答案是这些词之一时才算闭集题
_CLOSED_VOCAB = {
    "yes", "no", "true", "false", "是", "否",
    "yes.", "no.", "male", "female", "left", "right",
}


def _is_closed(answer: str) -> bool:
    return answer.strip().lower().rstrip(".") in {v.rstrip(".") for v in _CLOSED_VOCAB} \
        or answer.strip().lower() in _CLOSED_VOCAB


def _label(i: int) -> str:
    return chr(ord("A") + i)


def _finalize(records: list[dict], out_dir: str, name: str) -> dict:
    """给闭集题分配选项标签，写出 JSONL，返回统计。"""
    # 选项集合按数据集实际出现过的闭集答案来建，保证答案一定在选项里。
    # 顺序按"是/否"→"女/男"→"左/右"的习惯排，避免出现 A=否 B=是 这种别扭情况。
    _order = {"yes": 0, "是": 0, "true": 0, "female": 1, "女": 1,
              "no": 2, "否": 2, "false": 2, "male": 3, "男": 3,
              "left": 4, "右": 5, "right": 5}
    vocab = sorted({r["answer"] for r in records},
                   key=lambda s: (_order.get(s.strip().lower().rstrip("."), 9), s))
    if len(vocab) > 4:            # 超过 4 个候选的题不适合做 MCQ
        keep = {v for v in vocab if v.lower().rstrip(".") in {"yes", "no"}}
        vocab = sorted(keep) or vocab[:4]
        records = [r for r in records if r["answer"] in vocab]
    if len(vocab) == 1:           # 单选项是退化题（永远答对），补上对立项
        only = vocab[0]
        pair = {"yes": "no", "no": "yes", "是": "否", "否": "是",
                "true": "false", "false": "true"}
        alt = pair.get(only.strip().lower())
        vocab = sorted({only, alt} if alt else {only, "N/A"},
                       key=lambda s: (_order.get(s.strip().lower().rstrip("."), 9), s))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.jsonl")
    opt_line = "  ".join(f"{_label(i)}. {v}" for i, v in enumerate(vocab))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            r = dict(r)
            r["options"] = [_label(i) for i in range(len(vocab))]
            r["answer"] = _label(vocab.index(r["answer"]))
            r["question"] = f"{r['question']}\n选项：{opt_line}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cats: dict[str, int] = {}
    for r in records:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    return {"out": out_path, "n": len(records), "options": vocab,
            "category": cats}


# --------------------------------------------------------------------------
# SLAKE（中英双语，放射 CT/MRI/X-ray，含器官与内容类型标注）
# 目录结构：SLAKE/{imgs/, test.json, train.json, validation.json}
# 字段：qid img_name question answer answer_type(OPEN/CLOSED) content_type lang
# --------------------------------------------------------------------------
def convert_slake(src: str, out_dir: str, splits=("test",), lang: str | None = "zh") -> dict:
    recs, dropped = [], {"open": 0, "lang": 0}
    for split in splits:
        p = os.path.join(src, f"{split}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到 {p}")
        with open(p, encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            if lang and it.get("q_lang") != lang:
                dropped["lang"] += 1
                continue
            ans = str(it.get("answer", "")).strip()
            if it.get("answer_type", "").upper() == "OPEN" or not _is_closed(ans):
                dropped["open"] += 1
                continue
            img = os.path.join(src, "imgs", it["img_name"])
            recs.append({
                "id": f"slake-{split}-{it['qid']}",
                "question": it["question"],
                "answer": ans,
                "image": os.path.abspath(img),
                "category": "image_dependent",
                "meta": {"dataset": "slake", "split": split, "lang": it.get("q_lang"),
                         "content_type": it.get("content_type"),
                         "answer_type": it.get("answer_type"),
                         "modality": it.get("modality"),
                         "organ": it.get("location"),
                         "same_image": it.get("img_name")},
            })
    st = _finalize(recs, out_dir, "slake")
    st["dropped"] = dropped
    return st


# --------------------------------------------------------------------------
# VQA-RAD（315 张放射图 / 3515 QA，全开放获取，最经典的 baseline）
# 文件：VQA_RAD Dataset Public.json（list）
# 字段：qid image_name question answer answer_type question_type image_organ position
# --------------------------------------------------------------------------
def convert_vqa_rad(src: str, out_dir: str) -> dict:
    cand = [f for f in os.listdir(src) if f.lower().endswith(".json")]
    if not cand:
        raise FileNotFoundError(f"{src} 下没有 json")
    p = os.path.join(src, sorted(cand, key=len)[0])
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    recs, dropped = [], 0
    for it in items:
        # 真实数据答案大小写混乱（"Yes"/"YES"/"yes" 混杂），统一小写，
        # 否则精确匹配评分会把同一答案当成两个类（2026-10-29 实测发现）
        ans = str(it.get("answer", "")).strip().lower()
        if it.get("answer_type", "").upper() == "OPEN" or not _is_closed(ans):
            dropped += 1
            continue
        recs.append({
            "id": f"vqarad-{it['qid']}",
            "question": it["question"],
            "answer": ans,
            "image": os.path.abspath(os.path.join(src, "VQA_RAD Image Folder",
                                                  it["image_name"])),
            "category": "image_dependent",
            "meta": {"dataset": "vqa_rad", "lang": "en",
                     "question_type": it.get("question_type"),
                     "organ": it.get("image_organ"),
                     "position": it.get("position"),
                     "same_image": it.get("image_name")},
        })
    st = _finalize(recs, out_dir, "vqa_rad")
    st["dropped"] = {"open": dropped}
    return st


# --------------------------------------------------------------------------
# PathVQA（病理图，4998 QA，开放获取；答案是长句的比例高，闭集子集较小）
# 结构：PathVQA/{images/, qas/test.json, qas/train.json}
# 字段：qid image question answer
# --------------------------------------------------------------------------
def convert_pathvqa(src: str, out_dir: str, split: str = "test") -> dict:
    p = os.path.join(src, "qas", f"{split}.json")
    if not os.path.exists(p):
        p = os.path.join(src, f"{split}.json")
    with open(p, encoding="utf-8") as f:
        items = json.load(f)
    recs, dropped = [], 0
    for it in items:
        ans = str(it.get("answer", "")).strip()
        if not _is_closed(ans):
            dropped += 1
            continue
        recs.append({
            "id": f"pathvqa-{it['qid']}",
            "question": it["question"],
            "answer": ans,
            "image": os.path.abspath(os.path.join(src, "images", it["image"])),
            "category": "image_dependent",
            "meta": {"dataset": "pathvqa", "lang": "en",
                     "modality": "pathology", "same_image": it.get("image")},
        })
    st = _finalize(recs, out_dir, "pathvqa")
    st["dropped"] = {"open": dropped}
    return st


# --------------------------------------------------------------------------
# IU X-Ray / Open-I（完全开放、无需认证，3955 份报告，含同一患者的正位+侧位）
# 只用于"同患者配对"分析，不做 MCQ，所以单独输出配对清单而不是 JSONL。
# --------------------------------------------------------------------------
def convert_iu_xray_pairs(src: str, out_dir: str) -> dict:
    """输出 {(study_id, frontal, lateral)} 配对清单，供"同患者跨视角扰动"使用。"""
    rp = os.path.join(src, "NLMCXR_reports")
    if not os.path.isdir(rp):
        rp = os.path.join(src, "reports")
    if not os.path.isdir(rp):
        raise FileNotFoundError(f"{src} 下找不到报告目录（NLMCXR_reports/reports）")
    rows = []
    for fn in sorted(os.listdir(rp)):
        if not fn.endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(rp, fn)).getroot()
        except ET.ParseError:
            continue
        sid = fn[:-4]
        imgs = [e.text for e in root.iter("parentImage") if e.text]
        if not imgs:
            continue
        rows.append({"id": f"iuxray-{sid}", "frontal": imgs[0],
                     "lateral": imgs[1] if len(imgs) > 1 else None,
                     "meta": {"dataset": "iu_xray", "n_views": len(imgs)}})
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "iu_xray_pairs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"out": out, "n": len(rows),
            "paired": sum(1 for r in rows if r["lateral"])}


# --------------------------------------------------------------------------
# 自检：造一份"结构相同、内容假"的源数据，验证转换器不依赖真实下载
# --------------------------------------------------------------------------
def _selftest() -> None:
    tmp = tempfile.mkdtemp(prefix="adapter_selftest_")
    ok = True

    # SLAKE
    s = os.path.join(tmp, "SLAKE")
    os.makedirs(os.path.join(s, "imgs", "xmlab1"), exist_ok=True)
    open(os.path.join(s, "imgs", "xmlab1", "source.jpg"), "wb").write(b"\xff\xd8\xff")
    json.dump([
        {"qid": 1, "img_name": "xmlab1/source.jpg", "question": "肝脏是否增大？",
         "answer": "是", "answer_type": "CLOSED", "content_type": "Abnormality",
         "q_lang": "zh", "modality": "CT", "location": "Liver"},
        {"qid": 2, "img_name": "xmlab1/source.jpg", "question": "脾脏是否增大？",
         "answer": "否", "answer_type": "CLOSED", "content_type": "Abnormality",
         "q_lang": "zh", "modality": "CT", "location": "Spleen"},
        {"qid": 3, "img_name": "xmlab1/source.jpg", "question": "这是什么器官？",
         "answer": "Liver", "answer_type": "OPEN", "content_type": "Organ",
         "q_lang": "zh", "modality": "CT", "location": "Liver"},
        {"qid": 4, "img_name": "xmlab1/source.jpg", "question": "Is the liver enlarged?",
         "answer": "Yes", "answer_type": "CLOSED", "content_type": "Abnormality",
         "q_lang": "en", "modality": "CT", "location": "Liver"},
    ], open(os.path.join(s, "test.json"), "w", encoding="utf-8"), ensure_ascii=False)
    st = convert_slake(s, os.path.join(tmp, "out"), lang="zh")
    print("[selftest] SLAKE  :", st["n"], "闭集题 /", st["dropped"], "丢弃", st["options"])
    ok &= st["n"] == 2 and st["dropped"]["open"] == 1 and st["dropped"]["lang"] == 1

    # VQA-RAD
    v = os.path.join(tmp, "VQA_RAD")
    os.makedirs(os.path.join(v, "VQA_RAD Image Folder"), exist_ok=True)
    open(os.path.join(v, "VQA_RAD Image Folder", "1.png"), "wb").write(b"\x89PNG\r\n\x1a\n")
    json.dump([
        {"qid": 1, "image_name": "1.png", "question": "Is there a nodule?",
         "answer": "yes", "answer_type": "CLOSED", "question_type": "ABN",
         "image_organ": "LUNG", "position": "CHEST"},
        {"qid": 2, "image_name": "1.png", "question": "What organ is this?",
         "answer": "lung", "answer_type": "OPEN", "question_type": "ORGAN",
         "image_organ": "LUNG", "position": "CHEST"},
    ], open(os.path.join(v, "VQA_RAD Dataset Public.json"), "w", encoding="utf-8"))
    st = convert_vqa_rad(v, os.path.join(tmp, "out"))
    print("[selftest] VQA-RAD:", st["n"], "闭集题 /", st["dropped"], "丢弃", st["options"])
    ok &= st["n"] == 1 and st["dropped"]["open"] == 1

    # PathVQA（正确答案是 "yes"/"no"，另一题是长句）
    pw = os.path.join(tmp, "PathVQA")
    os.makedirs(os.path.join(pw, "images"), exist_ok=True)
    os.makedirs(os.path.join(pw, "qas"), exist_ok=True)
    open(os.path.join(pw, "images", "a.jpg"), "wb").write(b"\xff\xd8\xff")
    json.dump([
        {"qid": 1, "image": "a.jpg", "question": "Are there mitotic figures?",
         "answer": "yes"},
        {"qid": 2, "image": "a.jpg", "question": "What is the diagnosis?",
         "answer": "adenocarcinoma"},
    ], open(os.path.join(pw, "qas", "test.json"), "w", encoding="utf-8"))
    st = convert_pathvqa(pw, os.path.join(tmp, "out"))
    print("[selftest] PathVQA:", st["n"], "闭集题 /", st["dropped"], "丢弃", st["options"])
    ok &= st["n"] == 1 and st["dropped"]["open"] == 1

    # IU X-Ray 配对清单
    iu = os.path.join(tmp, "IU")
    os.makedirs(os.path.join(iu, "NLMCXR_reports"), exist_ok=True)
    xml = ("<eCitation><MedlineCitation><Article><Abstract>"
           "<AbstractText Label='FINDINGS'>normal</AbstractText></Abstract>"
           "</Article></MedlineCitation>"
           "<parentImage id='1'>CXR1_IM-0001-1001.png</parentImage>"
           "<parentImage id='2'>CXR1_IM-0001-2001.png</parentImage>"
           "</eCitation>")
    open(os.path.join(iu, "NLMCXR_reports", "CXR1.xml"), "w", encoding="utf-8").write(xml)
    st = convert_iu_xray_pairs(iu, os.path.join(tmp, "out"))
    print("[selftest] IU-Xray:", st["n"], "份报告 /", st["paired"], "含双视角")
    ok &= st["n"] == 1 and st["paired"] == 1

    # 转换结果能被 data.load_dataset 读回，且选项/答案自洽
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from medbench.data import load_dataset
    for name in ("slake", "vqa_rad", "pathvqa"):
        p = os.path.join(tmp, "out", f"{name}.jsonl")
        ds = load_dataset(p)
        assert ds, name
        for s in ds:
            assert s.answer in s.options, (name, s.answer, s.options)
        print(f"[selftest] 读回 {name}.jsonl: {len(ds)} 条，答案均在选项内；"
              f"示例 question={ds[0].question.splitlines()[0]!r} "
              f"answer={ds[0].answer} meta={ds[0].meta}")

    print("[selftest]", "全部通过" if ok else "有断言失败")
    print("[selftest] 临时目录:", tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description="公开医学多模态数据集 → medbench JSONL")
    ap.add_argument("--dataset", choices=["slake", "vqa_rad", "pathvqa", "iu_xray"])
    ap.add_argument("--src", help="原始数据集目录")
    ap.add_argument("--out", default="data/real", help="输出目录")
    ap.add_argument("--lang", default="zh", help="SLAKE 语言过滤：zh / en / all")
    ap.add_argument("--splits", default="test", help="SLAKE 使用的 split，逗号分隔")
    ap.add_argument("--selftest", action="store_true", help="用假数据验证转换器")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return
    if not (a.dataset and a.src):
        ap.error("需要 --dataset 和 --src（或加 --selftest）")

    if a.dataset == "slake":
        st = convert_slake(a.src, a.out, tuple(a.splits.split(",")),
                           None if a.lang == "all" else a.lang)
    elif a.dataset == "vqa_rad":
        st = convert_vqa_rad(a.src, a.out)
    elif a.dataset == "pathvqa":
        st = convert_pathvqa(a.src, a.out)
    else:
        st = convert_iu_xray_pairs(a.src, a.out)
    print(json.dumps(st, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

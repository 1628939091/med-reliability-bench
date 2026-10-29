# MedReliability-Bench — 医学多模态 VLM 可靠性评测基准

> 应用统计学研一 · 项目一 · 可拓展为毕业论文 + NLP/多模态顶会短文

## 一句话介绍

给医学多模态大模型（VLM）做"体检"的框架：**它是不是真的在看图？答案稳定吗？置信度可信吗？** 全部用统计方法量化，零 GPU 也能跑（API 推理 / Mock 演示）。

## 它解决什么问题（论文动机）

2025–2026 年多篇医学 VQA 研究发现了一个尴尬的现象：

1. **模态坍缩**：给模型一张空白图，准确率几乎不降——说明模型根本没看图，全靠语言先验在猜；
2. **语言捷径**：纯文本输入和图文输入的答案高度一致；
3. **答案不稳定**：同一道题问 5 次，答案可能不一样；
4. **校准差**：模型说"我 95% 确定"，实际正确率远没有 95%。

医疗场景里这四个问题每一个都是事故隐患，但现有医学 VLM 评测（如 MedVLMBench）只报准确率，**没有系统的扰动分析、没有统计校准、没有一致性指标**——这就是本项目（也是你的论文）的切入空隙。

## 已有实测结果（SLAKE 测试集 355 道闭集题 × 7 扰动 × 3 重复 = 7,454 次调用）

被测模型：`Qwen/Qwen3-VL-8B-Instruct`（硅基流动托管，temperature=0）

| 扰动 | 准确率 | 95%CI（按题聚类） | Yes 率 | 相对原图 |
|---|---|---|---|---|
| 原图 | **0.684** | [0.638, 0.732] | 0.590 | — |
| 高斯噪声 | 0.660 | [0.613, 0.711] | 0.581 | −0.023 |
| 上下翻转 | 0.669 | [0.620, 0.717] | 0.571 | −0.015 |
| 降分辨率 | 0.638 | [0.590, 0.687] | 0.466 | −0.045 |
| **图像置空** | **0.449** | [0.397, 0.498] | **0.122** | −0.235 |
| 换成别的题的图 | 0.446 | [0.394, 0.498] | 0.380 | −0.238 |
| 完全不给图 | 0.480 | [0.431, 0.532] | 0.577 | −0.204 |

**三个反直觉的结论（详见 `outputs/analysis_full.md`）：**

1. **聚合坍缩差 Δ=+0.235 是假象**：按答案方向拆开，gold=Yes 的题 Δ=**+0.712**、gold=No 的题 Δ=**−0.230**。图像的价值几乎全在"让模型敢答 Yes"；对"No 类"问题，不给图反而更准（条件反射答 No 蒙对）。单一 Δ 会同时低估图像作用、掩盖模型的"假会"。
2. **"置空图" ≠ "不给图"**：两者准确率接近（0.449 vs 0.480），但 Yes 率差 4.7 倍（0.122 vs 0.577）、答案熵 0.535 vs 0.983 bit。模型把空图读作"没有证据→答 No"，把缺图读作"从文字推断"。**评测设计混用这两种条件会得到互相矛盾的结论。**
3. **校准是最大短板**：平均置信度 0.961 vs 实际准确率 0.684，ECE **0.277**，1,058/1,065 条记录挤在置信度 0.9–1.0 区间——置信度几乎无区分度，不能用于临床决策阈值。

> 统计口径：同一题问 3 次不独立，置信区间用**按题目聚类的 Bootstrap**（B=2000）；配对比较用 McNemar 精确检验（p=4.3e−09）。子组 n<20 标 ⚠️ 仅作描述。

复现命令：

```bash
python run_demo.py configs/real_slake_api_full.json   # 全量评测（支持断点续跑）
python analyze_full.py                                # 分层统计 + 出图
```

## 项目结构

```
med-reliability-bench/
├── run_demo.py                 # 一键入口（配置驱动）
├── analyze_full.py             # 阶段 3：聚类 Bootstrap / McNemar / 分层 / 出图
├── debug_vlm.py                # 逐题调试：打印模型原始回复与解析结果
├── configs/
│   ├── demo_config.json        # Mock 演示（40 题合成数据，零成本）
│   ├── real_slake_mock.json    # 真实 SLAKE 数据 + Mock 模型
│   ├── real_slake_api_sanity.json  # 10 题 sanity check（几分钱）
│   └── real_slake_api_full.json    # 全量 355 题 × 7 扰动 × 3 次
├── medbench/
│   ├── adapters.py             # 适配层：SLAKE / VQA-RAD / PathVQA → 统一 JSONL
│   ├── data.py                 # 数据层：Sample 结构 + 演示集生成
│   ├── perturb.py              # 扰动套件：7 种破坏 + 缓存 + 降级可见
│   ├── models.py               # 模型层：MockVLM(4 种缺陷人格) + OpenAI 兼容 API
│   ├── metrics.py              # 指标层：Bootstrap CI / 坍缩差 / pass^k / 熵 / ECE
│   ├── runner.py               # 编排层：三级容错 + 断点续跑 + 拒答捕获
│   └── report.py               # 报告层：Markdown 表格 + 自动诊断
├── data/real_slake/dataset.jsonl       # 355 道闭集题（含 lang/modality 元数据）
├── data/real_vqarad/dataset.jsonl      # 1,193 道闭集题（交叉验证用）
└── outputs/                    # 评测记录 / 报告 / 图表（运行时生成）
```

> 原始数据（SLAKE 212MB、VQA-RAD 315 张图）不入库，按第 3 章「怎么换成真实实验数据」自行下载。

## 文档

| 文档 | 内容 |
|---|---|
| [项目一_从零到面试_完整教程.md](项目一_从零到面试_完整教程.md) | 13 章 + 3 附录：概念入门 → 架构 → 数据 → 扰动 → 指标 → 统计 → API 接入 → 结果解读 → **8 个踩坑复盘** → **20 题面试题库（含答案）** → 50 个术语表 |
| [项目一_行动路线.md](项目一_行动路线.md) | 六阶段路线 + 每日进度日志 + 坑的原始记录 |
| `outputs/analysis_full.md` | 分层分析报告（运行 `analyze_full.py` 生成） |

## 快速开始

```bash
# 1. Mock 演示（零成本，无需任何 API/网络/GPU）
python run_demo.py

# 2. 接真实模型：编辑 configs/demo_config.json，把 mock 条目换成：
# { "backend": "api", "name": "Qwen-VL-Max",
#   "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
#   "api_key": "sk-你的key", "model": "qwen-vl-max" }
python run_demo.py configs/demo_config.json

# 可选依赖：pip install pillow  （启用噪声/翻转/低分辨率像素级扰动）
```

## Mock 演示会看到什么

配置里 4 个 Mock 模型分别模拟真实 VLM 的 4 种典型缺陷，报告会自动诊断出来：

| Mock 模型 | 模拟的缺陷 | 报告中的信号 |
|---|---|---|
| 只看文字 | 模态坍缩 | 坍缩差 Δ = 0.000 |
| 正常看图 | 无（对照组） | Δ 显著 > 0，ECE 很小 |
| 答案不稳定 | 推理随机性 | pass^5 只有 0.05 |
| 过度自信 | 校准失败 | ECE = 0.3+ |

**这段"Mock 复现了真实论文报告的现象"本身就是你论文 Method 章的有效性验证（sanity check）。**

## 怎么换成真实实验数据

**已经写好了转换器**：`medbench/adapters.py`，一行命令把公开数据集转成本项目的 JSONL。

```bash
# 不下载数据也能先验证代码正确（造结构相同、内容假的源数据，跑通四个转换器）
python -m medbench.adapters --selftest

# SLAKE（中英双语，主力数据集；--lang zh 只取中文题）
python -m medbench.adapters --dataset slake --src /path/to/SLAKE --out data/real --lang zh
# VQA-RAD
python -m medbench.adapters --dataset vqa_rad --src /path/to/VQA_RAD --out data/real
# PathVQA（提供"病理"这个不同模态，用于分层对比）
python -m medbench.adapters --dataset pathvqa --src /path/to/PathVQA --out data/real
# IU X-Ray：不做选择题，输出"同患者正位+侧位"配对清单
python -m medbench.adapters --dataset iu_xray --src /path/to/IU --out data/real
```

三个已经定好的设计决定（写论文时要交代）：

1. **只取闭集题**。开放式题（"这是什么器官"）无法用选项准确率评估，混进来会让"坍缩差"
   变成"生成文风差异"。丢弃数量脚本会打印，作为数据筛选口径报告。
2. **选项标签与文本分离**：选项文本嵌进 `question`，`options` 只放标签 `A/B/C/D`。
   这样 Mock 后端（用 options 做配对实验）和真实 API 后端（需要看到选项文本）都不用改。
3. **保留 `meta`**（器官/模态/语言/答案类型/同图 id）。分层统计、混合效应模型全靠这几个字段——
   丢了就只能报总分，论文第 4 章的异质性分析做不了。

数据格式如下（`meta` 可选但强烈建议保留）：

```json
{"id":"slake-test-1","question":"肝脏是否增大？\n选项：A. 是  B. 否","options":["A","B"],
 "answer":"A","image":"/abs/path/imgs/xmlab1/source.jpg","category":"image_dependent",
 "meta":{"dataset":"slake","lang":"zh","modality":"CT","organ":"Liver","same_image":"xmlab1/source.jpg"}}
```

- `image_dependent`：看图才能答的题（扰动实验主要看这类）
- `text_prior`：语言先验即可猜的题（语言捷径分析看这类）
- 二元选择设计是为了统计功效：40 题二元题 ≈ 100 题开放题的信息量，且 McNemar 检验直接可用
- 注意：`category` 在真实数据里**不应该靠人工假设**，更严谨的做法是用 `text_only` 扰动实测
  "不给图能否答对"来**经验分层**——这是论文方法章的一个小贡献点

## 与论文的对应关系（重要）

| 代码模块 | 论文章节 |
|---|---|
| perturb.py | §3 Benchmark Design: Perturbation Suite |
| metrics.py 四个指标 | §3 Evaluation Dimensions |
| runner.py 复现协议 | §4 Experimental Setup（可复现性声明） |
| report.md 自动诊断 | §5 Results & Analysis |
| Mock sanity check | §3.5 Validity Check |

详细论文大纲见《论文拓展指南》文档。

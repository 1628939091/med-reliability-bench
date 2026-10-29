"""模型后端：MockVLM（本地演示用）与 OpenAICompatVLM（真实 API 用）。

所有模型实现统一接口：
    answer(question, options, gold, image_path, perturbation, rep) -> (pred, conf)
其中 gold 是标准答案（Mock 用它模拟"答对/答错"，API 模型忽略该参数），
perturbation 是扰动名（"identity" 表示干净图），rep 是重复采样序号。
Mock 模型用"可配置的缺陷行为"来模拟真实 VLM 的典型问题，
让你在零 API 成本下看到整套指标如何工作。
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import urllib.error
import urllib.request


# --------------------------------------------------------------------------
# Mock 模型：4 种典型行为，对应论文要揭露的 4 类问题
# --------------------------------------------------------------------------
class MockVLM:
    """模拟 VLM。strategy 说明：
      text_prior    : 完全不看图，永远回答第一个选项 → 会触发"模态坍缩"信号
      vision_ok     : 图像完好时答得不错且校准尚可，图像被破坏后掉到随机水平
      unstable      : 同一道题反复问，答案不稳定 → pass^k 很低
      overconfident : 正确率一般但置信度永远接近 1 → ECE 很大
    """

    def __init__(self, name: str, strategy: str, seed: int = 0):
        self.name = name
        self.strategy = strategy
        self.seed = seed

    def answer(self, question, options, gold, image_path, perturbation, rep):
        # 种子里必须带 image_path：不同题目各自独立采样，否则相关性会让
        # pass^k / 坍缩差等统计量失真（真实 API 模型天然满足这一点）
        rng = random.Random(f"{self.seed}|{image_path}|{question}|{perturbation}|{rep}")
        a_correct = rng.random()  # 仅用于让每次重复采样结果不同

        if self.strategy == "text_prior":
            # 永远选 A、置信度恒定 —— 图像怎么扰动都无所谓 → 坍缩差≈0
            return options[0], 0.90

        if self.strategy == "unstable":
            p_correct = 0.60
            conf = 0.75
        elif self.strategy == "overconfident":
            p_correct = 0.70
            conf = 0.98
        elif self.strategy == "vision_ok":
            if perturbation in ("identity", "low_res", "noise"):
                p_correct, conf = 0.85, 0.85
            else:  # blank / swap / flip：图像信息被破坏 → 退化到随机猜
                p_correct, conf = 0.50, 0.60
        else:
            raise ValueError(f"unknown strategy {self.strategy}")

        correct = a_correct < p_correct
        # 答对时输出标准答案，答错时输出另一个选项 —— 模拟真实 VLM 行为
        other = next(o for o in options if o != gold)
        return (gold if correct else other), conf


# --------------------------------------------------------------------------
# 真实模型：任何 OpenAI 兼容接口（Qwen-VL / GLM-4V / Gemini 中转 / vLLM 本地服务）
# --------------------------------------------------------------------------
class RefusalError(RuntimeError):
    """模型/平台拒答（内容审核 451）。评测中记为一次"拒答"观测，不中断整跑。"""


class APITransientError(RuntimeError):
    """重试全部用尽后仍失败的服务端/网络瞬态错误。
    runner 捕获后跳过该题继续跑（不写入 checkpoint，下次续跑会自动补测）。"""


class OpenAICompatVLM:
    def __init__(self, name: str, base_url: str, api_key: str,
                 model: str, temperature: float = 0.7):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def answer(self, question, options, gold, image_path, perturbation, rep):
        content: list[dict] = [
            {"type": "text",
             "text": (
                 f"问题：{question}\n选项："
                 + " ".join(f"{o}" for o in options)
                 + "\n请只回答一个 JSON：{\"answer\": \"选项字母\", "
                   "\"confidence\": 0到1之间的小数}"
             )},
        ]
        if image_path and perturbation != "text_only":
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }).encode()

        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        # 长跑保险：429 限流 / 5xx / 网络抖动自动重试（指数退避），避免几小时
        # 的评测因单次瞬态错误前功尽弃。仅重试"值得重试"的错误。
        import time
        data = None
        # 5 次尝试，退避 3/8/20/60s——服务器抖动常持续 1-2 分钟，3 次不够
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")[:800]
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    wait = [3, 8, 20, 60][attempt]
                    print(f"      [retry] API {e.code}，{wait}s 后第"
                          f"{attempt + 2}次尝试: {body[:120]}", flush=True)
                    time.sleep(wait)
                    continue
                # 451 / 内容审核：不可重试，但绝不能中断整跑。
                # 医疗影像常触发平台内容审核，拒答本身是可靠性观测指标。
                if e.code == 451 or "prohibited or sensitive" in body:
                    raise RefusalError(body) from None
                if e.code in (429, 500, 502, 503, 504):
                    raise APITransientError(
                        f"API {e.code} 重试4次仍失败 | {body}") from None
                raise RuntimeError(
                    f"API {e.code} {e.reason} | 响应体: {body}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 4:
                    wait = [3, 8, 20, 60][attempt]
                    print(f"      [retry] 网络异常 {e}，{wait}s 后重试",
                          flush=True)
                    time.sleep(wait)
                    continue
                raise APITransientError(f"网络异常重试4次仍失败: {e}") from None
        msg = data["choices"][0].get("message") or {}
        text = msg.get("content") or ""
        if not str(text).strip():
            # 思考型模型（GLM-4.5V / Qwen-Thinking 等）可能把正文放在
            # reasoning_content，content 为空。不回退就会全部解析失败。
            text = msg.get("reasoning_content") or ""
        text = str(text)
        self.last_raw = text   # 保留原始回复，便于调试解析问题
        pred, conf = self._parse(text, options)
        # 粗粒度格式遵循率统计：解析层完全没看到答案标记，说明是兜底值
        if '"answer"' not in text and "答案" not in text and "answer" not in text.lower():
            self.parse_fail = getattr(self, "parse_fail", 0) + 1
        return pred, conf

    @staticmethod
    def _parse(text: str, options):
        """从回复里抠出答案字母和置信度；解析失败按最保守处理。

        兼容三类常见输出（跨模型必需，Qwen 与 GLM 的格式习惯不同）：
          1. 严格 JSON：{"answer": "B", "confidence": 0.9}
          2. 代码块包裹：```json { ... } ```
          3. 自然语言：「答案：B」「The answer is B」
        思考型模型会在正式答案前输出大段推理，所以**从后往前**取最后一个
        答案标记——推理过程里出现的示例 JSON 不应被当成本题答案。
        """
        raw = (text or "")
        # 剥掉 think 包裹段（部分平台把推理塞进 content）
        raw = re.sub(r"<think[^>]*>.*?</think>", " ", raw, flags=re.S)

        # 优先取最后一个含 answer 的 JSON 对象
        cand = raw
        blocks = re.findall(r"\{[^{}]*\"answer\"[^{}]*\}", raw)
        if blocks:
            cand = blocks[-1]

        m = re.search(r'"answer"\s*:\s*"([A-Za-z])"', cand) or \
            re.search(r'"answer"\s*:\s*([A-Za-z])', cand)
        letter = m.group(1).upper() if m else None

        if letter is None:                       # 自然语言兜底
            opts = "".join(re.escape(o) for o in options)
            hits = re.findall(
                rf"(?:答案|answer\s*(?:is)?)\s*[:：]?\s*([{opts}])(?![A-Za-z])",
                raw, flags=re.I)
            if hits:
                letter = hits[-1].upper()
            else:
                lone = [o for o in options if re.search(rf"\b{o}\b", raw)]
                letter = lone[0] if len(lone) == 1 else None

        pred = next((o for o in options if o.upper() == (letter or "")), options[0])
        m2 = re.search(r'"confidence"\s*:\s*([0-9.]+)', cand) or \
            re.search(r"confidence[^0-9]{0,12}([0-9.]+)", raw, flags=re.I)
        conf = float(m2.group(1)) if m2 else 0.5
        if conf > 1.0:                            # 有的模型输出百分数
            conf /= 100.0
        return pred, min(max(conf, 0.0), 1.0)


def build_model(cfg: dict):
    """配置工厂：{"backend":"mock","name":...,"strategy":...}
                或 {"backend":"api","base_url":...,"api_key":...,"model":...}"""
    if cfg["backend"] == "mock":
        return MockVLM(cfg["name"], cfg["strategy"], cfg.get("seed", 0))
    if cfg["backend"] == "api":
        # key 不写进配置文件（防泄漏），优先环境变量 MRB_API_KEY
        api_key = cfg.get("api_key") or os.environ.get("MRB_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 API key：请先在终端设置 $env:MRB_API_KEY=\"sk-...\"")
        return OpenAICompatVLM(cfg["name"], cfg["base_url"], api_key,
                               cfg["model"], cfg.get("temperature", 0.7))
    raise ValueError(f"unknown backend: {cfg['backend']}")

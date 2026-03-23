"""
情感分析模块

使用 DeepSeek API 对预处理后的文本进行情感分类。

情感标签定义：
  - positive  : 文本表达正面、赞扬、感激、满意或积极期待的情感
  - negative  : 文本表达负面、批评、沮丧、失望或不满的情感
  - neutral   : 文本以客观陈述为主，几乎不含明显情感色彩
  - mixed     : 文本同时包含明显的正面和负面情感，难以归入单一类别

Prompt 设计原则：
  1. 清晰的标签定义 + 判断维度说明
  2. 正面/反面典型示例（few-shot）
  3. 严格的 JSON 输出格式要求
  4. 针对技术社区（GitHub Issues/PR）的领域适配说明
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一名专业的情感分析助手，擅长分析 GitHub 技术社区（Issues、Pull Requests、评论）中的文本情感。

## 情感标签定义

| 标签 | 含义 | 典型特征 |
|------|------|---------|
| positive | 正面情感 | 赞扬、感谢、满意、积极期待、支持、鼓励、庆祝功能合并/修复等 |
| negative | 负面情感 | 批评、抱怨、沮丧、失望、不满、愤怒、报告 Bug 时的强烈措辞、拒绝/否定 PR 等 |
| neutral  | 中性/客观 | 技术描述、功能说明、代码说明、步骤描述，几乎无情感色彩 |
| mixed    | 混合情感 | 同时包含明显正面和负面情感，例如"感谢你的贡献，但这个实现有严重问题" |

## 判断维度说明

1. **情感极性**：文本整体倾向是正面、负面还是中性？
2. **情感强度**：情感表达是强烈（感叹号、大写、极端词汇）还是温和？
3. **领域背景**：
   - "This is broken" / "bug" / "regression" → 负面
   - "Thank you" / "LGTM" / "Great work" / "Merged" → 正面
   - "Added support for X" / "This PR implements Y" → 通常中性
   - "I love this feature, but the performance is terrible" → 混合
4. **上下文敏感**：技术术语本身（如 "error", "warning"）不自动代表负面情感，需结合语境判断。

## 输出格式（严格 JSON）

```json
{
  "label": "<positive|negative|neutral|mixed>",
  "confidence": <0.0 到 1.0 之间的浮点数，表示你对此判断的把握程度>,
  "reasoning": "<不超过 100 字的简短分析理由>"
}
```

**重要**：只输出上述 JSON，不要添加任何额外解释、Markdown 代码块或其他内容。"""

# Few-shot 示例（作为 user/assistant 对话轮次注入）
FEW_SHOT_EXAMPLES = [
    # --- positive ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nThank you so much for this fix! I've been waiting for this for months. Works perfectly now.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "positive",
                "confidence": 0.97,
                "reasoning": "作者表达了强烈的感谢与满意，使用了'Thank you so much'、'perfectly'等正面词汇，情感明确且强烈。",
            },
            ensure_ascii=False,
        ),
    },
    # --- negative ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nThis is completely broken after the last update. The regression in #1234 was never fixed and now it's even worse. Very disappointing.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "negative",
                "confidence": 0.95,
                "reasoning": "作者明确表达不满，使用'completely broken'、'never fixed'、'even worse'、'Very disappointing'等强烈负面词汇。",
            },
            ensure_ascii=False,
        ),
    },
    # --- neutral ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nAdds support for async iteration in the new streaming API. Updated the documentation and added unit tests for the new endpoint.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "neutral",
                "confidence": 0.92,
                "reasoning": "文本为技术描述，客观陈述了新增功能和文档更新，无明显情感色彩。",
            },
            ensure_ascii=False,
        ),
    },
    # --- mixed ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nGreat idea and I appreciate the effort, but the current implementation has a serious memory leak that will cause production issues. We need to rethink this approach.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "mixed",
                "confidence": 0.88,
                "reasoning": "前半部分表示赞赏（'Great idea', 'appreciate'），后半部分指出严重问题（'serious memory leak', 'production issues'），正负情感并存。",
            },
            ensure_ascii=False,
        ),
    },
]

USER_PROMPT_TEMPLATE = "请分析以下文本的情感：\n\n{text}"


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """
    调用 DeepSeek API 完成情感分类。

    使用方式::

        analyzer = SentimentAnalyzer(config)
        result = analyzer.analyze("This is a great feature!")
        # result = {"label": "positive", "confidence": 0.95, "reasoning": "..."}
    """

    VALID_LABELS = {"positive", "negative", "neutral", "mixed"}

    def __init__(self, config: Config):
        self._config = config
        self._client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
        )
        self._model = config.deepseek_model

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        对单段文本进行情感分析。

        返回字典格式：
            {
                "label": "positive" | "negative" | "neutral" | "mixed",
                "confidence": float,   # 0.0 ~ 1.0
                "reasoning": str,
                "error": str | None,   # 分析失败时的错误信息
            }
        """
        if not text or not text.strip():
            return {"label": "neutral", "confidence": 1.0, "reasoning": "文本为空", "error": None}

        messages = self._build_messages(text)
        raw_response = self._call_api_with_retry(messages)

        if raw_response is None:
            return {
                "label": "neutral",
                "confidence": 0.0,
                "reasoning": "API 调用失败",
                "error": "API 调用失败，已超过最大重试次数",
            }

        return self._parse_response(raw_response)

    def analyze_batch(self, texts: list) -> list:
        """
        批量分析文本（顺序调用，自动加入请求间隔）。

        返回与输入顺序对应的结果列表。
        """
        results = []
        for i, text in enumerate(texts):
            result = self.analyze(text)
            results.append(result)
            if i < len(texts) - 1:
                time.sleep(self._config.request_delay)
        return results

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _build_messages(self, text: str) -> list:
        """构建包含系统提示和 few-shot 示例的消息列表"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append(
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        )
        return messages

    def _call_api_with_retry(self, messages: list) -> Optional[str]:
        """
        调用 DeepSeek API，失败时自动重试（指数退避）。
        返回模型原始输出字符串，失败返回 None。
        """
        last_error = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.0,   # 情感分类需要确定性输出
                    max_tokens=256,
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                wait = 60 * attempt
                logger.warning("速率限制，等待 %d 秒后重试（第 %d/%d 次）", wait, attempt, self._config.max_retries)
                time.sleep(wait)
                last_error = e
            except APITimeoutError as e:
                wait = 5 * attempt
                logger.warning("请求超时，等待 %d 秒后重试（第 %d/%d 次）", wait, attempt, self._config.max_retries)
                time.sleep(wait)
                last_error = e
            except APIError as e:
                logger.error("API 错误（第 %d/%d 次）: %s", attempt, self._config.max_retries, e)
                last_error = e
                if attempt < self._config.max_retries:
                    time.sleep(2 ** attempt)

        logger.error("已达最大重试次数，最后错误: %s", last_error)
        return None

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        解析模型返回的 JSON 字符串。
        若解析失败则尝试容错提取。
        """
        # 有时模型会包裹在 ```json ... ``` 中，先去除
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("JSON 解析失败，尝试容错提取。原始输出: %s", raw[:200])
            data = self._fallback_extract(cleaned)

        label = str(data.get("label", "neutral")).lower()
        if label not in self.VALID_LABELS:
            logger.warning("未知标签 '%s'，回退为 'neutral'", label)
            label = "neutral"

        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        return {
            "label": label,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")),
            "error": None,
        }

    @staticmethod
    def _fallback_extract(text: str) -> dict:
        """
        当 JSON 解析失败时，尝试用正则表达式提取关键字段。
        """
        result = {"label": "neutral", "confidence": 0.5, "reasoning": "解析失败"}

        label_match = re.search(r'"label"\s*:\s*"(\w+)"', text)
        if label_match:
            result["label"] = label_match.group(1)

        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        if reason_match:
            result["reasoning"] = reason_match.group(1)

        return result

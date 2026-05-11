"""
情感分析模块

使用 DeepSeek API 对预处理后的文本进行情感分类。

情感标签定义（面向软件开发场景的细粒度体系）：
  - functional_performance   : 表达对语言特性、性能、生态或开发体验的评价；
                               针对运行效率、稳定性、兼容性等方面的正面或负面反馈
  - usability_learning       : 围绕语法复杂度、API 设计、调试难度等提出的意见或反馈
  - documentation_ecosystem  : 针对文档质量、社区支持、第三方库生态的看法
  - neutral_factual          : 主要陈述问题现象或使用场景，不带明显情绪色彩
  - other_undetermined       : 包括讽刺、复杂情绪或信息不足样本，无法归入上述类别

Prompt 设计原则：
  1. 清晰的标签定义 + 判断维度说明
  2. 各类别典型示例（few-shot）
  3. 严格的 JSON 输出格式要求
  4. 针对技术社区（GitHub Issues/PR）的领域适配说明
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一名软件工程与自然语言处理领域的专家，正在进行方面级情感分析（Aspect-Based Sentiment Analysis）标注任务。

                   任务：
                   给定一段关于 Rust 编程语言的评论，请识别其中涉及的“评价维度（aspect）”，并判断每个维度的情感倾向与强度。

                   【aspect定义】
                   aspect 是“评价的角度”，例如：
                   - performance（性能）
                   - learning_curve（学习成本）
                   - maintainability（可维护性）

                   注意：
                   aspect 不是具体技术名词（如 borrow checker），而是抽象评价维度。

                   ---

                   【可选aspect列表】
                   请只从以下列表中选择：

                   Language:
                   - ownership
                   - type_system
                   - safety
                   - performance

                   Experience:
                   - learning_curve
                   - compile_time
                   - error_message
                   - debugging

                   Engineering:
                   - maintainability
                   - readability
                   - extensibility
                   - api_design

                   Ecosystem:
                   - package_manager
                   - libraries
                   - framework_support
                   - community

                   ---

                   【情感定义】
                   - positive：正面评价
                   - negative：负面评价
                   - neutral：客观描述 / 无明显情感

                   ---

                   【评分（score）定义】
                   score 用于表示情感强度，必须为以下离散值之一：

                   - 2：强正面（非常好、强烈推荐等）
                   - 1：弱正面（还不错、可以接受等）
                   - 0：中性（客观描述，无明显评价）
                   - -1：弱负面（有点问题、不太好等）
                   - -2：强负面（很差、无法接受等）

                   要求：
                   - score 必须与 sentiment 一致：
                     - positive → 1 或 2
                     - negative → -1 或 -2
                     - neutral → 0

                   ---

                   【标注要求】
                   1. 一条评论可以包含多个 aspect（多标签），但最多不要超过三个
                   2. 每个 aspect 必须对应一个 sentiment 和 score
                   3. score 必须从规定的离散值中选择
                   4. 如果没有涉及任何 aspect，返回空列表
                   5. 只允许使用给定的 aspect 列表，不要自行创造新标签
                   6. 不要将具体技术术语（如 borrow checker、lifetime）作为 aspect
                   7. 不要过度推断，仅根据文本中明确或较明显的信息判断

                   ---

                   【输入文本】
                   {text}

                   ---

                   【输出格式（严格JSON）】
                   {
                     "annotations": [
                       {
                         "aspect": "...",
                         "sentiment": "...",
                         "score": ...
                       }
                     ]
                   }"""

# Few-shot 示例（作为 user/assistant 对话轮次注入）
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nRust is memory safe but very hard to learn",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "annotations": [
                    {"aspect": "safety", "sentiment": "positive", "score": 1},
                    {"aspect": "learning_curve", "sentiment": "negative", "score": -2},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nRust compile time is too slow",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "annotations": [
                    {"aspect": "compile_time", "sentiment": "negative", "score": -2}
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nRust uses an ownership model",
    },
    {
        "role": "assistant",
        "content": json.dumps({"annotations": []}, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nRust has great performance\n\n### System\nOS: Linux",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "annotations": [
                    {"aspect": "performance", "sentiment": "positive", "score": 2}
                ]
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
    调用 DeepSeek API 完成方面级情感分类（JSON-only）。

    使用方式::

        analyzer = SentimentAnalyzer(config)
        result = analyzer.analyze("This is a great feature!")
        # result = {"annotations": [...], "error": None}
    """

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
                "text": str,
                "annotations": [
                    {
                        "aspect": str,
                        "sentiment": "positive" | "negative" | "neutral",
                        "score": -2 | -1 | 0 | 1 | 2,
                    }
                ],
                "error": str | None,
            }
        """
        if not text or not text.strip():
            return {"text": text or "", "annotations": [], "error": None}

        messages = self._build_messages(text)
        raw_response = self._call_api_with_retry(messages)

        if raw_response is None:
            return {
                "text": text,
                "annotations": [],
                "error": "API 调用失败，已超过最大重试次数",
            }

        return self._parse_response(raw_response, text)

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
                    response_format={"type": "json_object"},
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

    def _parse_response(self, raw: str, original_text: str) -> Dict[str, Any]:
        """
        解析模型返回的 JSON 字符串（仅接受 JSON）。
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("模型未返回合法 JSON。原始输出: %s", raw[:200])
            return {"text": original_text, "annotations": [], "error": "模型输出非 JSON"}

        if not isinstance(data, dict):
            return {"text": original_text, "annotations": [], "error": "模型输出 JSON 结构不正确"}

        annotations = data.get("annotations", [])
        if not isinstance(annotations, list):
            return {"text": original_text, "annotations": [], "error": "annotations 字段必须为数组"}

        normalized = []
        for item in annotations:
            if not isinstance(item, dict):
                continue
            aspect = str(item.get("aspect", "")).strip()
            sentiment = str(item.get("sentiment", "")).strip().lower()
            score = item.get("score")

            if sentiment not in {"positive", "negative", "neutral"}:
                continue
            if score not in {-2, -1, 0, 1, 2}:
                continue
            if not aspect:
                continue

            normalized.append(
                {"aspect": aspect, "sentiment": sentiment, "score": int(score)}
            )

        return {"text": original_text, "annotations": normalized, "error": None}

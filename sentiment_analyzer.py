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
本项目面向软件开发场景，采用结合开发体验维度的细粒度情感标签体系。

## 情感标签定义

| 标签 | 含义 | 典型特征 |
|------|------|---------|
| functional_performance  | 功能/性能类 | 表达对语言特性、运行效率、稳定性、兼容性、生态或整体开发体验的正面或负面评价，例如赞扬某特性性能好、抱怨内存占用高、报告兼容性问题等 |
| usability_learning      | 易用性与学习曲线类 | 围绕语法复杂度、API 设计、上手难度、调试体验等提出的意见或反馈，例如称赞 API 直观易用、批评学习曲线陡峭、抱怨错误信息不清晰等 |
| documentation_ecosystem | 文档与生态评价类 | 针对文档质量、社区活跃度、第三方库完善程度的看法，例如称赞官方文档详尽、抱怨 crate 数量不足、反映社区响应慢等 |
| neutral_factual         | 中性陈述与事实描述类 | 主要陈述问题现象、复现步骤或使用场景，不带明显情绪色彩，例如描述 Bug 触发条件、说明版本差异、汇报环境信息等 |
| other_undetermined      | 其他情感或无法判定类 | 包含讽刺、极度复杂情绪，或文本信息不足以归入上述任何类别的样本 |

## 判断维度说明

1. **维度归属**：文本主要关注的是功能/性能、易用性、文档/生态，还是纯粹的事实陈述？
2. **情感极性**：在该维度上，作者的态度是正面、负面还是中性？
3. **情感强度**：情感表达是强烈（感叹号、大写、极端词汇）还是温和？
4. **领域背景**：
   - 讨论执行速度、内存、并发、编译时间 → functional_performance
   - 讨论语法、API 设计、上手难度、调试 → usability_learning
   - 讨论文档、教程、社区、库生态 → documentation_ecosystem
   - 仅描述现象/步骤/环境，无明显情绪 → neutral_factual
   - 无法明确归类或含讽刺 → other_undetermined
5. **上下文敏感**：技术术语（如 "error", "warning"）本身不代表负面情感，需结合语境判断。

## 输出格式（严格 JSON）

```json
{
  "label": "<functional_performance|usability_learning|documentation_ecosystem|neutral_factual|other_undetermined>",
  "confidence": <0.0 到 1.0 之间的浮点数，表示你对此判断的把握程度>,
  "reasoning": "<不超过 100 字的简短分析理由>"
}
```

**重要**：只输出上述 JSON，不要添加任何额外解释、Markdown 代码块或其他内容。"""

# Few-shot 示例（作为 user/assistant 对话轮次注入）
FEW_SHOT_EXAMPLES = [
    # --- functional_performance ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nRust's zero-cost abstractions are phenomenal. The compiled binary runs almost as fast as hand-written C, and the memory footprint is minimal. Extremely impressed by the performance.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "functional_performance",
                "confidence": 0.95,
                "reasoning": "作者针对 Rust 的运行性能和内存占用给出了明确正面评价，使用'phenomenal'、'extremely impressed'等强烈赞扬词汇，属于功能/性能类反馈。",
            },
            ensure_ascii=False,
        ),
    },
    # --- usability_learning ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nThe borrow checker is incredibly frustrating for beginners. The error messages are cryptic and the learning curve is way too steep. I spent three days just trying to understand lifetimes.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "usability_learning",
                "confidence": 0.93,
                "reasoning": "作者对 borrow checker 和 lifetime 的学习难度、错误信息可读性提出强烈批评，涉及学习曲线和调试体验，属于易用性与学习曲线类。",
            },
            ensure_ascii=False,
        ),
    },
    # --- documentation_ecosystem ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nThe official Rust book is one of the best programming language books I've ever read. However, the async ecosystem still feels fragmented — too many competing runtimes with inconsistent documentation.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "documentation_ecosystem",
                "confidence": 0.90,
                "reasoning": "文本前半部分赞扬官方文档质量，后半部分批评异步生态碎片化和文档不一致，核心议题均指向文档与生态，属于文档与生态评价类。",
            },
            ensure_ascii=False,
        ),
    },
    # --- neutral_factual ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nThis issue occurs on Rust 1.75.0 but not on 1.74.1. Steps to reproduce: create a new project with `cargo new`, add the following dependency to Cargo.toml, then run `cargo build`.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "neutral_factual",
                "confidence": 0.94,
                "reasoning": "文本客观陈述了问题复现的版本范围和具体步骤，不含明显情绪色彩，属于中性陈述与事实描述类。",
            },
            ensure_ascii=False,
        ),
    },
    # --- other_undetermined ---
    {
        "role": "user",
        "content": "请分析以下文本的情感：\n\nOh great, another breaking change in the API. I'm sure nobody was using that function anyway. Who needs stability, right?",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "label": "other_undetermined",
                "confidence": 0.85,
                "reasoning": "文本使用讽刺语气（'Oh great'、'I'm sure nobody was using'、'Who needs stability, right?'），情感复杂，无法归入单一功能/性能或易用性维度，属于其他情感或无法判定类。",
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
        # result = {"label": "functional_performance", "confidence": 0.95, "reasoning": "..."}
    """

    VALID_LABELS = {
        "functional_performance",
        "usability_learning",
        "documentation_ecosystem",
        "neutral_factual",
        "other_undetermined",
    }

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
                "label": "functional_performance" | "usability_learning" |
                         "documentation_ecosystem" | "neutral_factual" | "other_undetermined",
                "confidence": float,   # 0.0 ~ 1.0
                "reasoning": str,
                "error": str | None,   # 分析失败时的错误信息
            }
        """
        if not text or not text.strip():
            return {"label": "other_undetermined", "confidence": 1.0, "reasoning": "文本为空", "error": None}

        messages = self._build_messages(text)
        raw_response = self._call_api_with_retry(messages)

        if raw_response is None:
            return {
                "label": "other_undetermined",
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

        label = str(data.get("label", "neutral_factual")).lower()
        if label not in self.VALID_LABELS:
            logger.warning("未知标签 '%s'，回退为 'neutral_factual'", label)
            label = "neutral_factual"

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
        result = {"label": "other_undetermined", "confidence": 0.5, "reasoning": "解析失败"}

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

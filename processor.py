"""
数据预处理模块

功能：
1. 结构解析与噪声剥离（去除代码块、日志行、URLs 等）
2. 去重（基于精确内容哈希）
3. 语言过滤（使用 langdetect 保留指定语言）
4. 文本规范化（合并空白、统一标点等）
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from langdetect import DetectorFactory, LangDetectException, detect

from config import Config

# 固定 langdetect 随机种子，保证结果可复现
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 噪声清除正则表达式
# ---------------------------------------------------------------------------

# 三反引号围住的代码块（含可选语言标识符）
_RE_FENCED_CODE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

# 单反引号行内代码
_RE_INLINE_CODE = re.compile(r"`[^`\n]+`")

# HTML 标签
_RE_HTML_TAGS = re.compile(r"<[^>]+>", re.MULTILINE)

# HTML 实体（如 &amp; &lt; &#39; 等）
_RE_HTML_ENTITIES = re.compile(r"&[a-zA-Z0-9#]+;")

# 超链接（http/https/ftp）
_RE_URL = re.compile(
    r"https?://[^\s\)\]\>\"\']+|ftp://[^\s\)\]\>\"\']+",
    re.IGNORECASE,
)

# 日志行：常见日志级别前缀，或带时间戳的行
# 示例：[2024-01-01 12:00:00] ERROR: ...  /  INFO: blah blah
_RE_LOG_LINE = re.compile(
    r"^\s*(?:"
    r"\[?\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]?"  # timestamp
    r"|(?:DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL|TRACE)\s*[:\[]"  # log level prefix
    r")\s*.*$",
    re.MULTILINE | re.IGNORECASE,
)

# Stack trace 行（Java/Python/Rust 等）
_RE_STACK_TRACE = re.compile(
    r"^\s*(?:"
    r"at\s+[\w\.$<>]+\(.*\)"           # Java stack frame
    r"|File\s+\".*\",\s+line\s+\d+"    # Python traceback
    r"|\s+\d+\s*\|.*"                  # Rust error source line
    r"|Traceback\s+\(most\s+recent"    # Python traceback header
    r")\s*$",
    re.MULTILINE,
)

# Diff / patch 行
_RE_DIFF = re.compile(r"^[-+]{3}\s.*$|^@@.*@@", re.MULTILINE)

# 引用行（以 > 开头的 Markdown 引用）
_RE_BLOCKQUOTE = re.compile(r"^>+\s*.*$", re.MULTILINE)

# 连续多个空行 → 单个空行
_RE_MULTI_BLANK = re.compile(r"\n{3,}")

# 首尾空白
_RE_LEADING_TRAILING = re.compile(r"^\s+|\s+$")


# ---------------------------------------------------------------------------
# TextProcessor
# ---------------------------------------------------------------------------


class TextProcessor:
    """
    对从数据库读取的原始文本执行完整的预处理流水线。

    使用方式::

        processor = TextProcessor(config)
        record = {"id": 1, "title": "...", "description": "..."}
        result = processor.process_record("issue", record, text_fields=["title", "description"])
    """

    def __init__(self, config: Config):
        self._config = config
        self._seen_hashes: Dict[str, int] = {}  # hash -> source_id (用于去重)

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def process_record(
        self,
        source_type: str,
        record: dict,
        text_fields: List[str],
    ) -> Tuple[int, str, str, Optional[str], bool]:
        """
        处理一条数据库记录，将多个文本字段拼接后进行完整预处理。

        返回 (source_id, raw_text, processed_text, language, is_duplicate)
        """
        source_id: int = record["id"]

        # 拼接文本字段，过滤 None
        parts = [str(record.get(f) or "").strip() for f in text_fields]
        raw_text = "\n\n".join(p for p in parts if p)

        processed_text, language, is_duplicate = self.process_text(raw_text, source_id)
        return source_id, raw_text, processed_text, language, is_duplicate

    def process_text(
        self, raw_text: str, source_id: Optional[int] = None
    ) -> Tuple[str, Optional[str], bool]:
        """
        对单段原始文本执行完整预处理流水线。

        返回 (processed_text, language, is_duplicate)
        """
        # 1. 噪声剥离
        text = self._strip_noise(raw_text)

        # 2. 文本规范化
        text = self._normalize(text)

        # 3. 长度过滤（过短则标记为空）
        if len(text) < self._config.min_text_length:
            return "", None, False

        # 4. 语言检测
        language = self._detect_language(text)
        if language not in self._config.allowed_languages:
            logger.debug("跳过语言 %s (source_id=%s)", language, source_id)
            return "", language, False

        # 5. 去重
        is_duplicate = self._is_duplicate(text, source_id)

        return text, language, is_duplicate

    # ------------------------------------------------------------------
    # 噪声剥离
    # ------------------------------------------------------------------

    def _strip_noise(self, text: str) -> str:
        """依次应用所有噪声清除规则"""
        # 代码块（先处理，避免代码内容干扰后续规则）
        text = _RE_FENCED_CODE.sub("", text)
        text = _RE_INLINE_CODE.sub("", text)

        # HTML
        text = _RE_HTML_TAGS.sub(" ", text)
        text = _RE_HTML_ENTITIES.sub(" ", text)

        # URL
        text = _RE_URL.sub("", text)

        # 日志行 & 堆栈跟踪
        text = _RE_LOG_LINE.sub("", text)
        text = _RE_STACK_TRACE.sub("", text)

        # Diff 行
        text = _RE_DIFF.sub("", text)

        # Markdown 引用块（通常是复制的他人内容，噪声较多）
        text = _RE_BLOCKQUOTE.sub("", text)

        return text

    # ------------------------------------------------------------------
    # 文本规范化
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """
        文本规范化步骤：
        - Unicode NFC 标准化
        - 将 Windows 换行符转换为 Unix 换行符
        - 合并多余空行
        - 去除首尾空白
        - 合并同行内连续空格
        """
        # Unicode 正规化
        text = unicodedata.normalize("NFC", text)

        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 合并多余空行
        text = _RE_MULTI_BLANK.sub("\n\n", text)

        # 每行内合并连续空白（保留换行）
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(line)
        text = "\n".join(lines)

        # 去除首尾空白
        text = text.strip()

        return text

    # ------------------------------------------------------------------
    # 语言检测
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        """
        使用 langdetect 检测文本语言。
        返回语言代码（如 'en', 'zh-cn'），检测失败返回 None。
        """
        try:
            return detect(text)
        except LangDetectException:
            return None

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------

    def _is_duplicate(self, text: str, source_id: Optional[int]) -> bool:
        """
        精确去重：基于 SHA-256 内容哈希。
        若同样的哈希已出现过，则标记为重复。
        """
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self._seen_hashes:
            logger.debug(
                "重复文本 source_id=%s，与 source_id=%s 相同",
                source_id,
                self._seen_hashes[h],
            )
            return True
        self._seen_hashes[h] = source_id if source_id is not None else -1
        return False

    def reset_dedup_cache(self):
        """清空去重缓存（在不同数据源之间可能需要调用）"""
        self._seen_hashes.clear()

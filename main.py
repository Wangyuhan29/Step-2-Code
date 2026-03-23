"""
程序入口

用法：
    python main.py [--config CONFIG] [--step {preprocess,analyze,all}]
                   [--api-key DEEPSEEK_API_KEY]

步骤：
  preprocess : 只执行数据预处理（噪声清除、去重、语言过滤、规范化）
  analyze    : 只执行情感分析（需先完成 preprocess）
  all        : 依次执行 preprocess 和 analyze（默认）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from tqdm import tqdm

from config import Config
from database import Database
from processor import TextProcessor
from sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 各数据源的配置（表名、文本字段、source_type 标签）
# ---------------------------------------------------------------------------

DATA_SOURCES = [
    {
        "source_type": "issue",
        "iter_method": "iter_issues",
        "text_fields": ["title", "description"],
    },
    {
        "source_type": "pr",
        "iter_method": "iter_prs",
        "text_fields": ["title", "description"],
    },
    {
        "source_type": "issue_comment",
        "iter_method": "iter_issue_comments",
        "text_fields": ["body"],
    },
    {
        "source_type": "pr_comment",
        "iter_method": "iter_pr_comments",
        "text_fields": ["body"],
    },
]


# ---------------------------------------------------------------------------
# 预处理流程
# ---------------------------------------------------------------------------


def run_preprocess(db: Database, config: Config):
    """
    从 github_rust_data 数据库读取所有原始数据，
    经过预处理后写入 processed_texts 表。
    """
    logger.info("=== 开始数据预处理 ===")
    processor = TextProcessor(config)

    total_processed = 0
    total_skipped = 0
    total_duplicates = 0

    for source_cfg in DATA_SOURCES:
        source_type = source_cfg["source_type"]
        iter_method = getattr(db, source_cfg["iter_method"])
        text_fields = source_cfg["text_fields"]

        logger.info("处理数据源: %s", source_type)
        source_count = 0

        for batch in iter_method():
            for record in tqdm(batch, desc=f"预处理 {source_type}", leave=False):
                source_id, raw_text, processed_text, language, is_duplicate = (
                    processor.process_record(source_type, record, text_fields)
                )

                # 原始文本为空或处理后为空 → 跳过写库
                if not raw_text:
                    total_skipped += 1
                    continue

                if config.save_to_db:
                    db.upsert_processed_text(
                        source_type=source_type,
                        source_id=source_id,
                        raw_text=raw_text,
                        processed_text=processed_text,
                        language=language,
                        is_duplicate=is_duplicate,
                    )

                source_count += 1
                total_processed += 1
                if is_duplicate:
                    total_duplicates += 1

        logger.info(
            "  %s 完成：处理 %d 条", source_type, source_count
        )

    logger.info(
        "=== 预处理完成 | 总处理: %d | 跳过: %d | 去重: %d ===",
        total_processed,
        total_skipped,
        total_duplicates,
    )


# ---------------------------------------------------------------------------
# 情感分析流程
# ---------------------------------------------------------------------------


def run_sentiment_analysis(db: Database, config: Config):
    """
    从 processed_texts 表读取尚未分析的文本，
    调用 DeepSeek API 进行情感分类，结果写入 sentiment_results 表。
    """
    logger.info("=== 开始情感分析 ===")
    analyzer = SentimentAnalyzer(config)

    total_analyzed = 0
    label_counts: dict = {}

    for batch in db.iter_unanalyzed_texts(batch_size=config.batch_size):
        texts = [row["processed_text"] for row in batch]
        results = analyzer.analyze_batch(texts)

        for row, result in zip(batch, results):
            processed_text_id = row["id"]
            label = result["label"]
            confidence = result.get("confidence")
            reasoning = result.get("reasoning")
            error = result.get("error")

            if error:
                logger.warning(
                    "情感分析失败 processed_text_id=%d: %s",
                    processed_text_id,
                    error,
                )

            if config.save_to_db:
                db.upsert_sentiment_result(
                    processed_text_id=processed_text_id,
                    sentiment_label=label,
                    confidence=confidence,
                    reasoning=reasoning,
                    model_name=config.deepseek_model,
                )

            label_counts[label] = label_counts.get(label, 0) + 1
            total_analyzed += 1

    logger.info("=== 情感分析完成 | 总分析: %d ===", total_analyzed)
    for label, count in sorted(label_counts.items()):
        logger.info("  %-10s: %d 条", label, count)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="GitHub Rust 数据处理与情感分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="config.ini",
        help="配置文件路径（默认：config.ini）",
    )
    parser.add_argument(
        "--step",
        choices=["preprocess", "analyze", "all"],
        default="all",
        help="执行步骤：preprocess（仅预处理）、analyze（仅情感分析）、all（全部，默认）",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="DeepSeek API Key（覆盖配置文件）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 通过命令行参数设置 API Key 环境变量
    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key

    # 加载配置
    config = Config(config_file=args.config)

    # 对需要 API 的步骤验证 API Key
    if args.step in ("analyze", "all"):
        try:
            config.validate()
        except ValueError as e:
            logger.error("配置验证失败：\n%s", e)
            sys.exit(1)

    # 建立数据库连接
    db = Database(config)
    try:
        db.connect()
        db.init_output_tables()

        if args.step in ("preprocess", "all"):
            run_preprocess(db, config)

        if args.step in ("analyze", "all"):
            run_sentiment_analysis(db, config)

    except Exception as e:
        logger.exception("运行时发生错误：%s", e)
        sys.exit(1)
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()

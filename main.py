"""
程序入口

用法：
    python main.py [--config CONFIG] [--api-key DEEPSEEK_API_KEY]
                   --sql-file QUERY.sql [--output-json OUTPUT_JSON]

说明：
  - 停用预处理流程，改为手动 SQL 筛选后直接调用情感分析 API。
  - 输出 JSON 仅包含 text_id（原始表 id）和 annotations。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from config import Config
from database import Database
from sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 情感分析流程
# ---------------------------------------------------------------------------


def _extract_text(row: dict) -> str:
    """从 SQL 结果行中提取可分析文本。"""
    if row.get("text"):
        return str(row["text"]).strip()

    parts = [
        row.get("title"),
        row.get("description"),
        row.get("body"),
        row.get("raw_text"),
        row.get("processed_text"),
    ]
    return "\n\n".join(str(p).strip() for p in parts if p and str(p).strip())


def run_sentiment_analysis(db: Database, config: Config, sql: str, output_json: str):
    """
    执行手动 SQL 筛选，调用 DeepSeek API 进行情感分类，结果写入 JSON 文件。
    """
    logger.info("=== 开始情感分析 ===")
    analyzer = SentimentAnalyzer(config)
    output_path = Path(output_json or config.output_json_file)

    total_input = 0
    output_records = []

    for batch in db.iter_rows_by_sql(sql=sql, batch_size=config.batch_size):
        if not batch:
            continue

        texts = []
        text_ids = []
        for row in batch:
            text_id = row.get("id", row.get("text_id"))
            text = _extract_text(row)
            if text_id is None or not text:
                continue
            text_ids.append(int(text_id))
            texts.append(text)

        if not texts:
            continue

        results = analyzer.analyze_batch(texts)
        total_input += len(texts)

        for text_id, result in zip(text_ids, results):
            output_records.append(
                {
                    "text_id": text_id,
                    "annotations": result.get("annotations", []),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    logger.info("=== 情感分析完成 | 输入文本: %d | 输出记录: %d ===", total_input, len(output_records))
    logger.info("结果已写入 JSON 文件: %s", output_path)


def _load_sql_from_file(sql_file: str) -> str:
    """从 SQL 文件读取查询语句。"""
    path = Path(sql_file)
    if not path.exists():
        raise FileNotFoundError(f"SQL 文件不存在: {path}")

    sql = path.read_text(encoding="utf-8-sig").strip()
    if not sql:
        raise ValueError(f"SQL 文件内容为空: {path}")

    return sql


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
        "--sql-file",
        required=True,
        help="SQL 文件路径（文件内容必须是 SELECT，且返回 id/text 或可拼接的文本列）",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="输出 JSON 文件路径（默认使用配置中的 sentiment.output_json_file）",
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
    sql = _load_sql_from_file(args.sql_file)

    try:
        config.validate()
    except ValueError as e:
        logger.error("配置验证失败：\n%s", e)
        sys.exit(1)

    # 建立数据库连接
    db = Database(config)
    try:
        db.connect()
        run_sentiment_analysis(db, config, sql=sql, output_json=args.output_json)

    except Exception as e:
        logger.exception("运行时发生错误：%s", e)
        sys.exit(1)
    finally:
        db.disconnect()


if __name__ == "__main__":
    main()

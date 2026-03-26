"""
数据库操作模块
负责从 github_rust_data MySQL 数据库读取 Issues/PR 数据，
以及将处理结果写回数据库。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Iterator, List, Optional

import mysql.connector
from mysql.connector import Error as MySQLError

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL：处理结果表
# ---------------------------------------------------------------------------

_CREATE_PROCESSED_TEXTS_TABLE = """
CREATE TABLE IF NOT EXISTS processed_texts (
    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
    source_type     VARCHAR(20)  NOT NULL COMMENT '来源类型: issue/pr/issue_comment/pr_comment',
    source_id       BIGINT       NOT NULL COMMENT '原始记录ID',
    raw_text        MEDIUMTEXT   NOT NULL COMMENT '原始文本',
    processed_text  MEDIUMTEXT   NOT NULL COMMENT '处理后的文本',
    language        VARCHAR(20)  DEFAULT NULL COMMENT '检测到的语言',
    is_duplicate    TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '是否为重复文本',
    created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_source (source_type, source_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='文本预处理结果';
"""

_CREATE_SENTIMENT_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS sentiment_results (
    id                  BIGINT PRIMARY KEY AUTO_INCREMENT,
    processed_text_id   BIGINT       NOT NULL COMMENT '关联的 processed_texts.id',
    sentiment_label     VARCHAR(30)  NOT NULL COMMENT '情感标签: functional_performance/usability_learning/documentation_ecosystem/neutral_factual/other_undetermined',
    confidence          FLOAT        DEFAULT NULL COMMENT '置信度 0-1',
    reasoning           TEXT         DEFAULT NULL COMMENT '模型给出的分析理由',
    model_name          VARCHAR(100) DEFAULT NULL COMMENT '使用的模型',
    created_at          DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_processed_text (processed_text_id),
    FOREIGN KEY (processed_text_id) REFERENCES processed_texts(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='情感分类结果';
"""


class Database:
    """MySQL 数据库连接与操作封装"""

    def __init__(self, config: Config):
        self._config = config
        self._conn: Optional[mysql.connector.MySQLConnection] = None

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self):
        """建立数据库连接"""
        self._conn = mysql.connector.connect(
            host=self._config.mysql_host,
            port=self._config.mysql_port,
            user=self._config.mysql_user,
            password=self._config.mysql_password,
            database=self._config.mysql_database,
            charset="utf8mb4",
            use_unicode=True,
            autocommit=False,
        )
        logger.info(
            "已连接到 MySQL %s:%s/%s",
            self._config.mysql_host,
            self._config.mysql_port,
            self._config.mysql_database,
        )

    def disconnect(self):
        """关闭数据库连接"""
        if self._conn and self._conn.is_connected():
            self._conn.close()
            logger.info("已断开数据库连接")

    @contextmanager
    def cursor(self) -> Generator:
        """提供带自动提交/回滚的游标上下文管理器"""
        cur = self._conn.cursor(dictionary=True)
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # 初始化输出表
    # ------------------------------------------------------------------

    def init_output_tables(self):
        """创建处理结果表（如果不存在）"""
        with self.cursor() as cur:
            cur.execute(_CREATE_PROCESSED_TEXTS_TABLE)
            cur.execute(_CREATE_SENTIMENT_RESULTS_TABLE)
        logger.info("输出表已就绪（processed_texts / sentiment_results）")

    # ------------------------------------------------------------------
    # 读取原始数据
    # ------------------------------------------------------------------

    def iter_issues(self, batch_size: int = 200) -> Iterator[List[dict]]:
        """分批迭代读取 issues 表"""
        yield from self._iter_table(
            table="issues",
            text_fields=["title", "description"],
            id_field="id",
            batch_size=batch_size,
        )

    def iter_prs(self, batch_size: int = 200) -> Iterator[List[dict]]:
        """分批迭代读取 pull_requests 表"""
        yield from self._iter_table(
            table="pull_requests",
            text_fields=["title", "description"],
            id_field="id",
            batch_size=batch_size,
        )

    def iter_issue_comments(self, batch_size: int = 200) -> Iterator[List[dict]]:
        """分批迭代读取 issue_comments 表"""
        yield from self._iter_table(
            table="issue_comments",
            text_fields=["body"],
            id_field="id",
            batch_size=batch_size,
        )

    def iter_pr_comments(self, batch_size: int = 200) -> Iterator[List[dict]]:
        """分批迭代读取 pr_comments 表"""
        yield from self._iter_table(
            table="pr_comments",
            text_fields=["body"],
            id_field="id",
            batch_size=batch_size,
        )

    def _iter_table(
        self,
        table: str,
        text_fields: List[str],
        id_field: str,
        batch_size: int,
    ) -> Iterator[List[dict]]:
        """通用分批读取迭代器"""
        _ALLOWED_TABLES = {"issues", "pull_requests", "issue_comments", "pr_comments"}
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"不允许的表名: {table!r}")

        offset = 0
        fields = ", ".join([id_field] + text_fields)
        while True:
            with self.cursor() as cur:
                cur.execute(
                    f"SELECT {fields} FROM `{table}` LIMIT %s OFFSET %s",
                    (batch_size, offset),
                )
                rows = cur.fetchall()
            if not rows:
                break
            yield rows
            offset += batch_size

    # ------------------------------------------------------------------
    # 写入处理结果
    # ------------------------------------------------------------------

    def upsert_processed_text(
        self,
        source_type: str,
        source_id: int,
        raw_text: str,
        processed_text: str,
        language: Optional[str],
        is_duplicate: bool,
    ) -> int:
        """插入或更新 processed_texts 记录，返回记录 id"""
        sql = """
            INSERT INTO processed_texts
                (source_type, source_id, raw_text, processed_text, language, is_duplicate)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                raw_text       = VALUES(raw_text),
                processed_text = VALUES(processed_text),
                language       = VALUES(language),
                is_duplicate   = VALUES(is_duplicate)
        """
        with self.cursor() as cur:
            cur.execute(
                sql,
                (source_type, source_id, raw_text, processed_text, language, int(is_duplicate)),
            )
            # ON DUPLICATE KEY UPDATE 时 lastrowid 返回已有行的 id
            if cur.lastrowid:
                return cur.lastrowid
            # 回退：查询已有记录
            cur.execute(
                "SELECT id FROM processed_texts WHERE source_type=%s AND source_id=%s",
                (source_type, source_id),
            )
            row = cur.fetchone()
            return row["id"] if row else 0

    def upsert_sentiment_result(
        self,
        processed_text_id: int,
        sentiment_label: str,
        confidence: Optional[float],
        reasoning: Optional[str],
        model_name: Optional[str],
    ):
        """插入或更新 sentiment_results 记录"""
        sql = """
            INSERT INTO sentiment_results
                (processed_text_id, sentiment_label, confidence, reasoning, model_name)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                sentiment_label = VALUES(sentiment_label),
                confidence      = VALUES(confidence),
                reasoning       = VALUES(reasoning),
                model_name      = VALUES(model_name)
        """
        with self.cursor() as cur:
            cur.execute(sql, (processed_text_id, sentiment_label, confidence, reasoning, model_name))

    # ------------------------------------------------------------------
    # 读取待分析的已处理文本
    # ------------------------------------------------------------------

    def iter_unanalyzed_texts(self, batch_size: int = 50) -> Iterator[List[dict]]:
        """
        迭代读取尚未进行情感分析的 processed_texts 记录
        （is_duplicate=0 且 processed_text 不为空且语言在允许范围内）
        """
        offset = 0
        sql = """
            SELECT pt.id, pt.processed_text, pt.source_type, pt.source_id
            FROM processed_texts pt
            LEFT JOIN sentiment_results sr ON sr.processed_text_id = pt.id
            WHERE pt.is_duplicate = 0
              AND pt.processed_text IS NOT NULL
              AND pt.processed_text != ''
              AND sr.id IS NULL
            ORDER BY pt.id
            LIMIT %s OFFSET %s
        """
        while True:
            with self.cursor() as cur:
                cur.execute(sql, (batch_size, offset))
                rows = cur.fetchall()
            if not rows:
                break
            yield rows
            offset += batch_size

"""
配置管理模块
负责读取和验证配置文件及环境变量
"""

import configparser
import os


class Config:
    """统一配置管理类"""

    DEFAULT_CONFIG = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        },
        "mysql": {
            "host": "localhost",
            "port": "3306",
            "user": "root",
            "password": "",
            "database": "github_rust_data",
        },
        "processor": {
            "allowed_languages": "en,zh-cn,zh-tw",
            "min_text_length": "10",
            "dedup_strategy": "exact",
            "save_to_db": "true",
        },
        "sentiment": {
            "batch_size": "10",
            "request_delay": "1.0",
            "max_retries": "3",
        },
    }

    def __init__(self, config_file: str = "config.ini"):
        self._parser = configparser.ConfigParser()
        # 设置默认值
        for section, values in self.DEFAULT_CONFIG.items():
            self._parser[section] = values

        # 读取配置文件（如果存在）
        if os.path.exists(config_file):
            self._parser.read(config_file, encoding="utf-8")

        # 环境变量覆盖
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """从环境变量读取敏感配置"""
        env_map = {
            "DEEPSEEK_API_KEY": ("deepseek", "api_key"),
            "DEEPSEEK_BASE_URL": ("deepseek", "base_url"),
            "MYSQL_HOST": ("mysql", "host"),
            "MYSQL_PORT": ("mysql", "port"),
            "MYSQL_USER": ("mysql", "user"),
            "MYSQL_PASSWORD": ("mysql", "password"),
            "MYSQL_DATABASE": ("mysql", "database"),
        }
        for env_var, (section, key) in env_map.items():
            value = os.environ.get(env_var)
            if value:
                if not self._parser.has_section(section):
                    self._parser.add_section(section)
                self._parser.set(section, key, value)

    # ---- DeepSeek ----
    @property
    def deepseek_api_key(self) -> str:
        return self._parser.get("deepseek", "api_key", fallback="")

    @property
    def deepseek_base_url(self) -> str:
        return self._parser.get("deepseek", "base_url")

    @property
    def deepseek_model(self) -> str:
        return self._parser.get("deepseek", "model")

    # ---- MySQL ----
    @property
    def mysql_host(self) -> str:
        return self._parser.get("mysql", "host")

    @property
    def mysql_port(self) -> int:
        return self._parser.getint("mysql", "port")

    @property
    def mysql_user(self) -> str:
        return self._parser.get("mysql", "user")

    @property
    def mysql_password(self) -> str:
        return self._parser.get("mysql", "password")

    @property
    def mysql_database(self) -> str:
        return self._parser.get("mysql", "database")

    # ---- Processor ----
    @property
    def allowed_languages(self) -> list:
        raw = self._parser.get("processor", "allowed_languages")
        return [lang.strip() for lang in raw.split(",") if lang.strip()]

    @property
    def min_text_length(self) -> int:
        return self._parser.getint("processor", "min_text_length")

    @property
    def dedup_strategy(self) -> str:
        return self._parser.get("processor", "dedup_strategy")

    @property
    def save_to_db(self) -> bool:
        return self._parser.getboolean("processor", "save_to_db")

    # ---- Sentiment ----
    @property
    def batch_size(self) -> int:
        return self._parser.getint("sentiment", "batch_size")

    @property
    def request_delay(self) -> float:
        return self._parser.getfloat("sentiment", "request_delay")

    @property
    def max_retries(self) -> int:
        return self._parser.getint("sentiment", "max_retries")

    def validate(self):
        """验证必填配置项"""
        errors = []
        if not self.deepseek_api_key:
            errors.append(
                "缺少 DeepSeek API Key。请在 config.ini 中设置 [deepseek] api_key "
                "或设置环境变量 DEEPSEEK_API_KEY。"
            )
        if errors:
            raise ValueError("\n".join(errors))

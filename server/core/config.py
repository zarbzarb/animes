# -*- coding: utf-8 -*-
"""应用层配置：从 `.env` 读取，**统一走 pydantic-settings**。

为什么不让各模块自己读 `os.environ`
------------------------------------
`docs/project-structure.md` 把 `server/core/config.py` 定为配置的唯一入口：
模型层（`models/`）**禁止读 .env**，否则消融实验的复现性就无法保证
（同一份代码在不同机器上会因为环境变量不同而算出不同结果）。
所以只有这里能碰环境变量，其它地方一律 `from server.core.config import settings`。

路径口径
--------
所有相对路径都相对**项目根**解析，而不是进程的工作目录 ——
与 `scripts/` 的 `anchor_paths()` 同一约定（见 `docs/dev-conventions.md`）。
这样 `uvicorn server.main:app` 不管从哪个目录启动都能找到 `.env` 与数据文件。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# server/core/config.py -> 上溯三层 = 项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """全部配置项。字段名与 `.env` 中的键一一对应（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",          # .env 里多写键不报错（它同时是文档）
        case_sensitive=False,
    )

    # ---------- 运行环境 ----------
    APP_ENV: str = "dev"
    APP_NAME: str = "anirec"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ---------- MySQL ----------
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = "anirec"
    MYSQL_POOL_SIZE: int = 10

    # ---------- Redis ----------
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    REC_CACHE_TTL: int = 86400

    # ---------- 连接串覆盖（测试注入 sqlite / 关闭 redis 时用）----------
    # 显式给出时优先级最高，便于 tests/ 与 CI 不依赖本机服务
    DATABASE_URL: Optional[str] = None
    REDIS_URL: Optional[str] = None
    CACHE_ENABLED: bool = True

    # ---------- LLM ----------
    LLM_PROVIDER: str = "dashscope"
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT: int = 30
    LLM_MAX_RETRY: int = 2

    # ---------- 模型与数据路径 ----------
    MODEL_DIR: str = "./data/checkpoints"
    SASREC_CKPT: str = "sasrec_best.pt"
    MULTI_INTEREST_CKPT: str = "multi_interest.pt"
    CONTENT_ENCODER_DIR: str = "./models/content_encoder/pretrained"
    ITEM_EMB_PATH: str = "./data/features/item_emb.npy"
    CONTENT_VEC_PATH: str = "./data/features/content_vec_512.npy"

    DATA_DIR: str = "./dataset"
    ANIME_META_PATH: str = "./dataset/animes.csv"
    RATING_PATH: str = "./dataset/ratings.csv"
    GENRE_MAP_PATH: str = "./dataset/id_to_genreids.json"
    DATASET_PKL: str = "./dataset/dataset.pkl"
    NEG_SAMPLE_PKL: str = "./dataset/random-sample_size100-seed98765.pkl"

    # ---------- 算法超参（与 configs/model.yaml 保持同值）----------
    MAX_SEQ_LEN: int = 50
    NUM_INTERESTS: int = 4
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 2
    NUM_HEADS: int = 2
    DROPOUT: float = 0.2
    BATCH_SIZE: int = 256
    LR: float = 0.001
    EPOCHS: int = 200
    EARLY_STOP_PATIENCE: int = 10
    COLD_START_THRESHOLD: int = 10
    FUSION_W_BEHAVIOR: float = 0.7
    FUSION_W_CONTENT: float = 0.3
    FUSION_W_CONTENT_COLD: float = 0.5

    # ---------- 定时任务 ----------
    CRON_OFFLINE_REC: str = "0 3 * * *"
    CRON_NEW_ANIME_SYNC: str = "0 4 * * 1"

    # ---------- 安全 ----------
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    # Agent 内部调用令牌（/internal/agents/*），默认与 JWT_SECRET 不同源
    INTERNAL_TOKEN: str = "internal-dev-token"

    # ---------- 前端 ----------
    VITE_API_BASE_URL: str = "http://127.0.0.1:8000/api/v1"

    # =================================================================
    # 派生量
    # =================================================================
    @property
    def database_url(self) -> str:
        """SQLAlchemy 连接串。`DATABASE_URL` 显式给出时优先（测试用 sqlite）。"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        pwd = self.MYSQL_PASSWORD
        auth = f"{self.MYSQL_USER}:{pwd}" if pwd else self.MYSQL_USER
        return (f"mysql+pymysql://{auth}@{self.MYSQL_HOST}:{self.MYSQL_PORT}"
                f"/{self.MYSQL_DB}?charset=utf8mb4")

    @property
    def redis_url(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def is_test(self) -> bool:
        return self.APP_ENV == "test"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def abs_path(self, relative: str) -> Path:
        """把 `.env` 里的相对路径锚到项目根（与 `scripts/` 的 anchor_paths 同一口径）。"""
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def model_dir(self) -> Path:
        return self.abs_path(self.MODEL_DIR)

    @property
    def content_vec_path(self) -> Path:
        return self.abs_path(self.CONTENT_VEC_PATH)

    @property
    def dataset_pkl(self) -> Path:
        return self.abs_path(self.DATASET_PKL)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试要换配置请 `get_settings.cache_clear()` 后重设环境变量。"""
    return Settings()


settings = get_settings()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "settings"]

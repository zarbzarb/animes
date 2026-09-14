# -*- coding: utf-8 -*-
"""Agent 层自己的配置。

为什么不直接 `from server.core.config import settings`
------------------------------------------------------
依赖方向是 **`server/ → agents/ → models/` 单向**（`docs/project-structure.md` §1），
由 `scripts/check_imports.py` 静态检查。Agent 要是 import 了 `server/`，
检查就会失败 —— 而这条约束不是为了好看：一旦 Agent 能反向依赖应用层，
「同一套 Agent 既能进程内调用、也能拆成独立服务」这个能力就没了
（协议 §二 M1 明说"生产可切 HTTP/gRPC"）。

所以这里只放 **Agent 层需要的那几个字段**，值从环境变量读（`agents/` 允许读 .env，
`models/` 不允许）。server 启动时会用它的 `Settings` 反向**注入**覆盖
（`configure()`），这样两边不会因为默认值不同而漂移。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class AgentSettings:
    """Agent 层关心的配置子集。字段名与 `.env` 的键保持一致（便于对照）。"""

    app_env: str = "dev"

    # 冷启动与融合（与 configs/model.yaml / .env 同值）
    cold_start_threshold: int = 10
    fusion_w_behavior: float = 0.7
    fusion_w_content: float = 0.3
    fusion_w_content_cold: float = 0.5

    # 召回规模
    recall_topk_behavior: int = 200
    recall_topk_content: int = 50
    recall_final_topk: int = 20
    max_seq_len: int = 50
    num_interests: int = 4

    # 模型路径（相对项目根或绝对路径）
    model_dir: str = "./data/checkpoints"
    sasrec_ckpt: str = "sasrec_best.pt"
    multi_interest_ckpt: str = "multi_interest.pt"
    item_emb_path: str = "./data/features/item_emb.npy"
    content_vec_path: str = "./data/features/content_vec_512.npy"

    # LLM
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_timeout: int = 30
    llm_max_retry: int = 2

    # 各 Agent 的超时预算（毫秒，协议 §3.2 要求 50–30000）
    timeout_ms: dict[str, int] = field(default_factory=lambda: {
        "A1": 120, "A2": 200, "A3": 150, "A4": 80, "A5": 200, "A7": 15_000,
    })

    @classmethod
    def from_env(cls) -> "AgentSettings":
        return cls(
            app_env=_env("APP_ENV", "dev"),
            cold_start_threshold=_env_int("COLD_START_THRESHOLD", 10),
            fusion_w_behavior=_env_float("FUSION_W_BEHAVIOR", 0.7),
            fusion_w_content=_env_float("FUSION_W_CONTENT", 0.3),
            fusion_w_content_cold=_env_float("FUSION_W_CONTENT_COLD", 0.5),
            recall_topk_behavior=_env_int("RECALL_TOPK_BEHAVIOR", 200),
            recall_topk_content=_env_int("RECALL_TOPK_CONTENT", 50),
            recall_final_topk=_env_int("RECALL_FINAL_TOPK", 20),
            max_seq_len=_env_int("MAX_SEQ_LEN", 50),
            num_interests=_env_int("NUM_INTERESTS", 4),
            model_dir=_env("MODEL_DIR", "./data/checkpoints"),
            sasrec_ckpt=_env("SASREC_CKPT", "sasrec_best.pt"),
            multi_interest_ckpt=_env("MULTI_INTEREST_CKPT", "multi_interest.pt"),
            item_emb_path=_env("ITEM_EMB_PATH", "./data/features/item_emb.npy"),
            content_vec_path=_env("CONTENT_VEC_PATH",
                                  "./data/features/content_vec_512.npy"),
            llm_base_url=_env("LLM_BASE_URL"),
            llm_api_key=_env("LLM_API_KEY"),
            llm_model=_env("LLM_MODEL", "qwen-plus"),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.3),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
            llm_timeout=_env_int("LLM_TIMEOUT", 30),
            llm_max_retry=_env_int("LLM_MAX_RETRY", 2),
        )

    @property
    def llm_configured(self) -> bool:
        """没配 base_url 或 api_key 就等于"没有 LLM" —— 此时必须走模板兜底，
        而不是发一个注定 401 的请求再等超时（那会白白吃掉 200ms 预算）。"""
        return bool(self.llm_base_url and self.llm_api_key)


_settings: Optional[AgentSettings] = None


@lru_cache(maxsize=1)
def _env_settings() -> AgentSettings:
    return AgentSettings.from_env()


def get_settings() -> AgentSettings:
    return _settings or _env_settings()


def configure(settings: AgentSettings) -> None:
    """由应用层（server 启动时）注入覆盖。测试也用它注入小超时。"""
    global _settings
    _settings = settings


def reset_settings() -> None:
    global _settings
    _settings = None


__all__ = ["AgentSettings", "configure", "get_settings", "reset_settings"]

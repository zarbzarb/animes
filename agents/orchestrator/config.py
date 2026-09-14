# -*- coding: utf-8 -*-
"""A0 调度 Agent 的配置。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OrchestratorConfig"]


@dataclass(frozen=True)
class OrchestratorConfig:
    # 自身总预算（协议 §5.2：A0 800ms）
    total_timeout_ms: int = 800
    # 规则置信度低于该值才启用 LLM 兜底意图识别
    intent_llm_min_conf: float = 0.7
    intent_llm_timeout_ms: int = 300

    # 未登录/无画像用户的兜底候选数（L4）
    anonymous_top_n: int = 20

    # 结果缓存 TTL（秒）。首页结果缓存 5 分钟 —— 比 A4 的落库 TTL 短，
    # 因为用户刚产生行为时希望立刻看到变化（行为事件会主动失效）
    result_cache_ttl: int = 300

    # 是否把完整流水线产物（含每步 payload）写进响应。
    # 默认 False：候选集可能 300 条，写进响应会让 payload 膨胀 10 倍（协议 §七）
    echo_pipeline: bool = False

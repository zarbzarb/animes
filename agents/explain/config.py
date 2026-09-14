# -*- coding: utf-8 -*-
"""A5 解释生成 Agent 的配置。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ExplainConfig"]


@dataclass(frozen=True)
class ExplainConfig:
    # 主链路 200ms 预算（协议 §5.2）。留 50ms 给网络与序列化。
    timeout_ms: int = 180
    # 单条解释接口可以用更长的预算（不在主链路上）
    standalone_timeout_ms: int = 3000

    max_reason_len: int = 60          # 协议：`reason` ≤ 60 字
    core_items_k: int = 3
    # 低于该匹配度就不做 LLM（信号太弱，LLM 只能编）
    min_overlap_for_llm: float = 0.0

    llm_temperature: float = 0.5
    llm_max_tokens: int = 160

    cache_ttl: int = 86400
    persist: bool = True

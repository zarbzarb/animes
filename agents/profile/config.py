# -*- coding: utf-8 -*-
"""A1 画像 Agent 的可调参数。

阈值集中在这里而不是散在 strategy 里，是为了能回答"为什么这个用户被判成低活跃"——
答辩时这是一个必被追问的问题，调参时也需要一处能改。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProfileConfig"]


@dataclass(frozen=True)
class ProfileConfig:
    # 活跃度分档（按总记录数）。三档而不是连续值，是因为前端只用三档配色，
    # 且连续值会让"画像摘要"的措辞变得不可控。
    activity_medium_min: int = 10
    activity_high_min: int = 50

    # 短期窗口（天）：`recent_count` 与 `trend` 都用它
    recent_window_days: int = 90

    # trend 的判定阈值：|trend| < 该值视为"持平"，避免把噪声说成"转向"
    trend_flat_eps: float = 0.05

    # 取前几个题材进画像（12 类题材的画像不用全给，前端雷达图另取全量）
    top_genres_k: int = 6

    # 最近观看列表长度（输出给"继续观看"）
    recent_items_k: int = 10

    # 画像缓存 TTL（秒）。行为事件触发主动失效，所以这里可以放长
    cache_ttl: int = 3600

    # 低于该交互数视为冷启动用户（A2 走纯内容路）
    cold_start_threshold: int = 10

    # LLM 生成摘要的超时（ms）—— 画像在主链路上，超时预算比 A5 更紧
    summarize_timeout_ms: int = 300

# -*- coding: utf-8 -*-
"""A4 融合排序 Agent 的配置（权重口径与 `.env` / `configs/model.yaml` 同源）。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FusionConfig"]


@dataclass(frozen=True)
class FusionConfig:
    # ---- 双路融合权重（协议 §A4：行为 0.7 : 内容 0.3）----
    w_behavior: float = 0.7
    w_content: float = 0.3
    # 冷启动候选（交互数 < cold_start_threshold）改用 5:5
    w_behavior_cold: float = 0.5
    w_content_cold: float = 0.5
    cold_start_threshold: int = 10

    # ---- 归一化 ----
    # 两路分数在不同量纲上（内积 vs 余弦），必须先各自 min-max 再融合。
    # `norm_eps` 防止"该路全是同一个值"时 0/0。
    norm_eps: float = 1e-9

    # ---- 多样性（MMR）----
    # λ=0.7 表示"相关性占 7 成、多样性占 3 成"
    diversity_lambda: float = 0.7
    # 保证每个兴趣维度至少曝光几条（K=4 → 每条兴趣至少有位置）
    min_per_interest: int = 1

    # ---- 输出 ----
    top_n: int = 20
    # 业务规则过滤
    drop_watched: bool = True
    drop_disliked: bool = True
    drop_forbidden: bool = True

    # ---- 落库 ----
    persist: bool = True
    cache_ttl: int = 86400          # rec:{uid} 24h（协议 §6.3）
    result_ttl_minutes: int = 1440

    timeout_ms: int = 100           # 协议 §5.2 给 A4 的预算

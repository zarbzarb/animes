# -*- coding: utf-8 -*-
"""A3 冷启动内容召回 Agent 的配置。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ColdStartConfig"]


@dataclass(frozen=True)
class ColdStartConfig:
    # ---- 内容向量 ----
    # 行序约定：第 i-1 行 = 物品池内索引 i（见 models/content_encoder/fusion.py）
    content_vec_path: str = "data/features/content_vec_512.npy"
    item_index_path: str = "data/features/item_index.json"

    # ---- 新番口径 ----
    # 「新番」= 交互数 < cold_start_threshold **或** 年份 >= min_year。
    # 两者取并集：前者是日志口径（真实冷启动），后者是实验口径
    # （E3 的 holdout 协议，训练集里根本没这些物品）。
    cold_start_threshold: int = 10
    min_year: int = 2021
    new_anime_only: bool = True

    # ---- 检索规模 ----
    top_k: int = 50
    max_candidates: int = 50
    # 历史里只用评分 >= 该值的番剧来构造兴趣内容向量
    # （低分番不能代表口味；rating 为空时按"已看"计入）
    min_rating_for_profile: int = 7
    # 历史条数上限
    max_history: int = 50

    # ---- 降级 ----
    # 索引/向量缺失时的规则打分：`题材重合度 * w1 + 新近度 * w2`
    fallback_w_genre: float = 0.7
    fallback_w_recency: float = 0.3
    fallback_topk: int = 50

    timeout_ms: int = 150
    cache_ttl: int = 1800

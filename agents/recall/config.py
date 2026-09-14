# -*- coding: utf-8 -*-
"""A2 序列召回 Agent 的配置。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RecallConfig"]


@dataclass(frozen=True)
class RecallConfig:
    # ---- 模型 ----
    # 线上召回用哪个架构：`multi_interest` 有 K 路兴趣，召回多样性更好；
    # `sasrec` 只有一路。默认走多兴趣（架构 §5.3 的主推口径）。
    arch: str = "multi_interest"
    checkpoint: str = "multi_interest_content_v1_m26b_mi_add_debug_seed42_best.pt"
    fallback_checkpoint: str = "multi_interest_content_v1_m26b_add_debug_seed42_best.pt"
    device: str = "cpu"                 # 生产配 cuda；答辩机留 cpu 更稳

    # ---- 召回规模 ----
    # 每个兴趣取多少。K=4 × 50 = 200 条原始候选，去重后通常 150~190。
    topk_per_interest: int = 50
    # 单路模型（sasrec）时用这个
    topk_single: int = 200
    # 输出给 A4 的候选上限（A4 自己还会再截）
    max_candidates: int = 300

    # ---- 序列 ----
    max_seq_len: int = 50
    input_cap: int = 48
    # 少于该长度的历史视为"无序列"，直接走降级
    min_seq_len: int = 2

    # ---- 过滤 ----
    # 是否屏蔽违规题材物品（离线 preprocess 导出的 forbidden_item_indices）
    block_forbidden: bool = True
    # 每个兴趣至少保留多少条（防止某个兴趣被过滤空导致 tag 缺失）
    min_per_interest: int = 1

    # ---- 降级 ----
    fallback_topk: int = 200
    # 单用户前向的硬超时（ms）。协议 §5.2 给 A2 的预算是 200ms
    forward_timeout_ms: int = 150

    # ---- 缓存 ----
    recall_cache_ttl: int = 600

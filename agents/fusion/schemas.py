# -*- coding: utf-8 -*-
"""A4 输入输出契约（Pydantic v2）。

`candidate_sets` 刻意用 `list[dict]` 而不是 `list[Candidate]`：
A4 要同时接受 A2（`interest_id` 语义）与 A3（`genre_overlap` 语义）两种候选，
强行统一成一种会让某一方的字段被迫填默认值 —— 那是"看起来对齐、实际丢信息"。
所以这里只约束**必填的三个字段**，其余原样透传。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CandidateIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anime_id: int = Field(..., description="池内索引")
    score: float = 0.0
    source: str = Field("sequence", description="sequence / content / fallback_hot")
    # ⚠️ `rank.diversify` 的输入是"已经算好分的候选"（A7 对话推荐场景），
    # 此时调用方传的是 `final_score`。若这里不接收，`extra=ignore` 会**静默丢掉**
    # 分数，`_select` 拿到的全是 0 → MMR 退化成"纯题材打散"，排序语义悄悄丢失。
    final_score: Optional[float] = None
    # 仅 A2 有
    interest_id: Optional[int] = None
    interest_ids: Optional[list[int]] = None
    hit_count: int = 1
    interest_label: Optional[str] = None
    # 仅 A3 有
    genre_overlap: Optional[float] = None
    matched_genres: Optional[list[str]] = None
    matched_genre_ids: Optional[list[int]] = None
    is_cold_start: Optional[bool] = None
    year: Optional[int] = None
    src_anime_id: Optional[int] = None


class FusionInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int = Field(..., gt=0)
    candidate_sets: list[list[CandidateIn]] = Field(
        default_factory=list, description="[A2 候选, A3 候选] 顺序不敏感")
    profile: Optional[dict] = None
    top_n: int = Field(20, ge=1, le=200)
    diversity_lambda: float = Field(0.7, ge=0.0, le=1.0)
    scene: int = Field(0, ge=0, le=3, description="0 综合 / 1 分题材 / 2 新番 / 3 对话")
    genre_id: Optional[int] = None
    batch_id: Optional[str] = None
    persist: bool = True
    exclude_items: list[int] = Field(default_factory=list)


class ExplainSignals(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_attn_items: list[dict] = Field(default_factory=list,
                                       description="Top3 历史番（注意力/画像来源）")
    genre_overlap: float = 0.0
    matched_genres: list[str] = Field(default_factory=list)
    interest_label: Optional[str] = None
    hit_count: int = 1


class RankedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anime_id: int
    src_anime_id: Optional[int] = None
    rank_no: int = 0
    final_score: float = 0.0
    behavior_score: Optional[float] = None
    content_score: Optional[float] = None
    interest_id: Optional[int] = None
    interest_label: Optional[str] = None
    is_cold_start: bool = False
    title: Optional[str] = None
    year: Optional[int] = None
    explain_signals: ExplainSignals = Field(default_factory=ExplainSignals)


class FusionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    items: list[RankedItem] = Field(default_factory=list)
    n_merged: int = 0
    n_after_filter: int = 0
    used_fallback: bool = False
    degraded_reason: Optional[str] = None
    batch_id: Optional[str] = None

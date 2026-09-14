# -*- coding: utf-8 -*-
"""A3 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ColdStartInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int = Field(..., gt=0)
    profile: Optional[dict] = Field(None, description="A1 画像（提供题材偏好）")
    new_anime_only: bool = Field(True, description="True=只推新番（新番专区）")
    top_k: int = Field(50, ge=1, le=500)
    min_year: Optional[int] = Field(None, description="覆盖默认年份阈值")
    exclude_items: list[int] = Field(default_factory=list)
    # 池内索引序列；缺省由 adapter 从 DB 取
    user_seq: list[int] = Field(default_factory=list)


class ColdCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anime_id: int = Field(..., description="池内索引")
    src_anime_id: Optional[int] = None
    content_score: float = Field(0.0, description="余弦相似度 [-1, 1]")
    genre_overlap: float = Field(0.0, ge=0.0, le=1.0)
    matched_genres: list[str] = Field(default_factory=list)
    matched_genre_ids: list[int] = Field(default_factory=list)
    is_cold_start: bool = True
    year: Optional[int] = None
    source: str = "content"


class ColdStartOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    candidates: list[ColdCandidate] = Field(default_factory=list)
    n_candidates: int = 0
    n_pool: int = 0
    used_fallback: bool = False
    degraded_reason: Optional[str] = None

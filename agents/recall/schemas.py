# -*- coding: utf-8 -*-
"""A2 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RecallInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int = Field(..., gt=0)
    # 池内索引序列（**不是** src_anime_id）。缺省时由 adapter 从 DB 取。
    user_seq: list[int] = Field(default_factory=list)
    profile: Optional[dict] = Field(None, description="A1 画像（可选，用于打标）")
    interests: int = Field(4, ge=1, le=8, description="K，兴趣路数")
    top_k_per_interest: int = Field(50, ge=1, le=500)
    exclude_items: list[int] = Field(default_factory=list,
                                     description="要排除的池内索引（已看/已曝光）")
    batch: bool = Field(False, description="True 时走批量前向（离线）")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anime_id: int = Field(..., description="**池内索引**（1..n_items）")
    src_anime_id: Optional[int] = Field(None, description="数据集 animeID，落库用")
    score: float = 0.0
    interest_id: int = Field(0, description="命中兴趣 0..K-1")
    source: Literal["sequence", "fallback_hot", "fallback_itemcf"] = "sequence"
    interest_label: Optional[str] = None


class RecallOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    candidates: list[Candidate] = Field(default_factory=list)
    n_raw: int = 0
    n_after_dedup: int = 0
    model_ver: Optional[str] = None
    interest_strength: list[float] = Field(default_factory=list,
                                           description="每路兴趣的相对强度（供 A4）")
    used_fallback: bool = False

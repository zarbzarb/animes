# -*- coding: utf-8 -*-
"""A1 输入输出契约（Pydantic v2）。

协议 P1「契约先行」：跨 Agent 调用禁止传裸 dict。这里定义的模型有两个用途 ——
① 入口 `model_validate` 挡掉脏参数；② 作为文档（字段含义写在 Field description 里）。
`extra="ignore"` 是协议 §九 的兼容规则：接收方必须容忍未知字段。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Scope = Literal["long", "short", "both"]


class ProfileInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int = Field(..., gt=0)
    scope: Scope = Field("both", description="long=长期画像 / short=短期偏好 / both")
    force_refresh: bool = Field(False, description="True 时跳过缓存强制重算")


class GenreStrength(BaseModel):
    model_config = ConfigDict(extra="ignore")

    genre_id: int
    genre: str = ""
    strength: float = Field(0.0, ge=0.0, le=1.0, description="归一化兴趣强度")
    count: int = 0
    avg_rating: Optional[float] = None
    trend: float = Field(0.0, description="近期相对历史的强度变化（正=升温）")


class ProfileOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    top_genres: list[GenreStrength] = Field(default_factory=list)
    activity_level: int = Field(0, ge=0, le=2, description="0=低 1=中 2=高")
    activity_label: str = "low"
    watch_intensity: float = Field(0.0, ge=0.0, le=1.0)
    avg_rating_tendency: Optional[float] = None
    dropped_rate: float = Field(0.0, ge=0.0, le=1.0)
    preferred_types: list[str] = Field(default_factory=list)
    preferred_era: list[int] = Field(default_factory=list, description="[start_year, end_year]")
    total_records: int = 0
    summary_text: Optional[str] = None
    user_tag: Optional[str] = None
    recent_items: list[dict] = Field(default_factory=list, description="最近 N 条观看")
    updated_at: Optional[str] = None
    is_cold_start: bool = Field(False, description="交互数 < COLD_START_THRESHOLD")


class SummarizeInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    top_genres: list[GenreStrength] = Field(default_factory=list)
    activity_label: str = "low"
    preferred_types: list[str] = Field(default_factory=list)
    style: Literal["concise", "detailed", "casual"] = "concise"

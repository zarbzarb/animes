# -*- coding: utf-8 -*-
"""A6 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DriftInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int = Field(..., gt=0)
    granularity: Literal["quarter", "month", "year"] = "quarter"
    start: Optional[str] = None
    end: Optional[str] = None
    drift_threshold: float = Field(0.35, ge=0.0, le=1.0)
    style: Literal["concise", "detailed", "casual"] = "concise"


class DriftPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    period: str
    prev_period: Optional[str] = None
    js: float = 0.0
    from_genre_id: Optional[int] = None
    to_genre_id: Optional[int] = None
    description: str = ""


class DriftTrendPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    period: str
    total: int = 0
    top_genre_id: Optional[int] = None
    top_share: float = 0.0


class DriftOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    granularity: str = "quarter"
    radar: list[float] = Field(default_factory=list, description="12 维题材强度")
    radar_labels: list[str] = Field(default_factory=list)
    trend: list[DriftTrendPoint] = Field(default_factory=list)
    drift_points: list[DriftPoint] = Field(default_factory=list)
    interpretation: Optional[str] = None
    is_stable: bool = False
    n_periods: int = 0
    degraded: bool = False
    tokens_used: int = 0

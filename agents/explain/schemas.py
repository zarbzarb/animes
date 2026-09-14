# -*- coding: utf-8 -*-
"""A5 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Style = Literal["concise", "detailed", "casual"]


class TargetAnime(BaseModel):
    model_config = ConfigDict(extra="ignore")

    anime_id: Optional[int] = None
    src_anime_id: Optional[int] = None
    title: str = ""
    genres: list[str] = Field(default_factory=list)
    year: Optional[int] = None


class ExplainInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int = Field(..., gt=0)
    target_anime: TargetAnime
    explain_signals: dict = Field(default_factory=dict,
                                  description="A4 的结构化信号（唯一事实来源）")
    style: Style = "concise"
    # 单条解释接口=True（预算放宽到 3s）；主链路=False
    standalone: bool = False
    persist: bool = True


class ExplainOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: int
    anime_id: Optional[int] = None
    reason: str = ""
    core_items: list[dict] = Field(default_factory=list, description="Top3 影响番剧")
    match_percent: int = Field(0, ge=0, le=100)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: Literal["llm", "template"] = "template"
    style: str = "concise"
    prompt_ver: Optional[str] = None
    llm_model: Optional[str] = None
    tokens_used: int = 0
    degraded: bool = False
    violations: list[str] = Field(default_factory=list,
                                  description="事实校验拦下的项（写进 trace 便于归因）")

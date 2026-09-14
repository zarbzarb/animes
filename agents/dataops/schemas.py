# -*- coding: utf-8 -*-
"""A8 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Task = Literal["sync_new_anime", "rebuild_index", "quality_check", "fix_data"]


class DataOpsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: Task = "quality_check"
    params: dict = Field(default_factory=dict)
    # sync_new_anime：待入库的新番元数据
    items: list[dict] = Field(default_factory=list)
    dry_run: bool = Field(False, description="只报告，不写库")
    operator_id: Optional[int] = None


class QualityItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    check_type: str
    target_table: str
    total_rows: int = 0
    problem_rows: int = 0
    problem_rate: float = 0.0
    detail: Optional[dict] = None
    alert_level: int = 0


class DataOpsOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task: str
    processed_count: int = 0
    failed_items: list[dict] = Field(default_factory=list)
    quality_report: list[QualityItem] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    dry_run: bool = False
    degraded: bool = False
    tokens_used: int = 0

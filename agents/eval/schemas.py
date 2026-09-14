# -*- coding: utf-8 -*-
"""A9 输入输出契约（Pydantic v2）。"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvalInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["offline", "online", "ablation", "trace"] = "online"
    exp_dir: Optional[str] = Field(None, description="offline/ablation：实验结果目录")
    date_range: Optional[list[str]] = Field(None, description="[start, end]")
    scene: Optional[int] = None
    days: int = Field(30, ge=1, le=365)
    # 告警阈值（环比跌幅）
    drop_threshold: float = Field(0.10, ge=0.0, le=1.0)
    persist: bool = True


class MetricRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_date: str
    scene: int = 0
    metric_type: str
    metric_value: float
    sample_size: int = 0
    model_ver: Optional[str] = None
    is_cold_start: int = 0
    genre_id: Optional[int] = None
    extra: Optional[dict] = None
    prev_value: Optional[float] = None
    change: Optional[float] = None


class Anomaly(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_type: str
    scene: int = 0
    current: float = 0.0
    previous: Optional[float] = None
    change: Optional[float] = None
    level: int = Field(1, ge=0, le=2, description="0 正常 1 警告 2 严重")
    message: str = ""


class EvalOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str
    metrics: list[MetricRow] = Field(default_factory=list)
    trend: list[dict] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    bottleneck_agent: Optional[str] = None
    agent_stats: list[dict] = Field(default_factory=list)
    alert_text: Optional[str] = None
    degraded: bool = False
    tokens_used: int = 0

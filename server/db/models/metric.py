# -*- coding: utf-8 -*-
"""指标 / 配置 / 数据质量：`metric_snapshot` / `system_config` / `data_quality_log`
（database-design.md §5.1–5.3）。

`metric_snapshot` 是**两套指标共用的同一张表**：
* 在线指标（CTR / CVR）由 A9 从 `user_feedback` 聚合；
* 离线指标（hr@5 / ndcg@10 …）由 A9 读实验结果回写。

用 `metric_type` 区分。之所以不拆两张表：管理后台的看板要**在同一张时间轴上
同时画线上 CTR 与离线 NDCG**，拆表就得在应用层做 union，而这正是"指标口径
容易分叉"的地方（本项目已经把这个问题栽过两次：指标唯一实现、
负样本唯一实现）。离线指标**必须**来自 `models/eval/metrics.py`，
不允许在这里重算 —— 数值可以复用，公式不许复用。
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Date, DateTime, ForeignKey, Index, Integer, JSON, Numeric, SmallInteger, String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import Base, BigIntPK, TimestampMixin, utcnow

# 离线指标（出处：models/eval/metrics.py）
OFFLINE_METRICS = ("hr5", "hr10", "ndcg5", "ndcg10", "mrr", "recall10")
# 在线指标（出处：user_feedback 聚合）
ONLINE_METRICS = ("ctr", "cvr", "fav_rate")

# data_quality_log.check_type
CHECK_MISSING = "missing"
CHECK_DUPLICATE = "duplicate"
CHECK_OUTLIER = "outlier"
CHECK_FORBIDDEN = "forbidden"
CHECK_GENRE_UNMAPPED = "genre_unmapped"
CHECK_TYPES = (CHECK_MISSING, CHECK_DUPLICATE, CHECK_OUTLIER,
               CHECK_FORBIDDEN, CHECK_GENRE_UNMAPPED)

ALERT_NORMAL, ALERT_WARN, ALERT_CRITICAL = 0, 1, 2


class MetricSnapshot(Base):
    __tablename__ = "metric_snapshot"
    __table_args__ = (
        UniqueConstraint("metric_date", "scene", "metric_type", "model_ver",
                         "is_cold_start", "genre_id", name="uk_date_scene_metric"),
        Index("idx_type_date", "metric_type", "metric_date"),
        Index("idx_model", "model_ver"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    metric_date: Mapped[date] = mapped_column(Date, nullable=False)
    scene: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    metric_type: Mapped[str] = mapped_column(String(32), nullable=False)
    metric_value: Mapped[Decimal] = mapped_column(Numeric(8, 5), nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_ver: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    is_cold_start: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    genre_id: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    extra: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class SystemConfig(Base, TimestampMixin):
    """配置优先级：`.env`（启动时）< 本表（运行时）< 请求参数（临时覆盖）。

    只有 `is_hot_reload=1` 的项允许运行时改（如 `COLD_START_THRESHOLD`、
    融合权重）—— 因为改它们不需要重建模型或重连数据库。
    """

    __tablename__ = "system_config"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    config_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    config_value: Mapped[str] = mapped_column(String(255), nullable=False)
    value_type: Mapped[str] = mapped_column(String(16), nullable=False, default="string")
    scope: Mapped[str] = mapped_column(String(16), nullable=False, default="global")
    description: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    is_hot_reload: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    updated_by: Mapped[Optional[int]] = mapped_column(BigIntPK, default=None)

    def typed_value(self):
        v = self.config_value
        try:
            if self.value_type == "int":
                return int(v)
            if self.value_type == "float":
                return float(v)
            if self.value_type == "bool":
                return v.strip().lower() in {"1", "true", "yes", "on"}
            if self.value_type == "json":
                import json
                return json.loads(v)
        except (TypeError, ValueError):
            return v
        return v


class DataQualityLog(Base):
    __tablename__ = "data_quality_log"
    __table_args__ = (Index("idx_check_date", "check_date", "check_type"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    check_date: Mapped[date] = mapped_column(Date, nullable=False)
    check_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_table: Mapped[str] = mapped_column(String(32), nullable=False)
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    problem_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    problem_rate: Mapped[Decimal] = mapped_column(
        Numeric(6, 5), nullable=False, default=Decimal("0"))
    detail: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    alert_level: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=ALERT_NORMAL)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


__all__ = [
    "ALERT_CRITICAL",
    "ALERT_NORMAL",
    "ALERT_WARN",
    "CHECK_DUPLICATE",
    "CHECK_FORBIDDEN",
    "CHECK_GENRE_UNMAPPED",
    "CHECK_MISSING",
    "CHECK_OUTLIER",
    "CHECK_TYPES",
    "OFFLINE_METRICS",
    "ONLINE_METRICS",
    "DataQualityLog",
    "MetricSnapshot",
    "SystemConfig",
]

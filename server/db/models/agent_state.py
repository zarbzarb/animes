# -*- coding: utf-8 -*-
"""Agent 运行时状态与调用链：`agent_state` / `agent_trace`
（database-design.md §4.1–4.2）。

这两张表是「**可观测**」这条架构要求的落地：
* `agent_state` → 管理后台的 Agent 健康面板（健康分、成功率、P95 耗时、熔断状态）；
* `agent_trace` → 一次推荐请求的完整调用树，前端 `AgentTraceViewer` 组件读它。

⚠️ `agent_trace` 是**全系统写入频率最高的表**（每次推荐 4–6 行）。文档给的结论
是必须按天分区 + 定期 `DROP PARTITION`。SQLAlchemy 不支持声明式分区，
MySQL 侧由 `scripts/init_db.py` 在建表后补 `PARTITION BY RANGE COLUMNS`；
SQLite（测试）不分区，靠 `idx_created` 清理。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    CHAR, DateTime, Index, Integer, JSON, Numeric, SmallInteger, String,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import Base, BigIntPK, TimestampMixin, utcnow

AGENT_ONLINE, AGENT_DEGRADED, AGENT_CIRCUIT_OPEN = 0, 2, 3
AGENT_HEALTHY = 1
AGENT_STATUS_LABEL = {AGENT_ONLINE: "下线", AGENT_HEALTHY: "正常",
                      AGENT_DEGRADED: "降级中", AGENT_CIRCUIT_OPEN: "熔断"}

# agent_trace.status（与协议 §4.2 的错误语义对齐）
TRACE_SUCCESS, TRACE_DEGRADED, TRACE_ERROR, TRACE_TIMEOUT = 0, 1, 2, 3
TRACE_STATUS_LABEL = {TRACE_SUCCESS: "success", TRACE_DEGRADED: "degraded",
                      TRACE_ERROR: "error", TRACE_TIMEOUT: "timeout"}

TRACE_RETENTION_DAYS = 30


class AgentState(Base, TimestampMixin):
    __tablename__ = "agent_state"
    __table_args__ = (Index("idx_status", "status"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(8), nullable=False, unique=True)
    agent_name: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=AGENT_HEALTHY)
    health_score: Mapped[Decimal] = mapped_column(
        Numeric(5, 4), nullable=False, default=Decimal("1.0"))
    success_cnt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fail_cnt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timeout_cnt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    degrade_cnt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_elapsed_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    p95_elapsed_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    circuit_open_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    version: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    config: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    @property
    def total_cnt(self) -> int:
        return self.success_cnt + self.fail_cnt + self.timeout_cnt


class AgentTrace(Base):
    __tablename__ = "agent_trace"
    # 索引名加 `at_` 前缀：SQLite 的索引名是库级唯一，重名会直接建表失败
    __table_args__ = (
        Index("idx_trace", "trace_id"),
        Index("idx_at_agent_time", "to_agent", "created_at"),
        Index("idx_at_status_time", "status", "created_at"),
        Index("idx_at_created", "created_at"),
    )

    # 文档里是 (id, created_at) 复合主键（分区表要求分区键进主键）；
    # SQLite 不支持，故这里用自增 id，MySQL 建表时由 init_db 补分区。
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(24), nullable=False)
    msg_id: Mapped[str] = mapped_column(String(32), nullable=False)
    parent_msg_id: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    from_agent: Mapped[str] = mapped_column(String(8), nullable=False)
    to_agent: Mapped[str] = mapped_column(String(8), nullable=False)
    action: Mapped[str] = mapped_column(String(48), nullable=False)
    priority: Mapped[str] = mapped_column(CHAR(4), nullable=False, default="P0")
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=TRACE_SUCCESS)
    error_code: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    elapsed_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_hit: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    call_path: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    detail: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


__all__ = [
    "AGENT_CIRCUIT_OPEN",
    "AGENT_DEGRADED",
    "AGENT_HEALTHY",
    "AGENT_ONLINE",
    "AGENT_STATUS_LABEL",
    "TRACE_DEGRADED",
    "TRACE_ERROR",
    "TRACE_RETENTION_DAYS",
    "TRACE_STATUS_LABEL",
    "TRACE_SUCCESS",
    "TRACE_TIMEOUT",
    "AgentState",
    "AgentTrace",
]

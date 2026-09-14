# -*- coding: utf-8 -*-
"""`agent_memory_short` —— 会话记忆持久化表（database-design.md §4.3）。

⚠️ **主记忆在 Redis**（`mem:short:{uid}:{sid}`，TTL 30min）。本表只落
**关键会话摘要**，用于 session 过期后复盘与 A7 对话推荐的训练数据积累。
把它当成"Redis 的备份"会走偏：这里只写摘要，不写逐字对话（也正因如此，
`content` 字段在写入前必须脱敏 —— 见 `agents/common/llm.py` 的 PII 过滤）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime, ForeignKey, Index, Integer, JSON, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import Base, BigIntPK, utcnow

ROLE_USER, ROLE_ASSISTANT = "user", "assistant"

# 保留 90 天后归档（文档 §4.3 的 idx_created 注释）
RETENTION_DAYS = 90


class SessionMemory(Base):
    __tablename__ = "agent_memory_short"
    # ⚠️ 索引名在 SQLite 里是**库级唯一**（MySQL 是表级），所以别处用过
    # `idx_user_created` / `idx_created` 会直接建表失败。这里统一加 `ams_` 前缀。
    __table_args__ = (
        Index("idx_ams_session_turn", "session_id", "turn_no"),
        Index("idx_ams_user_created", "user_id", "created_at"),
        Index("idx_ams_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    turn_no: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)      # 已脱敏
    extracted: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    shown_anime_ids: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


__all__ = ["RETENTION_DAYS", "ROLE_ASSISTANT", "ROLE_USER", "SessionMemory"]

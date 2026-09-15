# -*- coding: utf-8 -*-
"""`watch_record` —— 追番记录表（database-design.md §3.6）。

**这是整个系统的地基**：A2 序列召回的输入、A1 画像的统计来源、A6 漂移的时间序列
全部来自这张表。

⚠️ 序列顺序的口径：`dataset.pkl` 里用 `animeID` 原始顺序，线上按 `updated_at` 排序。
两者要对齐，必须依赖 `user.src_user_id` 与 `anime.src_anime_id` 的准确性 ——
否则离线实验的指标和线上表现对不上，而且**不会报错**。
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import (
    Date, ForeignKey, Index, Integer, JSON, SmallInteger, String, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.db.base import Base, BigIntPK, TimestampMixin
from server.db.models.anime import Anime

STATUS_PLAN = 0        # 想看
STATUS_WATCHING = 1    # 在看
STATUS_DONE = 2        # 已看
STATUS_DROPPED = 3     # 弃番

# A2 构造输入序列时使用的状态（想看不算"看过"）
STATUS_SEQUENCE = (STATUS_WATCHING, STATUS_DONE)

STATUS_LABEL = {STATUS_PLAN: "想看", STATUS_WATCHING: "在看",
                STATUS_DONE: "已看", STATUS_DROPPED: "弃番"}


class WatchRecord(Base, TimestampMixin):
    __tablename__ = "watch_record"
    __table_args__ = (
        UniqueConstraint("user_id", "anime_id", name="uk_user_anime"),
        Index("idx_user_updated", "user_id", "updated_at"),
        Index("idx_user_status", "user_id", "status"),
        Index("idx_anime_id", "anime_id"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    anime_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("anime.id"), nullable=False)
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False,
                                        default=STATUS_WATCHING)
    rating: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    watched_at: Mapped[Optional[date]] = mapped_column(Date, default=None)
    tags: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    # 文字评价（用户主观短评，≤500 字；与 1-10 的 rating 并列，可只填其一）
    review: Mapped[Optional[str]] = mapped_column(String(500), default=None)

    anime: Mapped[Anime] = relationship(lazy="selectin")


__all__ = [
    "STATUS_DONE",
    "STATUS_DROPPED",
    "STATUS_LABEL",
    "STATUS_PLAN",
    "STATUS_SEQUENCE",
    "STATUS_WATCHING",
    "WatchRecord",
]

# -*- coding: utf-8 -*-
"""`user_profile` —— 用户兴趣画像表（database-design.md §3.7）。

A1 画像 Agent 的唯一持久化落地。

`top_genres` 用 JSON 而不是 12 个列的原因（文档已论证）：题材数可能从 12 调到 16，
且每个题材带 `strength/count/avg_rating/trend` 四个属性，列式会膨胀到 60 个字段；
而实际查询永远是"整行取出"，不需要按题材过滤。将来若真要按题材聚合，
再抽 `user_genre_strength` 表 —— **不要**现在提前优化。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, DateTime, ForeignKey, Index, Integer, JSON, Numeric, SmallInteger,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import Base, BigIntPK, TimestampMixin

ACTIVITY_LOW, ACTIVITY_MEDIUM, ACTIVITY_HIGH = 0, 1, 2
ACTIVITY_LABEL = {ACTIVITY_LOW: "low", ACTIVITY_MEDIUM: "medium",
                  ACTIVITY_HIGH: "high"}

# 画像结构版本：加了字段就 +1。LLM 解读与前端雷达图都按这个版本解析 top_genres，
# 不升版本就改结构会让老数据在界面上错位（不报错、只是画错）。
PROFILE_SCHEMA_VERSION = 1


class UserProfile(Base, TimestampMixin):
    __tablename__ = "user_profile"
    __table_args__ = (Index("idx_activity", "activity_level"),
                      Index("idx_version", "version"))

    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)
    # [{"genre_id":1,"genre":"热血战斗","strength":0.42,"count":37,
    #   "avg_rating":8.1,"trend":0.12}, ...]
    top_genres: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    activity_level: Mapped[int] = mapped_column(SmallInteger, nullable=False,
                                                default=ACTIVITY_LOW)
    watch_intensity: Mapped[Decimal] = mapped_column(
        Numeric(5, 4), nullable=False, default=Decimal("0"))
    avg_rating_tendency: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(4, 2), default=None)
    dropped_rate: Mapped[Decimal] = mapped_column(
        Numeric(5, 4), nullable=False, default=Decimal("0"))
    preferred_types: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    preferred_era_start: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    preferred_era_end: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    total_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary_text: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    user_tag: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    # 乐观并发控制：A1 批量刷新与在线增量重排可能同时写同一用户
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    computed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    def top_genre_ids(self, top_n: int = 4) -> list[int]:
        ranked = sorted(self.top_genres or [],
                        key=lambda g: float(g.get("strength", 0.0)), reverse=True)
        return [int(g["genre_id"]) for g in ranked[:top_n] if "genre_id" in g]


__all__ = [
    "ACTIVITY_HIGH",
    "ACTIVITY_LABEL",
    "ACTIVITY_LOW",
    "ACTIVITY_MEDIUM",
    "PROFILE_SCHEMA_VERSION",
    "UserProfile",
]

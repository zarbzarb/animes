# -*- coding: utf-8 -*-
"""`user_interest_capsule` —— 兴趣胶囊表（database-design.md §3.8）。

每个用户固定 K 行（K=4）。向量本体不存表：`vec_ref = "cap_1024.npy#0"` 指向
用户胶囊矩阵文件的行，与 `anime_content.vec_ref` 同一约定。

用途分工：
* `strength`（路由系数归一化）→ 在线召回时决定"哪个兴趣出几条"、
  前端兴趣雷达图的扇区宽度；
* `label` → 把胶囊对齐回 12 类题材，让用户看得懂"这条兴趣是什么题材"；
* 向量本体 → A2 用胶囊向量与物品嵌入做内积召回。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, DateTime, ForeignKey, Index, Numeric, SmallInteger, String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import Base, BigIntPK, TimestampMixin

# 胶囊向量与 `models/multi_interest` 的产出口径绑定；换训练口径必须改这个值，
# 否则会拿旧向量配新模型（指标会变差但不会报错）
MODEL_VER = "multi_interest_v1"


class UserInterestCapsule(Base, TimestampMixin):
    __tablename__ = "user_interest_capsule"
    __table_args__ = (
        UniqueConstraint("user_id", "capsule_id", name="uk_user_capsule"),
        Index("idx_label", "label"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    capsule_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    strength: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    vec_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    model_ver: Mapped[str] = mapped_column(String(32), nullable=False, default=MODEL_VER)
    computed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


__all__ = ["MODEL_VER", "UserInterestCapsule"]

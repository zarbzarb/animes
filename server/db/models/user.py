# -*- coding: utf-8 -*-
"""`user` —— 用户表（database-design.md §3.1）。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Integer, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import Base, BigIntPK, TimestampMixin

ROLE_USER = 0
ROLE_ADMIN = 1
STATUS_DISABLED = 0
STATUS_ACTIVE = 1


class User(Base, TimestampMixin):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(120), default=None)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    nickname: Mapped[str] = mapped_column(String(50), nullable=False)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    role: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=ROLE_USER)
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=STATUS_ACTIVE)
    # 与 dataset.pkl 的 umap 对齐：线上序列与离线实验序列靠它对齐（§3.6 的提醒）
    src_user_id: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

    @property
    def is_admin(self) -> bool:
        return int(self.role) == ROLE_ADMIN

    @property
    def is_active(self) -> bool:
        return int(self.status) == STATUS_ACTIVE


__all__ = ["ROLE_ADMIN", "ROLE_USER", "STATUS_ACTIVE", "STATUS_DISABLED", "User"]

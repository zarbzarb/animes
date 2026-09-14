# -*- coding: utf-8 -*-
"""SQLAlchemy 声明式基类与通用列。

两条工程约定
------------
1. **主键统一 `BigInteger`，但在 SQLite 上退化为 `Integer`**
   （`BIGINT PRIMARY KEY` 在 SQLite 里不会自增，测试会全线炸在"id 为空"上）。
   测试用内存 SQLite、生产用 MySQL，同一个模型定义必须同时成立。
2. **表名/列名与 `docs/database-design.md` 完全一致**，不做"更好听"的重命名 ——
   文档里的每一条 SQL 都能直接对着 ORM 读，改表结构时不会出现两处打架。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import BigInteger, DateTime, Integer, JSON, MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 统一命名规则：让 Alembic 自动生成的索引/约束名稳定（否则每次 diff 都不一样）
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uk_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def to_dict(self, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
        """给 API 层用的浅序列化。**不要**把它当成通用序列化方案："""
        return {c.name: getattr(self, c.name)
                for c in self.__table__.columns if c.name not in exclude}


# SQLite 下自增主键必须是 INTEGER，故用 with_variant 退化
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


def utcnow() -> datetime:
    """统一用无时区 UTC 存库（协议 §3.2 要求时间戳为 UTC）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TimestampMixin:
    """`created_at` / `updated_at`（文档「通用字段」一节的约定）。"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow)


__all__ = ["Base", "BigIntPK", "JSON", "TimestampMixin", "utcnow"]

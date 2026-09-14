# -*- coding: utf-8 -*-
"""动漫与题材相关四张表：`anime` / `genre` / `anime_genre` / `anime_content`
（database-design.md §3.2–3.5）。

⚠️ **内容向量不存表里**。`anime_content` 只记 `vec_ref = "content_vec_512.npy#1234"`
（行号），15,687 × 512 float32 ≈ 32MB，存 BLOB 会让表膨胀且 FAISS 无法直接加载。
所以读向量的唯一方式是通过 `vec_ref` 去 `np.load(mmap_mode="r")` 按行取。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, CHAR, DateTime, ForeignKey, Index, Integer, JSON, Numeric,
    SmallInteger, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.db.base import Base, BigIntPK, TimestampMixin, utcnow


class Anime(Base, TimestampMixin):
    __tablename__ = "anime"
    __table_args__ = (
        Index("idx_year_type", "year", "type"),
        Index("idx_n_interactions", "n_interactions"),
        Index("idx_online_forbidden", "is_online", "is_forbidden"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    src_anime_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    alt_title: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    type: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    year: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    season: Mapped[Optional[str]] = mapped_column(CHAR(6), default=None)
    score: Mapped[Optional[Decimal]] = mapped_column(Numeric(4, 2), default=None)
    episodes: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    mal_url: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    image_url: Mapped[Optional[str]] = mapped_column(String(512), default=None)
    is_sequel: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    raw_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    genre_raw: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    genre_detail: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    # 冷启动判定的唯一依据（阈值见 .env 的 COLD_START_THRESHOLD）
    n_interactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_forbidden: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    is_online: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)

    genres: Mapped[list["AnimeGenre"]] = relationship(
        back_populates="anime", cascade="all, delete-orphan", lazy="selectin")

    @property
    def is_cold_start(self) -> bool:
        from server.core.config import settings
        return int(self.n_interactions or 0) < int(settings.COLD_START_THRESHOLD)

    @property
    def primary_genre_id(self) -> Optional[int]:
        for link in self.genres:
            if link.is_primary:
                return int(link.genre_id)
        return int(self.genres[0].genre_id) if self.genres else None


class Genre(Base):
    __tablename__ = "genre"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=True)
    name_cn: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    name_en: Mapped[str] = mapped_column(String(40), nullable=False)
    mal_genres: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    description: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    sort_order: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)


class AnimeGenre(Base):
    __tablename__ = "anime_genre"
    __table_args__ = (Index("idx_genre_anime", "genre_id", "anime_id"),)

    anime_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("anime.id", ondelete="CASCADE"), primary_key=True)
    genre_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("genre.id"), primary_key=True)
    is_primary: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    weight: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False,
                                            default=Decimal("1.000"))

    anime: Mapped[Anime] = relationship(back_populates="genres")


class AnimeContent(Base):
    __tablename__ = "anime_content"
    __table_args__ = (Index("idx_faiss_idx", "faiss_idx"),)

    anime_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("anime.id", ondelete="CASCADE"), primary_key=True)
    vec_dim: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=512)
    vec_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    faiss_idx: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    encode_model: Mapped[str] = mapped_column(
        String(64), nullable=False, default="distilbert-base-multilingual-cased")
    encode_ver: Mapped[str] = mapped_column(String(16), nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


__all__ = ["Anime", "AnimeContent", "AnimeGenre", "Genre"]

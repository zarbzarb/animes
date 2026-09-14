# -*- coding: utf-8 -*-
"""推荐结果 / 解释 / 冷启动池 / 反馈 —— 四张表
（database-design.md §3.9–3.12）。

这几个是**推荐链路的落地层**：
`recommend_result` 是 A4 融合排序 Agent 的批量产出（离线全量 + 在线增量），
`recommend_explain` 是 A5 解释生成 Agent 的产出（**带降级标记**），
`cold_start_pool` 是 A3/A8 的新番池，`user_feedback` 是在线指标与负反馈来源。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    CHAR, DateTime, ForeignKey, Index, Integer, JSON, Numeric, SmallInteger, String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from server.db.base import Base, BigIntPK, TimestampMixin, utcnow

# ---------------- recommend_result.scene ----------------
SCENE_FEED, SCENE_GENRE, SCENE_NEW_ANIME, SCENE_CHAT = 0, 1, 2, 3
SCENE_LABEL = {SCENE_FEED: "综合", SCENE_GENRE: "分题材",
               SCENE_NEW_ANIME: "新番", SCENE_CHAT: "对话"}

# ---------------- user_feedback.action ----------------
FB_EXPOSE, FB_CLICK, FB_FAV, FB_DISLIKE, FB_LIKE = 0, 1, 2, 3, 4
FB_LABEL = {FB_EXPOSE: "曝光", FB_CLICK: "点击", FB_FAV: "收藏",
            FB_DISLIKE: "不感兴趣", FB_LIKE: "感兴趣"}

# ---------------- recommend_explain.source ----------------
EXPLAIN_LLM, EXPLAIN_TEMPLATE = 0, 1


class RecommendResult(Base):
    __tablename__ = "recommend_result"
    __table_args__ = (
        UniqueConstraint("user_id", "scene", "genre_id", "rank_no", "batch_id",
                         name="uk_user_scene_rank"),
        Index("idx_user_scene", "user_id", "scene", "rank_no"),
        Index("idx_batch", "batch_id"),
        Index("idx_expire", "expire_at"),
        Index("idx_anime", "anime_id"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    scene: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=SCENE_FEED)
    genre_id: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    anime_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("anime.id"), nullable=False)
    rank_no: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    final_score: Mapped[Decimal] = mapped_column(Numeric(8, 5), nullable=False)
    behavior_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 5), default=None)
    content_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 5), default=None)
    interest_id: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    is_cold_start: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    # 结构化解释信号（core_items / genre_overlap）—— 模板兜底文案就靠它拼
    explain_signals: Mapped[Optional[dict]] = mapped_column(JSON, default=None)
    batch_id: Mapped[str] = mapped_column(String(32), nullable=False)
    expire_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class RecommendExplain(Base):
    __tablename__ = "recommend_explain"
    __table_args__ = (
        UniqueConstraint("user_id", "anime_id", "style", name="uk_user_anime_style"),
        Index("idx_user", "user_id"),
        Index("idx_source", "source"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    anime_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("anime.id"), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    core_items: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    match_percent: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    matched_genres: Mapped[Optional[list]] = mapped_column(JSON, default=None)
    style: Mapped[str] = mapped_column(String(16), nullable=False, default="concise")
    # source + prompt_ver 是**实验复现**的关键：能回答"这条解释是哪个 Prompt 版本生成的、
    # 有没有走降级"。没有这两列，降级率和 LLM 效果都无法归因。
    source: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=EXPLAIN_LLM)
    prompt_ver: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    llm_model: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class ColdStartPool(Base, TimestampMixin):
    __tablename__ = "cold_start_pool"
    __table_args__ = (Index("idx_active_season", "is_active", "season"),
                      Index("idx_ctr", "is_active", "exposure_cnt"))

    anime_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("anime.id", ondelete="CASCADE"), primary_key=True)
    season: Mapped[str] = mapped_column(CHAR(6), nullable=False)
    n_interactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_vec_ready: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)
    faiss_idx: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    exposure_cnt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    click_cnt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fav_cnt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enter_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    exit_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    is_active: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)

    @property
    def ctr(self) -> float:
        return float(self.click_cnt) / self.exposure_cnt if self.exposure_cnt else 0.0


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    __table_args__ = (
        Index("idx_user_created", "user_id", "created_at"),
        Index("idx_anime_action", "anime_id", "action"),
        Index("idx_scene_created", "scene", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    anime_id: Mapped[int] = mapped_column(
        BigIntPK, ForeignKey("anime.id"), nullable=False)
    scene: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=SCENE_FEED)
    action: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    position: Mapped[Optional[int]] = mapped_column(SmallInteger, default=None)
    reason: Mapped[Optional[str]] = mapped_column(String(255), default=None)
    batch_id: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


__all__ = [
    "EXPLAIN_LLM",
    "EXPLAIN_TEMPLATE",
    "FB_CLICK",
    "FB_DISLIKE",
    "FB_EXPOSE",
    "FB_FAV",
    "FB_LABEL",
    "FB_LIKE",
    "SCENE_CHAT",
    "SCENE_FEED",
    "SCENE_GENRE",
    "SCENE_LABEL",
    "SCENE_NEW_ANIME",
    "ColdStartPool",
    "RecommendExplain",
    "RecommendResult",
    "UserFeedback",
]

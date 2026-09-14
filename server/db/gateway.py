# -*- coding: utf-8 -*-
"""`SqlGateway` —— `agents.common.data.DataGateway` 协议的唯一实现。

为什么这一层必须存在
------------------
Agent 侧只认 `Protocol`（`agents/common/data.py`），因为依赖方向是
`server/ → agents/` 单向。所有"Agent 想要的数据，实际存在哪张表、字段叫什么、
要不要做 id 翻译"这件事，全部收口在本文件。换存储（MySQL→别的）只改这里。

⚠️ 两个 id 空间，混用不报错但全错
--------------------------------
| 空间 | 谁在用 | 例子 |
|---|---|---|
| **数据集 animeID**（`src_anime_id`） | `watch_record` / `anime` / 所有 Agent 侧查询 | 16498 |
| **池内索引**（`item_index.json`） | 模型输入与候选（A2/A3/A4 的 `anime_id` 字段） | 1234 |
| **`anime.id`**（自增内部主键） | `recommend_result.anime_id` 等外键列 | 1201 |

实测 `src_anime_id` 与池内索引有 **51.7% 不相等**，而 `anime.id` 与两者都无关。
本文件的纪律：
* 返回给 Agent 的 `anime_id` **一律是数据集 animeID**；
* 落 `recommend_result` / `recommend_explain` 时把池内索引翻译成 `anime.id`；
* 读回来时反向翻译，让 Agent 拿到的仍是池内索引（A7 的卡片要靠它做重排）。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import case, func, or_, select

from server.db import item_index as _ix
from server.db.models.agent_state import AgentState, AgentTrace
from server.db.models.anime import Anime, AnimeContent, AnimeGenre, Genre
from server.db.models.capsule import UserInterestCapsule
from server.db.models.memory import SessionMemory
from server.db.models.metric import DataQualityLog, MetricSnapshot, SystemConfig
from server.db.models.profile import UserProfile
from server.db.models.recommend import (
    RecommendExplain, RecommendResult, ColdStartPool, UserFeedback,
)
from server.db.models.record import STATUS_SEQUENCE, WatchRecord
from server.db.models.user import User
from server.db.session import session_scope

logger = logging.getLogger(__name__)

__all__ = ["SqlGateway"]

_SEQ_STATUS = tuple(int(x) for x in STATUS_SEQUENCE)
# 冷启动判定的兜底阈值（与 .env 的 COLD_START_THRESHOLD 同值；真正判定在 models/ 侧）
_COLD_DEFAULT = 10


def _f(v: Any) -> Optional[float]:
    """Decimal / None → float / None。**不要 `or 0.0`**：0 与"缺失"语义不同。"""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_date(v: Any) -> Optional[date]:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str) and v:
        try:
            return date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


class SqlGateway:
    """结构化子类型实现：**不继承** `DataGateway`（`typing.Protocol` 不需要继承），
    只要求方法名与签名一致。这样 `agents/` 与本文件之间没有任何 import 关系。"""

    def __init__(self, *, cold_start_threshold: int = _COLD_DEFAULT) -> None:
        self.cold_start_threshold = int(cold_start_threshold)

    # ================================================================ 用户
    def get_user(self, user_id: int) -> Optional[dict]:
        with session_scope() as s:
            u = s.get(User, int(user_id))
            return self._user_dict(u) if u else None

    def get_user_by_username(self, username: str) -> Optional[dict]:
        with session_scope() as s:
            u = s.scalars(select(User).where(User.username == str(username))).first()
            if not u:
                return None
            d = self._user_dict(u)
            d["password_hash"] = u.password_hash        # 仅供 server 鉴权
            return d

    def user_exists(self, user_id: int) -> bool:
        with session_scope() as s:
            return s.get(User, int(user_id)) is not None

    @staticmethod
    def _user_dict(u: User) -> dict:
        return {"id": int(u.id), "username": u.username, "nickname": u.nickname,
                "email": u.email, "avatar_url": u.avatar_url,
                "role": int(u.role), "status": int(u.status),
                "src_user_id": _i(u.src_user_id),
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None}

    # ---- 用户（管理/自助，超出 Protocol 的便利方法）----
    def list_users(self, *, limit: int = 20, offset: int = 0,
                   keyword: str = "") -> list[dict]:
        with session_scope() as s:
            stmt = select(User).order_by(User.id)
            if keyword:
                like = f"%{keyword}%"
                stmt = stmt.where(or_(User.username.like(like),
                                      User.nickname.like(like)))
            rows = s.scalars(stmt.limit(int(limit)).offset(int(offset))).all()
            return [self._user_dict(u) for u in rows]

    def count_users(self, *, keyword: str = "") -> int:
        with session_scope() as s:
            stmt = select(func.count()).select_from(User)
            if keyword:
                like = f"%{keyword}%"
                stmt = stmt.where(or_(User.username.like(like),
                                      User.nickname.like(like)))
            return int(s.scalar(stmt) or 0)

    def create_user(self, *, username: str, password_hash: str, nickname: str,
                    email: Optional[str] = None, role: int = 0) -> dict:
        with session_scope() as s:
            u = User(username=username, password_hash=password_hash,
                     nickname=nickname, email=email, role=int(role))
            s.add(u)
            s.flush()
            return self._user_dict(u)

    def update_user(self, user_id: int, data: dict) -> None:
        allowed = {"nickname", "email", "avatar_url", "role", "status",
                   "src_user_id"}
        with session_scope() as s:
            u = s.get(User, int(user_id))
            if u is None:
                return
            for k, v in (data or {}).items():
                if k in allowed:
                    setattr(u, k, v)

    def update_password(self, user_id: int, password_hash: str) -> None:
        with session_scope() as s:
            u = s.get(User, int(user_id))
            if u is not None:
                u.password_hash = password_hash

    def touch_login(self, user_id: int) -> None:
        with session_scope() as s:
            u = s.get(User, int(user_id))
            if u is not None:
                u.last_login_at = datetime.utcnow()

    def delete_user_memory(self, user_id: int) -> dict:
        """清空个人记忆数据（`DELETE /users/me/memory`）：画像 + 胶囊 + 会话摘要。"""
        uid = int(user_id)
        with session_scope() as s:
            n_p = s.query(UserProfile).filter(UserProfile.user_id == uid).delete()
            n_c = s.query(UserInterestCapsule).filter(
                UserInterestCapsule.user_id == uid).delete()
            n_m = s.query(SessionMemory).filter(
                SessionMemory.user_id == uid).delete()
        return {"profile": int(n_p or 0), "capsules": int(n_c or 0),
                "memory_turns": int(n_m or 0)}

    # ================================================================ 动漫
    @staticmethod
    def _anime_dict(a: Anime, *, genre_ids: Optional[list[int]] = None,
                    genre_names: Optional[list[str]] = None,
                    threshold: int = _COLD_DEFAULT) -> dict:
        return {
            "anime_id": int(a.src_anime_id),        # ← 对外统一是数据集 id
            "src_anime_id": int(a.src_anime_id),
            "id": int(a.id),
            "title": a.title, "alt_title": a.alt_title,
            "type": a.type, "year": _i(a.year), "season": a.season,
            "score": _f(a.score), "episodes": _i(a.episodes),
            "mal_url": a.mal_url, "image_url": a.image_url,
            "is_sequel": int(a.is_sequel or 0),
            "summary": a.summary,
            "n_interactions": int(a.n_interactions or 0),
            "is_online": int(a.is_online or 0),
            "is_forbidden": int(a.is_forbidden or 0),
            "is_cold_start": int(a.n_interactions or 0) < int(threshold),
            "genres": list(genre_ids or []),
            "genre_names": list(genre_names or []),
        }

    def get_anime(self, anime_id: int) -> Optional[dict]:
        """`anime_id` = **数据集 animeID**（Agent 侧口径）。"""
        with session_scope() as s:
            a = s.scalars(select(Anime).where(
                Anime.src_anime_id == int(anime_id))).first()
            if a is None:
                return None
            gids = self._genre_ids_of(s, [int(a.id)]).get(int(a.id), [])
            names = self._genre_names(s, gids)
            return self._anime_dict(a, genre_ids=gids, genre_names=names,
                                    threshold=self.cold_start_threshold)

    @staticmethod
    def _genre_ids_of(s, anime_pks: Sequence[int]) -> dict[int, list[int]]:
        if not anime_pks:
            return {}
        rows = s.execute(
            select(AnimeGenre.anime_id, AnimeGenre.genre_id,
                   AnimeGenre.is_primary)
            .where(AnimeGenre.anime_id.in_(list(anime_pks)))
            .order_by(AnimeGenre.is_primary.desc(), AnimeGenre.genre_id)
        ).all()
        out: dict[int, list[int]] = {}
        for aid, gid, _p in rows:
            out.setdefault(int(aid), []).append(int(gid))
        return out

    @staticmethod
    def _genre_name_map(s, genre_ids: Sequence[int]) -> dict[int, str]:
        """`{genre_id: 中文名}`。**批量**取，供"一次查多部番"的调用方复用。"""
        ids = {int(g) for g in genre_ids}
        if not ids:
            return {}
        return {int(g.id): g.name_cn for g in s.scalars(
            select(Genre).where(Genre.id.in_(ids))).all()}

    @staticmethod
    def _genre_names(s, genre_ids: Sequence[int]) -> list[str]:
        """按**传入顺序**返回题材中文名（`_genre_ids_of` 已把主标签排在最前）。"""
        ids = [int(g) for g in genre_ids]
        if not ids:
            return []
        m = SqlGateway._genre_name_map(s, ids)
        return [m[g] for g in ids if g in m]

    def get_animes(self, anime_ids: Sequence[int]) -> dict[int, dict]:
        """`{src_anime_id: {...}}` —— **键是数据集 id**（与 `get_watch_records` 对齐）。

        返回 dict 而不是 list：调用方几乎总是按 id 取用
        （`animes.get(int(r["anime_id"]))`），给 list 会逼每个人自己建索引。

        ⚠️ 这里**必须**补 `genre_names`。它是 `/recommend/feed` 的唯一元信息来源
        （`agent_bridge.shape_items`）、也是 A4 的 `anime_facts` 与 A5 的解释
        信号来源。早期漏了这个参数 → 推荐列表每条都是 `genres: []`、卡片上
        不显示任何题材标签，而 `/animes` 列表页却正常（它走 `list_anime`，
        那条路径传了 `genre_names`）—— 两处不一致，很难看出是同一个表查出来的。
        """
        srcs = [int(x) for x in (anime_ids or [])]
        if not srcs:
            return {}
        with session_scope() as s:
            rows = s.scalars(select(Anime).where(Anime.src_anime_id.in_(srcs))).all()
            gmap = self._genre_ids_of(s, [int(a.id) for a in rows])
            # 一次取回全部题材名再分配；逐部查会变成 N+1，20 条的首页也明显
            name_map = self._genre_name_map(
                s, [g for gids in gmap.values() for g in gids])
            out: dict[int, dict] = {}
            for a in rows:
                gids = gmap.get(int(a.id), [])
                out[int(a.src_anime_id)] = self._anime_dict(
                    a, genre_ids=gids,
                    genre_names=[name_map[g] for g in gids if g in name_map],
                    threshold=self.cold_start_threshold)
            return out

    def search_anime(self, keyword: str, limit: int = 10) -> list[dict]:
        kw = str(keyword or "").strip()
        if not kw:
            return []
        like = f"%{kw}%"
        with session_scope() as s:
            rows = s.scalars(
                select(Anime).where(Anime.is_online == 1,
                                    or_(Anime.title.like(like),
                                        Anime.alt_title.like(like)))
                .order_by(Anime.n_interactions.desc()).limit(int(limit))).all()
            gmap = self._genre_ids_of(s, [int(a.id) for a in rows])
            out = []
            for a in rows:
                gids = gmap.get(int(a.id), [])
                out.append(self._anime_dict(a, genre_ids=gids,
                                            genre_names=self._genre_names(s, gids),
                                            threshold=self.cold_start_threshold))
            return out

    def anime_genres(self, anime_ids: Sequence[int]) -> dict[int, list[int]]:
        """`{src_anime_id: [genre_id, ...]}`（不区分主次，主标签见 `get_anime`）。"""
        srcs = [int(x) for x in (anime_ids or [])]
        if not srcs:
            return {}
        with session_scope() as s:
            rows = s.execute(
                select(Anime.src_anime_id, AnimeGenre.genre_id,
                       AnimeGenre.is_primary)
                .join(AnimeGenre, AnimeGenre.anime_id == Anime.id)
                .where(Anime.src_anime_id.in_(srcs))
                .order_by(AnimeGenre.is_primary.desc(), AnimeGenre.genre_id)).all()
            out: dict[int, list[int]] = {}
            for src, gid, _p in rows:
                out.setdefault(int(src), []).append(int(gid))
            return out

    def genre_map(self) -> dict[int, str]:
        with session_scope() as s:
            return {int(g.id): g.name_cn for g in
                    s.scalars(select(Genre).order_by(Genre.sort_order)).all()}

    def list_genres(self) -> list[dict]:
        with session_scope() as s:
            return [{"genre_id": int(g.id), "name_cn": g.name_cn,
                     "name_en": g.name_en, "description": g.description,
                     "sort_order": int(g.sort_order or 0)}
                    for g in s.scalars(select(Genre).order_by(Genre.sort_order)).all()]

    def list_anime(self, *, limit: int = 100, offset: int = 0,
                   genre_id: Optional[int] = None, min_year: Optional[int] = None,
                   cold_only: bool = False, online_only: bool = True,
                   keyword: str = "", order_by: str = "n_interactions") -> list[dict]:
        with session_scope() as s:
            stmt = select(Anime)
            if online_only:
                stmt = stmt.where(Anime.is_online == 1)
            if min_year is not None:
                stmt = stmt.where(Anime.year >= int(min_year))
            if cold_only:
                stmt = stmt.where(Anime.n_interactions < self.cold_start_threshold)
            if keyword:
                like = f"%{keyword}%"
                stmt = stmt.where(or_(Anime.title.like(like),
                                      Anime.alt_title.like(like)))
            if genre_id is not None:
                sub = select(AnimeGenre.anime_id).where(
                    AnimeGenre.genre_id == int(genre_id))
                stmt = stmt.where(Anime.id.in_(sub))
            order = {"n_interactions": Anime.n_interactions.desc(),
                     "year": Anime.year.desc(),
                     "score": Anime.score.desc(),
                     "id": Anime.id.asc()}.get(order_by, Anime.n_interactions.desc())
            rows = s.scalars(stmt.order_by(order, Anime.id.asc())
                             .limit(int(limit)).offset(int(offset))).all()
            gmap = self._genre_ids_of(s, [int(a.id) for a in rows])
            out = []
            for a in rows:
                gids = gmap.get(int(a.id), [])
                out.append(self._anime_dict(a, genre_ids=gids,
                                            genre_names=self._genre_names(s, gids),
                                            threshold=self.cold_start_threshold))
            return out

    def count_anime(self, *, genre_id: Optional[int] = None,
                    min_year: Optional[int] = None, cold_only: bool = False,
                    online_only: bool = True, keyword: str = "") -> int:
        with session_scope() as s:
            stmt = select(func.count()).select_from(Anime)
            if online_only:
                stmt = stmt.where(Anime.is_online == 1)
            if min_year is not None:
                stmt = stmt.where(Anime.year >= int(min_year))
            if cold_only:
                stmt = stmt.where(Anime.n_interactions < self.cold_start_threshold)
            if keyword:
                like = f"%{keyword}%"
                stmt = stmt.where(or_(Anime.title.like(like),
                                      Anime.alt_title.like(like)))
            if genre_id is not None:
                sub = select(AnimeGenre.anime_id).where(
                    AnimeGenre.genre_id == int(genre_id))
                stmt = stmt.where(Anime.id.in_(sub))
            return int(s.scalar(stmt) or 0)

    def popular_anime_ids(self, limit: int = 200,
                          genre_id: Optional[int] = None) -> list[int]:
        with session_scope() as s:
            stmt = select(Anime.src_anime_id).where(Anime.is_online == 1)
            if genre_id is not None:
                sub = select(AnimeGenre.anime_id).where(
                    AnimeGenre.genre_id == int(genre_id))
                stmt = stmt.where(Anime.id.in_(sub))
            rows = s.scalars(stmt.order_by(Anime.n_interactions.desc())
                             .limit(int(limit))).all()
            return [int(x) for x in rows]

    def similar_anime_ids(self, anime_id: int, top_k: int = 20) -> list[int]:
        """共现近邻（**返回数据集 animeID**）。

        实现是"共同观看"计数（`watch_record` 自连接）。⚠️ 这是**规模敏感**的查询：
        没有预计算表时，它对热门作品的代价是 O(观看人数 × 人均记录数)。
        所以这里对"源作品的观看者"取前 500 人做截断 —— 结果是**近似**的，
        生产环境应换成离线算好的 `item_similar` 表（届时本方法只改实现、
        不改契约）。返回空列表 → 调用方（A7）走同题材热门兜底并如实说明。
        """
        aid = int(anime_id)
        with session_scope() as s:
            src_pk = s.scalar(select(Anime.id).where(Anime.src_anime_id == aid))
            if src_pk is None:
                return []
            watchers = select(WatchRecord.user_id).where(
                WatchRecord.anime_id == int(src_pk),
                WatchRecord.status.in_(list(_SEQ_STATUS))).limit(500).subquery()
            rows = s.execute(
                select(WatchRecord.anime_id, func.count().label("c"))
                .where(WatchRecord.user_id.in_(select(watchers.c.user_id)),
                       WatchRecord.anime_id != int(src_pk),
                       WatchRecord.status.in_(list(_SEQ_STATUS)))
                .group_by(WatchRecord.anime_id)
                .order_by(func.count().desc())
                .limit(int(top_k))).all()
            pks = [int(r[0]) for r in rows]
            if not pks:
                return []
            m = {int(a.id): int(a.src_anime_id) for a in s.scalars(
                select(Anime).where(Anime.id.in_(pks))).all()}
            return [m[p] for p in pks if p in m]

    # ---- 动漫（管理）----
    def upsert_anime(self, data: dict) -> dict:
        src = _i(data.get("src_anime_id"))
        with session_scope() as s:
            a = s.scalars(select(Anime).where(
                Anime.src_anime_id == int(src))).first() if src else None
            if a is None:
                if not src:
                    raise ValueError("缺少 src_anime_id")
                a = Anime(src_anime_id=int(src),
                          title=str(data.get("title") or f"anime-{src}"))
                s.add(a)
            for k in ("title", "alt_title", "type", "year", "season", "score",
                      "episodes", "mal_url", "image_url", "summary", "is_online",
                      "is_forbidden", "n_interactions", "genre_raw", "is_sequel"):
                if k in data and data[k] is not None:
                    setattr(a, k, data[k])
            s.flush()
            gids = [_i(x) for x in (data.get("genre_ids") or []) if _i(x)]
            if gids:
                s.query(AnimeGenre).filter(
                    AnimeGenre.anime_id == int(a.id)).delete()
                for i, gid in enumerate(gids):
                    s.add(AnimeGenre(anime_id=int(a.id), genre_id=int(gid),
                                     is_primary=1 if i == 0 else 0,
                                     weight=Decimal("1.000")))
            return self._anime_dict(a, genre_ids=gids,
                                    genre_names=self._genre_names(s, gids),
                                    threshold=self.cold_start_threshold)

    def set_anime_offline(self, anime_id: int) -> bool:
        with session_scope() as s:
            a = s.scalars(select(Anime).where(
                Anime.src_anime_id == int(anime_id))).first()
            if a is None:
                return False
            a.is_online = 0
            return True

    # ================================================================ 行为
    def get_watch_records(self, user_id: int, *,
                          statuses: Optional[Sequence[int]] = None,
                          limit: Optional[int] = None,
                          order: str = "asc") -> list[dict]:
        """`[{anime_id(=数据集id), status, rating, progress, watched_at, updated_at}]`。

        `order="asc"` 按 `updated_at` 升序 —— 这是构造模型输入序列的口径
        （`docs/database-design.md` §3.6）。
        """
        with session_scope() as s:
            stmt = (select(WatchRecord, Anime.src_anime_id)
                    .join(Anime, Anime.id == WatchRecord.anime_id)
                    .where(WatchRecord.user_id == int(user_id)))
            if statuses:
                stmt = stmt.where(WatchRecord.status.in_([int(x) for x in statuses]))
            stmt = stmt.order_by(
                WatchRecord.updated_at.asc() if order == "asc"
                else WatchRecord.updated_at.desc())
            if limit is not None:
                stmt = stmt.limit(int(limit))
            out = []
            for wr, src in s.execute(stmt).all():
                out.append({
                    "id": int(wr.id), "anime_id": int(src),
                    "src_anime_id": int(src),
                    "status": int(wr.status), "rating": _i(wr.rating),
                    "progress": int(wr.progress or 0),
                    "watched_at": wr.watched_at.isoformat() if wr.watched_at else None,
                    "updated_at": wr.updated_at.isoformat() if wr.updated_at else None,
                    "tags": wr.tags,
                })
            return out

    def count_watch_records(self, user_id: int, *,
                            status: Optional[int] = None,
                            genre_id: Optional[int] = None) -> int:
        with session_scope() as s:
            stmt = select(func.count()).select_from(WatchRecord).where(
                WatchRecord.user_id == int(user_id))
            if status is not None:
                stmt = stmt.where(WatchRecord.status == int(status))
            if genre_id is not None:
                sub = select(AnimeGenre.anime_id).where(
                    AnimeGenre.genre_id == int(genre_id))
                stmt = stmt.where(WatchRecord.anime_id.in_(sub))
            return int(s.scalar(stmt) or 0)

    def upsert_watch_record(self, user_id: int, anime_id: int, data: dict) -> dict:
        """新增/更新追番记录。`anime_id` = 数据集 animeID。"""
        with session_scope() as s:
            pk = s.scalar(select(Anime.id).where(
                Anime.src_anime_id == int(anime_id)))
            if pk is None:
                raise LookupError(f"anime {anime_id} 不存在")
            wr = s.scalars(select(WatchRecord).where(
                WatchRecord.user_id == int(user_id),
                WatchRecord.anime_id == int(pk))).first()
            is_new = wr is None
            if wr is None:
                wr = WatchRecord(user_id=int(user_id), anime_id=int(pk))
                s.add(wr)
            if "status" in data and data["status"] is not None:
                wr.status = int(data["status"])
            if "rating" in data:
                wr.rating = _i(data["rating"])
            if "progress" in data and data["progress"] is not None:
                wr.progress = int(data["progress"])
            if "tags" in data:
                wr.tags = data["tags"]
            wd = _as_date(data.get("watched_at"))
            if wd is not None:
                wr.watched_at = wd
            if wr.status == 2 and wr.watched_at is None:
                wr.watched_at = date.today()
            s.flush()
            self._refresh_anime_popularity(s, [int(pk)])
            return {"record_id": int(wr.id), "is_new": bool(is_new),
                    "anime_id": int(anime_id), "status": int(wr.status)}

    def delete_watch_record(self, user_id: int, record_id: int) -> bool:
        with session_scope() as s:
            wr = s.get(WatchRecord, int(record_id))
            if wr is None or int(wr.user_id) != int(user_id):
                return False
            pk = int(wr.anime_id)
            s.delete(wr)
            s.flush()
            self._refresh_anime_popularity(s, [pk])
            return True

    def get_watch_record(self, user_id: int, record_id: int) -> Optional[dict]:
        with session_scope() as s:
            row = s.execute(
                select(WatchRecord, Anime.src_anime_id)
                .join(Anime, Anime.id == WatchRecord.anime_id)
                .where(WatchRecord.id == int(record_id),
                       WatchRecord.user_id == int(user_id))).first()
            if row is None:
                return None
            wr, src = row
            return {"id": int(wr.id), "anime_id": int(src),
                    "status": int(wr.status), "rating": _i(wr.rating),
                    "progress": int(wr.progress or 0),
                    "watched_at": wr.watched_at.isoformat() if wr.watched_at else None,
                    "updated_at": wr.updated_at.isoformat() if wr.updated_at else None}

    @staticmethod
    def _refresh_anime_popularity(s, anime_pks: Sequence[int]) -> None:
        """记录变更后同步 `anime.n_interactions`（冷启动判定的依据）。"""
        if not anime_pks:
            return
        rows = s.execute(
            select(WatchRecord.anime_id, func.count())
            .where(WatchRecord.anime_id.in_(list(anime_pks)),
                   WatchRecord.status.in_(list(_SEQ_STATUS)))
            .group_by(WatchRecord.anime_id)).all()
        counts = {int(a): int(c) for a, c in rows}
        for pk in anime_pks:
            a = s.get(Anime, int(pk))
            if a is not None:
                a.n_interactions = int(counts.get(int(pk), 0))

    def get_genre_affinity(self, user_id: int) -> list[dict]:
        """按题材聚合的观看统计（A1 的唯一数据源）。

        `weighted` = 近期加权：以"最近一次观看"为基准，`1/(1+距今天数/180)`
        作权重。**这个折算必须在 SQL 侧一次算完** —— 放 Agent 侧意味着
        A1 要自己发明一套权重公式，那就会出现第二份口径。
        """
        with session_scope() as s:
            rows = s.execute(
                select(AnimeGenre.genre_id,
                       func.count(func.distinct(WatchRecord.anime_id)),
                       func.avg(WatchRecord.rating),
                       func.max(WatchRecord.updated_at))
                .join(WatchRecord, WatchRecord.anime_id == AnimeGenre.anime_id)
                .where(WatchRecord.user_id == int(user_id),
                       WatchRecord.status.in_(list(_SEQ_STATUS)))
                .group_by(AnimeGenre.genre_id)).all()
            if not rows:
                return []
            now = datetime.utcnow()
            recent_cut = now - timedelta(days=180)
            recent = dict(s.execute(
                select(AnimeGenre.genre_id,
                       func.count(func.distinct(WatchRecord.anime_id)))
                .join(WatchRecord, WatchRecord.anime_id == AnimeGenre.anime_id)
                .where(WatchRecord.user_id == int(user_id),
                       WatchRecord.status.in_(list(_SEQ_STATUS)),
                       WatchRecord.updated_at >= recent_cut)
                .group_by(AnimeGenre.genre_id)).all())
            out = []
            for gid, cnt, avg_r, last in rows:
                n_recent = int(recent.get(gid, 0) or 0)
                age = (now - last).days if last else 3650
                # 近期加权：180 天内衰减慢、更久衰减明显（分母下限 1 天）
                w = 1.0 / (1.0 + max(age, 0) / 180.0)
                out.append({"genre_id": int(gid), "count": int(cnt),
                            "avg_rating": _f(avg_r),
                            "recent_count": n_recent,
                            "weighted": round(float(cnt) * w, 6)})
            out.sort(key=lambda r: (-r["weighted"], r["genre_id"]))
            return out

    def get_user_activity(self, user_id: int) -> dict:
        with session_scope() as s:
            rows = s.execute(
                select(WatchRecord.status, func.count(),
                       func.min(WatchRecord.updated_at),
                       func.max(WatchRecord.updated_at))
                .where(WatchRecord.user_id == int(user_id))
                .group_by(WatchRecord.status)).all()
            by_status = {int(st): int(c) for st, c, _, _ in rows}
            last = max((r[3] for r in rows if r[3]), default=None)
            first = min((r[2] for r in rows if r[2]), default=None)
            n30 = int(s.scalar(
                select(func.count()).select_from(WatchRecord).where(
                    WatchRecord.user_id == int(user_id),
                    WatchRecord.updated_at >= datetime.utcnow() - timedelta(days=30))
            ) or 0)
            return {
                "total": sum(by_status.values()),
                "plan": by_status.get(0, 0), "watching": by_status.get(1, 0),
                "done": by_status.get(2, 0), "dropped": by_status.get(3, 0),
                "last_active_at": last.isoformat() if last else None,
                "first_active_at": first.isoformat() if first else None,
                "n_recent_30d": n30,
            }

    def user_genre_time_series(self, user_id: int, *, granularity: str = "quarter",
                               start: Optional[str] = None,
                               end: Optional[str] = None,
                               limit: Optional[int] = None) -> list[dict]:
        """`[{period: "2025Q2", genre_id, count}]`（A6 漂移分析的时间序列）。

        ⚠️ 分档不能用 `strftime('%Y-%m')` 再在 Python 里改：SQLite 与 MySQL
        的日期函数不同名。这里**按数据库方言**生成 period：
        SQLite 用 `strftime`，MySQL 用 `DATE_FORMAT`。少写这个分支的话，
        测试（SQLite）能过、生产（MySQL）直接 500。

        `limit`：只保留**最近 N 个周期**（按 period 升序取末尾，跨题材共享
        同一份周期清单 —— 否则"每个题材各取 N 条"会把窗口拉长 N 倍）。
        `None` 表示不限。调用方：`GET /records/timeline`（前端时间轴，
        默认 12 个周期）；A6 漂移分析不传，拿全量。

        ⚠️ 这个参数 2026-09-14 前不存在，而 `records.py::timeline` 一直在传 ——
        未捕获 TypeError 直接 500。冒烟没覆盖到这条（它只测了 feed/chat/admin），
        是 `tests/test_api/test_auth_guard.py` 的全端点探针抓出来的。
        """
        fmt = {"month": "%Y-%m", "quarter": "%Y", "year": "%Y"}[
            granularity if granularity in ("month", "quarter", "year") else "month"]
        with session_scope() as s:
            dialect = _dialect_of(s)
            if dialect == "mysql":
                period = func.date_format(WatchRecord.updated_at, fmt)
            else:
                period = func.strftime(fmt, WatchRecord.updated_at)
            stmt = (select(period.label("period"), AnimeGenre.genre_id,
                           func.count(func.distinct(WatchRecord.anime_id)))
                    .join(WatchRecord, WatchRecord.anime_id == AnimeGenre.anime_id)
                    .where(WatchRecord.user_id == int(user_id),
                           WatchRecord.status.in_(list(_SEQ_STATUS)),
                           WatchRecord.updated_at.isnot(None))
                    .group_by(period, AnimeGenre.genre_id))
            if start:
                stmt = stmt.where(WatchRecord.updated_at >= _start_dt(start))
            if end:
                stmt = stmt.where(WatchRecord.updated_at <= _end_dt(end))
            out: list[dict] = []
            for period_v, gid, cnt in s.execute(stmt).all():
                out.append({"period": _quarterize(str(period_v), granularity),
                            "genre_id": int(gid), "count": int(cnt)})
            # 同一季度可能被拆成 3 个月 → 合并
            merged: dict[tuple[str, int], int] = {}
            for r in out:
                k = (r["period"], r["genre_id"])
                merged[k] = merged.get(k, 0) + r["count"]
            rows = [{"period": k[0], "genre_id": k[1], "count": v}
                    for k, v in merged.items()]
            rows.sort(key=lambda r: (r["period"], r["genre_id"]))
            if limit is not None and rows:
                periods = sorted({r["period"] for r in rows})
                keep = set(periods[-int(limit):])
                rows = [r for r in rows if r["period"] in keep]
            return rows

    # ================================================================ 画像
    def get_profile(self, user_id: int) -> Optional[dict]:
        with session_scope() as s:
            p = s.get(UserProfile, int(user_id))
            if p is None:
                return None
            return {
                "user_id": int(p.user_id),
                "top_genres": list(p.top_genres or []),
                "activity_level": int(p.activity_level or 0),
                "activity_label": {0: "low", 1: "medium", 2: "high"}.get(
                    int(p.activity_level or 0), "low"),
                "watch_intensity": _f(p.watch_intensity),
                "avg_rating_tendency": _f(p.avg_rating_tendency),
                "dropped_rate": _f(p.dropped_rate),
                "preferred_types": list(p.preferred_types or []),
                "preferred_era_start": _i(p.preferred_era_start),
                "preferred_era_end": _i(p.preferred_era_end),
                "total_records": int(p.total_records or 0),
                "summary_text": p.summary_text, "user_tag": p.user_tag,
                "version": int(p.version or 0),
                "computed_at": p.computed_at.isoformat() if p.computed_at else None,
            }

    def upsert_profile(self, user_id: int, data: dict) -> None:
        allowed = ("top_genres", "activity_level", "watch_intensity",
                   "avg_rating_tendency", "dropped_rate", "preferred_types",
                   "preferred_era_start", "preferred_era_end", "total_records",
                   "summary_text", "user_tag")
        with session_scope() as s:
            p = s.get(UserProfile, int(user_id))
            if p is None:
                p = UserProfile(user_id=int(user_id))
                s.add(p)
                p.version = 1
            else:
                p.version = int(p.version or 0) + 1
            for k in allowed:
                if k in data and data[k] is not None:
                    setattr(p, k, data[k])
            p.computed_at = _parse_dt(data.get("computed_at")) or datetime.utcnow()

    # ================================================================ 兴趣胶囊
    def get_capsules(self, user_id: int) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(select(UserInterestCapsule)
                             .where(UserInterestCapsule.user_id == int(user_id))
                             .order_by(UserInterestCapsule.capsule_id)).all()
            return [{"capsule_id": int(c.capsule_id), "label": c.label,
                     "strength": _f(c.strength), "vec_ref": c.vec_ref,
                     "model_ver": c.model_ver} for c in rows]

    def replace_capsules(self, user_id: int, capsules: Sequence[dict]) -> None:
        with session_scope() as s:
            s.query(UserInterestCapsule).filter(
                UserInterestCapsule.user_id == int(user_id)).delete()
            for i, c in enumerate(capsules or []):
                s.add(UserInterestCapsule(
                    user_id=int(user_id),
                    capsule_id=int(c.get("capsule_id", i)),
                    label=str(c.get("label") or ""),
                    strength=Decimal(str(c.get("strength") or 0)),
                    vec_ref=str(c.get("vec_ref") or ""),
                    model_ver=str(c.get("model_ver") or "multi_interest_v1"),
                    computed_at=datetime.utcnow()))

    # ================================================================ 推荐结果
    def save_recommendations(self, user_id: int, scene: int,
                             items: Sequence[dict], *, batch_id: str,
                             ttl_minutes: int = 1440) -> int:
        """**先删同 (user, scene, batch_id) 再插**，保证重跑幂等。

        ⚠️ 入参 `item["anime_id"]` 是**池内索引**（A4 的候选空间），
        而表列是 `anime.id` 外键 —— 这里必须翻译（用 `src_anime_id` 最稳，
        A4 的装配结果一定带它；缺了才回退到池内索引反查）。
        """
        uid, sc = int(user_id), int(scene)
        bid = str(batch_id)[:32]
        expire = datetime.utcnow() + timedelta(minutes=int(ttl_minutes))
        srcs = [int(it["src_anime_id"]) if it.get("src_anime_id")
                else (_ix.pool_to_src(it.get("anime_id")) or 0) for it in items]
        with session_scope() as s:
            pk_map = {int(a.src_anime_id): int(a.id) for a in s.scalars(
                select(Anime).where(Anime.src_anime_id.in_(
                    [x for x in srcs if x]))) .all()} if any(srcs) else {}
            s.query(RecommendResult).filter(
                RecommendResult.user_id == uid, RecommendResult.scene == sc,
                RecommendResult.batch_id == bid).delete(synchronize_session=False)
            n = 0
            for it, src in zip(items, srcs):
                pk = pk_map.get(int(src))
                if pk is None:
                    logger.debug("跳过不在 anime 表中的候选 src=%s", src)
                    continue
                s.add(RecommendResult(
                    user_id=uid, scene=sc, genre_id=_i(it.get("genre_id")),
                    anime_id=int(pk), rank_no=int(it.get("rank_no") or n + 1),
                    final_score=Decimal(str(it.get("final_score") or 0)),
                    behavior_score=_dec(it.get("behavior_score")),
                    content_score=_dec(it.get("content_score")),
                    interest_id=_i(it.get("interest_id")),
                    is_cold_start=1 if it.get("is_cold_start") else 0,
                    explain_signals=it.get("explain_signals") or None,
                    batch_id=bid, expire_at=expire))
                n += 1
            return n

    def get_recommendations(self, user_id: int, scene: int,
                            limit: int = 20) -> list[dict]:
        """读回**最新一批**推荐结果。**`anime_id` 翻译回池内索引**（A7 的卡片要它做重排）。

        ⚠️ 必须先定位批次，不能只 `order_by(rank_no)`
        ------------------------------------------
        同一个 `(user, scene)` 会**累积多批**（用户每刷新一次首页就是一批，
        各批 `batch_id` 不同、TTL 内共存）。只按 `rank_no` 排的话，
        不同批次的第 1 名会混在一起 —— 调用方拿到的"当前推荐列表"
        其实是若干次历史结果的**拼盘**，`rank_no` 还会出现重复的 1、1、1。

        实测后果（排查了很久）：`explain_one` 从这里取 `explain_signals`，
        命中的却是**旧口径写入的那一批**，于是每条的推荐理由都退化成最空的
        模板句；而同一时刻用 `/recommend/feed` 重算出来的理由是具体的 ——
        "接口之间对不上"，但两边都不报错。
        """
        with session_scope() as s:
            latest = s.execute(
                select(RecommendResult.batch_id)
                .where(RecommendResult.user_id == int(user_id),
                       RecommendResult.scene == int(scene),
                       RecommendResult.expire_at >= datetime.utcnow())
                .order_by(RecommendResult.created_at.desc(),
                          RecommendResult.id.desc())
                .limit(1)).scalar()
            if latest is None:
                return []
            rows = s.execute(
                select(RecommendResult, Anime.src_anime_id, Anime.title,
                       Anime.year, Anime.image_url)
                .join(Anime, Anime.id == RecommendResult.anime_id)
                .where(RecommendResult.user_id == int(user_id),
                       RecommendResult.scene == int(scene),
                       RecommendResult.batch_id == str(latest),
                       RecommendResult.expire_at >= datetime.utcnow())
                .order_by(RecommendResult.rank_no).limit(int(limit))).all()
            out = []
            for rr, src, title, year, img in rows:
                out.append({
                    "anime_id": _ix.src_to_pool(int(src)),
                    "src_anime_id": int(src),
                    "rank_no": int(rr.rank_no),
                    "final_score": _f(rr.final_score),
                    "behavior_score": _f(rr.behavior_score),
                    "content_score": _f(rr.content_score),
                    "interest_id": _i(rr.interest_id),
                    "is_cold_start": bool(rr.is_cold_start),
                    "title": title, "year": _i(year), "image_url": img,
                    "explain_signals": rr.explain_signals or {},
                    "batch_id": rr.batch_id,
                })
            return out

    def invalidate_recommendations(self, user_id: int, scene: Optional[int] = None) -> int:
        """记录变更后清掉旧推荐（§7 的缓存失效配套：DB 侧也要作废）。"""
        with session_scope() as s:
            q = s.query(RecommendResult).filter(
                RecommendResult.user_id == int(user_id))
            if scene is not None:
                q = q.filter(RecommendResult.scene == int(scene))
            return int(q.delete(synchronize_session=False) or 0)

    # ================================================================ 解释
    def save_explain(self, user_id: int, anime_id: int, data: dict) -> None:
        """`anime_id` = **池内索引**（A5 的口径）→ 翻译成 `anime.id`。"""
        src = _ix.pool_to_src(anime_id)
        if src is None:
            logger.debug("解释落库跳过：池内索引 %s 无对应 src", anime_id)
            return
        with session_scope() as s:
            pk = s.scalar(select(Anime.id).where(Anime.src_anime_id == int(src)))
            if pk is None:
                return
            style = str(data.get("style") or "concise")
            row = s.scalars(select(RecommendExplain).where(
                RecommendExplain.user_id == int(user_id),
                RecommendExplain.anime_id == int(pk),
                RecommendExplain.style == style)).first()
            if row is None:
                row = RecommendExplain(user_id=int(user_id), anime_id=int(pk),
                                       style=style)
                s.add(row)
            row.reason = str(data.get("reason") or "")[:255]
            row.core_items = list(data.get("core_items") or [])
            row.match_percent = int(data.get("match_percent") or 0)
            row.matched_genres = data.get("matched_genres") or None
            row.source = int(data.get("source") if data.get("source") is not None else 0)
            row.prompt_ver = data.get("prompt_ver")
            row.llm_model = data.get("llm_model")
            row.tokens_used = int(data.get("tokens_used") or 0)

    def get_explain(self, user_id: int, anime_id: int,
                    style: str = "concise") -> Optional[dict]:
        src = _ix.pool_to_src(anime_id)
        if src is None:
            return None
        with session_scope() as s:
            row = s.scalars(
                select(RecommendExplain)
                .join(Anime, Anime.id == RecommendExplain.anime_id)
                .where(RecommendExplain.user_id == int(user_id),
                       Anime.src_anime_id == int(src),
                       RecommendExplain.style == str(style))).first()
            if row is None:
                return None
            return {"reason": row.reason, "core_items": list(row.core_items or []),
                    "match_percent": int(row.match_percent or 0),
                    "matched_genres": list(row.matched_genres or []),
                    "style": row.style,
                    "source": "llm" if int(row.source) == 0 else "template",
                    "prompt_ver": row.prompt_ver, "llm_model": row.llm_model,
                    "tokens_used": int(row.tokens_used or 0)}

    # ================================================================ 冷启动池
    def list_cold_start_pool(self, *, limit: int = 100,
                             active_only: bool = True) -> list[dict]:
        """返回 `src_anime_id`（Agent 需要，`ColdStartPool` 主键是 `anime.id`）。"""
        with session_scope() as s:
            stmt = (select(ColdStartPool, Anime.src_anime_id, Anime.title,
                           Anime.year)
                    .join(Anime, Anime.id == ColdStartPool.anime_id))
            if active_only:
                stmt = stmt.where(ColdStartPool.is_active == 1)
            rows = s.execute(stmt.order_by(
                ColdStartPool.exposure_cnt.asc()).limit(int(limit))).all()
            return [{"anime_id": int(src), "src_anime_id": int(src),
                     "title": title, "year": _i(year),
                     "season": p.season,
                     "n_interactions": int(p.n_interactions or 0),
                     "content_vec_ready": int(p.content_vec_ready or 0),
                     "exposure_cnt": int(p.exposure_cnt or 0),
                     "click_cnt": int(p.click_cnt or 0),
                     "fav_cnt": int(p.fav_cnt or 0),
                     "ctr": (float(p.click_cnt) / p.exposure_cnt
                             if p.exposure_cnt else 0.0),
                     "is_active": int(p.is_active or 0)}
                    for p, src, title, year in rows]

    def upsert_cold_start_pool(self, anime_id: int, data: dict) -> None:
        """`anime_id` = **数据集 animeID**（A8 传的就是它）。"""
        with session_scope() as s:
            pk = s.scalar(select(Anime.id).where(
                Anime.src_anime_id == int(anime_id)))
            if pk is None:
                logger.debug("冷启动池跳过：anime %s 不在表中", anime_id)
                return
            row = s.get(ColdStartPool, int(pk))
            if row is None:
                row = ColdStartPool(anime_id=int(pk),
                                    season=str(data.get("season") or "unknown"))
                s.add(row)
            for k in ("season", "n_interactions", "content_vec_ready",
                      "faiss_idx", "exposure_cnt", "click_cnt", "fav_cnt",
                      "is_active"):
                if k in data and data[k] is not None:
                    setattr(row, k, data[k])

    def cold_start_stats(self) -> dict:
        """管理后台冷启动看板（§3.6 `/admin/cold-start`）。"""
        with session_scope() as s:
            rows = s.execute(
                select(ColdStartPool.season, func.count(),
                       func.sum(ColdStartPool.exposure_cnt),
                       func.sum(ColdStartPool.click_cnt),
                       func.sum(ColdStartPool.fav_cnt))
                .where(ColdStartPool.is_active == 1)
                .group_by(ColdStartPool.season)
                .order_by(ColdStartPool.season.desc())).all()
            by_season, exp, clk, fav = [], 0, 0, 0
            for season, cnt, e, c, f in rows:
                e, c, f = int(e or 0), int(c or 0), int(f or 0)
                exp += e; clk += c; fav += f
                by_season.append({"season": season, "count": int(cnt),
                                  "exposure": e, "click": c,
                                  "ctr": round(c / e, 4) if e else 0.0})
            top = s.execute(
                select(Anime.src_anime_id, Anime.title,
                       ColdStartPool.exposure_cnt, ColdStartPool.click_cnt)
                .join(Anime, Anime.id == ColdStartPool.anime_id)
                .where(ColdStartPool.is_active == 1)
                .order_by(ColdStartPool.click_cnt.desc()).limit(10)).all()
            pool_size = int(s.scalar(
                select(func.count()).select_from(ColdStartPool)
                .where(ColdStartPool.is_active == 1)) or 0)
            return {
                "pool_size": pool_size,
                "funnel": {"exposure": exp, "click": clk, "fav": fav},
                "ctr": round(clk / exp, 4) if exp else 0.0,
                "fav_rate": round(fav / exp, 4) if exp else 0.0,
                "by_season": by_season,
                "top_items": [{"anime_id": int(src), "title": t,
                               "exposure": int(e or 0), "click": int(c or 0),
                               "ctr": round(int(c or 0) / int(e), 4) if e else 0.0}
                              for src, t, e, c in top],
            }

    # ================================================================ 反馈
    def save_feedback(self, user_id: int, anime_id: int, scene: int,
                      action: int, *, position: Optional[int] = None,
                      reason: Optional[str] = None,
                      batch_id: Optional[str] = None) -> bool:
        with session_scope() as s:
            pk = s.scalar(select(Anime.id).where(
                Anime.src_anime_id == int(anime_id)))
            if pk is None:
                return False
            s.add(UserFeedback(user_id=int(user_id), anime_id=int(pk),
                               scene=int(scene), action=int(action),
                               position=_i(position), reason=reason,
                               batch_id=(str(batch_id)[:32] if batch_id else None)))
            # 冷启动池的曝光/点击计数（CTR 看板的分母）
            if int(scene) == 2:
                cs = s.get(ColdStartPool, int(pk))
                if cs is not None:
                    if int(action) == 0:
                        cs.exposure_cnt = int(cs.exposure_cnt or 0) + 1
                    elif int(action) == 1:
                        cs.click_cnt = int(cs.click_cnt or 0) + 1
                    elif int(action) == 2:
                        cs.fav_cnt = int(cs.fav_cnt or 0) + 1
            return True

    def feedback_counts(self, *, start: Optional[str] = None,
                        end: Optional[str] = None,
                        scene: Optional[int] = None) -> dict:
        """`{action(int): count}` —— A9 的 `strategy.online_metrics` 按 int 取用。"""
        with session_scope() as s:
            stmt = select(UserFeedback.action, func.count()).group_by(
                UserFeedback.action)
            if scene is not None:
                stmt = stmt.where(UserFeedback.scene == int(scene))
            if start:
                stmt = stmt.where(UserFeedback.created_at >= _start_dt(start))
            if end:
                stmt = stmt.where(UserFeedback.created_at <= _end_dt(end))
            return {int(a): int(c) for a, c in s.execute(stmt).all()}

    def feedback_daily(self, *, days: int = 7, scene: Optional[int] = None) -> list[dict]:
        """按天聚合 CTR/CVR（管理后台 trend 曲线）。"""
        with session_scope() as s:
            dialect = _dialect_of(s)
            day = (func.date_format(UserFeedback.created_at, "%Y-%m-%d")
                   if dialect == "mysql"
                   else func.strftime("%Y-%m-%d", UserFeedback.created_at))
            stmt = (select(day.label("d"), UserFeedback.action, func.count())
                    .where(UserFeedback.created_at >=
                           datetime.utcnow() - timedelta(days=int(days))))
            if scene is not None:
                stmt = stmt.where(UserFeedback.scene == int(scene))
            agg: dict[str, dict[int, int]] = {}
            for d, a, c in s.execute(stmt.group_by(day, UserFeedback.action)).all():
                agg.setdefault(str(d), {})[int(a)] = int(c)
            out = []
            for d in sorted(agg):
                c = agg[d]
                expose, click, fav = c.get(0, 0), c.get(1, 0), c.get(2, 0)
                out.append({"date": d,
                            "exposure": expose, "click": click, "fav": fav,
                            "ctr": round(click / expose, 4) if expose else 0.0,
                            "cvr": round(fav / click, 4) if click else 0.0})
            return out

    # ================================================================ 指标
    def save_metrics(self, rows: Sequence[dict]) -> int:
        with session_scope() as s:
            n = 0
            for r in rows or []:
                md = _as_date(r.get("metric_date")) or date.today()
                mv = _dec(r.get("metric_value"))
                if mv is None:
                    continue
                scene = int(r.get("scene") or 0)
                mtype = str(r.get("metric_type") or "")
                mver = r.get("model_ver")
                cold = int(r.get("is_cold_start") or 0)
                gid = _i(r.get("genre_id"))
                row = s.scalars(select(MetricSnapshot).where(
                    MetricSnapshot.metric_date == md,
                    MetricSnapshot.scene == scene,
                    MetricSnapshot.metric_type == mtype,
                    MetricSnapshot.model_ver.is_(None) if mver is None
                    else MetricSnapshot.model_ver == mver,
                    MetricSnapshot.is_cold_start == cold,
                    MetricSnapshot.genre_id.is_(None) if gid is None
                    else MetricSnapshot.genre_id == gid)).first()
                if row is None:
                    row = MetricSnapshot(metric_date=md, scene=scene,
                                         metric_type=mtype, model_ver=mver,
                                         is_cold_start=cold, genre_id=gid)
                    s.add(row)
                row.metric_value = mv
                row.sample_size = int(r.get("sample_size") or 0)
                row.extra = r.get("extra") or None
                n += 1
            return n

    def query_metrics(self, *, metric_type: str, days: int = 30,
                      scene: Optional[int] = None) -> list[dict]:
        with session_scope() as s:
            stmt = select(MetricSnapshot).where(
                MetricSnapshot.metric_type == str(metric_type),
                MetricSnapshot.metric_date >=
                date.today() - timedelta(days=int(days)))
            if scene is not None:
                stmt = stmt.where(MetricSnapshot.scene == int(scene))
            rows = s.scalars(stmt.order_by(MetricSnapshot.metric_date)).all()
            return [{"metric_date": r.metric_date.isoformat(),
                     "metric_type": r.metric_type,
                     "metric_value": _f(r.metric_value),
                     "sample_size": int(r.sample_size or 0),
                     "scene": int(r.scene), "model_ver": r.model_ver,
                     "is_cold_start": int(r.is_cold_start or 0),
                     "extra": r.extra} for r in rows]

    def latest_model_ver(self) -> Optional[str]:
        with session_scope() as s:
            return s.scalar(select(MetricSnapshot.model_ver).where(
                MetricSnapshot.model_ver.isnot(None),
                MetricSnapshot.metric_type.in_(("hr10", "ndcg10"))
            ).order_by(MetricSnapshot.metric_date.desc()).limit(1))

    def save_data_quality(self, rows: Sequence[dict]) -> int:
        with session_scope() as s:
            n = 0
            for r in rows or []:
                s.add(DataQualityLog(
                    check_date=_as_date(r.get("check_date")) or date.today(),
                    check_type=str(r.get("check_type") or "missing"),
                    target_table=str(r.get("target_table") or "anime"),
                    total_rows=int(r.get("total_rows") or 0),
                    problem_rows=int(r.get("problem_rows") or 0),
                    problem_rate=_dec(r.get("problem_rate")) or Decimal("0"),
                    detail=r.get("detail") or None,
                    alert_level=int(r.get("alert_level") or 0)))
                n += 1
            return n

    # ================================================================ 配置
    def get_config(self, key: str) -> Optional[Any]:
        with session_scope() as s:
            row = s.scalars(select(SystemConfig).where(
                SystemConfig.config_key == str(key))).first()
            return row.typed_value() if row else None

    def set_config(self, key: str, value: Any, *, value_type: str = "string",
                   updated_by: Optional[int] = None) -> None:
        if isinstance(value, bool):
            value_type, raw = "bool", "1" if value else "0"
        elif isinstance(value, int):
            value_type, raw = "int", str(value)
        elif isinstance(value, float):
            value_type, raw = "float", str(value)
        elif isinstance(value, (dict, list)):
            value_type, raw = "json", json.dumps(value, ensure_ascii=False)
        else:
            raw = str(value)
        with session_scope() as s:
            row = s.scalars(select(SystemConfig).where(
                SystemConfig.config_key == str(key))).first()
            if row is None:
                row = SystemConfig(config_key=str(key), config_value=raw)
                s.add(row)
            row.config_value = raw
            row.value_type = value_type
            row.updated_by = _i(updated_by)

    def list_configs(self) -> list[dict]:
        with session_scope() as s:
            return [{"config_key": c.config_key, "config_value": c.typed_value(),
                     "value_type": c.value_type, "scope": c.scope,
                     "description": c.description,
                     "is_hot_reload": int(c.is_hot_reload or 0)}
                    for c in s.scalars(select(SystemConfig).order_by(
                        SystemConfig.config_key)).all()]

    # ================================================================ 会话记忆
    def append_memory_turn(self, user_id: int, session_id: str, turn_no: int,
                           role: str, content: str, *, tokens: int = 0,
                           shown_anime_ids: Optional[Sequence[int]] = None,
                           extracted: Optional[dict] = None) -> None:
        with session_scope() as s:
            s.add(SessionMemory(
                user_id=int(user_id), session_id=str(session_id),
                turn_no=int(turn_no), role=str(role), content=str(content),
                tokens=int(tokens or 0),
                shown_anime_ids=list(shown_anime_ids or []),
                extracted=dict(extracted or {}) or None))

    def get_memory_turns(self, user_id: int, session_id: str,
                         limit: int = 20) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(select(SessionMemory).where(
                SessionMemory.user_id == int(user_id),
                SessionMemory.session_id == str(session_id))
                .order_by(SessionMemory.turn_no).limit(int(limit))).all()
            return [{"turn_no": int(m.turn_no), "role": m.role,
                     "content": m.content,
                     "shown_anime_ids": list(m.shown_anime_ids or []),
                     "tokens": int(m.tokens or 0),
                     "created_at": m.created_at.isoformat() if m.created_at else None}
                    for m in rows]

    def latest_session_id(self, user_id: int) -> Optional[str]:
        with session_scope() as s:
            return s.scalar(select(SessionMemory.session_id).where(
                SessionMemory.user_id == int(user_id))
                .order_by(SessionMemory.created_at.desc()).limit(1))

    # ================================================================ Agent 状态
    def list_agent_states(self) -> list[dict]:
        with session_scope() as s:
            return [{"agent_id": a.agent_id, "agent_name": a.agent_name,
                     "status": int(a.status or 0),
                     "health_score": _f(a.health_score),
                     "success_cnt": int(a.success_cnt or 0),
                     "fail_cnt": int(a.fail_cnt or 0),
                     "timeout_cnt": int(a.timeout_cnt or 0),
                     "degrade_cnt": int(a.degrade_cnt or 0),
                     "avg_elapsed_ms": int(a.avg_elapsed_ms or 0),
                     "p95_elapsed_ms": int(a.p95_elapsed_ms or 0),
                     "version": a.version,
                     "last_active_at": (a.last_active_at.isoformat()
                                        if a.last_active_at else None)}
                    for a in s.scalars(select(AgentState).order_by(
                        AgentState.agent_id)).all()]

    def upsert_agent_state(self, agent_id: str, data: dict) -> None:
        with session_scope() as s:
            row = s.scalars(select(AgentState).where(
                AgentState.agent_id == str(agent_id))).first()
            if row is None:
                row = AgentState(agent_id=str(agent_id),
                                 agent_name=str(data.get("agent_name")
                                                or str(agent_id).lower()))
                s.add(row)
            for k in ("agent_name", "status", "health_score", "success_cnt",
                      "fail_cnt", "timeout_cnt", "degrade_cnt",
                      "avg_elapsed_ms", "p95_elapsed_ms", "version", "config"):
                if k in data and data[k] is not None:
                    setattr(row, k, data[k])

    def trace_stats(self, *, since: Optional[str] = None) -> list[dict]:
        """按 Agent 聚合调用链：**一次查询算完**（A9 的健康分与瓶颈定位都用它）。"""
        with session_scope() as s:
            stmt = select(
                AgentTrace.to_agent,
                func.count(),
                # ⚠️ 必须用 `sqlalchemy.case`，**不是** `func.case`：
                # `func.case` 会生成 `case(1, 0, 0)` 这种函数调用语法，
                # SQLite/MySQL 都不认（CASE 需要 WHEN/THEN 子句）。
                func.sum(case((AgentTrace.status == 0, 1), else_=0)),
                func.sum(case((AgentTrace.status == 1, 1), else_=0)),
                func.sum(case((AgentTrace.status == 2, 1), else_=0)),
                func.sum(case((AgentTrace.status == 3, 1), else_=0)),
                func.avg(AgentTrace.elapsed_ms),
                func.max(AgentTrace.elapsed_ms),
                func.sum(AgentTrace.tokens_used),
            )
            if since:
                stmt = stmt.where(AgentTrace.created_at >= _start_dt(since))
            stmt = stmt.group_by(AgentTrace.to_agent)
            out = []
            for aid, calls, suc, deg, err, tmo, avg_ms, max_ms, toks in s.execute(stmt).all():
                calls = int(calls or 0)
                out.append({
                    "agent_id": aid, "to_agent": aid,
                    "calls": calls, "success": int(suc or 0),
                    "degrade": int(deg or 0), "fail": int(err or 0),
                    "timeout": int(tmo or 0),
                    "avg_elapsed_ms": int(avg_ms or 0),
                    "p95_elapsed_ms": int(max_ms or 0),   # 近似：用最大值代表尾部
                    "total_ms": int((avg_ms or 0) * calls),
                    "tokens_used": int(toks or 0),
                    "success_rate": round(int(suc or 0) / calls, 4) if calls else None,
                })
            out.sort(key=lambda r: r["agent_id"])
            return out

    def get_trace(self, trace_id: str) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(select(AgentTrace).where(
                AgentTrace.trace_id == str(trace_id))
                .order_by(AgentTrace.created_at, AgentTrace.id)).all()
            return [{"msg_id": t.msg_id, "parent_msg_id": t.parent_msg_id,
                     "from_agent": t.from_agent, "to_agent": t.to_agent,
                     "action": t.action, "priority": t.priority,
                     "status": int(t.status or 0), "error_code": _i(t.error_code),
                     "elapsed_ms": int(t.elapsed_ms or 0),
                     "cache_hit": bool(t.cache_hit),
                     "tokens_used": int(t.tokens_used or 0),
                     "call_path": list(t.call_path or []),
                     "detail": t.detail or {},
                     "created_at": t.created_at.isoformat() if t.created_at else None}
                    for t in rows]

    # ================================================================ 健康检查
    def ping(self) -> bool:
        try:
            with session_scope() as s:
                s.execute(select(func.count()).select_from(Genre))
            return True
        except Exception as exc:
            logger.warning("数据库 ping 失败：%s", exc)
            return False

    def counts(self) -> dict:
        with session_scope() as s:
            return {
                "user": int(s.scalar(select(func.count()).select_from(User)) or 0),
                "anime": int(s.scalar(select(func.count()).select_from(Anime)) or 0),
                "genre": int(s.scalar(select(func.count()).select_from(Genre)) or 0),
                "watch_record": int(s.scalar(
                    select(func.count()).select_from(WatchRecord)) or 0),
                "recommend_result": int(s.scalar(
                    select(func.count()).select_from(RecommendResult)) or 0),
                "agent_trace": int(s.scalar(
                    select(func.count()).select_from(AgentTrace)) or 0),
            }


# ---------------------------------------------------------------- 小工具
def _dialect_of(session: Any) -> str:
    """取方言名。

    ⚠️ `Session.bind` 在 SQLAlchemy 2.0 已被移除（1.x 才有），
    正确入口是 `Session.get_bind()`。用错会在运行期抛 `AttributeError`，
    而且只在**日期分档**这条冷路径上触发（`user_genre_time_series` /
    `feedback_daily`），单测不覆盖就很容易漏到生产。
    """
    try:
        return str(session.get_bind().dialect.name)
    except Exception:
        return "sqlite"


def _dec(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


def _parse_dt(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str) and v:
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def _start_dt(v: Any) -> datetime:
    """把 `2026-09-10` / `2026-09` / `2025Q2` 统一成区间起点。

    ⚠️ 季度串（`2025Q2`）不能丢给 `fromisoformat` —— 会抛。A6 的
    `start`/`end` 参数允许季度写法，所以这里必须显式解析。
    """
    s = str(v or "").strip()
    if len(s) >= 6 and s[4:5].upper() == "Q" and s[:4].isdigit():
        y, q = int(s[:4]), int(s[5:6])
        return datetime(y, 1 + 3 * max(1, min(4, q)) - 3, 1)
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s[:len(fmt.replace("%Y", "2000"))], fmt)
        except ValueError:
            continue
    return datetime(1970, 1, 1)


def _end_dt(v: Any) -> datetime:
    s = str(v or "").strip()
    if len(s) >= 6 and s[4:5].upper() == "Q" and s[:4].isdigit():
        y, q = int(s[:4]), int(s[5:6])
        m = 1 + 3 * max(1, min(4, q))
        return (datetime(y + 1, 1, 1) if m > 12
                else datetime(y, m, 1) - timedelta(seconds=1))
    for fmt, tail in (("%Y-%m-%d", "%Y-%m-%d 23:59:59"),
                      ("%Y-%m", "%Y-%m"), ("%Y", "%Y")):
        try:
            base = datetime.strptime(s[:len(fmt.replace("%Y", "2000"))], fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d":
            return base.replace(hour=23, minute=59, second=59)
        if fmt == "%Y-%m":
            nxt = base.replace(day=28) + timedelta(days=4)
            return nxt.replace(day=1) - timedelta(seconds=1)
        return base.replace(month=12, day=31, hour=23, minute=59, second=59)
    return datetime.utcnow()


def _quarterize(period: str, granularity: str) -> str:
    """`2025-05` → `2025Q2`（quarter 档）。其它档原样返回。"""
    if granularity != "quarter":
        return period
    try:
        y, m = period.split("-")[:2]
        return f"{int(y)}Q{(int(m) - 1) // 3 + 1}"
    except Exception:
        return period

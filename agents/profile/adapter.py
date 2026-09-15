# -*- coding: utf-8 -*-
"""A1 的数据取数适配层 —— Agent 里**唯一**碰数据源的地方。

本文件只做两件事：把网关的返回**规整成 strategy 能吃的普通结构**；
把算好的画像**写回**。任何"怎么算"的判断都不在这里（在 `strategy.py`）。

为什么把取数与计算分开
--------------------
* 测试：`strategy` 的测试不需要网关；`adapter` 的测试只需要一个假网关。
* 复核：`docs/result-analysis.md` 要求"每个数字都能追溯"，取数 SQL 与计算式
  分开写在两个文件里，复核者读一遍就知道数字怎么来的。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from agents.common.data import require_gateway
from agents.common.memory import get_history
from agents.profile.config import ProfileConfig

logger = logging.getLogger(__name__)

__all__ = ["collect_facts", "load_cached_profile", "record_count", "recent_items",
           "save_profile", "recent_history"]

# watch_record.status（与 server/db/models/record.py 对齐；AGENTS 层不 import server，
# 故这里按协议把常量抄一份，用测试锁定一致性）
STATUS_PLAN, STATUS_WATCHING, STATUS_DONE, STATUS_DROPPED = 0, 1, 2, 3


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _recent_cutoff(records: list[dict], window_days: int) -> Optional[str]:
    """按数据里的**最后一个时间戳**回推窗口起点。

    刻意不用 `now()`：离线数据集的时间戳可能是几年前的，用 now() 会让
    所有用户的 recent_count = 0，trend 全为 -1（看起来像全站集体转向）。
    """
    stamps = [r.get("updated_at") for r in records if r.get("updated_at")]
    if not stamps:
        return None
    try:
        latest = max(datetime.fromisoformat(str(s)) for s in stamps)
    except (TypeError, ValueError):
        return None
    from datetime import timedelta
    return (latest - timedelta(days=window_days)).isoformat(timespec="seconds")


def record_count(user_id: int) -> int:
    """该用户的实际追番记录数（单条 COUNT，画像新鲜度校验用）。

    为什么需要它（2026-09-15 真实事故）：注册时会落一行 `total_records=0`
    的空画像，`_usable` 判定"零记录零题材"是合法新用户状态 → 直接复用，
    用户之后加再多番，A1 也永远返回这行过期画像（且无任何报错）。
    记录数对不上 = 画像过期，必须重算。
    """
    return int(require_gateway().count_watch_records(user_id) or 0)


def collect_facts(user_id: int, cfg: ProfileConfig) -> dict:
    """把网关数据装配成 `strategy.compute_profile` 的输入。"""
    gw = require_gateway()

    affinity = gw.get_genre_affinity(user_id) or []
    activity = gw.get_user_activity(user_id) or {}

    # 已看/在看/想看都要，因为 preferred_types / era 依赖"看过什么"
    records = gw.get_watch_records(user_id, order="asc") or []
    animes = gw.get_animes([r["anime_id"] for r in records]) if records else {}
    genre_map = gw.genre_map() or {}

    records_meta: list[dict] = []
    for r in records:
        meta = animes.get(int(r["anime_id"])) or {}
        records_meta.append({
            "anime_id": int(r["anime_id"]),
            "status": int(r.get("status", STATUS_WATCHING)),
            "rating": r.get("rating"),
            "title": meta.get("title"),
            "type": meta.get("type"),
            "year": meta.get("year"),
            "updated_at": r.get("updated_at"),
        })

    # 题材名回填（gateway 的 affinity 可能只给 id）
    for a in affinity:
        gid = int(a.get("genre_id", 0))
        if not a.get("genre"):
            a["genre"] = genre_map.get(gid, f"genre_{gid}")

    recent = [m for m in records_meta if m.get("status") in (STATUS_WATCHING, STATUS_DONE)]
    recent.sort(key=lambda m: str(m.get("updated_at") or ""), reverse=True)

    return {
        "user_id": user_id,
        "affinity": affinity,
        "activity": activity,
        "records_meta": records_meta,
        "recent_items": [
            {"anime_id": m["anime_id"], "title": m.get("title"),
             "status": m.get("status"), "rating": m.get("rating"),
             "updated_at": m.get("updated_at")}
            for m in recent[:cfg.recent_items_k]
        ],
    }


def recent_items(user_id: int, cfg: ProfileConfig) -> list[dict]:
    """只要"最近在看/已看"的那几条（`recent_items`）。

    为什么单独抽一个函数，而不是复用 `collect_facts`
    ---------------------------------------------
    `user_profile` 表**没有** `recent_items` 列（它是短期信息，不适合入库），
    所以凡是命中"已落库画像"的请求，返回的 `recent_items` 都是空的。
    而它是 A4 `explain_signals.top_attn_items` 的唯一来源 ——
    空掉之后最好的那句解释模板
    「你近期观看了《A》《B》等热血战斗番」**永远走不到**，
    全部退化成最空的「根据你近期的观看偏好」，且没有任何报错。

    `collect_facts` 一次要跑 5 个查询（题材亲和、活跃度、记录、番剧、题材表），
    为了补一个小字段而全跑一遍不划算；这里只查"记录 + 番剧"两个。
    """
    gw = require_gateway()
    try:
        records = gw.get_watch_records(
            user_id, statuses=[STATUS_WATCHING, STATUS_DONE], order="desc") or []
    except Exception as exc:          # 取数失败不该让整个画像请求失败
        logger.warning("A1 补取 recent_items 失败（置空）：%s", exc)
        return []
    if not records:
        return []
    rows = records[: cfg.recent_items_k]
    try:
        animes = gw.get_animes([r["anime_id"] for r in rows]) or {}
    except Exception:
        animes = {}
    return [
        {"anime_id": int(r["anime_id"]),
         "src_anime_id": int(r["anime_id"]),
         "title": (animes.get(int(r["anime_id"])) or {}).get("title"),
         "status": int(r.get("status", STATUS_WATCHING)),
         "rating": r.get("rating"),
         "updated_at": r.get("updated_at")}
        for r in rows
    ]


def load_cached_profile(user_id: int) -> Optional[dict]:
    """读已落库的画像（含 `computed_at`）。"""
    gw = require_gateway()
    row = gw.get_profile(user_id)
    if not row:
        return None
    return dict(row)


def save_profile(user_id: int, profile: dict) -> None:
    """写回 `user_profile`（`version` 自增由网关负责）。"""
    gw = require_gateway()
    gw.upsert_profile(user_id, {**profile, "computed_at": _iso_now()})


async def recent_history(user_id: int, session_id: str, limit: int = 10) -> list[dict]:
    """短期会话偏好（Redis）。缓存不可用 → 空列表。"""
    return await get_history(user_id, session_id, limit=limit)

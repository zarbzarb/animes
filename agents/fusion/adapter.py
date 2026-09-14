# -*- coding: utf-8 -*-
"""A4 的取数与落库适配层（排序算法在 `strategy.py`）。"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "anime_facts",
    "batch_id_of",
    "cache_key",
    "load_cached_result",
    "save_result",
]


def cache_key(user_id: int, scene: int, genre_id: Optional[int] = None) -> str:
    """`rec:{uid}:{scene}` / `rec:{uid}:{scene}:{gid}`。

    为什么 Agent 侧自己拼串而不 import `server/core/cache.py`
    ------------------------------------------------------
    `agents/` 不许反向依赖 `server/`（`docs/dev-conventions.md` §3.2 的 R3，
    由 `scripts/check_imports.py` 把关）。

    为什么键里必须有 `scene` / `genre_id`
    -----------------------------------
    同一个用户在**综合(0)/分题材(1)/新番(2)/对话(3)** 四个场景下的结果
    完全不同；分题材还要再按题材分。少了这两维就会互相串缓存。

    ⚠️ 与 server 侧的一致性（这里曾经是个真 bug）
    ------------------------------------------
    `server/core/cache.py` 原来的 `rec_key(uid)` 返回 `rec:{uid}`（不带 scene），
    而失效逻辑直接拿它去 `delete` —— **一个真实键都删不中**，表现为
    "用户新增追番后推荐 24 小时内不变"，且不报任何错。
    现在 server 侧改为 `rec_scope_prefix(uid)` = `rec:{uid}:` + `delete_prefix`，
    `tests/test_agents/test_cache_keys.py` 会断言本函数生成的**每一个键**
    都落在那个前缀下（这是失效能生效的充分条件）。
    """
    base = f"rec:{int(user_id)}:{int(scene)}"
    return base if genre_id is None else f"{base}:{int(genre_id)}"


def anime_facts(gateway: Any, indices: Sequence[int]) -> dict[int, dict]:
    """池内索引 → `{src_anime_id, title, year, type, n_interactions, genres[]}`。

    `n_interactions` 是**冷启动权重的唯一依据**（协议 §A4 第 ③ 步），
    所以即使某条元数据不全，也要把它带出来。
    """
    from agents.recall.adapter import load_item_index
    inv = load_item_index()["index_to_src"]
    src_map = {int(i): inv.get(int(i)) for i in indices}
    src_ids = [s for s in src_map.values() if s]
    try:
        by_src = gateway.get_animes(src_ids) if src_ids else {}
        genres_src = gateway.anime_genres(src_ids) if src_ids else {}
    except Exception as exc:
        logger.warning("A4 取动漫元数据失败：%s", exc)
        by_src, genres_src = {}, {}

    out: dict[int, dict] = {}
    for idx, src in src_map.items():
        meta = dict(by_src.get(src) or {})
        meta["src_anime_id"] = src
        meta["genres"] = list(genres_src.get(src, []) or meta.get("genres") or [])
        out[idx] = meta
    return out


async def load_cached_result(user_id: int, scene: int, genre_id: Optional[int],
                             cache: Any) -> Optional[list[dict]]:
    try:
        data = await cache.get_json(cache_key(user_id, scene, genre_id))
    except Exception as exc:
        logger.debug("A4 读结果缓存失败：%s", exc)
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("items")
    return items if isinstance(items, list) else None


async def save_result(user_id: int, scene: int, genre_id: Optional[int],
                      items: list[dict], *, batch_id: str, cache: Any,
                      gateway: Any, ttl: int, ttl_minutes: int,
                      persist: bool = True) -> None:
    """写缓存 + 落库。**两者独立失败**：缓存挂了不该连带丢 DB 写入。"""
    payload = {"user_id": user_id, "scene": scene, "batch_id": batch_id,
               "items": items}
    try:
        await cache.set_json(cache_key(user_id, scene, genre_id), payload, ttl=ttl)
    except Exception as exc:
        logger.warning("A4 写结果缓存失败（已忽略）：%s", exc)

    if not persist or gateway is None:
        return
    try:
        gateway.save_recommendations(user_id, scene, items, batch_id=batch_id,
                                     ttl_minutes=ttl_minutes)
    except Exception as exc:
        logger.warning("A4 落库失败（已忽略）：%s", exc)


def batch_id_of(user_id: int, scene: int, provided: Optional[str] = None) -> str:
    """批次号：离线任务传入固定值，在线请求生成一个短随机值。

    长度 32 是表结构约束（`recommend_result.batch_id VARCHAR(32)`）。
    """
    if provided:
        return str(provided)[:32]
    import os
    return f"b{int(user_id)}_{int(scene)}_{os.urandom(4).hex()}"[:32]

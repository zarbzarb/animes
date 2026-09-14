# -*- coding: utf-8 -*-
"""动漫模块（api-specification.md §3.2 前的只读部分）。

`/animes` 与 `/animes/{id}` **不需要登录**（协议 §1.3 明确只读接口免鉴权），
但带上 token 时会附带 `is_favorited` 之类的个性化标记。这里保持最小：
免鉴权可浏览，登录后额外返回该用户的追番状态。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from server.core.exceptions import not_found
from server.core.response import make_router, Enveloped, paginate
from server.deps import get_gw, optional_user

logger = logging.getLogger(__name__)
router = make_router(tags=["动漫"])


@router.get("/genres", summary="12 类题材列表")
async def list_genres() :
    gw = get_gw()
    rows = gw.list_genres()
    return Enveloped(data={"list": rows, "n": len(rows)})


@router.get("/animes", summary="动漫列表（搜索/筛选/分页）")
async def list_animes(
    keyword: str = Query("", description="标题模糊匹配"),
    genre_id: int | None = Query(None, ge=1, le=12),
    min_year: int | None = Query(None, ge=1900, le=2100),
    cold_only: bool = Query(False, description="只看新番（交互数低于阈值）"),
    order_by: str = Query("n_interactions",
                          pattern="^(n_interactions|year|score|id)$"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    user: dict | None = Depends(optional_user),
) :
    gw = get_gw()
    kw = keyword.strip()
    rows = gw.list_anime(limit=size, offset=(page - 1) * size, genre_id=genre_id,
                         min_year=min_year, cold_only=cold_only,
                         keyword=kw, order_by=order_by)
    total = gw.count_anime(genre_id=genre_id, min_year=min_year,
                           cold_only=cold_only, keyword=kw)
    data = paginate([_brief(r) for r in rows], page, size, total)
    return Enveloped(data=data)


@router.get("/animes/{anime_id}", summary="动漫详情")
async def anime_detail(anime_id: int,
                       user: dict | None = Depends(optional_user)) :
    gw = get_gw()
    a = gw.get_anime(int(anime_id))
    if a is None or int(a.get("is_forbidden", 0)) == 1:
        raise not_found(f"动漫 {anime_id} 不存在")
    data = {
        "id": a["id"],
        "src_anime_id": a["src_anime_id"],
        "title": a["title"],
        "alt_title": a.get("alt_title"),
        "type": a.get("type"),
        "year": a.get("year"),
        "score": a.get("score"),
        "episodes": a.get("episodes"),
        "image_url": a.get("image_url"),
        "mal_url": a.get("mal_url"),
        "summary": a.get("summary"),
        "genres": [{"genre_id": gid, "genre": name} for gid, name
                   in zip(a.get("genres") or [], a.get("genre_names") or [])],
        "n_interactions": a.get("n_interactions"),
        "is_cold_start": bool(a.get("is_cold_start")),
    }
    if user:
        data["my_record"] = _my_record(int(user["id"]), int(anime_id))
    return Enveloped(data=data)


@router.get("/animes/{anime_id}/similar", summary="相似动漫")
async def similar(anime_id: int, size: int = Query(12, ge=1, le=50)) :
    """相似度来自 `watch_record` 的**共现统计**（`gateway.similar_anime_ids`）。

    不新写一套向量相似度：M2.7 已证明该协议的共现统计极强（hr@10 0.96），
    另起一套只会制造"两个入口推荐的相似番不一样"的怪象。

    共现为空时（冷门番 / 数据稀疏）降级为**同题材热门**并如实标注 `method`，
    而不是返回空列表 —— 前端一个"相似推荐"区域空着比推几部同题材更差。
    """
    gw = get_gw()
    anime = gw.get_anime(int(anime_id))
    if anime is None:
        raise not_found(f"动漫 {anime_id} 不存在")

    src_ids = gw.similar_anime_ids(int(anime_id), top_k=int(size)) or []
    method = "itemcf_cooccurrence"
    if not src_ids:
        gids = list(anime.get("genres") or [])
        if gids:
            rows = gw.list_anime(limit=int(size), genre_id=int(gids[0]),
                                 order_by="n_interactions")
            src_ids = [int(r["src_anime_id"]) for r in rows
                       if int(r["src_anime_id"]) != int(anime_id)]
            method = "genre_popularity"

    metas = gw.get_animes(src_ids)
    items = [_brief(metas[s]) for s in src_ids if s in metas]
    return Enveloped(data={"anime_id": int(anime_id), "list": items,
                           "n": len(items), "method": method})


# ---------------------------------------------------------------- 私有


def _brief(a: dict) -> dict:
    return {
        "id": a.get("id"),
        "src_anime_id": a.get("src_anime_id"),
        "title": a.get("title"),
        "type": a.get("type"),
        "year": a.get("year"),
        "score": a.get("score"),
        "episodes": a.get("episodes"),
        "image_url": a.get("image_url"),
        "genres": list(a.get("genre_names") or []),
        "n_interactions": a.get("n_interactions"),
        "is_cold_start": bool(a.get("is_cold_start")),
    }


def _my_record(user_id: int, anime_id: int) -> dict | None:
    try:
        for r in get_gw().get_watch_records(user_id):
            if int(r.get("src_anime_id") or 0) == int(anime_id):
                return {"status": r.get("status"), "rating": r.get("rating"),
                        "progress": r.get("progress"),
                        "watched_at": r.get("watched_at")}
    except Exception as exc:
        logger.debug("读取追番状态失败：%s", exc)
    return None

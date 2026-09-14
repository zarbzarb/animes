# -*- coding: utf-8 -*-
"""推荐模块（api-specification.md §3.3）—— 系统的核心。

5 个接口里 4 个走 A0 编排，1 个（feedback）只写库。
**注意**：这里不实现任何排序/召回逻辑，只做「入参校验 + 形态转换 + 落库」。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel, Field

from server.core.exceptions import invalid_param, not_found
from server.core.response import make_router, Enveloped
from server.deps import current_user, get_gw, rate_limit
from server.services.agent_bridge import explain_one, feed

logger = logging.getLogger(__name__)
router = make_router(prefix="/recommend", tags=["推荐"])


def _feed_envelope(data: dict, meta: dict) :
    """把 `meta.degraded` 体现在 `message` 上：前端据此显示"部分能力降级"。

    协议 §6 明确「降级不是错误」：HTTP 仍 200，`code` 仍 0/60401。
    """
    deg = meta.get("degraded") or []
    if deg:
        return Enveloped(data={"items": data.get("items") or [], "meta": meta},
                         code=60401,
                         message=f"部分能力已降级：{'、'.join(str(d) for d in deg)}")
    return Enveloped(data=data | {"meta": meta}, code=0, message="ok")


@router.get("/feed", summary="综合推荐 Top20",
            dependencies=[Depends(rate_limit("recommend_feed"))])
async def recommend_feed(
    size: int = Query(20, ge=1, le=20),
    with_explain: bool = Query(True, description="是否附带解释理由"),
    refresh: bool = Query(False, description="强制绕过缓存重算"),
    query: str = Query("", description="可选的自然语言（用于意图识别）"),
    user: dict = Depends(current_user),
) :
    data, meta = await feed(int(user["id"]), size=size, with_explain=with_explain,
                            intent_hint="RECOMMEND_FEED",
                            raw_query=query.strip() or None,
                            params={"_refresh": refresh})
    return _feed_envelope(data, meta)


@router.get("/by-genre", summary="分类推荐",
            dependencies=[Depends(rate_limit("recommend_feed"))])
async def by_genre(
    genre_id: int = Query(..., ge=1, le=12),
    size: int = Query(20, ge=1, le=20),
    user: dict = Depends(current_user),
) :
    gw = get_gw()
    names = {int(g["genre_id"]): g["name_cn"] for g in (gw.list_genres() or [])}
    if int(genre_id) not in names:
        raise invalid_param(f"genre_id 必须在 1-12 之间，收到 {genre_id}")
    data, meta = await feed(int(user["id"]), size=size, intent_hint="RECOMMEND_BY_GENRE",
                            params={"genre_id": int(genre_id)})
    data["genre_id"] = int(genre_id)
    data["genre"] = names[int(genre_id)]
    return _feed_envelope(data, meta)


@router.get("/new-anime", summary="新番专属推荐",
            dependencies=[Depends(rate_limit("recommend_feed"))])
async def new_anime(
    season: str = Query("", description="季度键，如 2026Q3；留空=不限定"),
    size: int = Query(20, ge=1, le=20),
    user: dict = Depends(current_user),
) :
    data, meta = await feed(int(user["id"]), size=size,
                            intent_hint="RECOMMEND_NEW_ANIME",
                            params={"season": season} if season else {})
    data["season"] = season or "all"
    data["cold_start_note"] = (
        "本数据集无交互数 <10 的番（H4：主协议无真冷启动物品），"
        "新番由 cold_start_pool 表定义，其候选带 is_cold_start=True 标记")
    return _feed_envelope(data, meta)


@router.get("/{anime_id}/explain", summary="单条推荐解释",
            dependencies=[Depends(rate_limit("recommend_explain"))])
async def explain(
    anime_id: int = Path(..., ge=1),
    style: str = Query("concise", pattern="^(concise|detailed|casual)$"),
    user: dict = Depends(current_user),
) :
    data, meta = await explain_one(int(user["id"]), int(anime_id), style=style)
    deg = meta.get("degraded")
    return Enveloped(data=data | {"meta": meta},
                     code=60401 if deg else 0,
                     message="解释已降级为模板文案" if deg else "ok")


class FeedbackIn(BaseModel):
    anime_id: int = Field(..., ge=1, description="数据集 animeID")
    action: str = Field(..., pattern="^(exposure|click|fav|dislike|like)$")
    scene: int = Field(0, ge=0, le=3)
    reason: str | None = None


# 字符串 action → `user_feedback.action` 的整型编码（见 models/recommend.py）
_FB_CODE = {"exposure": 0, "click": 1, "fav": 2, "dislike": 3, "like": 4}


@router.post("/feedback", summary="推荐反馈（曝光/点击/不感兴趣）")
async def feedback(body: FeedbackIn, user: dict = Depends(current_user)) :
    """写 `user_feedback`。**不等待**增量重排（协议 §3.2 的异步约定）。

    协议要求「同 (user, anime, action) 短窗口去重」：前端一次滑动会连发多次
    `exposure`，不去重会把曝光数刷成噪声，直接毁掉 CTR 的分母。
    """
    gw = get_gw()
    uid = int(user["id"])
    if gw.get_anime(body.anime_id) is None:
        raise not_found(f"动漫 {body.anime_id} 不存在")

    from server.core.cache import get_cache, rate_key

    cache = get_cache()
    dedup_key = rate_key(f"fb:{body.action}", f"{uid}:{body.anime_id}")
    try:
        if not await cache.incr(dedup_key, 300):    # 5 分钟窗口
            return Enveloped(data={"recorded": False, "deduped": True},
                             code=0, message="重复反馈已忽略")
    except Exception as exc:
        logger.debug("反馈去重不可用（放行）：%s", exc)

    ok = gw.save_feedback(uid, int(body.anime_id), int(body.scene),
                          _FB_CODE[body.action], reason=body.reason)
    if not ok:
        raise not_found(f"动漫 {body.anime_id} 不存在")
    return Enveloped(data={"recorded": True, "action": body.action}, code=0,
                     message="已记录")

# -*- coding: utf-8 -*-
"""追番管理（api-specification.md §3.2）。

`POST /records` 是**唯一会改变推荐结果**的用户接口：写入后协议要求
「异步触发增量重排（A1 → A2 → A4）并让 `rec:{uid}` 缓存失效，响应不等待重排」。
本实现用 FastAPI 的 `BackgroundTasks` 满足"不等待"，并把是否已投递如实返回
（`recompute_scheduled`）—— 不等待不等于假装成功。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from pydantic import BaseModel, Field

from server.core.exceptions import not_found
from server.core.response import make_router, Enveloped, paginate
from server.deps import current_user, get_gw, is_visible_anime

logger = logging.getLogger(__name__)
router = make_router(prefix="/records", tags=["追番"])

STATUS_LABEL = {0: "想看", 1: "在看", 2: "已看", 3: "弃番"}


class RecordIn(BaseModel):
    anime_id: int = Field(..., ge=1, description="数据集 animeID")
    status: int = Field(1, ge=0, le=3)
    rating: int | None = Field(None, ge=1, le=10)
    progress: int | None = Field(None, ge=0)
    watched_at: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    tags: list[str] | None = None
    review: str | None = Field(None, max_length=500, description="文字评价，空串=清除")


class RecordPatch(BaseModel):
    status: int | None = Field(None, ge=0, le=3)
    rating: int | None = Field(None, ge=1, le=10)
    progress: int | None = Field(None, ge=0)
    watched_at: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    tags: list[str] | None = None
    review: str | None = Field(None, max_length=500, description="文字评价，空串=清除")


def _decorate(r: dict) -> dict:
    """补上 `anime{}` 与 `status_label`（接口规范 §3.2 的响应形状）。"""
    gw = get_gw()
    a = gw.get_anime(int(r.get("src_anime_id") or r.get("anime_id") or 0)) or {}
    return {
        "id": r.get("id"),
        "anime": {
            "id": a.get("id"), "src_anime_id": a.get("src_anime_id"),
            "title": a.get("title"), "title_cn": a.get("title_cn"),
            "type": a.get("type"),
            "year": a.get("year"), "score": a.get("score"),
            "episodes": a.get("episodes"), "image_url": a.get("image_url"),
            "genres": list(a.get("genre_names") or []),
        },
        "status": r.get("status"),
        "status_label": STATUS_LABEL.get(int(r.get("status") or 0), "未知"),
        "rating": r.get("rating"), "progress": r.get("progress"),
        "watched_at": r.get("watched_at"), "updated_at": r.get("updated_at"),
        "tags": r.get("tags"), "review": r.get("review"),
    }


@router.get("", summary="我的追番列表")
async def list_records(
    status: int | None = Query(None, ge=0, le=3),
    genre_id: int | None = Query(None, ge=1, le=12),
    sort: str = Query("updated_desc",
                      pattern="^(updated_desc|rating_desc|watched_desc)$"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    user: dict = Depends(current_user),
) :
    gw = get_gw()
    uid = int(user["id"])
    order = "asc" if sort == "watched_desc" else "desc"
    rows = gw.get_watch_records(uid, statuses=[status] if status is not None else None,
                                order=order)
    if sort == "rating_desc":
        rows.sort(key=lambda r: -(int(r.get("rating") or 0)))
    if genre_id is not None:
        allowed = {int(a["src_anime_id"]) for a in
                   gw.list_anime(limit=5000, genre_id=int(genre_id))}
        rows = [r for r in rows if int(r.get("src_anime_id") or 0) in allowed]

    total = len(rows)
    page_rows = rows[(page - 1) * size: page * size]
    return Enveloped(data=paginate([_decorate(r) for r in page_rows], page, size, total))


@router.get("/stats", summary="追番统计概览")
async def stats(user: dict = Depends(current_user)) :
    gw = get_gw()
    uid = int(user["id"])
    rows = gw.get_watch_records(uid)
    by_status = {STATUS_LABEL[s]: 0 for s in STATUS_LABEL}
    ratings = []
    for r in rows:
        by_status[STATUS_LABEL.get(int(r.get("status") or 0), "未知")] += 1
        if r.get("rating"):
            ratings.append(int(r["rating"]))
    return Enveloped(data={
        "total": len(rows),
        "by_status": by_status,
        "n_rated": len(ratings),
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
    })


@router.get("/timeline", summary="观看时间轴")
async def timeline(
    granularity: str = Query("month", pattern="^(month|quarter|year)$"),
    limit: int = Query(12, ge=1, le=60),
    user: dict = Depends(current_user),
) :
    """按周期聚合的观看时间轴（A6 的同一份数据源，这里只做展示形态）。"""
    gw = get_gw()
    uid = int(user["id"])
    rows = gw.user_genre_time_series(uid, granularity=granularity, limit=limit) or []
    gmap = {int(g["genre_id"]): g["name_cn"] for g in (gw.list_genres() or [])}

    buckets: dict[str, dict] = {}
    for r in rows:
        period = str(r.get("period"))
        b = buckets.setdefault(period, {"period": period, "count": 0, "genres": {},
                                        "items": []})
        cnt = int(r.get("count") or 0)
        b["count"] += cnt
        name = gmap.get(int(r.get("genre_id") or 0))
        if name and cnt:
            b["genres"][name] = b["genres"].get(name, 0) + cnt

    out = []
    for period in sorted(buckets, reverse=True)[:limit]:
        b = buckets[period]
        b["genres"] = [{"genre": k, "count": v} for k, v in
                       sorted(b["genres"].items(), key=lambda kv: -kv[1])]
        out.append(b)
    return Enveloped(data={"timeline": out, "granularity": granularity})


@router.post("", summary="新增/更新追番记录")
async def upsert_record(body: RecordIn, background: BackgroundTasks,
                        user: dict = Depends(current_user)) :
    gw = get_gw()
    uid = int(user["id"])
    if not is_visible_anime(gw.get_anime(body.anime_id)):
        raise not_found(f"动漫 {body.anime_id} 不存在或已下架")

    data = body.model_dump(exclude_none=True)
    data.pop("anime_id", None)
    res = gw.upsert_watch_record(uid, int(body.anime_id), data)

    # 缓存失效：必须在**投递重排之前**做，否则重排写完后又有请求把旧值写回缓存
    await _invalidate(uid)
    background.add_task(_recompute, uid)
    return Enveloped(data={
        "record_id": res.get("record_id"),
        "is_new": bool(res.get("is_new")),
        "recompute_scheduled": True,
        "recompute_eta_ms": 800,
    }, code=0, message="已保存")


@router.put("/{record_id}", summary="修改追番记录")
async def update_record(record_id: int, body: RecordPatch, background: BackgroundTasks,
                        user: dict = Depends(current_user)) :
    gw = get_gw()
    uid = int(user["id"])
    row = gw.get_watch_record(uid, int(record_id))
    if row is None:
        raise not_found(f"记录 {record_id} 不存在")
    src = int(row.get("src_anime_id") or row.get("anime_id") or 0)
    data = body.model_dump(exclude_none=True)
    res = gw.upsert_watch_record(uid, src, data)
    await _invalidate(uid)
    background.add_task(_recompute, uid)
    return Enveloped(data={"record_id": res.get("record_id"), "is_new": False,
                           "recompute_scheduled": True, "recompute_eta_ms": 800},
                     code=0, message="已更新")


@router.delete("/{record_id}", summary="删除追番记录")
async def delete_record(record_id: int, background: BackgroundTasks,
                        user: dict = Depends(current_user)) :
    gw = get_gw()
    uid = int(user["id"])
    if not gw.delete_watch_record(uid, int(record_id)):
        raise not_found(f"记录 {record_id} 不存在")
    await _invalidate(uid)
    background.add_task(_recompute, uid)
    return Enveloped(data={"deleted": True, "recompute_scheduled": True},
                     code=0, message="已删除")


# ---------------------------------------------------------------- 内部


async def _invalidate(user_id: int) -> None:
    """清推荐缓存 + 作废 DB 侧推荐结果（两处都要，缺一会出现"缓存清了但读到旧结果"）。

    ⚠️ 缓存必须**按前缀**清：推荐结果是 `rec:{uid}:{scene}[:{gid}]`，
    一个用户在 4 个场景 × 12 个题材上各有一份。这里只有 `user_id`，
    用精确键删等于什么都没删（旧实现就是这样，且不报错）。
    """
    from server.core.cache import get_cache, rec_scope_prefix

    gw = get_gw()
    try:
        gw.invalidate_recommendations(int(user_id))
    except Exception as exc:
        logger.warning("作废推荐结果失败：%s", exc)
    try:
        cache = get_cache()
        await cache.delete_prefix(rec_scope_prefix(int(user_id)))
        # 画像缓存也要清：A1 的键是 `profile:{uid}`（不在 rec: 前缀下），
        # 不清的话记录变更后 1h 内 A1 还在返回旧画像（2026-09-15 事故）。
        from server.core.cache import profile_key
        await cache.delete(profile_key(int(user_id)))
    except Exception as exc:
        logger.warning("清推荐/画像缓存失败：%s", exc)


async def _recompute(user_id: int) -> None:
    """后台增量重排：A1（画像失效）→ A2 → A4。

    复用离线任务的实现，保证线上线下同一套流水线（否则反馈归因会错位）。
    """
    from server.tasks.offline import run_offline_recommend

    try:
        r = await run_offline_recommend(user_id=int(user_id), limit=1)
        logger.info("增量重排完成 user=%s ok=%s avg_ms=%s", user_id, r.get("ok"),
                    r.get("avg_ms"))
    except Exception as exc:
        logger.warning("增量重排失败 user=%s：%s", user_id, exc)
